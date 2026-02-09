/*
 * nano_vm - NanoVM bytecode executor
 *
 * Loads an .nvm file and executes it via the VM.
 *
 * Usage: nano_vm [--daemon] <file.nvm>
 *
 * With --daemon: sends the .nvm blob to the nano_vmd daemon for execution
 *                (lazy-launches the daemon if not running).
 * Without:       executes directly in-process (original behavior).
 */

#include "vm.h"
#include "vm_ffi.h"
#include "vmd_client.h"
#include "../nanoisa/nvm_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* Global flag for co-process FFI isolation */
static bool g_isolate_ffi = false;

static uint8_t *read_file(const char *path, uint32_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open '%s'\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || size > 100 * 1024 * 1024) { /* 100 MB limit */
        fprintf(stderr, "Error: Invalid file size (%ld bytes)\n", size);
        fclose(f);
        return NULL;
    }

    uint8_t *data = malloc((size_t)size);
    if (!data) {
        fprintf(stderr, "Error: Out of memory\n");
        fclose(f);
        return NULL;
    }

    size_t read = fread(data, 1, (size_t)size, f);
    fclose(f);

    if ((long)read != size) {
        fprintf(stderr, "Error: Short read (%zu of %ld bytes)\n", read, size);
        free(data);
        return NULL;
    }

    *out_size = (uint32_t)size;
    return data;
}

static int run_standalone(const char *path) {
    uint32_t file_size = 0;
    uint8_t *data = read_file(path, &file_size);
    if (!data) return 1;

    NvmModule *module = nvm_deserialize(data, file_size);
    free(data);

    if (!module) {
        fprintf(stderr, "Error: Failed to load '%s' (invalid .nvm format)\n", path);
        return 1;
    }

    /* Preload FFI modules if the .nvm has imports */
    if (module->import_count > 0) {
        vm_ffi_init();

        /* Load modules referenced by name in the import table */
        for (uint32_t i = 0; i < module->import_count; i++) {
            const char *mod_name = nvm_get_string(module,
                                                   module->imports[i].module_name_idx);
            if (mod_name && mod_name[0] != '\0') {
                vm_ffi_load_module(mod_name);
            }
        }

        /* For imports with empty module names (bare extern fn declarations),
         * try loading well-known standard modules by function name prefix. */
        static const struct { const char *prefix; const char *module; } known_modules[] = {
            {"path_",    "std/fs"},
            {"fs_",      "std/fs"},
            {"file_",    "std/fs"},
            {"dir_",     "std/fs"},
            {"regex_",   "std/regex"},
            {"process_", "std/process"},
            {"json_",    "std/json"},
            {"bstr_",    "std/bstring"},
            {NULL, NULL}
        };

        for (uint32_t i = 0; i < module->import_count; i++) {
            const char *fn_name = nvm_get_string(module,
                                                  module->imports[i].function_name_idx);
            const char *mod_name = nvm_get_string(module,
                                                   module->imports[i].module_name_idx);
            if (fn_name && (!mod_name || mod_name[0] == '\0')) {
                for (int k = 0; known_modules[k].prefix; k++) {
                    if (strncmp(fn_name, known_modules[k].prefix,
                               strlen(known_modules[k].prefix)) == 0) {
                        vm_ffi_load_module(known_modules[k].module);
                        break;
                    }
                }
            }
        }
    }

    VmState vm;
    vm_init(&vm, module);

    /* Enable co-process FFI isolation if requested.
     * The cop is launched lazily on first extern call, not here. */
    if (g_isolate_ffi) {
        vm.isolate_ffi = true;
    }

    VmResult result = vm_execute(&vm);

    int exit_code = 0;
    if (result != VM_OK) {
        fprintf(stderr, "Runtime error: %s\n", vm_error_string(result));
        if (vm.error_msg[0]) {
            fprintf(stderr, "  %s\n", vm.error_msg);
        }
        exit_code = 1;
    }

    /* Stop co-process if it was launched */
    if (vm.cop_pid > 0) {
        vm_ffi_cop_stop(&vm);
    }

    vm_destroy(&vm);
    vm_ffi_shutdown();
    nvm_module_free(module);

    return exit_code;
}

static int run_daemon(const char *path) {
    uint32_t file_size = 0;
    uint8_t *data = read_file(path, &file_size);
    if (!data) return 1;

    VmdClient *client = vmd_connect(5000);  /* 5 second timeout */
    if (!client) {
        fprintf(stderr, "Error: Cannot connect to nano_vmd daemon\n");
        free(data);
        return 1;
    }

    int exit_code = vmd_execute(client, data, file_size);
    free(data);

    vmd_disconnect(client);

    if (exit_code < 0) {
        fprintf(stderr, "Error: Communication error with daemon\n");
        return 1;
    }

    return exit_code;
}

int main(int argc, char *argv[]) {
    g_argc = argc;
    g_argv = argv;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [--daemon] <file.nvm>\n", argv[0]);
        return 1;
    }

    bool daemon_mode = false;
    const char *nvm_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--daemon") == 0 || strcmp(argv[i], "-d") == 0) {
            daemon_mode = true;
        } else if (strcmp(argv[i], "--isolate-ffi") == 0 || strcmp(argv[i], "--cop") == 0) {
            g_isolate_ffi = true;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        } else {
            nvm_path = argv[i];
        }
    }

    if (!nvm_path) {
        fprintf(stderr, "Error: No .nvm file specified\n");
        return 1;
    }

    if (daemon_mode) {
        return run_daemon(nvm_path);
    } else {
        return run_standalone(nvm_path);
    }
}
