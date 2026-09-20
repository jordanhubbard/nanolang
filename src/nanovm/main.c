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

#include "../nanoisa/file_public.h"
#include "../nanoisa/file_cyclic_public.h"
#include "../nanoisa/file_cli.h"
#include "vm.h"
#include "vm_ffi.h"
#include "vmd_client.h"
#include "../nanoisa/verifier.h"
#include "../nanoisa/nvm_format.h"
#include "../../modules/nanoisa/nanoisa.h"
#include "../runtime/shadow_runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* Global flag for co-process FFI isolation */
static bool g_isolate_ffi = false;

/* Global flag for debug mode (--debug or DEBUG env var) */
static bool g_debug_mode = false;
static const char *g_profile_path = NULL;
/* Iterations of the loaded module per process. Benchmarks use this to push
 * process startup below the execution time they are trying to measure. */
static uint32_t g_repeat = 1;

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

/* Explicit bounded File route: input-file I/O is separate from service grants.
 * Generic module/VM/FFI readiness never sees this invocation. */
static bool file_instruction_limit(const char *text, uint64_t *out) {
    if (!text || !*text) return false;
    uint64_t value = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (NVM_FILE_CYCLIC_FUEL_MAX - digit) / 10) return false;
        value = value * 10 + digit;
    }
    *out = value;
    return true;
}
static int run_file_standalone(const char *path, const NvmFileCyclicOptions *options) {
    uint8_t *bytes = NULL;
    size_t size = 0;
    char diagnostic[256] = {0};
    if (!nvm_file_cli_read(path,&bytes,&size,diagnostic,sizeof diagnostic)) {
        fprintf(stderr,"%s\n",diagnostic);
        return 1;
    }
    NvmFileHostGrant *grant = NULL;
    NvmFileHostStatus created = nvm_file_host_grant_create_temporary_files(&grant);
    if (created != NVM_FILE_HOST_OK) {
        free(bytes);
        fprintf(stderr,"I cannot create a temporary-file grant (%u)\n",(unsigned)created);
        return 1;
    }
    NvmFileScalar scalar = {0};
    NvmFileCyclicExecutionReport cyclic = {0};
    NvmFileRuntimeReport report;
    if (options) {
        cyclic = nvm_file_execute_cyclic_bytes(grant,bytes,size,options,&scalar);
        report = cyclic.runtime;
    } else report = nvm_file_execute_bytes(grant,bytes,size,&scalar);
    free(bytes);
    NvmFileHostStatus destroyed = nvm_file_host_grant_destroy(&grant);
    if (report.status != NVM_FILE_RUNTIME_OK || destroyed != NVM_FILE_HOST_OK) {
        if (options) fprintf(stderr,
                "I refuse cyclic File execution (status %u, site %u:%u, limit %llu, started %llu, exhausted %u, cleanup %llu, grant %u)\n",
                (unsigned)report.status,report.function,report.instruction,
                (unsigned long long)cyclic.instruction_limit,
                (unsigned long long)cyclic.instructions_started,(unsigned)cyclic.fuel_exhausted,
                (unsigned long long)report.cleanup.cleanup_failures,(unsigned)destroyed);
        else fprintf(stderr,"I refuse File execution (status %u, site %u:%u, cleanup %llu, grant %u)\n",
                (unsigned)report.status,report.function,report.instruction,
                (unsigned long long)report.cleanup.cleanup_failures,(unsigned)destroyed);
        return 1;
    }
    return (int)((uint64_t)scalar.value & UINT64_C(255));
}

static int run_standalone(const char *path, bool verify_only) {
    NanoisaErr err;
    NvmModule *module = nanoisa_load_file(path, &err);
    if (!module) {
        fprintf(stderr, "Error: Failed to load '%s': %s\n",
                path, err.message);
        return 1;
    }

    /* Verify bytecode safety before execution */
    NvmVerifyResult vr = nvm_verify(module);
    if (!vr.ok) {
        fprintf(stderr, "Error: Bytecode verification failed for '%s': %s\n",
                path, vr.error_msg);
        nvm_module_free(module);
        return 1;
    }

    if (verify_only) {
        nvm_module_free(module);
        return 0;
    }

    /* Preload FFI modules if the .nvm has imports */
    if (module->import_count > 0) {
        vm_ffi_init();

        /* Load modules referenced by name in the import table */
        for (uint32_t i = 0; i < module->import_count; i++) {
            vm_ffi_load_import(module, i);
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

    /* Repeat the program in this process, reusing the loaded module.
     *
     * Without this, a benchmark measures process startup and almost nothing
     * else: the workloads here retire between 78 and 32,082 instructions and
     * every one of them takes about the same 17 ms of wall clock, because the
     * spawn dominates by three orders of magnitude. A harness built on that
     * cannot see an interpreter change of any size. Running the module N times
     * behind one startup lets the per-iteration cost be recovered by
     * comparing two values of N. */
    VmResult result = VM_OK;
    for (uint32_t iteration = 1; iteration < g_repeat; iteration++) {
        VmState warm;
        vm_init(&warm, module);
        if (g_isolate_ffi) warm.isolate_ffi = true;
        VmResult r = vm_execute(&warm);
        vm_destroy(&warm);
        if (r != VM_OK) {
            fprintf(stderr, "Runtime error on repeat iteration %u\n", iteration);
            nvm_module_free(module);
            return 1;
        }
    }

    VmState vm;
    vm_init(&vm, module);
    vm_profile_enable(&vm, g_profile_path != NULL);

    /* Enable co-process FFI isolation if requested.
     * The cop is launched lazily on first extern call, not here. */
    if (g_isolate_ffi) {
        vm.isolate_ffi = true;
    }

    if (g_debug_mode) {
        vm.debug_mode = true;
    }

    result = vm_execute(&vm);

    if (g_profile_path) {
        FILE *profile = strcmp(g_profile_path, "-") == 0
            ? stdout : fopen(g_profile_path, "w");
        if (!profile) {
            fprintf(stderr, "Error: Cannot write NanoISA profile '%s'\n",
                    g_profile_path);
            result = VM_ERR_MEMORY;
        } else {
            if (!vm_profile_write_json(&vm, profile)) {
                fprintf(stderr, "Error: Failed to write NanoISA profile\n");
                result = VM_ERR_MEMORY;
            }
            if (profile != stdout) fclose(profile);
        }
    }

    int exit_code = 0;
    if (result != VM_OK) {
        /* I report failures even when debug metadata exists but no trace ran. */
        fprintf(stderr, "I could not execute the module: %s\n", vm_error_string(result));
        if (vm.error_msg[0]) {
            fprintf(stderr, "  %s\n", vm.error_msg);
        }
        exit_code = 1;
    } else {
        NanoValue value = vm_get_result(&vm);
        if (value.tag == TAG_INT) exit_code = (int)value.as.i64;
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

/* I report completion only after verified VM execution returns normally. */
static const char *shadow_module_path;
static int run_shadow_module(void) {
    return run_standalone(shadow_module_path, false);
}

int main(int argc, char *argv[]) {
    bool allow_temporary_files = false;
    bool file_cyclic = false, file_limit_set = false;
    NvmFileCyclicOptions file_options = {NVM_FILE_CYCLIC_RUNTIME_REVISION, 0};
    g_argc = argc;
    g_argv = argv;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [--verify-only | --daemon | --check-shadows | --allow-temporary-files] [--debug] [--profile-isa FILE] <file.nvm> [-- guest-args...]\n", argv[0]);
        fprintf(stderr, "For cyclic File execution I require --allow-temporary-files --file-cyclic --file-instruction-limit N.\n");
        return 1;
    }

    bool daemon_mode = false;
    bool verify_only = false;
    bool check_shadows = false;
    bool repeat_requested = false;
    const char *nvm_path = NULL;
    int module_index = 0;
    int guest_start = 0;

    /* Honour DEBUG env var before parsing flags */
    if (getenv("DEBUG")) g_debug_mode = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) {
            if (!nvm_path) {
                fprintf(stderr, "I require a module path before guest arguments.\n");
                return 1;
            }
            guest_start = i;
            break;
        } else if (strcmp(argv[i], "--allow-temporary-files") == 0) {
            allow_temporary_files = true;
        } else if (strcmp(argv[i], "--file-cyclic") == 0) {
            if (file_cyclic) {
                fprintf(stderr,"I accept --file-cyclic once.\n");
                return 1;
            }
            file_cyclic = true;
        } else if (strcmp(argv[i], "--file-instruction-limit") == 0) {
            if (file_limit_set || i + 1 >= argc ||
                !file_instruction_limit(argv[++i], &file_options.instruction_limit)) {
                fprintf(stderr,"I require one decimal File instruction limit from 0 through 1000000.\n");
                return 1;
            }
            file_limit_set = true;
        } else if (strcmp(argv[i], "--check-shadows") == 0) {
            check_shadows = true;
        } else if (strcmp(argv[i], "--verify-only") == 0) {
            verify_only = true;
        } else if (strcmp(argv[i], "--daemon") == 0 || strcmp(argv[i], "-d") == 0) {
            daemon_mode = true;
        } else if (strcmp(argv[i], "--isolate-ffi") == 0 || strcmp(argv[i], "--cop") == 0) {
            g_isolate_ffi = true;
        } else if (strcmp(argv[i], "--debug") == 0) {
            g_debug_mode = true;
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat_requested = true;
            long n = strtol(argv[++i], NULL, 10);
            g_repeat = (n > 0 && n <= 1000000) ? (uint32_t)n : 1;
        } else if (strcmp(argv[i], "--profile-isa") == 0 && i + 1 < argc) {
            g_profile_path = argv[++i];
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        } else {
            if (nvm_path) {
                fprintf(stderr, "I accept one module path; place guest arguments after --.\n");
                return 1;
            }
            nvm_path = argv[i];
            module_index = i;
        }
    }

    if (!nvm_path) {
        fprintf(stderr, "Error: No .nvm file specified\n");
        return 1;
    }

    if (file_cyclic != file_limit_set || (file_cyclic && !allow_temporary_files)) {
        fprintf(stderr,"I require --file-cyclic, --file-instruction-limit and --allow-temporary-files together.\n");
        return 1;
    }
    if (allow_temporary_files) {
        if (check_shadows || verify_only || daemon_mode || repeat_requested ||
            g_profile_path || g_isolate_ffi || g_debug_mode || guest_start) {
            fprintf(stderr,"I run the explicit temporary-file profile without other execution modes or guest arguments.\n");
            return 1;
        }
        return run_file_standalone(nvm_path, file_cyclic ? &file_options : NULL);
    }

    if (check_shadows && (verify_only || daemon_mode || repeat_requested ||
                          g_profile_path || guest_start)) {
        fprintf(stderr, "I run shadows once, without daemon, verification-only, profiling or guest arguments.\n");
        return 1;
    }

    if (verify_only && (daemon_mode || g_profile_path || g_isolate_ffi || repeat_requested)) {
        fprintf(stderr, "I cannot combine verification-only mode with execution options.\n");
        return 1;
    }

    if (daemon_mode && g_profile_path) {
        fprintf(stderr, "Error: --profile-isa requires in-process execution\n");
        return 1;
    }

    if (guest_start && (daemon_mode || verify_only)) {
        fprintf(stderr, "I accept guest arguments only for standalone execution.\n");
        return 1;
    }

    /* Give the guest its module name, without the VM options or delimiter. */
    if (guest_start) {
        argv[guest_start] = argv[module_index];
        g_argc = argc - guest_start;
        g_argv = argv + guest_start;
    } else {
        g_argc = 1;
        g_argv = argv + module_index;
        g_argv[1] = NULL;
    }

    if (check_shadows) {
        shadow_module_path = nvm_path;
        return nl_run_shadow_entry(run_shadow_module, 10);
    } else if (daemon_mode) {
        return run_daemon(nvm_path);
    } else {
        return run_standalone(nvm_path, verify_only);
    }
}
