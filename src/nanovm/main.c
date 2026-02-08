/*
 * nano_vm - NanoVM bytecode executor
 *
 * Loads an .nvm file and executes it via the VM.
 *
 * Usage: nano_vm <file.nvm>
 */

#include "vm.h"
#include "../nanoisa/nvm_format.h"
#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.nvm>\n", argv[0]);
        return 1;
    }

    /* Load the .nvm file */
    uint32_t file_size = 0;
    uint8_t *data = read_file(argv[1], &file_size);
    if (!data) return 1;

    /* Deserialize into a module */
    NvmModule *module = nvm_deserialize(data, file_size);
    free(data);

    if (!module) {
        fprintf(stderr, "Error: Failed to load '%s' (invalid .nvm format)\n", argv[1]);
        return 1;
    }

    /* Execute */
    VmState vm;
    vm_init(&vm, module);

    VmResult result = vm_execute(&vm);

    int exit_code = 0;
    if (result != VM_OK) {
        fprintf(stderr, "Runtime error: %s\n", vm_error_string(result));
        if (vm.error_msg[0]) {
            fprintf(stderr, "  %s\n", vm.error_msg);
        }
        exit_code = 1;
    }

    vm_destroy(&vm);
    nvm_module_free(module);

    return exit_code;
}
