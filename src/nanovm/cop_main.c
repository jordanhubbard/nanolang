/*
 * nano_cop - NanoVM Co-Process for FFI Isolation
 *
 * Runs as a child process of nano_vm (or nano_vmd threads).
 * Reads FFI requests from stdin, executes them, writes results to stdout.
 * Provides complete address-space isolation for external C function calls.
 *
 * Protocol:
 *   1. Parent sends COP_MSG_INIT with serialized .nvm import table
 *   2. Co-process initializes FFI, responds with COP_MSG_READY
 *   3. Parent sends COP_MSG_FFI_REQ for each extern call
 *   4. Co-process calls the C function, responds with COP_MSG_FFI_RESULT
 *   5. Parent sends COP_MSG_SHUTDOWN (or closes pipe) to terminate
 */

#include "cop_protocol.h"
#include "vm_ffi.h"
#include "heap.h"
#include "../nanoisa/nvm_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* Local heap for deserializing strings */
static VmHeap g_heap;

/* The module whose import table we serve */
static NvmModule *g_module = NULL;

static bool handle_init(int in_fd, uint32_t payload_len) {
    /* Receive serialized .nvm module blob */
    uint8_t *blob = malloc(payload_len);
    if (!blob) return false;
    if (!cop_recv_payload(in_fd, blob, payload_len)) {
        free(blob);
        return false;
    }

    /* Deserialize the module to get its import table */
    g_module = nvm_deserialize(blob, payload_len);
    free(blob);
    if (!g_module) return false;

    /* Initialize FFI and load all referenced modules */
    vm_ffi_init();
    for (uint32_t i = 0; i < g_module->import_count; i++) {
        const char *mod_name = nvm_get_string(g_module, g_module->imports[i].module_name_idx);
        if (mod_name && mod_name[0]) {
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

    for (uint32_t i = 0; i < g_module->import_count; i++) {
        const char *fn_name = nvm_get_string(g_module,
                                              g_module->imports[i].function_name_idx);
        const char *mod_name = nvm_get_string(g_module,
                                               g_module->imports[i].module_name_idx);
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

    /* Signal ready */
    cop_send_simple(STDOUT_FILENO, COP_MSG_READY);
    return true;
}

static bool handle_ffi_req(int in_fd, uint32_t payload_len) {
    uint8_t *payload = malloc(payload_len);
    if (!payload) return false;
    if (!cop_recv_payload(in_fd, payload, payload_len)) {
        free(payload);
        return false;
    }

    /* Parse: u32 import_idx + u16 argc + serialized args */
    if (payload_len < 6) {
        free(payload);
        cop_send(STDOUT_FILENO, COP_MSG_FFI_ERROR, "bad request", 11);
        return true;
    }

    uint32_t import_idx;
    uint16_t argc;
    memcpy(&import_idx, payload, 4);
    memcpy(&argc, payload + 4, 2);

    NanoValue args[16] = {0};
    uint32_t pos = 6;
    for (int i = 0; i < argc && i < 16; i++) {
        uint32_t consumed = cop_deserialize_value(payload + pos, payload_len - pos,
                                                   &args[i], &g_heap);
        if (consumed == 0) {
            free(payload);
            cop_send(STDOUT_FILENO, COP_MSG_FFI_ERROR, "bad arg", 7);
            return true;
        }
        pos += consumed;
    }
    free(payload);

    /* Call the C function */
    NanoValue result;
    char error_msg[256] = {0};
    if (!vm_ffi_call(g_module, import_idx, args, argc, &result, &g_heap,
                     error_msg, sizeof(error_msg))) {
        uint32_t err_len = (uint32_t)strlen(error_msg);
        cop_send(STDOUT_FILENO, COP_MSG_FFI_ERROR, error_msg, err_len);
    } else {
        /* Serialize and send result.
         * Use a small stack buffer for simple values, dynamically
         * allocate for large results (arrays, deeply nested structs). */
        uint8_t stack_buf[4096];
        uint32_t result_len = cop_serialize_value(&result, stack_buf, sizeof(stack_buf));
        if (result_len > 0) {
            cop_send(STDOUT_FILENO, COP_MSG_FFI_RESULT, stack_buf, result_len);
        } else {
            /* Stack buffer too small — retry with a larger heap buffer */
            uint32_t big_size = 1024 * 1024;  /* 1 MB */
            uint8_t *big_buf = malloc(big_size);
            if (big_buf) {
                result_len = cop_serialize_value(&result, big_buf, big_size);
                cop_send(STDOUT_FILENO, COP_MSG_FFI_RESULT, big_buf, result_len);
                free(big_buf);
            } else {
                cop_send(STDOUT_FILENO, COP_MSG_FFI_ERROR,
                         "OOM serializing result", 22);
            }
        }
        vm_release(&g_heap, result);
    }

    /* Release deserialized args */
    for (int i = 0; i < argc && i < 16; i++) {
        vm_release(&g_heap, args[i]);
    }

    return true;
}

int main(void) {
    /* Co-process reads from stdin, writes to stdout */
    vm_heap_init(&g_heap);

    /* Main message loop */
    for (;;) {
        CopMsgHeader hdr;
        if (!cop_recv_header(STDIN_FILENO, &hdr)) {
            break;  /* Parent closed pipe or error */
        }

        switch (hdr.msg_type) {
        case COP_MSG_INIT:
            if (!handle_init(STDIN_FILENO, hdr.payload_len)) {
                goto cleanup;
            }
            break;

        case COP_MSG_FFI_REQ:
            if (!handle_ffi_req(STDIN_FILENO, hdr.payload_len)) {
                goto cleanup;
            }
            break;

        case COP_MSG_SHUTDOWN:
            goto cleanup;

        default:
            /* Unknown message, drain payload and continue */
            if (hdr.payload_len > 0) {
                uint8_t *discard = malloc(hdr.payload_len);
                if (discard) {
                    cop_recv_payload(STDIN_FILENO, discard, hdr.payload_len);
                    free(discard);
                }
            }
            break;
        }
    }

cleanup:
    vm_ffi_shutdown();
    if (g_module) nvm_module_free(g_module);
    vm_heap_destroy(&g_heap);
    return 0;
}
