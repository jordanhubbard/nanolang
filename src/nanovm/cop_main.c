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
#include "../../modules/nanoisa/nanoisa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* Local heap for deserializing strings */
static VmHeap g_heap;

/* The module whose import table we serve */
static NvmModule *g_module = NULL;
static CopOpaqueWorker g_opaque;

static bool handle_init(int in_fd, uint32_t payload_len) {
    /* Receive serialized .nvm module blob */
    uint8_t *blob = malloc(payload_len);
    if (!blob) return false;
    if (!cop_recv_payload(in_fd, blob, payload_len)) {
        free(blob);
        return false;
    }

    NanoisaErr load_error;
    g_module = nanoisa_load_bytes(blob, payload_len, &load_error);
    free(blob);
    if (!g_module) return false;

    if (!cop_worker_load_imports(g_module)) return false;

    /* Signal ready */
    cop_send_simple(STDOUT_FILENO, COP_MSG_READY);
    return true;
}

static bool handle_ffi_req(int in_fd, uint32_t payload_len) {
    if (payload_len > COP_MAX_PAYLOAD) return false;
    uint8_t *payload = malloc(payload_len ? payload_len : 1);
    if (!payload) return false;
    if (!cop_recv_payload(in_fd, payload, payload_len)) {
        free(payload);
        return false;
    }
    uint8_t *reply = NULL;
    uint32_t reply_size = 0;
    char error[256] = {0};
    bool ok = cop_execute_request_owned(payload, payload_len, g_module, &g_heap,
                                  &reply, &reply_size, error, sizeof error, &g_opaque);
    free(payload);
    bool sent = ok ? cop_send(STDOUT_FILENO, COP_MSG_FFI_RESULT, reply, reply_size)
                   : cop_send(STDOUT_FILENO, COP_MSG_FFI_ERROR, error, (uint32_t)strlen(error));
    free(reply);
    return sent;
}

/* My exec mode accepts only the fixed private descriptor protocol. */
static int mailbox_main(void) {
    struct stat mailbox_stat, module_stat;
    if (fstat(3, &mailbox_stat) != 0 || !S_ISREG(mailbox_stat.st_mode) ||
        mailbox_stat.st_size != (off_t)sizeof(CopMailbox) ||
        fstat(8, &module_stat) != 0 || !S_ISREG(module_stat.st_mode) ||
        module_stat.st_size <= 0 || module_stat.st_size > COP_MAX_PAYLOAD) return 1;
    for (int i = 3; i <= 8; ++i)
        if (fcntl(i, F_SETFD, FD_CLOEXEC) != 0) return 1;
    void *bytes = mmap(NULL, (size_t)module_stat.st_size, PROT_READ, MAP_PRIVATE, 8, 0);
    if (bytes == MAP_FAILED) return 1;
    NanoisaErr error;
    NvmModule *module = nanoisa_load_bytes(bytes, (uint32_t)module_stat.st_size, &error);
    munmap(bytes, (size_t)module_stat.st_size);
    close(8);
    if (!module) return 1;
    CopMailbox *mailbox = mmap(NULL, sizeof(CopMailbox), PROT_READ | PROT_WRITE,
                               MAP_SHARED, 3, 0);
    close(3);
    if (mailbox == MAP_FAILED) { nvm_module_free(module); return 1; }
    cop_child_main(mailbox, sizeof(CopMailbox), 4, 5, 6, 7, module);
    for (int i = 4; i <= 7; ++i) close(i);
    munmap(mailbox, sizeof(CopMailbox));
    nvm_module_free(module);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--mailbox-v1") == 0) return mailbox_main();
    if (argc != 1) return 1;

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
    cop_opaque_worker_clear(&g_opaque);
    vm_ffi_shutdown();
    if (g_module) nvm_module_free(g_module);
    vm_heap_destroy(&g_heap);
    return 0;
}
