/*
 * cop_protocol.c - Co-Process FFI Protocol Implementation
 *
 * Serialization/deserialization for NanoValues, pipe-based I/O,
 * and the shared-memory mailbox child main loop.
 */

#include "cop_protocol.h"
#include "vm_ffi.h"
#include "heap.h"
#include "../nanoisa/nvm_format.h"
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <poll.h>

/* MAP_ANON compat */
#ifndef MAP_ANON
#  ifdef MAP_ANONYMOUS
#    define MAP_ANON MAP_ANONYMOUS
#  else
#    define MAP_ANON 0x1000
#  endif
#endif

/* ========================================================================
 * NanoValue Serialization
 * ======================================================================== */

uint32_t cop_serialize_value(const NanoValue *val, uint8_t *buf, uint32_t buf_size) {
    if (buf_size < 1) return 0;
    buf[0] = val->tag;
    uint32_t pos = 1;

    switch (val->tag) {
    case TAG_INT: {
        if (pos + 8 > buf_size) return 0;
        int64_t v = val->as.i64;
        memcpy(buf + pos, &v, 8);
        pos += 8;
        break;
    }
    case TAG_FLOAT: {
        if (pos + 8 > buf_size) return 0;
        double v = val->as.f64;
        memcpy(buf + pos, &v, 8);
        pos += 8;
        break;
    }
    case TAG_BOOL: {
        if (pos + 1 > buf_size) return 0;
        buf[pos++] = val->as.boolean ? 1 : 0;
        break;
    }
    case TAG_STRING: {
        const char *s = "";
        uint32_t len = 0;
        if (val->as.string) {
            s = val->as.string->data;
            len = val->as.string->length;
        }
        if (pos + 4 + len > buf_size) return 0;
        memcpy(buf + pos, &len, 4);
        pos += 4;
        if (len > 0) {
            memcpy(buf + pos, s, len);
            pos += len;
        }
        break;
    }
    case TAG_OPAQUE: {
        if (pos + 8 > buf_size) return 0;
        int64_t v = val->as.i64;
        memcpy(buf + pos, &v, 8);
        pos += 8;
        break;
    }
    case TAG_ARRAY: {
        VmArray *arr = val->as.array;
        uint32_t count = arr ? arr->length : 0;
        uint8_t etype = arr ? arr->elem_type : 0;
        if (pos + 5 > buf_size) return 0;
        buf[pos++] = etype;
        memcpy(buf + pos, &count, 4);
        pos += 4;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t n = cop_serialize_value(&arr->elements[i],
                                              buf + pos, buf_size - pos);
            if (n == 0) return 0;
            pos += n;
        }
        break;
    }
    case TAG_VOID:
    default:
        /* No payload for void */
        break;
    }

    return pos;
}

uint32_t cop_deserialize_value(const uint8_t *buf, uint32_t buf_size,
                               NanoValue *out, VmHeap *heap) {
    if (buf_size < 1) return 0;
    uint8_t tag = buf[0];
    uint32_t pos = 1;

    switch (tag) {
    case TAG_INT: {
        if (pos + 8 > buf_size) return 0;
        int64_t v;
        memcpy(&v, buf + pos, 8);
        pos += 8;
        *out = val_int(v);
        break;
    }
    case TAG_FLOAT: {
        if (pos + 8 > buf_size) return 0;
        double v;
        memcpy(&v, buf + pos, 8);
        pos += 8;
        *out = val_float(v);
        break;
    }
    case TAG_BOOL: {
        if (pos + 1 > buf_size) return 0;
        *out = val_bool(buf[pos] != 0);
        pos += 1;
        break;
    }
    case TAG_STRING: {
        if (pos + 4 > buf_size) return 0;
        uint32_t len;
        memcpy(&len, buf + pos, 4);
        pos += 4;
        if (pos + len > buf_size) return 0;
        VmString *s = vm_string_new(heap, (const char *)(buf + pos), len);
        pos += len;
        *out = val_string(s);
        break;
    }
    case TAG_OPAQUE: {
        if (pos + 8 > buf_size) return 0;
        int64_t v;
        memcpy(&v, buf + pos, 8);
        pos += 8;
        NanoValue ov = {0};
        ov.tag = TAG_OPAQUE;
        ov.as.i64 = v;
        *out = ov;
        break;
    }
    case TAG_ARRAY: {
        if (pos + 5 > buf_size) return 0;
        uint8_t etype = buf[pos++];
        uint32_t count;
        memcpy(&count, buf + pos, 4);
        pos += 4;
        VmArray *arr = vm_array_new(heap, etype, count > 0 ? count : 4);
        for (uint32_t i = 0; i < count; i++) {
            NanoValue elem;
            uint32_t n = cop_deserialize_value(buf + pos, buf_size - pos,
                                                &elem, heap);
            if (n == 0) { *out = val_void(); return 0; }
            pos += n;
            vm_array_push(arr, elem);
        }
        *out = val_array(arr);
        break;
    }
    case TAG_VOID:
    default:
        *out = val_void();
        break;
    }

    return pos;
}

/* ========================================================================
 * Pipe I/O Helpers
 * ======================================================================== */

static bool write_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= (size_t)n;
    }
    return true;
}

static bool read_all(int fd, void *buf, size_t len) {
    uint8_t *p = buf;
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;  /* EOF */
        p += n;
        len -= (size_t)n;
    }
    return true;
}

bool cop_send(int fd, CopMsgType type, const void *payload, uint32_t payload_len) {
    CopMsgHeader hdr = {0};
    hdr.version = COP_PROTO_VERSION;
    hdr.msg_type = (uint8_t)type;
    hdr.payload_len = payload_len;

    if (!write_all(fd, &hdr, COP_HEADER_SIZE)) return false;
    if (payload_len > 0 && payload) {
        if (!write_all(fd, payload, payload_len)) return false;
    }
    return true;
}

bool cop_recv_header(int fd, CopMsgHeader *hdr) {
    if (!read_all(fd, hdr, COP_HEADER_SIZE)) return false;
    if (hdr->version != COP_PROTO_VERSION) return false;
    if (hdr->payload_len > COP_MAX_PAYLOAD) return false;
    return true;
}

bool cop_recv_payload(int fd, void *buf, uint32_t len) {
    return read_all(fd, buf, len);
}

bool cop_send_simple(int fd, CopMsgType type) {
    return cop_send(fd, type, NULL, 0);
}

/* ========================================================================
 * cop_child_main — shared-memory mailbox fast path (in-process child)
 *
 * Called after fork() in vm_ffi_cop_start(). The module pointer is valid
 * because we forked (not exec'd), so the parent's deserialized module is
 * directly accessible.  Communication uses two 1-byte signal pipes and the
 * mmap'd CopMailbox for all request/response data.
 * ======================================================================== */

void cop_child_main(CopMailbox *mailbox, size_t mailbox_size,
                    int sig_in_fd, int sig_out_fd,
                    const NvmModule *module) {
    (void)mailbox_size;

    VmHeap heap;
    vm_heap_init(&heap);

    /* Initialize FFI — module is already deserialized in the parent's address space */
    vm_ffi_init();
    for (uint32_t i = 0; i < module->import_count; i++) {
        const char *mod_name = nvm_get_string(module, module->imports[i].module_name_idx);
        if (mod_name && mod_name[0]) {
            vm_ffi_load_module(mod_name);
        }
    }

    /* Prefix-based fallback for bare extern declarations */
    static const struct { const char *prefix; const char *mod; } known[] = {
        {"path_",    "std/fs"},    {"fs_",      "std/fs"},
        {"file_",    "std/fs"},    {"dir_",     "std/fs"},
        {"regex_",   "std/regex"}, {"process_", "std/process"},
        {"json_",    "std/json"},  {"bstr_",    "std/bstring"},
        {NULL, NULL}
    };
    for (uint32_t i = 0; i < module->import_count; i++) {
        const char *fn  = nvm_get_string(module, module->imports[i].function_name_idx);
        const char *mod = nvm_get_string(module, module->imports[i].module_name_idx);
        if (fn && (!mod || mod[0] == '\0')) {
            for (int k = 0; known[k].prefix; k++) {
                if (strncmp(fn, known[k].prefix, strlen(known[k].prefix)) == 0) {
                    vm_ffi_load_module(known[k].mod);
                    break;
                }
            }
        }
    }

    /* Signal ready to parent */
    uint8_t sig = 1;
    if (write(sig_out_fd, &sig, 1) != 1) goto done;

    {
        struct pollfd pfd = { .fd = sig_in_fd, .events = POLLIN };
        for (;;) {
            if (poll(&pfd, 1, -1) <= 0) break;

            uint8_t trigger;
            if (read(sig_in_fd, &trigger, 1) != 1) break;

            uint32_t import_idx = mailbox->req_import_idx;
            uint16_t argc       = mailbox->req_argc;
            uint16_t data_size  = mailbox->req_data_size;

            NanoValue args[16] = {0};
            int actual_argc = 0;
            uint32_t pos = 0;
            for (int i = 0; i < argc && i < 16 && pos < data_size; i++) {
                uint32_t consumed = cop_deserialize_value(mailbox->req_data + pos,
                                                          data_size - pos,
                                                          &args[i], &heap);
                if (consumed == 0) break;
                pos += consumed;
                actual_argc++;
            }

            NanoValue result;
            char errmsg[256] = {0};
            if (!vm_ffi_call(module, import_idx, args, actual_argc, &result, &heap,
                             errmsg, sizeof(errmsg))) {
                mailbox->resp_is_error  = 1;
                mailbox->resp_data_size = 0;
                strncpy(mailbox->resp_error, errmsg, sizeof(mailbox->resp_error) - 1);
                mailbox->resp_error[sizeof(mailbox->resp_error) - 1] = '\0';
            } else {
                uint32_t rlen = cop_serialize_value(&result,
                                                    mailbox->resp_data,
                                                    COP_MAILBOX_SLOT_SIZE);
                mailbox->resp_is_error  = 0;
                mailbox->resp_data_size = rlen;
                vm_release(&heap, result);
            }

            for (int i = 0; i < actual_argc; i++) vm_release(&heap, args[i]);

            if (write(sig_out_fd, &sig, 1) != 1) break;
        }
    }

done:
    vm_ffi_shutdown();
    vm_heap_destroy(&heap);
    _exit(0);
}
