/*
 * cop_protocol.c - Co-Process FFI Protocol Implementation
 *
 * Serialization/deserialization for NanoValues and pipe-based I/O.
 */

#include "cop_protocol.h"
#include "heap.h"
#include <string.h>
#include <unistd.h>
#include <errno.h>

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
