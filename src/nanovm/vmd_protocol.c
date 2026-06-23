/*
 * vmd_protocol.c - NanoVM Daemon Wire Protocol Implementation
 *
 * Handles partial writes/reads, EINTR, and payload size validation.
 */

#include "vmd_protocol.h"
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <arpa/inet.h>

/* ========================================================================
 * Path helpers
 * ======================================================================== */

void vmd_socket_path(char *buf, size_t size) {
    snprintf(buf, size, "/tmp/nanolang_vm_%u.sock", (unsigned)getuid());
}

void vmd_pid_path(char *buf, size_t size) {
    snprintf(buf, size, "/tmp/nanolang_vm_%u.pid", (unsigned)getuid());
}

/* ========================================================================
 * Low-level I/O helpers (handle partial transfers and EINTR)
 * ======================================================================== */

static bool write_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t n = write(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;  /* shouldn't happen on write */
        p += n;
        remaining -= (size_t)n;
    }
    return true;
}

static bool read_all(int fd, void *buf, size_t len) {
    uint8_t *p = buf;
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t n = read(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;  /* peer closed */
        p += n;
        remaining -= (size_t)n;
    }
    return true;
}

/* ========================================================================
 * Endianness helpers (protocol is little-endian)
 * ======================================================================== */

static inline uint16_t to_le16(uint16_t v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return v;
#else
    return __builtin_bswap16(v);
#endif
}

static inline uint32_t to_le32(uint32_t v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return v;
#else
    return __builtin_bswap32(v);
#endif
}

static inline uint16_t from_le16(uint16_t v) { return to_le16(v); }
static inline uint32_t from_le32(uint32_t v) { return to_le32(v); }

/* ========================================================================
 * Send / Receive
 * ======================================================================== */

bool vmd_msg_send(int fd, VmdMsgType type, const void *payload, uint32_t payload_len) {
    VmdMsgHeader hdr;
    hdr.version     = VMD_PROTO_VERSION;
    hdr.msg_type    = (uint8_t)type;
    hdr.flags       = 0;
    hdr.payload_len = to_le32(payload_len);

    if (!write_all(fd, &hdr, VMD_HEADER_SIZE))
        return false;

    if (payload_len > 0 && payload) {
        if (!write_all(fd, payload, payload_len))
            return false;
    }

    return true;
}

bool vmd_msg_recv_header(int fd, VmdMsgHeader *hdr) {
    if (!read_all(fd, hdr, VMD_HEADER_SIZE))
        return false;

    /* Convert from little-endian */
    hdr->flags       = from_le16(hdr->flags);
    hdr->payload_len = from_le32(hdr->payload_len);

    /* Validate */
    if (hdr->version != VMD_PROTO_VERSION)
        return false;
    if (hdr->payload_len > VMD_MAX_PAYLOAD)
        return false;

    return true;
}

bool vmd_msg_recv_payload(int fd, void *buf, uint32_t len) {
    return read_all(fd, buf, len);
}

/* ========================================================================
 * Convenience helpers
 * ======================================================================== */

bool vmd_msg_send_simple(int fd, VmdMsgType type) {
    return vmd_msg_send(fd, type, NULL, 0);
}

bool vmd_msg_send_output(int fd, const char *text, uint32_t len) {
    return vmd_msg_send(fd, VMD_MSG_OUTPUT, text, len);
}

bool vmd_msg_send_exit(int fd, int32_t code) {
    int32_t le_code = (int32_t)to_le32((uint32_t)code);
    return vmd_msg_send(fd, VMD_MSG_EXIT_CODE, &le_code, sizeof(le_code));
}

bool vmd_msg_send_error(int fd, const char *msg) {
    uint32_t len = msg ? (uint32_t)strlen(msg) : 0;
    return vmd_msg_send(fd, VMD_MSG_ERROR, msg, len);
}
