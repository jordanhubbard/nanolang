/*
 * vmd_protocol.h - NanoVM Daemon Wire Protocol
 *
 * 8-byte header + variable payload, little-endian integers.
 * Used for communication between nano_vm clients and the nano_vmd daemon
 * over Unix domain sockets.
 */

#ifndef NANOVM_VMD_PROTOCOL_H
#define NANOVM_VMD_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Protocol version */
#define VMD_PROTO_VERSION 1

/* ========================================================================
 * Message Types
 * ======================================================================== */

typedef enum {
    /* Client → Server */
    VMD_MSG_LOAD_EXEC  = 0x01,  /* Payload: .nvm blob */
    VMD_MSG_PING       = 0x02,  /* No payload */
    VMD_MSG_STATUS     = 0x03,  /* No payload */
    VMD_MSG_SHUTDOWN   = 0x04,  /* No payload */

    /* Server → Client */
    VMD_MSG_OUTPUT     = 0x10,  /* Payload: text (no NUL terminator) */
    VMD_MSG_EXIT_CODE  = 0x11,  /* Payload: int32_t exit code */
    VMD_MSG_ERROR      = 0x12,  /* Payload: error string (no NUL) */
    VMD_MSG_PONG       = 0x13,  /* No payload */
    VMD_MSG_STATUS_RSP = 0x14   /* Payload: status string */
} VmdMsgType;

/* ========================================================================
 * Wire Header (8 bytes, little-endian)
 * ======================================================================== */

typedef struct {
    uint8_t  version;       /* VMD_PROTO_VERSION */
    uint8_t  msg_type;      /* VmdMsgType */
    uint16_t flags;         /* Reserved, must be 0 */
    uint32_t payload_len;   /* Bytes following this header */
} __attribute__((packed)) VmdMsgHeader;

#define VMD_HEADER_SIZE 8

/* Maximum payload size (100 MB, matching .nvm file size limit) */
#define VMD_MAX_PAYLOAD (100 * 1024 * 1024)

/* ========================================================================
 * Send / Receive API
 *
 * All functions handle partial writes/reads and EINTR.
 * Returns true on success, false on error (connection closed, etc.).
 * ======================================================================== */

/* Send a message (header + payload). payload may be NULL if payload_len == 0. */
bool vmd_msg_send(int fd, VmdMsgType type, const void *payload, uint32_t payload_len);

/* Receive a message header. Returns true on success. */
bool vmd_msg_recv_header(int fd, VmdMsgHeader *hdr);

/* Receive exactly `len` bytes of payload into `buf`.
 * Caller must ensure buf has at least `len` bytes. */
bool vmd_msg_recv_payload(int fd, void *buf, uint32_t len);

/* Convenience: send a message with no payload. */
bool vmd_msg_send_simple(int fd, VmdMsgType type);

/* Convenience: send an OUTPUT message with a text string. */
bool vmd_msg_send_output(int fd, const char *text, uint32_t len);

/* Convenience: send an EXIT_CODE message. */
bool vmd_msg_send_exit(int fd, int32_t code);

/* Convenience: send an ERROR message with a string. */
bool vmd_msg_send_error(int fd, const char *msg);

/* ========================================================================
 * Path helpers (shared between client and server)
 * ======================================================================== */

/* Get the socket path for the current user (/tmp/nanolang_vm_<uid>.sock) */
void vmd_socket_path(char *buf, size_t size);

/* Get the PID file path (/tmp/nanolang_vm_<uid>.pid) */
void vmd_pid_path(char *buf, size_t size);

#endif /* NANOVM_VMD_PROTOCOL_H */
