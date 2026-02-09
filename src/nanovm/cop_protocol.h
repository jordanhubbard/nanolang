/*
 * cop_protocol.h - Co-Process FFI Protocol
 *
 * Wire format for FFI isolation: the VM core runs in the main process,
 * FFI calls are dispatched to a separate co-process (nano_cop) via pipes.
 *
 * Reuses VmdMsgHeader (8-byte header + payload) wire format.
 */

#ifndef NANOVM_COP_PROTOCOL_H
#define NANOVM_COP_PROTOCOL_H

#include "value.h"
#include "heap.h"
#include "../nanoisa/nvm_format.h"
#include <stdint.h>
#include <stdbool.h>

/* ========================================================================
 * Message Types (co-process ↔ VM)
 * ======================================================================== */

typedef enum {
    /* VM → Co-process */
    COP_MSG_INIT       = 0x01,  /* Payload: serialized import table */
    COP_MSG_FFI_REQ    = 0x02,  /* Payload: import_idx + serialized args */
    COP_MSG_SHUTDOWN   = 0x03,  /* No payload */

    /* Co-process → VM */
    COP_MSG_FFI_RESULT = 0x10,  /* Payload: serialized NanoValue result */
    COP_MSG_FFI_ERROR  = 0x11,  /* Payload: error string */
    COP_MSG_READY      = 0x12   /* No payload (init complete) */
} CopMsgType;

/* Wire Header (same 8-byte format as VmdMsgHeader) */
typedef struct {
    uint8_t  version;       /* 1 */
    uint8_t  msg_type;      /* CopMsgType */
    uint16_t reserved;      /* Must be 0 */
    uint32_t payload_len;   /* Bytes following this header */
} __attribute__((packed)) CopMsgHeader;

#define COP_HEADER_SIZE 8
#define COP_PROTO_VERSION 1
#define COP_MAX_PAYLOAD (16 * 1024 * 1024)  /* 16 MB max for FFI payloads */

/* ========================================================================
 * NanoValue Serialization
 *
 * Format: tag(u8) + payload
 *   TAG_INT:    i64 (8 bytes, little-endian)
 *   TAG_FLOAT:  f64 (8 bytes, IEEE 754)
 *   TAG_BOOL:   u8 (0 or 1)
 *   TAG_STRING: len(u32) + data (UTF-8 bytes)
 *   TAG_OPAQUE: i64 (proxy ID)
 *   TAG_VOID:   nothing (0 extra bytes)
 * ======================================================================== */

/* Serialize a NanoValue into a buffer. Returns bytes written.
 * Buffer must be at least 13 bytes (1 tag + max 8 payload + 4 len). */
uint32_t cop_serialize_value(const NanoValue *val, uint8_t *buf, uint32_t buf_size);

/* Deserialize a NanoValue from a buffer. Returns bytes consumed, 0 on error.
 * heap is needed for allocating strings. */
uint32_t cop_deserialize_value(const uint8_t *buf, uint32_t buf_size,
                               NanoValue *out, VmHeap *heap);

/* ========================================================================
 * Send / Receive Helpers (pipe-based I/O)
 * ======================================================================== */

/* Send a header + payload over a file descriptor. */
bool cop_send(int fd, CopMsgType type, const void *payload, uint32_t payload_len);

/* Receive a header. Returns true on success. */
bool cop_recv_header(int fd, CopMsgHeader *hdr);

/* Receive exactly len bytes. */
bool cop_recv_payload(int fd, void *buf, uint32_t len);

/* Send a simple message (no payload). */
bool cop_send_simple(int fd, CopMsgType type);

#endif /* NANOVM_COP_PROTOCOL_H */
