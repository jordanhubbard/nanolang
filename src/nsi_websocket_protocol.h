#ifndef NL_NSI_WEBSOCKET_PROTOCOL_H
#define NL_NSI_WEBSOCKET_PROTOCOL_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define NL_WS_MESSAGE_MAX (1024u * 1024u)
#define NL_WS_HEADER_MAX 8192u
#define NL_WS_FEED_MAX 65536u
#define NL_WS_FRAME_OVERHEAD 14u

typedef enum {
    NL_WS_OK, NL_WS_MORE, NL_WS_EVENT, NL_WS_ARGUMENT, NL_WS_PROTOCOL,
    NL_WS_LIMIT, NL_WS_MEMORY, NL_WS_CLOSED, NL_WS_CRYPTO
} NlWsStatus;
typedef enum { NL_WS_TEXT=1, NL_WS_BINARY=2, NL_WS_CLOSE=8, NL_WS_PING=9, NL_WS_PONG=10 } NlWsOpcode;
typedef struct NlWsDecoder NlWsDecoder;
typedef struct { NlWsOpcode opcode; const uint8_t *bytes; size_t length; } NlWsEvent;
/* I own bounded message storage. Decoder calls are serialized. Valid caller
 * objects and input bytes must be disjoint from each other and opaque storage. */
NlWsStatus nl_ws_decoder_create(NlWsDecoder **out);
void nl_ws_decoder_destroy(NlWsDecoder *);
/* I consume at most NL_WS_FEED_MAX bytes and stop at one complete event. MORE
 * can consume all bytes without publishing an event. Event bytes borrow decoder
 * storage until the next feed/destroy. Close and protocol failures are terminal.
 * Argument failures preserve outputs/state; other outcomes publish consumed.
 * Unconsumed suffixes belong to the caller, including coalesced later frames. */
NlWsStatus nl_ws_decoder_feed(NlWsDecoder *, const uint8_t *, size_t,
                             size_t *consumed, NlWsEvent *out);
/* I reject EOF inside a frame/fragment or without a received close handshake. */
NlWsStatus nl_ws_decoder_eof(NlWsDecoder *);
/* I encode one final masked client frame. The caller MUST provide a fresh
 * unpredictable mask for each frame. I acquire no randomness or network rights.
 * Insufficient capacity or invalid payload preserves output bytes and *written.
 * Input/mask/output/written storage must be disjoint. */
NlWsStatus nl_ws_client_frame(NlWsOpcode, const uint8_t *, size_t,
                             const uint8_t mask[4], uint8_t *, size_t, size_t *written);
/* The trusted host supplies a fresh unpredictable 16-byte nonce. I compute the
 * RFC 6455 key and expected accept value; output arrays must be disjoint. */
NlWsStatus nl_ws_handshake_key(const uint8_t nonce[16], char key[25], char accept[29]);
/* I accept no negotiated extensions/subprotocols. On OK I publish the header
 * byte count, leaving any coalesced frame bytes with the caller. MORE and errors
 * preserve *consumed. I bound the header, not the caller's whole input buffer. */
NlWsStatus nl_ws_upgrade_validate(const char *, size_t, const char accept[29], size_t *consumed);
#endif
