#ifndef NANOLANG_WEBSOCKET_H
#define NANOLANG_WEBSOCKET_H

#include <stdint.h>

/* Connect to a WebSocket server.
 * url: ws://host:port/path  or  wss:// (wss not supported, use ws://)
 * Returns a handle (fd-based context pointer) or 0 on failure. */
int64_t nl_ws_connect(const char *url);

/* Send a UTF-8 text frame. Returns 0 on success. */
int64_t nl_ws_send(int64_t handle, const char *message);

/* Receive the next text frame. Blocks until a frame arrives or timeout.
 * Returns a heap-allocated string (caller should not free; overwritten on next call).
 * Returns "" on error or close. */
const char *nl_ws_receive(int64_t handle);

/* Receive with timeout_ms milliseconds. Returns "" on timeout. */
const char *nl_ws_receive_timeout(int64_t handle, int64_t timeout_ms);

/* Close the connection. Returns 0 on success. */
int64_t nl_ws_close(int64_t handle);

/* Returns 1 if handle is a valid open connection, 0 otherwise. */
int64_t nl_ws_is_connected(int64_t handle);

/* Returns last error string for this handle. */
const char *nl_ws_last_error(int64_t handle);

#endif
