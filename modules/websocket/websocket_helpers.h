#ifndef NANOLANG_WEBSOCKET_H
#define NANOLANG_WEBSOCKET_H

#include <stdint.h>

/* Connect to a WebSocket server.
 * I accept ws://host:port/path and refuse wss:// until TLS is implemented.
 * I return a monotone registry identity, never a context pointer, or zero.
 * My connect/upgrade I/O deadline is 10 seconds; host DNS is synchronous.
 * I serialize calls and refuse concurrent entry. Handles are legacy unsafe
 * API identities, not verified affine source resources. */
int64_t nl_ws_connect(const char *url);

/* Send a UTF-8 text frame. Returns 0 on success. */
int64_t nl_ws_send(int64_t handle, const char *message);

/* I receive a complete UTF-8 text message, with a 10-second I/O deadline.
 * I process control frames and skip binary messages. Embedded NUL text is
 * refused by this legacy C-string API. Partial input survives a timeout.
 * I return borrowed text valid until the next receive or close; do not free it.
 * Returns "" on error or close. */
const char *nl_ws_receive(int64_t handle);

/* I accept timeout_ms from zero through 60000; zero polls available input.
 * I return "" on timeout or refusal. */
const char *nl_ws_receive_timeout(int64_t handle, int64_t timeout_ms);

/* I send close and wait for the peer close within the remaining ten-second
 * I/O budget, then retire the handle even on failure. I return zero on success. */
int64_t nl_ws_close(int64_t handle);

/* Returns 1 if handle is a valid open connection, 0 otherwise. */
int64_t nl_ws_is_connected(int64_t handle);

/* Returns last error string for this handle. */
const char *nl_ws_last_error(int64_t handle);

#endif
