/*
 * vmd_client.h - NanoVM Daemon Client Library
 *
 * Connects to the nano_vmd daemon (lazy-launching if needed), sends
 * an .nvm blob for execution, and streams output back to the caller.
 */

#ifndef NANOVM_VMD_CLIENT_H
#define NANOVM_VMD_CLIENT_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/* ========================================================================
 * Client API
 * ======================================================================== */

/* Opaque client handle */
typedef struct VmdClient VmdClient;

/* Connect to daemon. If none is running, fork+exec nano_vmd and wait
 * for it to become ready (up to timeout_ms). Returns NULL on failure. */
VmdClient *vmd_connect(int timeout_ms);

/* Send an .nvm blob for execution. Output is streamed to stdout.
 * Returns the program's exit code (0=success, 1=error, -1=comm error). */
int vmd_execute(VmdClient *client, const uint8_t *blob, uint32_t blob_size);

/* Disconnect from daemon. */
void vmd_disconnect(VmdClient *client);

/* Send a ping to check if daemon is alive. Returns true if pong received. */
bool vmd_ping(VmdClient *client);

#endif /* NANOVM_VMD_CLIENT_H */
