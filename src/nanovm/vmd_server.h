/*
 * vmd_server.h - NanoVM Daemon Server
 *
 * Long-lived daemon that accepts program execution requests over a Unix
 * domain socket. Each client gets its own thread with a private VmState.
 */

#ifndef NANOVM_VMD_SERVER_H
#define NANOVM_VMD_SERVER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/* ========================================================================
 * Server Configuration
 * ======================================================================== */

typedef struct {
    int  idle_timeout_sec;   /* Shut down after N seconds with no clients (0=never) */
    bool foreground;         /* Stay in foreground (don't daemonize) */
    bool verbose;            /* Verbose logging */
} VmdServerConfig;

/* Default config values */
#define VMD_DEFAULT_IDLE_TIMEOUT  300  /* 5 minutes */

/* ========================================================================
 * Server API
 * ======================================================================== */

/* Run the daemon server (blocks until shutdown). Returns 0 on clean exit. */
int vmd_server_run(const VmdServerConfig *cfg);

#endif /* NANOVM_VMD_SERVER_H */
