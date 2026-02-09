/*
 * vmd_main.c - NanoVM Daemon entry point
 *
 * Usage: nano_vmd [--foreground] [--verbose] [--idle-timeout N]
 */

#include "vmd_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

int main(int argc, char *argv[]) {
    g_argc = argc;
    g_argv = argv;
    VmdServerConfig cfg = {
        .idle_timeout_sec = VMD_DEFAULT_IDLE_TIMEOUT,
        .foreground = false,
        .verbose = false,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--foreground") == 0 || strcmp(argv[i], "-f") == 0) {
            cfg.foreground = true;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            cfg.verbose = true;
        } else if (strcmp(argv[i], "--idle-timeout") == 0 && i + 1 < argc) {
            cfg.idle_timeout_sec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-timeout") == 0) {
            cfg.idle_timeout_sec = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n"
                    "\n"
                    "Options:\n"
                    "  --foreground, -f    Stay in foreground (don't daemonize)\n"
                    "  --verbose, -v       Verbose logging\n"
                    "  --idle-timeout N    Shutdown after N seconds idle (default: %d)\n"
                    "  --no-timeout        Never auto-shutdown\n"
                    "  --help, -h          Show this help\n",
                    argv[0], VMD_DEFAULT_IDLE_TIMEOUT);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    return vmd_server_run(&cfg);
}
