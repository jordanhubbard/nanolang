#ifndef NANO_CSEED_CAPTURE_H
#define NANO_CSEED_CAPTURE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>

/* I keep direct C-seed capture strings valid until process exit. Each
 * translation unit owns its captures and releases them through atexit. */
typedef struct NanoSeedCapture {
    struct NanoSeedCapture *next;
    char text[65536];
} NanoSeedCapture;

static NanoSeedCapture *nano_seed_captures;
static atomic_flag nano_seed_capture_lock = ATOMIC_FLAG_INIT;
static int nano_seed_capture_registered;

static void nano_seed_capture_acquire(void) {
    while (atomic_flag_test_and_set_explicit(&nano_seed_capture_lock, memory_order_acquire)) {}
}

static void nano_seed_capture_release_all(void) {
    nano_seed_capture_acquire();
    NanoSeedCapture *capture = nano_seed_captures;
    nano_seed_captures = NULL;
    atomic_flag_clear_explicit(&nano_seed_capture_lock, memory_order_release);
    while (capture) {
        NanoSeedCapture *next = capture->next;
        free(capture);
        capture = next;
    }
}

static inline const char *nl_exec_capture(const char *command) {
    FILE *pipe = popen(command, "r");
    if (!pipe) return "";
    NanoSeedCapture *capture = malloc(sizeof(*capture));
    if (!capture) { pclose(pipe); return ""; }
    size_t total = 0;
    while (total < sizeof(capture->text) - 1) {
        size_t count = fread(capture->text + total, 1,
                             sizeof(capture->text) - 1 - total, pipe);
        if (!count) break;
        total += count;
    }
    capture->text[total] = '\0';
    pclose(pipe);
    nano_seed_capture_acquire();
    if (!nano_seed_capture_registered) {
        if (atexit(nano_seed_capture_release_all) != 0) {
            atomic_flag_clear_explicit(&nano_seed_capture_lock, memory_order_release);
            free(capture);
            return "";
        }
        nano_seed_capture_registered = 1;
    }
    capture->next = nano_seed_captures;
    nano_seed_captures = capture;
    atomic_flag_clear_explicit(&nano_seed_capture_lock, memory_order_release);
    return capture->text;
}

#endif
