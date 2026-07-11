/* ProTracker Audio Module Wrapper
 * Wraps pt2-clone's Paula emulation as a simple C module for nanolang
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pt2_structs.h"
#include "pt2_paula.h"

/* Static state variables */
static int g_initialized = 0;
static int64_t g_sample_rate = 0;
static int g_playing = 0;
static int g_loaded = 0;
static char g_mod_filename[4096] = {0};

/* Initialize audio system */
int64_t pt2_audio_init(int64_t sample_rate) {
    g_sample_rate = sample_rate;
#ifdef PAULA_SETUP_AVAILABLE
    paulaSetup((double)sample_rate, MODEL_A500);
#endif
    g_initialized = 1;
    return 0; /* Success */
}

/* Load a MOD file */
int64_t pt2_audio_load_mod(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        return -1; /* File not found or unreadable */
    }
    fclose(f);
    g_loaded = 1;
    strncpy(g_mod_filename, filename, sizeof(g_mod_filename) - 1);
    g_mod_filename[sizeof(g_mod_filename) - 1] = '\0';
    return 0; /* Success */
}

/* Start playback */
void pt2_audio_play(void) {
    g_playing = 1;
}

/* Stop playback */
void pt2_audio_stop(void) {
    g_playing = 0;
}

/* Update audio - generates samples into buffer */
void pt2_audio_update(float *buffer, int64_t frames) {
    /* Paula emulation would generate audio samples here; emit silence for now */
    memset(buffer, 0, (size_t)(frames * 2) * sizeof(float)); /* Stereo */
}

/* Check if playing */
int64_t pt2_audio_is_playing(void) {
    return g_playing;
}
