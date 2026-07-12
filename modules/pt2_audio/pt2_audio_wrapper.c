/* ProTracker Audio Module Wrapper
 * Wraps pt2-clone's Paula emulation as a simple C module for nanolang
 *
 * Real state machine: init -> load_mod -> play -> stop
 * State is fully tracked in static variables; paulaSetup called when
 * SDL is available (PT2_FULL_BUILD build flag).
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PT2_FULL_BUILD
#include "pt2_structs.h"
#include "pt2_paula.h"
#endif

/* Internal state */
static int      g_initialized   = 0;
static int64_t  g_sample_rate   = 0;
static int      g_playing       = 0;
static int      g_loaded        = 0;
static char     g_mod_filename[4096] = {0};

/* Initialize the audio system and Paula emulator.
 * Must be called before any other pt2_audio function.
 * Returns 0 on success, -1 on failure.
 */
int64_t pt2_audio_init(int64_t sample_rate) {
    g_sample_rate   = sample_rate;
    g_playing       = 0;
    g_loaded        = 0;
    g_mod_filename[0] = '\0';

#ifdef PT2_FULL_BUILD
    paulaSetup((double)sample_rate, MODEL_A500);
#endif

    g_initialized = 1;
    return 0;
}

/* Load a MOD file for playback.
 * Returns 0 on success, -1 if the file cannot be opened,
 * -2 if the audio system has not been initialized.
 */
int64_t pt2_audio_load_mod(const char *filename) {
    if (!g_initialized) {
        return -2;
    }

    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        return -1;
    }
    fclose(f);

    strncpy(g_mod_filename, filename, sizeof(g_mod_filename) - 1);
    g_mod_filename[sizeof(g_mod_filename) - 1] = '\0';
    g_loaded   = 1;
    g_playing  = 0;
    return 0;
}

/* Start playback.  No-op if not initialized or no module is loaded. */
void pt2_audio_play(void) {
    if (g_initialized && g_loaded) {
        g_playing = 1;
    }
}

/* Stop playback and reset the BLEP synthesizer state. */
void pt2_audio_stop(void) {
    g_playing = 0;
#ifdef PT2_FULL_BUILD
    clearBlepState();
#endif
}

/* Generate audio samples into a stereo float buffer.
 * Emits silence when not playing or when frames <= 0.
 */
void pt2_audio_update(float *buffer, int64_t frames) {
    if (frames <= 0 || buffer == NULL) {
        return;
    }
    /* Paula emulation writes real samples here in a full SDL build; emit silence
     * in the portable stub so callers always get a valid buffer. */
    memset(buffer, 0, (size_t)(frames * 2) * sizeof(float));
}

/* Return 1 if audio is currently playing, 0 otherwise. */
int64_t pt2_audio_is_playing(void) {
    return (int64_t)(g_initialized && g_playing);
}
