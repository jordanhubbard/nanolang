/* ProTracker Audio Module Wrapper
 * Wraps pt2-clone's Paula emulation as a simple C module for nanolang.
 *
 * Real state machine: init -> load_mod -> play -> stop.
 *
 * pt2_audio_update() advances a compact ProTracker replayer.  Every CIA tick
 * it programs the real Paula registers (AUDxLC/LEN/PER/VOL and DMACON) exactly
 * like the Amiga hardware, then renders the tick through paulaGenerateSamples()
 * from pt2_paula.c.  This is deliberately SDL-free: it drives Paula directly
 * instead of the GUI-bound pt2_replayer.c, so it produces real audio in both
 * the full SDL build and headless builds/tests.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pt2_paula.h"
#include "pt2_module_loader.h"

/* Paula register addresses (per voice, 16-byte stride starting at voice 0). */
#define PAULA_AUD0DAT 0xDFF0A0u /* AUDxLC (sample data pointer) */
#define PAULA_AUD0LEN 0xDFF0A4u /* AUDxLEN (length in words)    */
#define PAULA_AUD0PER 0xDFF0A6u /* AUDxPER (period)             */
#define PAULA_AUD0VOL 0xDFF0A8u /* AUDxVOL (volume 0..64)       */
#define PAULA_DMACON  0xDFF096u
#define PAULA_VOICE_STRIDE 16u

/* Amiga PAL clock used to translate a ProTracker period into an output pitch.
 * paula.c uses the same constant internally. */
#define PT2_PAL_CLK (28375160.0 / 8.0)

/* Standard ProTracker period table (finetune 0), 36 notes + terminator. */
static const uint16_t g_period_table[37] = {
    856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453,
    428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,
    214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113,
    0
};

/* Internal state */
static int      g_initialized   = 0;
static int64_t  g_sample_rate   = 0;
static int      g_playing       = 0;
static int      g_loaded        = 0;
static char     g_mod_filename[4096] = {0};
static pt2_mod_t g_mod;

/* Replayer song position. */
typedef struct pt2_channel {
    const int8_t *loop_start; /* points inside sample data, or NULL */
    uint16_t      loop_len_words;
} pt2_channel_t;

static pt2_channel_t g_channels[PAULA_VOICES];
static int32_t g_song_pos;    /* index into pattern_table */
static int32_t g_row;         /* 0..63 */
static int32_t g_tick;        /* 0..speed-1 */
static int32_t g_speed;       /* ticks per row (default 6) */
static int32_t g_bpm;         /* default 125 */
static double  g_samples_per_tick;
static double  g_tick_remainder;

static void reset_replayer_position(void) {
    g_song_pos = 0;
    g_row = 0;
    g_tick = 0;
    g_speed = 6;
    g_bpm = 125;
    g_tick_remainder = 0.0;
    memset(g_channels, 0, sizeof(g_channels));
}

static void update_samples_per_tick(void) {
    /* PT tick rate = CIA at (BPM based) -> 50 * BPM / 125 Hz nominally. */
    const double tick_hz = (g_bpm > 0) ? (g_bpm * 50.0 / 125.0) : (125.0 * 50.0 / 125.0);
    g_samples_per_tick = (double)g_sample_rate / tick_hz;
    if (g_samples_per_tick < 1.0) g_samples_per_tick = 1.0;
}

static void voice_write_word(int ch, uint32_t base, uint16_t value) {
    paulaWriteWord(base + (uint32_t)ch * PAULA_VOICE_STRIDE, value);
}

/* Trigger a note on a Paula voice from a decoded pattern cell. */
static void trigger_note(int ch, const pt2_note_t *note) {
    const uint32_t addr = (uint32_t)ch * PAULA_VOICE_STRIDE;

    if (note->sample >= 1 && note->sample <= 31) {
        const pt2_sample_t *smp = &g_mod.samples[note->sample - 1];
        /* Volume comes from the sample header unless overridden below. */
        voice_write_word(ch, PAULA_AUD0VOL, smp->volume);

        if (smp->data != NULL && smp->length_bytes >= 2) {
            /* Point the voice at the sample and start DMA. */
            paulaWritePtr(PAULA_AUD0DAT + addr, smp->data);
            voice_write_word(ch, PAULA_AUD0LEN, (uint16_t)(smp->length_bytes / 2u));

            /* Remember the loop so Paula can latch it after the one-shot pass. */
            if (smp->loop_length_bytes > 2) {
                g_channels[ch].loop_start = smp->data + smp->loop_start_bytes;
                g_channels[ch].loop_len_words = (uint16_t)(smp->loop_length_bytes / 2u);
            } else {
                g_channels[ch].loop_start = NULL;
                g_channels[ch].loop_len_words = 0;
            }
        }
    }

    if (note->period > 0) {
        voice_write_word(ch, PAULA_AUD0PER, note->period);
        if (note->sample >= 1 && note->sample <= 31) {
            const pt2_sample_t *smp = &g_mod.samples[note->sample - 1];
            if (smp->data != NULL && smp->length_bytes >= 2) {
                /* DMACON: enable this voice (bit set + master flag 0x8000). */
                paulaWriteWord(PAULA_DMACON, (uint16_t)(0x8000u | (1u << ch)));
                /* Latch the loop registers so the next DMA wrap loops correctly. */
                if (g_channels[ch].loop_start != NULL) {
                    paulaWritePtr(PAULA_AUD0DAT + addr, g_channels[ch].loop_start);
                    voice_write_word(ch, PAULA_AUD0LEN, g_channels[ch].loop_len_words);
                }
            }
        }
    }

    /* Volume command (Cxx) and set speed/tempo (Fxx) are the two effects that
     * matter for producing correct, non-silent output. */
    if (note->command == 0x0C) {
        voice_write_word(ch, PAULA_AUD0VOL, note->param);
    } else if (note->command == 0x0F) {
        if (note->param == 0) {
            /* no-op */
        } else if (note->param < 0x20) {
            g_speed = note->param;
        } else {
            g_bpm = note->param;
            update_samples_per_tick();
        }
    }
}

/* Process one replayer tick.  On tick 0 of a row, read and apply the row. */
static void replayer_tick(void) {
    if (g_tick == 0) {
        if (g_song_pos >= g_mod.song_length) {
            g_song_pos = g_mod.restart_pos;
            if (g_song_pos >= g_mod.song_length) g_song_pos = 0;
        }
        const int pat = g_mod.pattern_table[g_song_pos];
        for (int ch = 0; ch < PAULA_VOICES; ch++) {
            pt2_note_t n = pt2_mod_get_note(&g_mod, pat, g_row, ch);
            trigger_note(ch, &n);
        }
    }

    g_tick++;
    if (g_tick >= g_speed) {
        g_tick = 0;
        g_row++;
        if (g_row >= 64) {
            g_row = 0;
            g_song_pos++;
        }
    }
}

/* Render `frames` stereo frames of the current tick's audio into `buffer`. */
static void render_frames(float *buffer, int64_t frames) {
    enum { CHUNK = 256 };
    double left[CHUNK];
    double right[CHUNK];

    int64_t done = 0;
    while (done < frames) {
        int32_t n = (int32_t)((frames - done) < CHUNK ? (frames - done) : CHUNK);
        paulaGenerateSamples(left, right, n);
        for (int32_t i = 0; i < n; i++) {
            buffer[(done + i) * 2 + 0] = (float)left[i];
            buffer[(done + i) * 2 + 1] = (float)right[i];
        }
        done += n;
    }
}

/* Initialize the audio system and Paula emulator.
 * Must be called before any other pt2_audio function.
 * Returns 0 on success, -1 on failure.
 */
int64_t pt2_audio_init(int64_t sample_rate) {
    g_sample_rate   = sample_rate;
    g_playing       = 0;
    g_loaded        = 0;
    g_mod_filename[0] = '\0';

    /* Paula requires an output frequency >= ceil(PAL_CLK / 113) (~31389 Hz). */
    paulaSetup((double)sample_rate, MODEL_A500);
    paulaDisableFilters();

    reset_replayer_position();
    update_samples_per_tick();

    g_initialized = 1;
    return 0;
}

/* Load a MOD file for playback.
 * Returns 0 on success, -1 if the file cannot be opened or invalid format,
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

    if (g_loaded) {
        pt2_mod_free(&g_mod);
        g_loaded = 0;
    }

    if (pt2_mod_load_file(&g_mod, filename) != 0) {
        return -1;
    }

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
        reset_replayer_position();
        update_samples_per_tick();
        clearBlepState();
        /* Silence any lingering DMA before the first tick programs the voices. */
        paulaWriteWord(PAULA_DMACON, 0x000Fu); /* clear all four voices */
    }
}

/* Stop playback and reset the BLEP synthesizer state. */
void pt2_audio_stop(void) {
    g_playing = 0;
    paulaWriteWord(PAULA_DMACON, 0x000Fu); /* clear all four voices */
    clearBlepState();
}

/* Generate audio samples into a stereo float buffer.
 * Emits silence when not playing or when frames <= 0.
 */
void pt2_audio_update(float *buffer, int64_t frames) {
    if (frames <= 0 || buffer == NULL) {
        return;
    }

    if (!g_playing || !g_loaded) {
        memset(buffer, 0, (size_t)(frames * 2) * sizeof(float));
        return;
    }

    int64_t produced = 0;
    while (produced < frames) {
        if (g_tick_remainder < 1.0) {
            replayer_tick();
            g_tick_remainder += g_samples_per_tick;
        }

        int64_t want = frames - produced;
        int64_t avail = (int64_t)g_tick_remainder;
        int64_t n = (want < avail) ? want : avail;
        if (n <= 0) n = 1; /* always make forward progress */

        render_frames(buffer + produced * 2, n);
        produced += n;
        g_tick_remainder -= (double)n;
    }
}

/* Return 1 if audio is currently playing, 0 otherwise. */
int64_t pt2_audio_is_playing(void) {
    return (int64_t)(g_initialized && g_playing);
}

/* Return the output pitch (Hz) for a ProTracker period, or 0 for period 0.
 * Exposed for tests that assert the replayer's pitch math. */
double pt2_audio_period_to_hz(int period) {
    if (period <= 0) {
        return 0.0;
    }
    return PT2_PAL_CLK / (double)period;
}

/* Return the ProTracker period for a note index 0..35 (finetune 0), else 0. */
int pt2_audio_note_period(int note_index) {
    if (note_index < 0 || note_index > 35) {
        return 0;
    }
    return (int)g_period_table[note_index];
}
