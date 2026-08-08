/* Regression test for pt2_audio playback.
 *
 * Historically pt2_audio_update() just memset the output buffer to zero and
 * never called paulaGenerateSamples(), so playback was silent.  These tests
 * assert that:
 *   1. Rendering a synthetic module produces a buffer that is not all zeroes.
 *   2. Rendering the checked-in gabba-studies-12.mod produces non-silent audio.
 *   3. The replayer's period->pitch math matches the Amiga PAL clock.
 *
 * The test drives the pt2_audio wrapper directly (no nanolang runtime and no
 * SDL), exercising the real Paula emulator from pt2_paula.c.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* Wrapper API (declared here to avoid pulling in the .nano FFI header). */
int64_t pt2_audio_init(int64_t sample_rate);
int64_t pt2_audio_load_mod(const char *filename);
void    pt2_audio_play(void);
void    pt2_audio_stop(void);
void    pt2_audio_update(float *buffer, int64_t frames);
int64_t pt2_audio_is_playing(void);
double  pt2_audio_period_to_hz(int period);
int     pt2_audio_note_period(int note_index);

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do { \
        g_checks++; \
        if (!(cond)) { \
            g_failures++; \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        } \
    } while (0)

/* Count how many samples in a rendered stereo buffer are non-zero. */
static int64_t count_nonzero(const float *buf, int64_t frames) {
    int64_t n = 0;
    for (int64_t i = 0; i < frames * 2; i++) {
        if (buf[i] != 0.0f) n++;
    }
    return n;
}

/* Build a minimal but valid 4-channel M.K. MOD in memory and write it to path.
 * One sample (a square wave), one pattern whose first row plays it on channel 0
 * at period 428 (C-2), enough to exercise the full render path. */
static int write_synthetic_mod(const char *path) {
    const int sample_words = 64;          /* 128 bytes of PCM */
    const int sample_bytes = sample_words * 2;

    unsigned char hdr[1084];
    memset(hdr, 0, sizeof(hdr));

    memcpy(hdr, "SYNTHMOD", 8);            /* song name */

    /* Sample 1 header at offset 20. */
    unsigned char *s = &hdr[20];
    memcpy(s, "square", 6);                /* name */
    s[22] = (unsigned char)(sample_words >> 8);
    s[23] = (unsigned char)(sample_words & 0xFF);
    s[24] = 0;                             /* finetune */
    s[25] = 64;                            /* volume */
    s[26] = 0; s[27] = 0;                  /* loop start */
    s[28] = (unsigned char)(sample_words >> 8);
    s[29] = (unsigned char)(sample_words & 0xFF); /* loop length = whole sample */

    hdr[950] = 1;                          /* song length */
    hdr[951] = 0;                          /* restart pos */
    hdr[952] = 0;                          /* order 0 -> pattern 0 */
    memcpy(&hdr[1080], "M.K.", 4);         /* signature */

    /* Pattern 0: 64 rows * 4 channels * 4 bytes. Row 0 ch 0 plays sample 1. */
    unsigned char pattern[64 * 4 * 4];
    memset(pattern, 0, sizeof(pattern));
    const uint16_t period = 428;           /* C-2 */
    const uint8_t sample = 1;
    pattern[0] = (unsigned char)((sample & 0xF0) | ((period >> 8) & 0x0F));
    pattern[1] = (unsigned char)(period & 0xFF);
    pattern[2] = (unsigned char)((sample & 0x0F) << 4); /* command 0 */
    pattern[3] = 0;

    /* Sample PCM: a full-scale square wave so playback is unmistakably audible. */
    int8_t pcm[128];
    for (int i = 0; i < sample_bytes; i++) {
        pcm[i] = (i < sample_bytes / 2) ? (int8_t)100 : (int8_t)-100;
    }

    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ok = 1;
    ok &= (fwrite(hdr, 1, sizeof(hdr), f) == sizeof(hdr));
    ok &= (fwrite(pattern, 1, sizeof(pattern), f) == sizeof(pattern));
    ok &= (fwrite(pcm, 1, sample_bytes, f) == (size_t)sample_bytes);
    fclose(f);
    return ok ? 0 : -1;
}

/* Render up to `max_seconds` of `mod_path` and return the number of non-zero
 * output samples across the whole render.  Rendering stops early once audio has
 * been produced, which keeps the test fast while still covering intros that
 * start with a Cxx volume ramp (as gabba-studies-12.mod does). */
static int64_t render_and_count(const char *mod_path, int64_t sample_rate,
                                double max_seconds) {
    if (pt2_audio_init(sample_rate) != 0) {
        fprintf(stderr, "FAIL: pt2_audio_init failed for %s\n", mod_path);
        g_failures++;
        return 0;
    }
    if (pt2_audio_load_mod(mod_path) != 0) {
        fprintf(stderr, "FAIL: pt2_audio_load_mod failed for %s\n", mod_path);
        g_failures++;
        return 0;
    }
    pt2_audio_play();
    CHECK(pt2_audio_is_playing() == 1, "is_playing should be 1 after play");

    const int64_t frames = 4410; /* 0.1s blocks at 44100 Hz */
    const int64_t max_blocks = (int64_t)((max_seconds * (double)sample_rate) / frames) + 1;
    float *buf = (float *)malloc((size_t)frames * 2 * sizeof(float));
    if (!buf) { g_failures++; return 0; }

    int64_t nonzero = 0;
    for (int64_t block = 0; block < max_blocks; block++) {
        memset(buf, 0, (size_t)frames * 2 * sizeof(float));
        pt2_audio_update(buf, frames);
        int64_t nz = count_nonzero(buf, frames);
        nonzero += nz;
        if (nonzero > 0) break; /* audio confirmed; no need to render further */
    }

    pt2_audio_stop();
    CHECK(pt2_audio_is_playing() == 0, "is_playing should be 0 after stop");

    /* When stopped, the buffer must be silent. */
    memset(buf, 0, (size_t)frames * 2 * sizeof(float));
    pt2_audio_update(buf, frames);
    CHECK(count_nonzero(buf, frames) == 0, "update must be silent when stopped");

    free(buf);
    return nonzero;
}

int main(void) {
    const int64_t sample_rate = 44100;

    /* --- 1. Synthetic module --- */
    char tmpl[] = "/tmp/pt2_synth_XXXXXX.mod";
    char synth_path[64];
    snprintf(synth_path, sizeof(synth_path), "/tmp/pt2_synth_%d.mod", (int)getpid());
    (void)tmpl;
    CHECK(write_synthetic_mod(synth_path) == 0, "write synthetic mod");

    int64_t nz_synth = render_and_count(synth_path, sample_rate, 2.0);
    CHECK(nz_synth > 0, "synthetic module playback must not be all zeroes");
    remove(synth_path);

    /* --- 2. Checked-in real module --- */
    const char *real_mod = "examples/audio/gabba-studies-12.mod";
    FILE *probe = fopen(real_mod, "rb");
    if (probe) {
        fclose(probe);
        int64_t nz_real = render_and_count(real_mod, sample_rate, 20.0);
        CHECK(nz_real > 0, "gabba-studies-12.mod playback must not be all zeroes");
    } else {
        fprintf(stderr, "WARN: %s not found; skipping real-module check\n", real_mod);
    }

    /* --- 3. Pitch math --- */
    /* Period 428 (C-2) should map to ~8287.14 Hz on a PAL Amiga. */
    double hz = pt2_audio_period_to_hz(428);
    CHECK(fabs(hz - 8287.137) < 1.0, "period 428 should be ~8287 Hz");
    CHECK(pt2_audio_period_to_hz(0) == 0.0, "period 0 maps to 0 Hz");
    /* Note index 24 in the table is C-3 (period 214), one octave above C-2. */
    CHECK(pt2_audio_note_period(0) == 856, "note 0 period is 856");
    CHECK(pt2_audio_note_period(12) == 428, "note 12 period is 428");
    CHECK(pt2_audio_period_to_hz(214) > pt2_audio_period_to_hz(428),
          "shorter period is higher pitch");

    if (g_failures == 0) {
        printf("PASS: pt2_audio playback (%d checks)\n", g_checks);
        return 0;
    }
    fprintf(stderr, "FAILED: %d/%d checks failed\n", g_failures, g_checks);
    return 1;
}
