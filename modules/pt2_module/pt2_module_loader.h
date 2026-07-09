#ifndef NANOLANG_PT2_MODULE_LOADER_H
#define NANOLANG_PT2_MODULE_LOADER_H

#include <stdint.h>
#include <stddef.h>

typedef struct pt2_note {
    uint16_t period;
    uint8_t sample;  /* 1..31, 0 if none */
    uint8_t command; /* 0..15 */
    uint8_t param;   /* 0..255 */
} pt2_note_t;

typedef struct pt2_sample {
    char name[23]; /* NUL-terminated */
    uint32_t length_bytes;
    uint8_t finetune;
    uint8_t volume;
    uint32_t loop_start_bytes;
    uint32_t loop_length_bytes;
} pt2_sample_t;

typedef struct pt2_mod {
    char name[21]; /* NUL-terminated */
    uint8_t song_length;
    uint8_t restart_pos;
    uint8_t pattern_table[128];
    uint8_t pattern_count;
    pt2_sample_t samples[31];
    pt2_note_t *patterns; /* pattern_count * 64 * 4 */
} pt2_mod_t;

int pt2_mod_load_file(pt2_mod_t *out, const char *filename);
void pt2_mod_free(pt2_mod_t *m);

static inline pt2_note_t pt2_mod_get_note(const pt2_mod_t *m, int pattern, int row, int channel) {
    /* Caller must validate indices. */
    return m->patterns[(pattern * 64 * 4) + (row * 4) + channel];
}

#endif
