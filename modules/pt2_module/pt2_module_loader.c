#include "pt2_module_loader.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static uint16_t read_be16(const uint8_t *p) {
    return (uint16_t)((p[0] << 8) | p[1]);
}

static void copy_text_trim(char *dst, size_t dst_cap, const uint8_t *src, size_t src_len) {
    if (!dst || dst_cap == 0) return;
    size_t n = (src_len < (dst_cap - 1)) ? src_len : (dst_cap - 1);
    memcpy(dst, src, n);
    dst[n] = 0;
    /* Trim trailing spaces/NULs. */
    while (n > 0) {
        char c = dst[n - 1];
        if (c == 0 || c == ' ') {
            dst[n - 1] = 0;
            n--;
            continue;
        }
        break;
    }
}

static int is_supported_sig(const uint8_t sig[4]) {
    /* Accept common 4ch signatures. */
    if (memcmp(sig, "M.K.", 4) == 0) return 1;
    if (memcmp(sig, "M!K!", 4) == 0) return 1;
    if (memcmp(sig, "4CHN", 4) == 0) return 1;
    if (memcmp(sig, "FLT4", 4) == 0) return 1;
    return 0;
}

static void decode_note(pt2_note_t *out, const uint8_t b[4]) {
    const uint8_t b0 = b[0];
    const uint8_t b1 = b[1];
    const uint8_t b2 = b[2];
    const uint8_t b3 = b[3];

    out->period = (uint16_t)(((b0 & 0x0F) << 8) | b1);
    out->sample = (uint8_t)((b0 & 0xF0) | (b2 >> 4));
    out->command = (uint8_t)(b2 & 0x0F);
    out->param = b3;
}

int pt2_mod_load_file(pt2_mod_t *out, const char *filename) {
    if (!out || !filename) return -1;
    memset(out, 0, sizeof(*out));

    FILE *f = fopen(filename, "rb");
    if (!f) return -1;

    uint8_t hdr[1084];
    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fclose(f);
        return -1;
    }

    if (!is_supported_sig(&hdr[1080])) {
        fclose(f);
        return -1;
    }

    copy_text_trim(out->name, sizeof(out->name), &hdr[0], 20);

    /* Samples: 31 headers * 30 bytes starting at offset 20. */
    const size_t sample_base = 20;
    for (int i = 0; i < 31; i++) {
        const uint8_t *s = &hdr[sample_base + (size_t)i * 30];
        copy_text_trim(out->samples[i].name, sizeof(out->samples[i].name), &s[0], 22);

        const uint16_t len_words = read_be16(&s[22]);
        out->samples[i].length_bytes = (uint32_t)len_words * 2u;
        out->samples[i].finetune = (uint8_t)(s[24] & 0x0F);
        out->samples[i].volume = s[25];

        const uint16_t loop_start_words = read_be16(&s[26]);
        const uint16_t loop_len_words = read_be16(&s[28]);
        out->samples[i].loop_start_bytes = (uint32_t)loop_start_words * 2u;
        out->samples[i].loop_length_bytes = (uint32_t)loop_len_words * 2u;
    }

    out->song_length = hdr[950];
    out->restart_pos = hdr[951];
    memcpy(out->pattern_table, &hdr[952], 128);

    uint8_t max_pat = 0;
    for (int i = 0; i < 128; i++) {
        if (out->pattern_table[i] > max_pat) max_pat = out->pattern_table[i];
    }
    out->pattern_count = (uint8_t)(max_pat + 1);
    if (out->pattern_count == 0) {
        fclose(f);
        return -1;
    }

    const size_t pat_entries = (size_t)out->pattern_count * 64u * 4u;
    out->patterns = (pt2_note_t *)calloc(pat_entries, sizeof(pt2_note_t));
    if (!out->patterns) {
        fclose(f);
        return -1;
    }

    /* Pattern data starts immediately after header. */
    const size_t pat_bytes = (size_t)out->pattern_count * 64u * 4u * 4u;
    uint8_t *pat_raw = (uint8_t *)malloc(pat_bytes);
    if (!pat_raw) {
        pt2_mod_free(out);
        fclose(f);
        return -1;
    }

    if (fread(pat_raw, 1, pat_bytes, f) != pat_bytes) {
        free(pat_raw);
        pt2_mod_free(out);
        fclose(f);
        return -1;
    }

    for (size_t i = 0; i < pat_entries; i++) {
        decode_note(&out->patterns[i], &pat_raw[i * 4]);
    }

    free(pat_raw);
    fclose(f);
    return 0;
}

void pt2_mod_free(pt2_mod_t *m) {
    if (!m) return;
    free(m->patterns);
    m->patterns = NULL;
    m->pattern_count = 0;
}
