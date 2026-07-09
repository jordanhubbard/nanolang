#include <stdint.h>
#include <string.h>

#include "pt2_module_loader.h"
#include "pt2_module_saver.h"

static pt2_mod_t g_mod;
static int g_loaded = 0;

static int valid_pat_row_ch(int pattern, int row, int channel) {
    if (!g_loaded) return 0;
    if (pattern < 0 || pattern >= (int)g_mod.pattern_count) return 0;
    if (row < 0 || row >= 64) return 0;
    if (channel < 0 || channel >= 4) return 0;
    return 1;
}

int64_t pt2_module_load(const char *filename) {
    if (g_loaded) {
        pt2_mod_free(&g_mod);
        g_loaded = 0;
    }

    if (pt2_mod_load_file(&g_mod, filename) != 0) {
        g_loaded = 0;
        return -1;
    }

    g_loaded = 1;
    return 0;
}

const char *pt2_module_get_name(void) {
    if (!g_loaded) return "";
    return g_mod.name;
}

int64_t pt2_module_get_song_length(void) {
    if (!g_loaded) return 0;
    return (int64_t)g_mod.song_length;
}

int64_t pt2_module_get_pattern_count(void) {
    if (!g_loaded) return 0;
    return (int64_t)g_mod.pattern_count;
}

int64_t pt2_module_get_pattern_table(int64_t position) {
    if (!g_loaded) return 0;
    if (position < 0 || position >= 128) return 0;
    return (int64_t)g_mod.pattern_table[position];
}

int64_t pt2_module_get_sample_length(int64_t sample_num) {
    if (!g_loaded) return 0;
    if (sample_num < 1 || sample_num > 31) return 0;
    return (int64_t)g_mod.samples[sample_num - 1].length_bytes;
}

int64_t pt2_module_get_sample_volume(int64_t sample_num) {
    if (!g_loaded) return 0;
    if (sample_num < 1 || sample_num > 31) return 0;
    return (int64_t)g_mod.samples[sample_num - 1].volume;
}

const char *pt2_module_get_sample_name(int64_t sample_num) {
    if (!g_loaded) return "";
    if (sample_num < 1 || sample_num > 31) return "";
    return g_mod.samples[sample_num - 1].name;
}

int64_t pt2_module_get_note_period(int64_t pattern, int64_t row, int64_t channel) {
    if (!valid_pat_row_ch((int)pattern, (int)row, (int)channel)) return 0;
    const pt2_note_t n = pt2_mod_get_note(&g_mod, (int)pattern, (int)row, (int)channel);
    return (int64_t)n.period;
}

int64_t pt2_module_get_note_sample(int64_t pattern, int64_t row, int64_t channel) {
    if (!valid_pat_row_ch((int)pattern, (int)row, (int)channel)) return 0;
    const pt2_note_t n = pt2_mod_get_note(&g_mod, (int)pattern, (int)row, (int)channel);
    return (int64_t)n.sample;
}

int64_t pt2_module_get_note_command(int64_t pattern, int64_t row, int64_t channel) {
    if (!valid_pat_row_ch((int)pattern, (int)row, (int)channel)) return 0;
    const pt2_note_t n = pt2_mod_get_note(&g_mod, (int)pattern, (int)row, (int)channel);
    return (int64_t)n.command;
}

int64_t pt2_module_get_note_param(int64_t pattern, int64_t row, int64_t channel) {
    if (!valid_pat_row_ch((int)pattern, (int)row, (int)channel)) return 0;
    const pt2_note_t n = pt2_mod_get_note(&g_mod, (int)pattern, (int)row, (int)channel);
    return (int64_t)n.param;
}

int64_t pt2_module_save(const char *filename) {
    (void)filename;
    return -1;
}

void pt2_module_free(void) {
    if (!g_loaded) return;
    pt2_mod_free(&g_mod);
    memset(&g_mod, 0, sizeof(g_mod));
    g_loaded = 0;
}
