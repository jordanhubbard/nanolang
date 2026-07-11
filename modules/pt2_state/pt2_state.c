/*
 * ProTracker State Management
 * Workaround for nanolang limitation: mutable variables not allowed at top level
 * Stores replayer state in C to enable state mutation from nanolang
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Module state
static int g_playing = 0;
static int g_current_pattern = 0;
static int g_current_row = 0;
static int g_song_pos = 0;
static int g_speed = 6;
static int g_bpm = 125;
static int g_tick = 0;

// Pattern order table
static int g_pattern_table[128];
static int g_song_length = 1;

// Channel state (4 channels)
static int g_ch_period[4] = {0, 0, 0, 0};
static int g_ch_volume[4] = {0, 0, 0, 0};
static int g_ch_sample[4] = {0, 0, 0, 0};
static int g_ch_note[4] = {0, 0, 0, 0};

// Sample data (31 samples)
static int g_sample_length[31];
static int g_sample_volume[31];
static int g_sample_loop_start[31];
static int g_sample_loop_length[31];

// === INITIALIZATION ===

int64_t pt2_init_state(void) {
    g_playing = 0;
    g_current_pattern = 0;
    g_current_row = 0;
    g_song_pos = 0;
    g_speed = 6;
    g_bpm = 125;
    g_tick = 0;

    memset(g_pattern_table, 0, sizeof(g_pattern_table));
    g_pattern_table[0] = 0;
    g_song_length = 1;

    memset(g_ch_period, 0, sizeof(g_ch_period));
    memset(g_ch_volume, 0, sizeof(g_ch_volume));
    memset(g_ch_sample, 0, sizeof(g_ch_sample));
    memset(g_ch_note, 0, sizeof(g_ch_note));

    return 0;
}

// === PLAYBACK CONTROL ===

int64_t pt2_start_playback(void) {
    g_playing = 1;
    g_tick = 0;
    return 0;
}

int64_t pt2_stop_playback(void) {
    g_playing = 0;
    g_tick = 0;
    return 0;
}

int64_t pt2_is_playing(void) {
    return (int64_t)g_playing;
}

// === STATE GETTERS ===

int64_t pt2_get_current_row(void) {
    return (int64_t)g_current_row;
}

int64_t pt2_get_current_pattern(void) {
    return (int64_t)g_current_pattern;
}

int64_t pt2_get_speed(void) {
    return (int64_t)g_speed;
}

int64_t pt2_get_bpm(void) {
    return (int64_t)g_bpm;
}

int64_t pt2_get_tick(void) {
    return (int64_t)g_tick;
}

int64_t pt2_get_song_pos(void) {
    return (int64_t)g_song_pos;
}

// === STATE SETTERS ===

void pt2_set_current_row(int64_t row) {
    g_current_row = (int)row;
}

void pt2_set_current_pattern(int64_t pattern) {
    g_current_pattern = (int)pattern;
}

void pt2_set_speed(int64_t speed) {
    g_speed = (int)speed;
}

void pt2_set_bpm(int64_t bpm) {
    g_bpm = (int)bpm;
}

void pt2_set_tick(int64_t tick) {
    g_tick = (int)tick;
}

void pt2_set_song_pos(int64_t pos) {
    g_song_pos = (int)pos;
}

// === CHANNEL STATE ===

void pt2_set_channel_period(int64_t channel, int64_t period) {
    if (channel >= 0 && channel < 4) {
        g_ch_period[channel] = (int)period;
    }
}

int64_t pt2_get_channel_period(int64_t channel) {
    if (channel >= 0 && channel < 4) {
        return (int64_t)g_ch_period[channel];
    }
    return 0;
}

void pt2_set_channel_volume(int64_t channel, int64_t volume) {
    if (channel >= 0 && channel < 4) {
        g_ch_volume[channel] = (int)volume;
    }
}

int64_t pt2_get_channel_volume(int64_t channel) {
    if (channel >= 0 && channel < 4) {
        return (int64_t)g_ch_volume[channel];
    }
    return 0;
}

void pt2_set_channel_sample(int64_t channel, int64_t sample) {
    if (channel >= 0 && channel < 4) {
        g_ch_sample[channel] = (int)sample;
    }
}

int64_t pt2_get_channel_sample(int64_t channel) {
    if (channel >= 0 && channel < 4) {
        return (int64_t)g_ch_sample[channel];
    }
    return 0;
}

void pt2_set_channel_note(int64_t channel, int64_t note) {
    if (channel >= 0 && channel < 4) {
        g_ch_note[channel] = (int)note;
    }
}

int64_t pt2_get_channel_note(int64_t channel) {
    if (channel >= 0 && channel < 4) {
        return (int64_t)g_ch_note[channel];
    }
    return 0;
}

// === PATTERN TABLE ===

void pt2_set_pattern_table_entry(int64_t pos, int64_t pattern) {
    if (pos >= 0 && pos < 128) {
        g_pattern_table[pos] = (int)pattern;
    }
}

int64_t pt2_get_pattern_table_entry(int64_t pos) {
    if (pos >= 0 && pos < 128) {
        return (int64_t)g_pattern_table[pos];
    }
    return 0;
}

void pt2_set_song_length(int64_t length) {
    if (length > 0 && length <= 128) {
        g_song_length = (int)length;
    }
}

int64_t pt2_get_song_length(void) {
    return (int64_t)g_song_length;
}

// === SAMPLE DATA ===

void pt2_set_sample_length(int64_t sample, int64_t length) {
    if (sample >= 0 && sample < 31) {
        g_sample_length[sample] = (int)length;
    }
}

int64_t pt2_get_sample_length(int64_t sample) {
    if (sample >= 0 && sample < 31) {
        return (int64_t)g_sample_length[sample];
    }
    return 0;
}

void pt2_set_sample_volume(int64_t sample, int64_t volume) {
    if (sample >= 0 && sample < 31) {
        g_sample_volume[sample] = (int)volume;
    }
}

int64_t pt2_get_sample_volume(int64_t sample) {
    if (sample >= 0 && sample < 31) {
        return (int64_t)g_sample_volume[sample];
    }
    return 0;
}

void pt2_set_sample_loop_start(int64_t sample, int64_t start) {
    if (sample >= 0 && sample < 31) {
        g_sample_loop_start[sample] = (int)start;
    }
}

int64_t pt2_get_sample_loop_start(int64_t sample) {
    if (sample >= 0 && sample < 31) {
        return (int64_t)g_sample_loop_start[sample];
    }
    return 0;
}

void pt2_set_sample_loop_length(int64_t sample, int64_t length) {
    if (sample >= 0 && sample < 31) {
        g_sample_loop_length[sample] = (int)length;
    }
}

int64_t pt2_get_sample_loop_length(int64_t sample) {
    if (sample >= 0 && sample < 31) {
        return (int64_t)g_sample_loop_length[sample];
    }
    return 0;
}
