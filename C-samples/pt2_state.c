/*
 * ProTracker State Management
 * Workaround for nanolang limitation: mutable variables not allowed at top level
 * Stores replayer state in C to enable state mutation from nanolang
 */

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

// === INITIALIZATION ===

extern int pt2_init_state(void) {
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

extern int pt2_start_playback(void) {
    g_playing = 1;
    g_tick = 0;
    return 0;
}

extern int pt2_stop_playback(void) {
    g_playing = 0;
    g_tick = 0;
    return 0;
}

extern int pt2_is_playing(void) {
    return g_playing;
}

// === STATE GETTERS ===

extern int pt2_get_current_row(void) {
    return g_current_row;
}

extern int pt2_get_current_pattern(void) {
    return g_current_pattern;
}

extern int pt2_get_speed(void) {
    return g_speed;
}

extern int pt2_get_bpm(void) {
    return g_bpm;
}

extern int pt2_get_tick(void) {
    return g_tick;
}

// === STATE SETTERS ===

extern void pt2_set_current_row(int row) {
    g_current_row = row;
}

extern void pt2_set_current_pattern(int pattern) {
    g_current_pattern = pattern;
}

extern void pt2_set_speed(int speed) {
    g_speed = speed;
}

extern void pt2_set_bpm(int bpm) {
    g_bpm = bpm;
}

extern void pt2_set_tick(int tick) {
    g_tick = tick;
}

extern void pt2_set_song_pos(int pos) {
    g_song_pos = pos;
}

extern int pt2_get_song_pos(void) {
    return g_song_pos;
}

// === CHANNEL STATE ===

extern void pt2_set_channel_period(int channel, int period) {
    if (channel >= 0 && channel < 4) {
        g_ch_period[channel] = period;
    }
}

extern int pt2_get_channel_period(int channel) {
    if (channel >= 0 && channel < 4) {
        return g_ch_period[channel];
    }
    return 0;
}

extern void pt2_set_channel_volume(int channel, int volume) {
    if (channel >= 0 && channel < 4) {
        g_ch_volume[channel] = volume;
    }
}

extern int pt2_get_channel_volume(int channel) {
    if (channel >= 0 && channel < 4) {
        return g_ch_volume[channel];
    }
    return 0;
}

extern void pt2_set_channel_sample(int channel, int sample) {
    if (channel >= 0 && channel < 4) {
        g_ch_sample[channel] = sample;
    }
}

extern int pt2_get_channel_sample(int channel) {
    if (channel >= 0 && channel < 4) {
        return g_ch_sample[channel];
    }
    return 0;
}

extern void pt2_set_channel_note(int channel, int note) {
    if (channel >= 0 && channel < 4) {
        g_ch_note[channel] = note;
    }
}

extern int pt2_get_channel_note(int channel) {
    if (channel >= 0 && channel < 4) {
        return g_ch_note[channel];
    }
    return 0;
}

// === PATTERN TABLE ===

extern void pt2_set_pattern_table_entry(int pos, int pattern) {
    if (pos >= 0 && pos < 128) {
        g_pattern_table[pos] = pattern;
    }
}

extern int pt2_get_pattern_table_entry(int pos) {
    if (pos >= 0 && pos < 128) {
        return g_pattern_table[pos];
    }
    return 0;
}

extern void pt2_set_song_length(int length) {
    if (length > 0 && length <= 128) {
        g_song_length = length;
    }
}

extern int pt2_get_song_length(void) {
    return g_song_length;
}
