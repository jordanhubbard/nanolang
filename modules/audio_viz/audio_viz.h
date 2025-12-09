#ifndef AUDIO_VIZ_H
#define AUDIO_VIZ_H

// Initialize audio visualization system
void nl_audio_viz_init(int format, int channels);

// Get volume level for a channel (0-3)
// Returns integer 0-100 representing volume level
int nl_audio_viz_get_channel_volume_int(int channel);

// Get waveform sample at index for oscilloscope display
// channel: 0 = left, 1 = right
// Returns value between -1.0 and 1.0
float nl_audio_viz_get_waveform_sample(int channel, int index);

// Get waveform buffer size
int nl_audio_viz_get_waveform_size(void);

// Shutdown audio visualization
void nl_audio_viz_shutdown(void);

#endif // AUDIO_VIZ_H
