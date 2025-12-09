#include <SDL2/SDL_mixer.h>
#include <math.h>
#include <string.h>

// Audio visualization state
static float channel_volumes[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static float smoothed_volumes[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static int audio_format = 0;
static int audio_channels = 0;

// Waveform buffer for oscilloscope display
#define WAVEFORM_BUFFER_SIZE 1024
static float waveform_left[WAVEFORM_BUFFER_SIZE];
static float waveform_right[WAVEFORM_BUFFER_SIZE];
static int waveform_write_pos = 0;

// Audio callback that analyzes the audio stream
static void audio_callback(void *udata, Uint8 *stream, int len) {
    (void)udata;  // Unused
    
    if (audio_format == 0) return;  // Not initialized
    
    // Calculate number of samples (16-bit stereo = 4 bytes per frame)
    int sample_count = len / 4;  // 2 bytes per sample * 2 channels
    Sint16 *samples = (Sint16 *)stream;
    
    // For stereo audio, we'll split into left/right
    // For MOD files, this gives us a rough approximation
    float left_sum = 0.0f, right_sum = 0.0f;
    
    // Store samples for waveform display (decimated to fit buffer)
    int decimation = (sample_count > WAVEFORM_BUFFER_SIZE) ? 
                     (sample_count / WAVEFORM_BUFFER_SIZE) : 1;
    
    for (int i = 0; i < sample_count; i++) {
        Sint16 left = samples[i * 2];
        Sint16 right = samples[i * 2 + 1];
        
        // Store waveform samples (decimated)
        if (i % decimation == 0 && waveform_write_pos < WAVEFORM_BUFFER_SIZE) {
            waveform_left[waveform_write_pos] = (float)left / 32768.0f;
            waveform_right[waveform_write_pos] = (float)right / 32768.0f;
            waveform_write_pos++;
        }
        
        // Calculate absolute values (peak detection)
        float left_val = fabsf((float)left / 32768.0f);
        float right_val = fabsf((float)right / 32768.0f);
        
        left_sum += left_val * left_val;  // RMS
        right_sum += right_val * right_val;
    }
    
    // Reset waveform buffer position for next frame
    waveform_write_pos = 0;
    
    // Calculate RMS and scale up for visibility
    float left_rms = sqrtf(left_sum / sample_count) * 2.0f;
    float right_rms = sqrtf(right_sum / sample_count) * 2.0f;
    
    // Clamp to 0-1 range
    if (left_rms > 1.0f) left_rms = 1.0f;
    if (right_rms > 1.0f) right_rms = 1.0f;
    
    // For 4 channels, we'll distribute the stereo into pairs
    // Channels 1,3 use left, channels 2,4 use right
    // Add some variation to make it more interesting
    channel_volumes[0] = left_rms;
    channel_volumes[1] = right_rms;
    channel_volumes[2] = left_rms * 0.8f;
    channel_volumes[3] = right_rms * 0.9f;
    
    // Smooth the values for better visual appearance
    float smoothing = 0.3f;
    for (int i = 0; i < 4; i++) {
        smoothed_volumes[i] = smoothed_volumes[i] * (1.0f - smoothing) + 
                              channel_volumes[i] * smoothing;
    }
}

// Initialize audio visualization
void nl_audio_viz_init(int format, int channels) {
    audio_format = format;
    audio_channels = channels;
    Mix_SetPostMix(audio_callback, NULL);
}

// Get volume for a specific channel (0-3)
// Returns value between 0.0 and 1.0
float nl_audio_viz_get_channel_volume(int channel) {
    if (channel < 0 || channel >= 4) return 0.0f;
    return smoothed_volumes[channel];
}

// Get volume as integer (0-100) for easier use in nanolang
int nl_audio_viz_get_channel_volume_int(int channel) {
    return (int)(nl_audio_viz_get_channel_volume(channel) * 100.0f);
}

// Get waveform sample at index for oscilloscope display
// channel: 0 = left, 1 = right
// index: 0 to WAVEFORM_BUFFER_SIZE-1
// Returns value between -1.0 and 1.0
float nl_audio_viz_get_waveform_sample(int channel, int index) {
    if (index < 0 || index >= WAVEFORM_BUFFER_SIZE) return 0.0f;
    
    if (channel == 0) {
        return waveform_left[index];
    } else if (channel == 1) {
        return waveform_right[index];
    }
    return 0.0f;
}

// Get waveform buffer size
int nl_audio_viz_get_waveform_size(void) {
    return WAVEFORM_BUFFER_SIZE;
}

// Shutdown audio visualization
void nl_audio_viz_shutdown(void) {
    Mix_SetPostMix(NULL, NULL);
    memset(channel_volumes, 0, sizeof(channel_volumes));
    memset(smoothed_volumes, 0, sizeof(smoothed_volumes));
    memset(waveform_left, 0, sizeof(waveform_left));
    memset(waveform_right, 0, sizeof(waveform_right));
    waveform_write_pos = 0;
}
