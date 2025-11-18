/* Audio Helper Functions
 * Utilities for ProTracker sample conversion and playback
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* WAV file header structure */
typedef struct {
    char riff[4];              // "RIFF"
    uint32_t file_size;        // File size - 8
    char wave[4];              // "WAVE"
    char fmt[4];               // "fmt "
    uint32_t fmt_size;         // 16 for PCM
    uint16_t audio_format;     // 1 for PCM
    uint16_t num_channels;     // 1 = mono, 2 = stereo
    uint32_t sample_rate;      // 8363 for Amiga
    uint32_t byte_rate;        // sample_rate * num_channels * bits_per_sample/8
    uint16_t block_align;      // num_channels * bits_per_sample/8
    uint16_t bits_per_sample;  // 8 or 16
    char data[4];              // "data"
    uint32_t data_size;        // Size of PCM data
} wav_header_t;

/* Convert raw 8-bit signed PCM to WAV file
 * Returns 0 on success, -1 on error
 */
int64_t audio_convert_raw_to_wav(const char *raw_file, const char *wav_file, int64_t sample_rate) {
    FILE *in = fopen(raw_file, "rb");
    if (!in) {
        printf("Error: Cannot open %s\n", raw_file);
        return -1;
    }
    
    // Get file size
    fseek(in, 0, SEEK_END);
    long data_size = ftell(in);
    fseek(in, 0, SEEK_SET);
    
    // Read PCM data
    uint8_t *pcm_data = (uint8_t*)malloc(data_size);
    if (!pcm_data) {
        fclose(in);
        return -1;
    }
    
    fread(pcm_data, 1, data_size, in);
    fclose(in);
    
    // Convert 8-bit signed to unsigned (WAV format)
    for (long i = 0; i < data_size; i++) {
        pcm_data[i] = ((int8_t)pcm_data[i]) + 128;
    }
    
    // Create WAV header
    wav_header_t header;
    memcpy(header.riff, "RIFF", 4);
    header.file_size = 36 + data_size;
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt, "fmt ", 4);
    header.fmt_size = 16;
    header.audio_format = 1;  // PCM
    header.num_channels = 1;  // Mono
    header.sample_rate = (uint32_t)sample_rate;
    header.bits_per_sample = 8;
    header.byte_rate = (uint32_t)sample_rate * 1 * 1;  // rate * channels * bytes
    header.block_align = 1;  // channels * bytes
    memcpy(header.data, "data", 4);
    header.data_size = (uint32_t)data_size;
    
    // Write WAV file
    FILE *out = fopen(wav_file, "wb");
    if (!out) {
        free(pcm_data);
        printf("Error: Cannot create %s\n", wav_file);
        return -1;
    }
    
    fwrite(&header, sizeof(wav_header_t), 1, out);
    fwrite(pcm_data, 1, data_size, out);
    fclose(out);
    free(pcm_data);
    
    return 0;
}
