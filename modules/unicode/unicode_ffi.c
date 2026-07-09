/* Unicode FFI bindings for NanoLang
 * Provides grapheme-aware string operations via utf8proc
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <utf8proc.h>

/* Get byte length of UTF-8 string (explicit, replaces ambiguous str_length) */
int64_t nl_str_byte_length(const char* str) {
    if (!str) return 0;
    return (int64_t)strlen(str);
}

/* Get grapheme cluster count (user-perceived characters) */
int64_t nl_str_grapheme_length(const char* str) {
    if (!str) return 0;
    
    const uint8_t* input = (const uint8_t*)str;
    int64_t grapheme_count = 0;
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t bytes_read;
    size_t offset = 0;
    size_t len = strlen(str);
    
    utf8proc_int32_t prev_codepoint = -1;
    
    while (offset < len) {
        bytes_read = utf8proc_iterate(input + offset, len - offset, &codepoint);
        
        if (bytes_read <= 0) {
            // Invalid UTF-8 sequence, count as one grapheme
            grapheme_count++;
            offset++;
            continue;
        }
        
        // Check if this codepoint breaks from previous grapheme
        if (prev_codepoint >= 0) {
            // Use grapheme break algorithm
            bool should_break = utf8proc_grapheme_break(prev_codepoint, codepoint);
            if (should_break) {
                grapheme_count++;
            }
        } else {
            // First character
            grapheme_count++;
        }
        
        prev_codepoint = codepoint;
        offset += bytes_read;
    }
    
    return grapheme_count;
}

/* Get Unicode codepoint at byte index */
int64_t nl_str_codepoint_at(const char* str, int64_t byte_index) {
    if (!str || byte_index < 0) return -1;
    
    const uint8_t* input = (const uint8_t*)str;
    size_t len = strlen(str);
    
    if ((size_t)byte_index >= len) return -1;
    
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t bytes_read = utf8proc_iterate(input + byte_index, len - byte_index, &codepoint);
    
    if (bytes_read <= 0) return -1;
    return (int64_t)codepoint;
}

/* Get grapheme cluster at grapheme index (returns string) */
char* nl_str_grapheme_at(const char* str, int64_t grapheme_index) {
    if (!str || grapheme_index < 0) return NULL;
    
    const uint8_t* input = (const uint8_t*)str;
    int64_t current_grapheme = 0;
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t bytes_read;
    size_t offset = 0;
    size_t len = strlen(str);
    size_t grapheme_start = 0;
    size_t grapheme_end = 0;
    
    utf8proc_int32_t prev_codepoint = -1;
    
    while (offset < len) {
        bytes_read = utf8proc_iterate(input + offset, len - offset, &codepoint);
        
        if (bytes_read <= 0) {
            offset++;
            continue;
        }
        
        // Check grapheme boundary
        if (prev_codepoint >= 0) {
            bool should_break = utf8proc_grapheme_break(prev_codepoint, codepoint);
            if (should_break) {
                if (current_grapheme == grapheme_index) {
                    // Found target grapheme, return substring
                    grapheme_end = offset;
                    size_t grapheme_len = grapheme_end - grapheme_start;
                    char* result = malloc(grapheme_len + 1);
                    if (!result) return NULL;
                    memcpy(result, str + grapheme_start, grapheme_len);
                    result[grapheme_len] = '\0';
                    return result;
                }
                current_grapheme++;
                grapheme_start = offset;
            }
        } else {
            grapheme_start = 0;
        }
        
        prev_codepoint = codepoint;
        offset += bytes_read;
    }
    
    // Check if we're at the last grapheme
    if (current_grapheme == grapheme_index) {
        grapheme_end = len;
        size_t grapheme_len = grapheme_end - grapheme_start;
        char* result = malloc(grapheme_len + 1);
        if (!result) return NULL;
        memcpy(result, str + grapheme_start, grapheme_len);
        result[grapheme_len] = '\0';
        return result;
    }
    
    return NULL;
}

/* Convert to lowercase (Unicode-aware) */
char* nl_str_to_lowercase(const char* str) {
    if (!str) return NULL;
    
    const uint8_t* input = (const uint8_t*)str;
    utf8proc_uint8_t* result = NULL;
    
    utf8proc_map(
        input,
        0,  // 0 means string is null-terminated
        &result,
        UTF8PROC_NULLTERM | UTF8PROC_STABLE | UTF8PROC_COMPOSE | UTF8PROC_CASEFOLD | UTF8PROC_COMPAT
    );
    
    return (char*)result;
}

/* Convert to uppercase (Unicode-aware) */
char* nl_str_to_uppercase(const char* str) {
    if (!str) return NULL;
    
    const uint8_t* input = (const uint8_t*)str;
    utf8proc_uint8_t* result = NULL;
    
    // Note: utf8proc doesn't have a direct uppercase function
    // We need to iterate and apply toupper to each codepoint
    size_t len = strlen(str);
    size_t result_capacity = len * 4;  // Allocate generous buffer
    char* output = malloc(result_capacity);
    if (!output) return NULL;
    
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t bytes_read;
    size_t input_offset = 0;
    size_t output_offset = 0;
    
    while (input_offset < len) {
        bytes_read = utf8proc_iterate(input + input_offset, len - input_offset, &codepoint);
        
        if (bytes_read <= 0) {
            output[output_offset++] = input[input_offset++];
            continue;
        }
        
        // Convert to uppercase
        utf8proc_int32_t upper = utf8proc_toupper(codepoint);
        
        // Encode back to UTF-8
        utf8proc_uint8_t buffer[4];
        utf8proc_ssize_t encoded_bytes = utf8proc_encode_char(upper, buffer);
        
        if (encoded_bytes > 0 && output_offset + encoded_bytes < result_capacity) {
            memcpy(output + output_offset, buffer, encoded_bytes);
            output_offset += encoded_bytes;
        }
        
        input_offset += bytes_read;
    }
    
    output[output_offset] = '\0';
    return output;
}

/* Unicode normalization */
char* nl_str_normalize(const char* str, int64_t form) {
    if (!str) return NULL;
    
    const uint8_t* input = (const uint8_t*)str;
    utf8proc_uint8_t* result = NULL;
    
    utf8proc_option_t options = UTF8PROC_NULLTERM | UTF8PROC_STABLE;
    
    switch (form) {
        case 0:  // NFC (Canonical Composition)
            options |= UTF8PROC_COMPOSE;
            break;
        case 1:  // NFD (Canonical Decomposition)
            options |= UTF8PROC_DECOMPOSE;
            break;
        case 2:  // NFKC (Compatibility Composition)
            options |= UTF8PROC_COMPOSE | UTF8PROC_COMPAT;
            break;
        case 3:  // NFKD (Compatibility Decomposition)
            options |= UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT;
            break;
        default:
            return NULL;
    }
    
    utf8proc_map(input, 0, &result, options);
    
    return (char*)result;
}

/* Check if string contains only ASCII */
bool nl_str_is_ascii(const char* str) {
    if (!str) return true;
    
    const uint8_t* bytes = (const uint8_t*)str;
    size_t len = strlen(str);
    
    for (size_t i = 0; i < len; i++) {
        if (bytes[i] > 127) {
            return false;
        }
    }
    
    return true;
}

/* Check if string is valid UTF-8 */
bool nl_str_is_valid_utf8(const char* str) {
    if (!str) return true;
    
    const uint8_t* input = (const uint8_t*)str;
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t bytes_read;
    size_t offset = 0;
    size_t len = strlen(str);
    
    while (offset < len) {
        bytes_read = utf8proc_iterate(input + offset, len - offset, &codepoint);
        
        if (bytes_read <= 0) {
            return false;  // Invalid UTF-8 sequence
        }
        
        offset += bytes_read;
    }
    
    return true;
}

