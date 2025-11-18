/**
 * nl_string.c - Nanolang String Implementation
 */

#include "nl_string.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * UTF-8 Helper Functions
 * ============================================================================ */

/**
 * Get number of bytes in UTF-8 sequence from first byte
 * Returns 0 for invalid lead byte
 */
static int utf8_sequence_length(unsigned char first_byte) {
    if ((first_byte & 0x80) == 0) return 1;      // 0xxxxxxx
    if ((first_byte & 0xE0) == 0xC0) return 2;   // 110xxxxx
    if ((first_byte & 0xF0) == 0xE0) return 3;   // 1110xxxx
    if ((first_byte & 0xF8) == 0xF0) return 4;   // 11110xxx
    return 0; // Invalid
}

/**
 * Check if byte is a valid UTF-8 continuation byte
 */
static bool is_utf8_continuation(unsigned char byte) {
    return (byte & 0xC0) == 0x80; // 10xxxxxx
}

/* ============================================================================
 * Creation & Destruction
 * ============================================================================ */

nl_string_t* nl_string_new(const char *cstr) {
    if (!cstr) return NULL;
    
    size_t len = strlen(cstr);
    nl_string_t *str = malloc(sizeof(nl_string_t));
    if (!str) return NULL;
    
    str->capacity = len + 1;
    str->data = malloc(str->capacity);
    if (!str->data) {
        free(str);
        return NULL;
    }
    
    memcpy(str->data, cstr, len);
    str->data[len] = '\0';
    str->length = len;
    str->null_terminated = true;
    str->is_utf8 = false; // Will validate on demand
    
    return str;
}

nl_string_t* nl_string_new_binary(const void *data, size_t length) {
    if (!data && length > 0) return NULL;
    
    nl_string_t *str = malloc(sizeof(nl_string_t));
    if (!str) return NULL;
    
    str->capacity = length;
    str->data = malloc(str->capacity);
    if (!str->data) {
        free(str);
        return NULL;
    }
    
    if (length > 0) {
        memcpy(str->data, data, length);
    }
    str->length = length;
    str->null_terminated = false;
    str->is_utf8 = false;
    
    return str;
}

nl_string_t* nl_string_from_utf8(const char *utf8_data, size_t length) {
    nl_string_t *str = nl_string_new_binary(utf8_data, length);
    if (!str) return NULL;
    
    if (!nl_string_validate_utf8(str)) {
        nl_string_free(str);
        return NULL;
    }
    
    return str;
}

nl_string_t* nl_string_with_capacity(size_t capacity) {
    nl_string_t *str = malloc(sizeof(nl_string_t));
    if (!str) return NULL;
    
    str->capacity = capacity;
    str->data = malloc(capacity);
    if (!str->data) {
        free(str);
        return NULL;
    }
    
    str->length = 0;
    str->null_terminated = false;
    str->is_utf8 = true; // Empty string is valid UTF-8
    
    return str;
}

void nl_string_free(nl_string_t *str) {
    if (!str) return;
    free(str->data);
    free(str);
}

/* ============================================================================
 * Basic Operations
 * ============================================================================ */

size_t nl_string_length(const nl_string_t *str) {
    return str ? str->length : 0;
}

char nl_string_byte_at(const nl_string_t *str, size_t index) {
    return str->data[index];
}

bool nl_string_byte_at_safe(const nl_string_t *str, size_t index, char *out) {
    if (!str || index >= str->length || !out) return false;
    *out = str->data[index];
    return true;
}

nl_string_t* nl_string_concat(const nl_string_t *a, const nl_string_t *b) {
    if (!a || !b) return NULL;
    
    size_t new_len = a->length + b->length;
    nl_string_t *result = nl_string_with_capacity(new_len + 1);
    if (!result) return NULL;
    
    memcpy(result->data, a->data, a->length);
    memcpy(result->data + a->length, b->data, b->length);
    result->data[new_len] = '\0';
    result->length = new_len;
    result->null_terminated = true;
    result->is_utf8 = a->is_utf8 && b->is_utf8;
    
    return result;
}

nl_string_t* nl_string_substring(const nl_string_t *str, size_t start, size_t length) {
    if (!str || start >= str->length) {
        return nl_string_with_capacity(0);
    }
    
    if (start + length > str->length) {
        length = str->length - start;
    }
    
    nl_string_t *result = nl_string_new_binary(str->data + start, length);
    if (result && str->is_utf8) {
        // Substring of UTF-8 string may not be valid UTF-8 if split mid-character
        nl_string_validate_utf8(result);
    }
    
    return result;
}

bool nl_string_equals(const nl_string_t *a, const nl_string_t *b) {
    if (!a || !b) return a == b;
    if (a->length != b->length) return false;
    return memcmp(a->data, b->data, a->length) == 0;
}

bool nl_string_equals_cstr(const nl_string_t *str, const char *cstr) {
    if (!str || !cstr) return false;
    size_t cstr_len = strlen(cstr);
    if (str->length != cstr_len) return false;
    return memcmp(str->data, cstr, cstr_len) == 0;
}

/* ============================================================================
 * UTF-8 Operations
 * ============================================================================ */

bool nl_string_validate_utf8(nl_string_t *str) {
    if (!str) return false;
    
    size_t i = 0;
    while (i < str->length) {
        unsigned char byte = str->data[i];
        int seq_len = utf8_sequence_length(byte);
        
        if (seq_len == 0) {
            str->is_utf8 = false;
            return false;
        }
        
        // Check we have enough bytes
        if (i + seq_len > str->length) {
            str->is_utf8 = false;
            return false;
        }
        
        // Validate continuation bytes
        for (int j = 1; j < seq_len; j++) {
            if (!is_utf8_continuation(str->data[i + j])) {
                str->is_utf8 = false;
                return false;
            }
        }
        
        i += seq_len;
    }
    
    str->is_utf8 = true;
    return true;
}

int64_t nl_string_utf8_length(const nl_string_t *str) {
    if (!str) return -1;
    if (!str->is_utf8) return -1;
    
    int64_t count = 0;
    size_t i = 0;
    
    while (i < str->length) {
        int seq_len = utf8_sequence_length(str->data[i]);
        if (seq_len == 0) return -1; // Shouldn't happen if validated
        i += seq_len;
        count++;
    }
    
    return count;
}

int32_t nl_string_utf8_char_at(const nl_string_t *str, size_t char_index) {
    if (!str || !str->is_utf8) return -1;
    
    size_t current_char = 0;
    size_t i = 0;
    
    while (i < str->length) {
        if (current_char == char_index) {
            // Decode UTF-8 codepoint
            unsigned char byte = str->data[i];
            int seq_len = utf8_sequence_length(byte);
            
            if (seq_len == 1) {
                return byte;
            } else if (seq_len == 2) {
                return ((byte & 0x1F) << 6) | (str->data[i+1] & 0x3F);
            } else if (seq_len == 3) {
                return ((byte & 0x0F) << 12) | 
                       ((str->data[i+1] & 0x3F) << 6) |
                       (str->data[i+2] & 0x3F);
            } else if (seq_len == 4) {
                return ((byte & 0x07) << 18) |
                       ((str->data[i+1] & 0x3F) << 12) |
                       ((str->data[i+2] & 0x3F) << 6) |
                       (str->data[i+3] & 0x3F);
            }
        }
        
        int seq_len = utf8_sequence_length(str->data[i]);
        i += seq_len;
        current_char++;
    }
    
    return -1; // Index out of range
}

nl_string_t* nl_string_utf8_substring(const nl_string_t *str, size_t char_start, size_t char_length) {
    if (!str || !str->is_utf8) return NULL;
    
    // Find byte positions for character positions
    size_t byte_start = 0;
    size_t byte_end = 0;
    size_t current_char = 0;
    size_t i = 0;
    bool found_start = false;
    
    while (i < str->length && current_char <= char_start + char_length) {
        if (current_char == char_start) {
            byte_start = i;
            found_start = true;
        }
        if (current_char == char_start + char_length) {
            byte_end = i;
            break;
        }
        
        int seq_len = utf8_sequence_length(str->data[i]);
        i += seq_len;
        current_char++;
    }
    
    if (!found_start) return nl_string_with_capacity(0);
    if (byte_end == 0) byte_end = str->length;
    
    return nl_string_substring(str, byte_start, byte_end - byte_start);
}

/* ============================================================================
 * Conversion & Interoperability
 * ============================================================================ */

const char* nl_string_to_cstr(nl_string_t *str) {
    if (!str) return "";
    
    if (!str->null_terminated) {
        nl_string_ensure_null_terminated(str);
    }
    
    return str->data;
}

const void* nl_string_to_binary(const nl_string_t *str, size_t *out_length) {
    if (!str) {
        if (out_length) *out_length = 0;
        return NULL;
    }
    
    if (out_length) *out_length = str->length;
    return str->data;
}

void nl_string_ensure_null_terminated(nl_string_t *str) {
    if (!str || str->null_terminated) return;
    
    if (str->length >= str->capacity) {
        // Need to reallocate
        size_t new_capacity = str->length + 1;
        char *new_data = realloc(str->data, new_capacity);
        if (!new_data) return; // Can't allocate, leave unchanged
        str->data = new_data;
        str->capacity = new_capacity;
    }
    
    str->data[str->length] = '\0';
    str->null_terminated = true;
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

void nl_string_reserve(nl_string_t *str, size_t new_capacity) {
    if (!str || new_capacity <= str->capacity) return;
    
    char *new_data = realloc(str->data, new_capacity);
    if (!new_data) return;
    
    str->data = new_data;
    str->capacity = new_capacity;
}

void nl_string_shrink_to_fit(nl_string_t *str) {
    if (!str || str->capacity == str->length) return;
    
    size_t new_capacity = str->length + (str->null_terminated ? 1 : 0);
    char *new_data = realloc(str->data, new_capacity);
    if (!new_data) return;
    
    str->data = new_data;
    str->capacity = new_capacity;
}

nl_string_t* nl_string_clone(const nl_string_t *str) {
    if (!str) return NULL;
    
    nl_string_t *clone = malloc(sizeof(nl_string_t));
    if (!clone) return NULL;
    
    clone->capacity = str->capacity;
    clone->data = malloc(clone->capacity);
    if (!clone->data) {
        free(clone);
        return NULL;
    }
    
    memcpy(clone->data, str->data, str->length);
    if (str->null_terminated) {
        clone->data[str->length] = '\0';
    }
    
    clone->length = str->length;
    clone->is_utf8 = str->is_utf8;
    clone->null_terminated = str->null_terminated;
    
    return clone;
}
