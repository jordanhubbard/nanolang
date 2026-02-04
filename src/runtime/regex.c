/*
 * POSIX Regex Implementation for Nanolang
 * Built-in runtime support (not a module)
 */

#define _POSIX_C_SOURCE 200809L  /* For strdup() */

#include <regex.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "dyn_array.h"
#include "gc.h"

typedef struct {
    regex_t compiled;
    int is_valid;
} nl_regex_t;

/* Internal cleanup function (used as GC finalizer) */
static void nl_regex_cleanup(void* regex) {
    if (!regex) return;
    nl_regex_t* re = (nl_regex_t*)regex;
    if (re->is_valid) {
        regfree(&re->compiled);
        re->is_valid = 0;
    }
}

// Compile regex pattern (GC-managed with automatic cleanup)
void* nl_regex_compile(const char* pattern) {
    if (!pattern) return NULL;

    /* Allocate using GC with finalizer for automatic cleanup */
    nl_regex_t* re = (nl_regex_t*)gc_alloc_opaque(sizeof(nl_regex_t), nl_regex_cleanup);
    if (!re) return NULL;

    // REG_EXTENDED = modern regex syntax
    int result = regcomp(&re->compiled, pattern, REG_EXTENDED);
    if (result != 0) {
        /* No need to manually free - GC will handle it */
        return NULL;
    }

    re->is_valid = 1;
    return re;
}

// Check if text matches pattern
int64_t nl_regex_match(void* regex, const char* text) {
    if (!regex || !text) return -1;
    
    nl_regex_t* re = (nl_regex_t*)regex;
    if (!re->is_valid) return -1;
    
    int result = regexec(&re->compiled, text, 0, NULL, 0);
    return (result == 0) ? 1 : 0;
}

// Find first match position
int64_t nl_regex_find(void* regex, const char* text) {
    if (!regex || !text) return -1;
    
    nl_regex_t* re = (nl_regex_t*)regex;
    if (!re->is_valid) return -1;
    
    regmatch_t match;
    int result = regexec(&re->compiled, text, 1, &match, 0);
    
    if (result == 0) {
        return (int64_t)match.rm_so;
    }
    return -1;
}

// Find all matches
DynArray* nl_regex_find_all(void* regex, const char* text) {
    DynArray* positions = dyn_array_new(ELEM_INT);
    if (!regex || !text) return positions;
    
    nl_regex_t* re = (nl_regex_t*)regex;
    if (!re->is_valid) return positions;
    
    const char* p = text;
    regmatch_t match;
    size_t offset = 0;
    
    while (regexec(&re->compiled, p, 1, &match, 0) == 0) {
        int64_t pos = (int64_t)(offset + match.rm_so);
        positions = dyn_array_push_int(positions, pos);
        
        // Move past this match
        offset += match.rm_eo;
        p += match.rm_eo;
        
        if (match.rm_eo == 0) break;  // Avoid infinite loop on empty matches
    }
    
    return positions;
}

// Extract capture groups
DynArray* nl_regex_groups(void* regex, const char* text) {
    DynArray* groups = dyn_array_new(ELEM_STRING);
    if (!regex || !text) return groups;
    
    nl_regex_t* re = (nl_regex_t*)regex;
    if (!re->is_valid) return groups;
    
    #define MAX_GROUPS 10
    regmatch_t matches[MAX_GROUPS];
    
    int result = regexec(&re->compiled, text, MAX_GROUPS, matches, 0);
    if (result != 0) return groups;
    
    for (int i = 0; i < MAX_GROUPS && matches[i].rm_so != -1; i++) {
        size_t len = matches[i].rm_eo - matches[i].rm_so;
        char* group = (char*)malloc(len + 1);
        if (group) {
            strncpy(group, text + matches[i].rm_so, len);
            group[len] = '\0';
            /* Copy into DynArray: `group` is freed below. */
            groups = dyn_array_push_string_copy(groups, group);
            free(group);
        }
    }
    
    return groups;
}

// Replace first occurrence
const char* nl_regex_replace(void* regex, const char* text, const char* replacement) {
    if (!regex || !text || !replacement) return text;
    
    nl_regex_t* re = (nl_regex_t*)regex;
    if (!re->is_valid) return text;
    
    regmatch_t match;
    int result = regexec(&re->compiled, text, 1, &match, 0);
    if (result != 0) return strdup(text);
    
    size_t prefix_len = match.rm_so;
    size_t suffix_len = strlen(text) - match.rm_eo;
    size_t replacement_len = strlen(replacement);
    
    char* output = (char*)malloc(prefix_len + replacement_len + suffix_len + 1);
    if (!output) return text;
    
    strncpy(output, text, prefix_len);
    strcpy(output + prefix_len, replacement);
    strcpy(output + prefix_len + replacement_len, text + match.rm_eo);
    
    return output;
}

// Replace all occurrences
const char* nl_regex_replace_all(void* regex, const char* text, const char* replacement) {
    if (!regex || !text || !replacement) return text;
    
    nl_regex_t* re = (nl_regex_t*)regex;
    if (!re->is_valid) return text;
    
    char* result = strdup(text);
    const char* p = result;
    regmatch_t match;
    size_t offset = 0;
    
    while (regexec(&re->compiled, p, 1, &match, 0) == 0) {
        // Calculate sizes
        size_t prefix_len = offset + match.rm_so;
        size_t replacement_len = strlen(replacement);
        size_t suffix_start = offset + match.rm_eo;
        size_t suffix_len = strlen(text) - suffix_start;
        
        // Allocate new buffer
        char* new_result = (char*)malloc(prefix_len + replacement_len + suffix_len + 1);
        if (!new_result) break;
        
        // Build new string
        strncpy(new_result, result, prefix_len);
        strcpy(new_result + prefix_len, replacement);
        strcpy(new_result + prefix_len + replacement_len, text + suffix_start);
        
        free(result);
        result = new_result;
        
        // Move forward
        offset = prefix_len + replacement_len;
        p = result + offset;
        
        if (match.rm_eo == 0) break;  // Avoid infinite loop
    }
    
    return result;
}

// Split string by regex
DynArray* nl_regex_split(void* regex, const char* text) {
    DynArray* parts = dyn_array_new(ELEM_STRING);
    if (!regex || !text) return parts;
    
    nl_regex_t* re = (nl_regex_t*)regex;
    if (!re->is_valid) return parts;
    
    const char* p = text;
    regmatch_t match;
    
    while (regexec(&re->compiled, p, 1, &match, 0) == 0) {
        // Add part before match
        size_t len = match.rm_so;
        char* part = (char*)malloc(len + 1);
        if (part) {
            strncpy(part, p, len);
            part[len] = '\0';
            parts = dyn_array_push_string_copy(parts, part);
            free(part);
        }
        
        // Move past match
        p += match.rm_eo;
        
        if (match.rm_eo == 0) break;
    }
    
    // Add remaining text
    if (*p) {
        parts = dyn_array_push_string_copy(parts, p);
    }
    
    return parts;
}

// Free regex (now a no-op - GC handles cleanup automatically)
// Kept for backward compatibility but no longer required
void nl_regex_free(void* regex) {
    /* Complete no-op - GC handles cleanup automatically when ref count reaches 0 */
    /* Calling gc_release() here would cause double-free since env_free_value also calls it */
    (void)regex;  /* Suppress unused parameter warning */
}

