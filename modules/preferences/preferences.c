#define _POSIX_C_SOURCE 200809L
#include "preferences.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

NANO_EXPORT_ARRAY_ABI(nl_prefs_save_playlist);
NANO_EXPORT_ARRAY_ABI(nl_prefs_load_playlist);

// Save playlist to file
int64_t nl_prefs_save_playlist(const char* filename, DynArray* items, int64_t count) {
    /* I validate the selected prefix before opening a file destructively. */
    if (!filename || !items || items->elem_type != ELEM_STRING ||
        items->elem_size != sizeof(char*) || items->length < 0 ||
        items->capacity < items->length || count < 0 || count > items->length ||
        (uint64_t)items->capacity > SIZE_MAX / sizeof(char*) ||
        (items->capacity && !items->data)) return 0;
    for (int64_t i = 0; i < count; i++) {
        const char* item = ((const char**)items->data)[i];
        if (!item || strchr(item, '\n')) return 0;
    }
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        return 0;
    }
    
    // Write each item on a separate line
    int ok = 1;
    for (int64_t i = 0; i < count; i++) {
        const char* item = ((const char**)items->data)[i];
        if (fprintf(fp, "%s\n", item) < 0) { ok = 0; break; }
    }
    
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

// Load playlist from file
DynArray* nl_prefs_load_playlist(const char* filename) {
    if (!filename) return NULL;
    FILE* fp = fopen(filename, "r");
    if (!fp) return errno == ENOENT
        ? dyn_array_new_with_capacity(ELEM_STRING, 0) : NULL;

    /* I stage owned lines before allocating the canonical result, avoiding
     * runtime growth's abort-on-OOM contract. Nothing partial escapes. */
    char **lines = NULL;
    size_t count = 0, capacity = 0;
    char *line = NULL;
    size_t line_capacity = 0;
    int ok = 1;
    for (;;) {
        ssize_t n = getline(&line, &line_capacity, fp);
        if (n < 0) { if (!feof(fp) || ferror(fp)) ok = 0; break; }
        if (memchr(line, 0, (size_t)n)) { ok = 0; break; }
        if (n && line[n - 1] == '\n') line[--n] = 0;
        if (!n) continue;
        if (count == capacity) {
            if (capacity > SIZE_MAX / sizeof(char*) / 2 ||
                capacity > INT64_MAX / 2) { ok = 0; break; }
            size_t next = capacity ? capacity * 2 : 32;
            char **grown = realloc(lines, next * sizeof(char*));
            if (!grown) { ok = 0; break; }
            lines = grown;
            capacity = next;
        }
        lines[count++] = line;
        line = NULL;
        line_capacity = 0;
    }
    free(line);
    if (fclose(fp) != 0) ok = 0;
    DynArray *result = ok
        ? dyn_array_new_with_capacity(ELEM_STRING, (int64_t)count) : NULL;
    if (result) {
        for (size_t i = 0; i < count; i++) dyn_array_push_string(result, lines[i]);
    } else {
        for (size_t i = 0; i < count; i++) free(lines[i]);
    }
    free(lines);
    return result;
}

// Get user's home directory
const char* nl_prefs_get_home() {
    const char* home = getenv("HOME");
    if (home) {
        return home;
    }
    
    // Fallback
    return "/tmp";
}

// Build preference file path
const char* nl_prefs_get_path(const char* app_name) {
    static char path[1024];
    const char* home = nl_prefs_get_home();
    if (!app_name) return NULL;
    int length = snprintf(path, sizeof(path), "%s/.%s_prefs", home, app_name);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        path[0] = 0;
        return NULL;
    }
    return path;
}
