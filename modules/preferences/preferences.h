#ifndef NANOLANG_PREFERENCES_H
#define NANOLANG_PREFERENCES_H

#include <stdint.h>

// Array type matching DynArray layout
#ifndef NL_ARRAY_T_DEFINED
#define NL_ARRAY_T_DEFINED
typedef struct {
    int64_t length;
    int64_t capacity;
    int elem_type;
    unsigned char elem_size;
    void* data;
} nl_array_t;
#endif

// Save playlist to file (one path per line)
int64_t nl_prefs_save_playlist(const char* filename, nl_array_t* items, int64_t count);

// Load playlist from file
nl_array_t* nl_prefs_load_playlist(const char* filename);

// Get user's home directory
const char* nl_prefs_get_home();

// Build preference file path (~/.app_prefs)
const char* nl_prefs_get_path(const char* app_name);

#endif // NANOLANG_PREFERENCES_H
