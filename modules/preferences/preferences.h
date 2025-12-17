#ifndef NANOLANG_PREFERENCES_H
#define NANOLANG_PREFERENCES_H

#include <stdint.h>
#include "../../src/runtime/dyn_array.h"

// Save playlist to file (one path per line)
int64_t nl_prefs_save_playlist(const char* filename, DynArray* items, int64_t count);

// Load playlist from file
DynArray* nl_prefs_load_playlist(const char* filename);

// Get user's home directory
const char* nl_prefs_get_home();

// Build preference file path (~/.app_prefs)
const char* nl_prefs_get_path(const char* app_name);

#endif // NANOLANG_PREFERENCES_H
