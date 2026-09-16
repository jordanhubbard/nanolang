#ifndef NANOLANG_PREFERENCES_H
#define NANOLANG_PREFERENCES_H

#include <stdint.h>
#include "../../src/runtime/dyn_array.h"

// I save exactly count entries, rejecting invalid metadata, null entries and LF.
// Invalid input leaves the file untouched. I report I/O failure as 0; a write
// failure can leave partial output. A successful save returns 1.
int64_t nl_prefs_save_playlist(const char* filename, DynArray* items, int64_t count);

// I return a canonical array, skipping blank lines and preserving complete lines.
// A missing file yields an empty array; invalid input, NUL bytes, allocation or
// I/O failure yields NULL. Successful strings are separate owned allocations;
// automatic reclamation of escaped native strings remains runtime work.
DynArray* nl_prefs_load_playlist(const char* filename);

// Get user's home directory
const char* nl_prefs_get_home();

// I return a borrowed static path, or NULL for null input or truncation.
const char* nl_prefs_get_path(const char* app_name);

#endif // NANOLANG_PREFERENCES_H
