#include "preferences.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Helper: Create array for strings
static DynArray* create_string_array(int64_t initial_capacity) {
    DynArray* arr = (DynArray*)malloc(sizeof(DynArray));
    if (!arr) return NULL;
    
    arr->length = 0;
    arr->capacity = initial_capacity;
    arr->elem_type = ELEM_STRING;  // ElementType enum from dyn_array.h
    arr->elem_size = sizeof(char*);
    arr->data = calloc(initial_capacity, sizeof(char*));
    
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    
    return arr;
}

// Helper: Append string to array
static void array_append_string(DynArray* arr, const char* str) {
    if (!arr || !str) return;
    
    // Grow if needed
    if (arr->length >= arr->capacity) {
        int64_t new_capacity = arr->capacity * 2;
        char** new_data = (char**)realloc(arr->data, new_capacity * sizeof(char*));
        if (!new_data) return;
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    
    // Duplicate string and add
    char* dup = strdup(str);
    if (dup) {
        ((char**)arr->data)[arr->length++] = dup;
    }
}

// Save playlist to file
int64_t nl_prefs_save_playlist(const char* filename, DynArray* items, int64_t count) {
    if (!filename || !items) return 0;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        return 0;
    }
    
    // Write each item on a separate line
    for (int64_t i = 0; i < count && i < items->length; i++) {
        const char* item = ((const char**)items->data)[i];
        if (item) {
            fprintf(fp, "%s\n", item);
        }
    }
    
    fclose(fp);
    return 1;
}

// Load playlist from file
DynArray* nl_prefs_load_playlist(const char* filename) {
    DynArray* result = create_string_array(32);
    if (!result) return NULL;
    
    // Check if file exists
    if (access(filename, F_OK) != 0) {
        // File doesn't exist - return empty array
        return result;
    }
    
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        return result;
    }
    
    // Read lines
    char line[2048];
    while (fgets(line, sizeof(line), fp)) {
        // Remove trailing newline
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }
        
        // Skip empty lines
        if (strlen(line) == 0) continue;
        
        array_append_string(result, line);
    }
    
    fclose(fp);
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
    snprintf(path, sizeof(path), "%s/.%s_prefs", home, app_name);
    return path;
}
