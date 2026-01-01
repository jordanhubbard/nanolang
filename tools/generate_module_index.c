/* Generate modules/index.json from module manifests
 * Uses existing fs.c infrastructure + cJSON for JSON handling
 * This is a critical build tool, so C (not NanoLang) for reliability
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../src/runtime/dyn_array.h"
#include "../src/cJSON.h"

/* Import fs module infrastructure */
extern DynArray* fs_walkdir(const char* root);

/* Read entire file into string */
static char* read_file(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* content = malloc(size + 1);
    if (!content) {
        fclose(f);
        return NULL;
    }
    
    fread(content, 1, size, f);
    content[size] = '\0';
    fclose(f);
    
    return content;
}

/* Check if string ends with suffix */
static int ends_with(const char* str, const char* suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    
    if (str_len < suffix_len) return 0;
    
    return strcmp(str + (str_len - suffix_len), suffix) == 0;
}

/* Find all module.manifest.json files */
static DynArray* find_manifests(const char* modules_dir) {
    DynArray* all_files = fs_walkdir(modules_dir);
    if (!all_files) return NULL;
    
    DynArray* manifests = dyn_array_new_with_capacity(ELEM_STRING, 32);
    
    for (int64_t i = 0; i < all_files->length; i++) {
        const char* path = dyn_array_get_string(all_files, i);
        if (ends_with(path, "module.manifest.json")) {
            dyn_array_push_string_copy(manifests, path);
        }
    }
    
    return manifests;
}

int main(int argc, char** argv) {
    const char* modules_dir = "modules";
    const char* output_path = "modules/index.json";
    
    printf("Generating module index...\n");
    
    /* Find all manifests */
    DynArray* manifest_paths = find_manifests(modules_dir);
    if (!manifest_paths) {
        fprintf(stderr, "Error: Failed to scan modules directory\n");
        return 1;
    }
    
    printf("Found %lld module manifests\n", (long long)manifest_paths->length);
    
    /* Create index structure */
    cJSON* index = cJSON_CreateObject();
    cJSON_AddStringToObject(index, "version", "1.0.0");
    
    /* Add timestamp */
    time_t now = time(NULL);
    char timestamp[64];
    struct tm* tm_info = gmtime(&now);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", tm_info);
    cJSON_AddStringToObject(index, "generated_at", timestamp);
    
    cJSON_AddNumberToObject(index, "total_modules", manifest_paths->length);
    
    /* Add modules array */
    cJSON* modules_array = cJSON_CreateArray();
    
    for (int64_t i = 0; i < manifest_paths->length; i++) {
        const char* manifest_path = dyn_array_get_string(manifest_paths, i);
        
        /* Read and parse manifest */
        char* manifest_content = read_file(manifest_path);
        if (!manifest_content) {
            fprintf(stderr, "Warning: Failed to read %s\n", manifest_path);
            continue;
        }
        
        cJSON* manifest = cJSON_Parse(manifest_content);
        free(manifest_content);
        
        if (!manifest) {
            fprintf(stderr, "Warning: Failed to parse %s\n", manifest_path);
            continue;
        }
        
        /* Get module name for logging */
        cJSON* name = cJSON_GetObjectItem(manifest, "name");
        if (name && cJSON_IsString(name)) {
            printf("  ✓ %s\n", name->valuestring);
        }
        
        cJSON_AddItemToArray(modules_array, manifest);
    }
    
    cJSON_AddItemToObject(index, "modules", modules_array);
    
    /* Add empty indices structure (could be populated later) */
    cJSON* indices = cJSON_CreateObject();
    cJSON_AddItemToObject(indices, "capabilities", cJSON_CreateObject());
    cJSON_AddItemToObject(indices, "keywords", cJSON_CreateObject());
    cJSON_AddItemToObject(indices, "io_surfaces", cJSON_CreateObject());
    cJSON_AddItemToObject(index, "indices", indices);
    
    /* Write output */
    char* json_str = cJSON_Print(index);
    FILE* out = fopen(output_path, "w");
    if (!out) {
        fprintf(stderr, "Error: Failed to open %s for writing\n", output_path);
        cJSON_Delete(index);
        free(json_str);
        return 1;
    }
    
    fprintf(out, "%s\n", json_str);
    fclose(out);
    
    printf("\n✓ Generated %s\n", output_path);
    printf("  %lld modules indexed\n", (long long)manifest_paths->length);
    
    cJSON_Delete(index);
    free(json_str);
    
    return 0;
}

