// Module Build System Implementation
// Handles automatic compilation of C sources, caching, and dependency tracking

#include "module_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

// JSON parsing (simple, minimal implementation for module.json)
#include "cJSON.h"

bool module_builder_verbose = false;

static void append_flag_move_to_end(char **out_flags, size_t *out_count, size_t out_cap, const char *flag) {
    if (!flag || flag[0] == '\0' || !out_flags || !out_count) return;

    for (size_t i = 0; i < *out_count; i++) {
        if (out_flags[i] && strcmp(out_flags[i], flag) == 0) {
            free(out_flags[i]);
            for (size_t j = i; j + 1 < *out_count; j++) {
                out_flags[j] = out_flags[j + 1];
            }
            (*out_count)--;
            break;
        }
    }

    if (*out_count >= out_cap) return;
    out_flags[(*out_count)++] = strdup(flag);
}

static void append_split_flags_move_to_end(char **out_flags, size_t *out_count, size_t out_cap, const char *flags) {
    if (!flags || flags[0] == '\0') return;

    char *copy = strdup(flags);
    if (!copy) return;

    char *saveptr = NULL;
    for (char *tok = strtok_r(copy, " \t\r\n", &saveptr); tok; tok = strtok_r(NULL, " \t\r\n", &saveptr)) {
        append_flag_move_to_end(out_flags, out_count, out_cap, tok);
    }

    free(copy);
}

// Helper: Check if file exists
static bool file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

// Helper: Check if directory exists
static bool dir_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

// Helper: Get file modification time
static time_t get_mtime(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return 0;
    }
    return st.st_mtime;
}

// Helper: Create directory (mkdir -p)
static bool mkdir_p(const char *path) {
    char tmp[1024];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = 0;
    }

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (!dir_exists(tmp)) {
                if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                    return false;
                }
            }
            *p = '/';
        }
    }

    if (!dir_exists(tmp)) {
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }

    return true;
}

// Helper: Run command and capture output
static char* run_command(const char *cmd) {
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        return NULL;
    }

    char *output = malloc(4096);
    if (!output) {
        pclose(fp);
        return NULL;
    }
    
    // Initialize buffer to empty string in case command produces no output
    output[0] = '\0';

    size_t total = 0;
    size_t capacity = 4096;
    
    while (fgets(output + total, capacity - total, fp) != NULL) {
        total = strlen(output);
        if (total + 1024 > capacity) {
            capacity *= 2;
            char *new_output = realloc(output, capacity);
            if (!new_output) {
                free(output);
                pclose(fp);
                return NULL;
            }
            output = new_output;
        }
    }

    pclose(fp);

    // Remove trailing newline
    if (total > 0 && output[total - 1] == '\n') {
        output[total - 1] = '\0';
    }

    return output;
}

// Install system packages from module metadata
static bool install_system_packages(ModuleBuildMetadata *meta) {
    bool all_installed = true;
    char cmd[2048];
    
    #ifdef __APPLE__
    // macOS: Install Homebrew packages
    if (meta->brew_packages_count > 0) {
        if (access("/opt/homebrew/bin/brew", F_OK) == 0 || access("/usr/local/bin/brew", F_OK) == 0) {
            printf("[Module] Installing Homebrew packages for '%s'...\n", meta->name);
            for (size_t i = 0; i < meta->brew_packages_count; i++) {
                // Check if already installed
                snprintf(cmd, sizeof(cmd), "brew list %s >/dev/null 2>&1", meta->brew_packages[i]);
                if (system(cmd) == 0) {
                    printf("[Module]   ✓ %s already installed\n", meta->brew_packages[i]);
                    continue;
                }
                
                // Install package
                snprintf(cmd, sizeof(cmd), "brew install %s", meta->brew_packages[i]);
                printf("[Module]   Installing %s...\n", meta->brew_packages[i]);
                int result = system(cmd);
                if (result == 0) {
                    printf("[Module]   ✓ Successfully installed %s\n", meta->brew_packages[i]);
                } else {
                    fprintf(stderr, "[Module]   ❌ Failed to install %s\n", meta->brew_packages[i]);
                    all_installed = false;
                }
            }
        } else {
            fprintf(stderr, "[Module] ⚠️  Homebrew not found, cannot auto-install packages\n");
            fprintf(stderr, "[Module]    Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"\n");
            all_installed = false;
        }
    }
    #else
    // Linux: Install apt or dnf packages
    if (access("/usr/bin/apt-get", F_OK) == 0 || access("/usr/bin/apt", F_OK) == 0) {
        // Debian/Ubuntu
        if (meta->apt_packages_count > 0) {
            printf("[Module] Installing apt packages for '%s'...\n", meta->name);
            
            // Build list of packages to install
            char packages[1024] = "";
            for (size_t i = 0; i < meta->apt_packages_count; i++) {
                // Check if already installed
                snprintf(cmd, sizeof(cmd), "dpkg -l %s 2>/dev/null | grep -q '^ii'", meta->apt_packages[i]);
                if (system(cmd) == 0) {
                    printf("[Module]   ✓ %s already installed\n", meta->apt_packages[i]);
                    continue;
                }
                
                // Add to install list
                if (strlen(packages) > 0) strcat(packages, " ");
                strncat(packages, meta->apt_packages[i], sizeof(packages) - strlen(packages) - 1);
            }
            
            // Install all needed packages at once
            if (strlen(packages) > 0) {
                snprintf(cmd, sizeof(cmd), "sudo apt-get update -qq && sudo apt-get install -y %s", packages);
                printf("[Module]   Running: sudo apt-get install -y %s\n", packages);
                printf("[Module]   (You may be prompted for your password)\n");
                int result = system(cmd);
                if (result == 0) {
                    printf("[Module]   ✓ Successfully installed packages\n");
                } else {
                    fprintf(stderr, "[Module]   ❌ Failed to install packages\n");
                    all_installed = false;
                }
            }
        }
    } else if (access("/usr/bin/dnf", F_OK) == 0) {
        // Fedora/RHEL
        if (meta->dnf_packages_count > 0) {
            printf("[Module] Installing dnf packages for '%s'...\n", meta->name);
            
            // Build list of packages to install
            char packages[1024] = "";
            for (size_t i = 0; i < meta->dnf_packages_count; i++) {
                // Check if already installed
                snprintf(cmd, sizeof(cmd), "rpm -q %s >/dev/null 2>&1", meta->dnf_packages[i]);
                if (system(cmd) == 0) {
                    printf("[Module]   ✓ %s already installed\n", meta->dnf_packages[i]);
                    continue;
                }
                
                // Add to install list
                if (strlen(packages) > 0) strcat(packages, " ");
                strncat(packages, meta->dnf_packages[i], sizeof(packages) - strlen(packages) - 1);
            }
            
            // Install all needed packages at once
            if (strlen(packages) > 0) {
                snprintf(cmd, sizeof(cmd), "sudo dnf install -y %s", packages);
                printf("[Module]   Running: sudo dnf install -y %s\n", packages);
                printf("[Module]   (You may be prompted for your password)\n");
                int result = system(cmd);
                if (result == 0) {
                    printf("[Module]   ✓ Successfully installed packages\n");
                } else {
                    fprintf(stderr, "[Module]   ❌ Failed to install packages\n");
                    all_installed = false;
                }
            }
        }
    } else {
        if (meta->apt_packages_count > 0 || meta->dnf_packages_count > 0) {
            fprintf(stderr, "[Module] ⚠️  No supported package manager found (apt-get/dnf)\n");
            all_installed = false;
        }
    }
    #endif
    
    return all_installed;
}

// Get pkg-config flags, with automatic package installation on failure
static char* get_pkg_config_flags(const char *package, const char *flag_type) {
    char cmd[512];
    
    // Try standard pkg-config first
    snprintf(cmd, sizeof(cmd), "pkg-config %s %s 2>/dev/null", flag_type, package);
    char *result = run_command(cmd);
    
    // If that failed, try Homebrew's pkg-config on macOS
    if (!result || strlen(result) == 0) {
        free(result);
        snprintf(cmd, sizeof(cmd), "/opt/homebrew/bin/pkg-config %s %s 2>/dev/null", flag_type, package);
        result = run_command(cmd);
    }
    
    // If still failed, try /usr/local/bin (Intel Mac Homebrew)
    if (!result || strlen(result) == 0) {
        free(result);
        snprintf(cmd, sizeof(cmd), "/usr/local/bin/pkg-config %s %s 2>/dev/null", flag_type, package);
        result = run_command(cmd);
    }
    
    
    // If result is empty or only whitespace, return NULL instead
    if (result) {
        // Trim leading/trailing whitespace
        char *start = result;
        while (*start && (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r')) {
            start++;
        }
        if (*start == '\0') {
            // Empty after trimming
            free(result);
            return NULL;
        }
        // If start is not at beginning, shift the string
        if (start != result) {
            memmove(result, start, strlen(start) + 1);
        }
    }
    
    return result;
}

// Simple C header parser to extract #define constants
// Returns array of ConstantDef, or NULL if parsing fails
// Note: This is a basic parser - it handles simple integer #define patterns only
#include "nanolang.h"

ConstantDef* parse_c_header_constants(const char *header_path, int *count_out) {
    *count_out = 0;
    
    FILE *fp = fopen(header_path, "r");
    if (!fp) {
        return NULL;  /* Header not found - not an error, just skip */
    }
    
    /* First pass: count #define integer constants */
    char line[1024];
    int const_count = 0;
    while (fgets(line, sizeof(line), fp)) {
        /* Look for #define NAME VALUE patterns */
        char name[256];
        long long value;
        char *trimmed = line;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
        
        /* Try hex format: #define NAME 0x1234 */
        if (sscanf(trimmed, "#define %255s 0x%llx", name, (unsigned long long *)&value) == 2) {
            const_count++;
        }
        /* Try decimal format: #define NAME 1234 */
        else if (sscanf(trimmed, "#define %255s %lld", name, &value) == 2) {
            const_count++;
        }
    }
    
    if (const_count == 0) {
        fclose(fp);
        return NULL;
    }
    
    /* Second pass: extract constants */
    ConstantDef *constants = malloc(sizeof(ConstantDef) * const_count);
    rewind(fp);
    
    int idx = 0;
    while (fgets(line, sizeof(line), fp) && idx < const_count) {
        char name[256];
        long long value;
        char *trimmed = line;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
        
        bool parsed = false;
        /* Try hex format */
        if (sscanf(trimmed, "#define %255s 0x%llx", name, (unsigned long long *)&value) == 2) {
            parsed = true;
        }
        /* Try decimal format */
        else if (sscanf(trimmed, "#define %255s %lld", name, &value) == 2) {
            parsed = true;
        }
        
        if (parsed) {
            constants[idx].name = strdup(name);
            constants[idx].value = value;
            constants[idx].type = TYPE_INT;
            idx++;
        }
    }
    
    fclose(fp);
    *count_out = idx;
    return constants;
}

// Module metadata functions

ModuleBuildMetadata* module_load_metadata(const char *module_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/module.json", module_dir);

    if (!file_exists(path)) {
        // No module.json = pure nanolang module
        return NULL;
    }

    // Read file
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s\n", path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *content = malloc(size + 1);
    if (!content) {
        fclose(fp);
        return NULL;
    }

    fread(content, 1, size, fp);
    content[size] = '\0';
    fclose(fp);

    // Parse JSON
    cJSON *json = cJSON_Parse(content);
    free(content);

    if (!json) {
        fprintf(stderr, "Error: Invalid JSON in %s\n", path);
        return NULL;
    }

    ModuleBuildMetadata *meta = calloc(1, sizeof(ModuleBuildMetadata));
    if (!meta) {
        cJSON_Delete(json);
        return NULL;
    }

    // Parse fields
    cJSON *name = cJSON_GetObjectItem(json, "name");
    if (name && cJSON_IsString(name)) {
        meta->name = strdup(name->valuestring);
    }

    cJSON *version = cJSON_GetObjectItem(json, "version");
    if (version && cJSON_IsString(version)) {
        meta->version = strdup(version->valuestring);
    }

    cJSON *description = cJSON_GetObjectItem(json, "description");
    if (description && cJSON_IsString(description)) {
        meta->description = strdup(description->valuestring);
    }

    // Parse arrays
    #define PARSE_STRING_ARRAY(field_name, dest, count) do { \
        cJSON *arr = cJSON_GetObjectItem(json, field_name); \
        if (arr && cJSON_IsArray(arr)) { \
            int arr_size = cJSON_GetArraySize(arr); \
            meta->dest = calloc(arr_size, sizeof(char*)); \
            meta->count = 0; \
            for (int i = 0; i < arr_size; i++) { \
                cJSON *item = cJSON_GetArrayItem(arr, i); \
                if (cJSON_IsString(item)) { \
                    meta->dest[meta->count++] = strdup(item->valuestring); \
                } \
            } \
        } \
    } while(0)

    PARSE_STRING_ARRAY("headers", headers, headers_count);
    PARSE_STRING_ARRAY("c_sources", c_sources, c_sources_count);
    PARSE_STRING_ARRAY("system_libs", system_libs, system_libs_count);
    PARSE_STRING_ARRAY("pkg_config", pkg_config, pkg_config_count);
    PARSE_STRING_ARRAY("include_dirs", include_dirs, include_dirs_count);
    PARSE_STRING_ARRAY("cflags", cflags, cflags_count);
    PARSE_STRING_ARRAY("ldflags", ldflags, ldflags_count);
    PARSE_STRING_ARRAY("frameworks", frameworks, frameworks_count);
    PARSE_STRING_ARRAY("dependencies", dependencies, dependencies_count);
    PARSE_STRING_ARRAY("apt_packages", apt_packages, apt_packages_count);
    PARSE_STRING_ARRAY("dnf_packages", dnf_packages, dnf_packages_count);
    PARSE_STRING_ARRAY("brew_packages", brew_packages, brew_packages_count);
    PARSE_STRING_ARRAY("owned_string_returns", owned_string_returns, owned_string_returns_count);

    #undef PARSE_STRING_ARRAY

    // Parse header_priority (default = 0)
    cJSON *header_priority = cJSON_GetObjectItem(json, "header_priority");
    if (header_priority && cJSON_IsNumber(header_priority)) {
        meta->header_priority = header_priority->valueint;
    } else {
        meta->header_priority = 0;  // Default priority
    }

    meta->module_dir = strdup(module_dir);

    cJSON_Delete(json);
    return meta;
}

void module_metadata_free(ModuleBuildMetadata *meta) {
    if (!meta) return;

    free(meta->name);
    free(meta->version);
    free(meta->description);
    free(meta->module_dir);

    #define FREE_STRING_ARRAY(arr, count) do { \
        for (size_t i = 0; i < meta->count; i++) { \
            free(meta->arr[i]); \
        } \
        free(meta->arr); \
    } while(0)

    FREE_STRING_ARRAY(headers, headers_count);
    FREE_STRING_ARRAY(c_sources, c_sources_count);
    FREE_STRING_ARRAY(system_libs, system_libs_count);
    FREE_STRING_ARRAY(pkg_config, pkg_config_count);
    FREE_STRING_ARRAY(include_dirs, include_dirs_count);
    FREE_STRING_ARRAY(cflags, cflags_count);
    FREE_STRING_ARRAY(ldflags, ldflags_count);
    FREE_STRING_ARRAY(frameworks, frameworks_count);
    FREE_STRING_ARRAY(dependencies, dependencies_count);
    FREE_STRING_ARRAY(apt_packages, apt_packages_count);
    FREE_STRING_ARRAY(dnf_packages, dnf_packages_count);
    FREE_STRING_ARRAY(brew_packages, brew_packages_count);
    FREE_STRING_ARRAY(owned_string_returns, owned_string_returns_count);

    #undef FREE_STRING_ARRAY

    free(meta);
}

// Module build directory management

char* module_get_build_dir(const char *module_dir) {
    char *build_dir = malloc(1024);
    if (!build_dir) return NULL;
    snprintf(build_dir, 1024, "%s/.build", module_dir);
    return build_dir;
}

bool module_ensure_build_dir(const char *module_dir) {
    char *build_dir = module_get_build_dir(module_dir);
    if (!build_dir) return false;

    bool success = mkdir_p(build_dir);
    free(build_dir);
    return success;
}

// Check if module needs rebuild

bool module_needs_rebuild(const char *module_dir, ModuleBuildMetadata *meta) {
    if (!meta || meta->c_sources_count == 0) {
        // No C sources = no rebuild needed
        return false;
    }

    char *build_dir = module_get_build_dir(module_dir);
    if (!build_dir) return true;

    char object_file[1024];
    snprintf(object_file, sizeof(object_file), "%s/%s.o", build_dir, meta->name);
    free(build_dir);

    if (!file_exists(object_file)) {
        if (module_builder_verbose) {
            printf("[Module] %s needs build: object file missing\n", meta->name);
        }
        return true;
    }

    /* If the shared library is missing, rebuild so interpreter FFI can load it */
    char shared_lib[1024];
    #ifdef __APPLE__
    snprintf(shared_lib, sizeof(shared_lib), "%s/.build/lib%s.dylib", module_dir, meta->name);
    #else
    snprintf(shared_lib, sizeof(shared_lib), "%s/.build/lib%s.so", module_dir, meta->name);
    #endif
    if (!file_exists(shared_lib)) {
        if (module_builder_verbose) {
            printf("[Module] %s needs build: shared library missing\n", meta->name);
        }
        return true;
    }

    time_t object_mtime = get_mtime(object_file);

    // Check if any C source is newer
    for (size_t i = 0; i < meta->c_sources_count; i++) {
        char source_path[1024];
        snprintf(source_path, sizeof(source_path), "%s/%s", module_dir, meta->c_sources[i]);

        if (!file_exists(source_path)) {
            fprintf(stderr, "Error: C source not found: %s\n", source_path);
            return true;
        }

        time_t source_mtime = get_mtime(source_path);
        if (source_mtime > object_mtime) {
            if (module_builder_verbose) {
                printf("[Module] %s needs rebuild: %s modified\n", meta->name, meta->c_sources[i]);
            }
            return true;
        }
    }

    // Check if module.json is newer
    char module_json[1024];
    snprintf(module_json, sizeof(module_json), "%s/module.json", module_dir);
    time_t json_mtime = get_mtime(module_json);
    if (json_mtime > object_mtime) {
        if (module_builder_verbose) {
            printf("[Module] %s needs rebuild: module.json modified\n", meta->name);
        }
        return true;
    }

    return false;
}

// Build module

struct ModuleBuilder {
    char *module_path;
    char **include_paths;
    size_t include_paths_count;
};

ModuleBuilder* module_builder_new(const char *module_path) {
    ModuleBuilder *builder = calloc(1, sizeof(ModuleBuilder));
    if (!builder) return NULL;

    builder->module_path = strdup(module_path ? module_path : "modules");
    return builder;
}

void module_builder_free(ModuleBuilder *builder) {
    if (!builder) return;

    free(builder->module_path);
    
    for (size_t i = 0; i < builder->include_paths_count; i++) {
        free(builder->include_paths[i]);
    }
    free(builder->include_paths);

    free(builder);
}

ModuleBuildInfo* module_build(ModuleBuilder *builder __attribute__((unused)), ModuleBuildMetadata *meta) {
    if (!meta || meta->c_sources_count == 0) {
        // No C sources = nothing to build, but still need link/compile flags
        ModuleBuildInfo *info = calloc(1, sizeof(ModuleBuildInfo));
        if (!info) return NULL;

        // Collect link flags from pkg-config and system_libs
        size_t total_link_flags = 0;
        char **link_flags = calloc(1024, sizeof(char*));

        // Add pkg-config link flags
        for (size_t i = 0; i < meta->pkg_config_count; i++) {
            char *pkg_flags = get_pkg_config_flags(meta->pkg_config[i], "--libs");
            if (pkg_flags) {
                append_split_flags_move_to_end(link_flags, &total_link_flags, 1024, pkg_flags);
                free(pkg_flags);
            }
        }

        // Add custom ldflags
        for (size_t i = 0; i < meta->ldflags_count; i++) {
            link_flags[total_link_flags++] = strdup(meta->ldflags[i]);
        }

        // Add macOS frameworks
        #ifdef __APPLE__
        for (size_t i = 0; i < meta->frameworks_count; i++) {
            link_flags[total_link_flags++] = strdup("-framework");
            link_flags[total_link_flags++] = strdup(meta->frameworks[i]);
        }
        #endif

        // Add system libs
        for (size_t i = 0; i < meta->system_libs_count; i++) {
            char *lib_flag = malloc(256);
            snprintf(lib_flag, 256, "-l%s", meta->system_libs[i]);
            link_flags[total_link_flags++] = lib_flag;
        }

        info->link_flags = link_flags;
        info->link_flags_count = total_link_flags;

        // Collect compile flags (include paths from pkg-config)
        size_t total_compile_flags = 0;
        char **compile_flags = calloc(1024, sizeof(char*));

        // Add pkg-config compile flags (include paths, defines)
        for (size_t i = 0; i < meta->pkg_config_count; i++) {
            char *pkg_cflags = get_pkg_config_flags(meta->pkg_config[i], "--cflags");
            if (pkg_cflags) {
                append_split_flags_move_to_end(compile_flags, &total_compile_flags, 1024, pkg_cflags);
                free(pkg_cflags);
            }
        }

        // Add custom include dirs
        for (size_t i = 0; i < meta->include_dirs_count; i++) {
            char *include_flag = malloc(256);
            snprintf(include_flag, 256, "-I%s", meta->include_dirs[i]);
            compile_flags[total_compile_flags++] = include_flag;
        }

        // Add custom cflags
        for (size_t i = 0; i < meta->cflags_count; i++) {
            compile_flags[total_compile_flags++] = strdup(meta->cflags[i]);
        }

        info->compile_flags = compile_flags;
        info->compile_flags_count = total_compile_flags;
        info->needs_rebuild = false;
        info->object_file = NULL;

        return info;
    }

    // Ensure build directory exists
    if (!module_ensure_build_dir(meta->module_dir)) {
        fprintf(stderr, "Error: Could not create build directory for %s\n", meta->name);
        return NULL;
    }

    // Check if rebuild needed
    bool needs_rebuild = module_needs_rebuild(meta->module_dir, meta);

    char *build_dir = module_get_build_dir(meta->module_dir);
    char object_file[1024];
    snprintf(object_file, sizeof(object_file), "%s/%s.o", build_dir, meta->name);

    if (needs_rebuild) {
        if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
            printf("[Module] Building %s...\n", meta->name);
        }

        // Install system package dependencies (only when rebuilding)
        if (meta->apt_packages_count > 0 || meta->dnf_packages_count > 0 || meta->brew_packages_count > 0) {
            if (!install_system_packages(meta)) {
                fprintf(stderr, "[Module] Warning: Some system packages failed to install for '%s'\n", meta->name);
                fprintf(stderr, "[Module] Continuing anyway - build may fail if dependencies are missing\n");
            }
        }

        // Get CC from environment or use POSIX cc
        const char *cc = getenv("NANO_CC");
        if (!cc) cc = getenv("CC");
        if (!cc) cc = "cc";

        // Build a reusable compile prefix (flags only)
        char compile_prefix[4096];
        int prefix_pos = 0;
        prefix_pos += snprintf(compile_prefix + prefix_pos, sizeof(compile_prefix) - prefix_pos, "%s -c -fPIC", cc);

        // Add pkg-config cflags
        for (size_t i = 0; i < meta->pkg_config_count; i++) {
            char *pkg_cflags = get_pkg_config_flags(meta->pkg_config[i], "--cflags");
            if (pkg_cflags) {
                prefix_pos += snprintf(compile_prefix + prefix_pos, sizeof(compile_prefix) - prefix_pos, " %s", pkg_cflags);
                free(pkg_cflags);
            }
        }

        // Add include dirs
        for (size_t i = 0; i < meta->include_dirs_count; i++) {
            prefix_pos += snprintf(compile_prefix + prefix_pos, sizeof(compile_prefix) - prefix_pos, " -I%s", meta->include_dirs[i]);
        }

        // Add custom cflags
        for (size_t i = 0; i < meta->cflags_count; i++) {
            prefix_pos += snprintf(compile_prefix + prefix_pos, sizeof(compile_prefix) - prefix_pos, " %s", meta->cflags[i]);
        }

        if (meta->c_sources_count == 1) {
            // Single source can compile directly to the module object.
            char compile_cmd[8192];
            snprintf(compile_cmd, sizeof(compile_cmd), "%s %s/%s -o %s",
                     compile_prefix, meta->module_dir, meta->c_sources[0], object_file);

            if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
                printf("[Module] %s\n", compile_cmd);
            }

            int result = system(compile_cmd);
            if (result != 0) {
                fprintf(stderr, "Error: Failed to compile module %s\n", meta->name);
                free(build_dir);
                return NULL;
            }
        } else {
            // Multiple sources: compile each to its own object, then combine.
            char **src_objects = calloc(meta->c_sources_count, sizeof(char*));
            if (!src_objects) {
                fprintf(stderr, "Error: Out of memory building module %s\n", meta->name);
                free(build_dir);
                return NULL;
            }

            for (size_t i = 0; i < meta->c_sources_count; i++) {
                char obj_path[1024];
                snprintf(obj_path, sizeof(obj_path), "%s/%s_%zu.o", build_dir, meta->name, i);
                src_objects[i] = strdup(obj_path);

                char compile_cmd[8192];
                snprintf(compile_cmd, sizeof(compile_cmd), "%s %s/%s -o %s",
                         compile_prefix, meta->module_dir, meta->c_sources[i], obj_path);

                if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
                    printf("[Module] %s\n", compile_cmd);
                }

                int result = system(compile_cmd);
                if (result != 0) {
                    fprintf(stderr, "Error: Failed to compile module %s (%s)\n", meta->name, meta->c_sources[i]);
                    for (size_t j = 0; j < meta->c_sources_count; j++) free(src_objects[j]);
                    free(src_objects);
                    free(build_dir);
                    return NULL;
                }
            }

            char combine_cmd[8192];
            int combine_pos = 0;
            combine_pos += snprintf(combine_cmd + combine_pos, sizeof(combine_cmd) - combine_pos, "%s -r -o %s", cc, object_file);
            for (size_t i = 0; i < meta->c_sources_count; i++) {
                combine_pos += snprintf(combine_cmd + combine_pos, sizeof(combine_cmd) - combine_pos, " %s", src_objects[i]);
            }

            if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
                printf("[Module] %s\n", combine_cmd);
            }

            int combine_result = system(combine_cmd);
            for (size_t i = 0; i < meta->c_sources_count; i++) free(src_objects[i]);
            free(src_objects);

            if (combine_result != 0) {
                fprintf(stderr, "Error: Failed to combine objects for module %s\n", meta->name);
                free(build_dir);
                return NULL;
            }
        }

        if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
            printf("[Module] ✓ Built %s\n", meta->name);
        }
        
        /* Also create shared library for interpreter FFI */
        #ifdef __APPLE__
        char shared_lib[1024];
        snprintf(shared_lib, sizeof(shared_lib), "%s/.build/lib%s.dylib", 
                 meta->module_dir, meta->name);
        #else
        char shared_lib[1024];
        snprintf(shared_lib, sizeof(shared_lib), "%s/.build/lib%s.so", 
                 meta->module_dir, meta->name);
        #endif
        
        /* Build shared library command */
        char lib_cmd[4096];
        size_t lib_pos = 0;
        
        #ifdef __APPLE__
        /* On macOS, allow unresolved symbols so modules can reference symbols
         * provided by the host process (compiler/interpreter) at dlopen() time.
         */
        lib_pos += snprintf(lib_cmd + lib_pos, sizeof(lib_cmd) - lib_pos,
                           "%s -dynamiclib -undefined dynamic_lookup -fPIC -o %s",
                           cc, shared_lib);
        #else
        lib_pos += snprintf(lib_cmd + lib_pos, sizeof(lib_cmd) - lib_pos,
                           "%s -shared -fPIC -o %s", cc, shared_lib);
        #endif
        
        /* Link the shared library from the module object (supports multi-source modules) */
        lib_pos += snprintf(lib_cmd + lib_pos, sizeof(lib_cmd) - lib_pos, " %s", object_file);
        
        /* Add pkg-config flags (deduplicated) */
        char *shared_cflags[1024] = {0};
        size_t shared_cflags_count = 0;
        for (size_t i = 0; i < meta->pkg_config_count; i++) {
            char *pkg_cflags = get_pkg_config_flags(meta->pkg_config[i], "--cflags");
            if (pkg_cflags) {
                append_split_flags_move_to_end(shared_cflags, &shared_cflags_count, 1024, pkg_cflags);
                free(pkg_cflags);
            }
        }
        for (size_t i = 0; i < shared_cflags_count; i++) {
            lib_pos += snprintf(lib_cmd + lib_pos, sizeof(lib_cmd) - lib_pos, " %s", shared_cflags[i]);
            free(shared_cflags[i]);
        }

        char *shared_ldflags[1024] = {0};
        size_t shared_ldflags_count = 0;
        for (size_t i = 0; i < meta->pkg_config_count; i++) {
            char *pkg_libs = get_pkg_config_flags(meta->pkg_config[i], "--libs");
            if (pkg_libs) {
                append_split_flags_move_to_end(shared_ldflags, &shared_ldflags_count, 1024, pkg_libs);
                free(pkg_libs);
            }
        }
        for (size_t i = 0; i < meta->system_libs_count; i++) {
            char buf[256];
            snprintf(buf, sizeof(buf), "-l%s", meta->system_libs[i]);
            append_flag_move_to_end(shared_ldflags, &shared_ldflags_count, 1024, buf);
        }
        for (size_t i = 0; i < meta->ldflags_count; i++) {
            append_split_flags_move_to_end(shared_ldflags, &shared_ldflags_count, 1024, meta->ldflags[i]);
        }
        #ifdef __APPLE__
        for (size_t i = 0; i < meta->frameworks_count; i++) {
            append_flag_move_to_end(shared_ldflags, &shared_ldflags_count, 1024, "-framework");
            append_flag_move_to_end(shared_ldflags, &shared_ldflags_count, 1024, meta->frameworks[i]);
        }
        #endif

        for (size_t i = 0; i < shared_ldflags_count; i++) {
            lib_pos += snprintf(lib_cmd + lib_pos, sizeof(lib_cmd) - lib_pos, " %s", shared_ldflags[i]);
            free(shared_ldflags[i]);
        }
        
        /* Add custom cflags */
        for (size_t i = 0; i < meta->cflags_count; i++) {
            lib_pos += snprintf(lib_cmd + lib_pos, sizeof(lib_cmd) - lib_pos,
                               " %s", meta->cflags[i]);
        }
        
        /* Note: ldflags/system libs/frameworks are included above via shared_ldflags */
        
        /* Build shared library */
        if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
            printf("[Module] Building shared library: %s\n", lib_cmd);
        }
        
        int lib_result = system(lib_cmd);
        if (lib_result != 0) {
            fprintf(stderr, "Warning: Failed to build shared library for %s (interpreter FFI unavailable)\n", 
                    meta->name);
        } else if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
            printf("[Module] ✓ Built shared library %s\n", shared_lib);
        }
    } else {
        if (module_builder_verbose) {
            printf("[Module] %s up to date (using cache)\n", meta->name);
        }
    }

    free(build_dir);

    // Create build info with object file and link flags
    ModuleBuildInfo *info = calloc(1, sizeof(ModuleBuildInfo));
    if (!info) return NULL;

    info->object_file = strdup(object_file);
    info->needs_rebuild = needs_rebuild;

    // Collect link flags
    size_t total_link_flags = 0;
    char **link_flags = calloc(1024, sizeof(char*));

    // Add object file
    link_flags[total_link_flags++] = strdup(object_file);

    // Add pkg-config link flags
    for (size_t i = 0; i < meta->pkg_config_count; i++) {
        char *pkg_flags = get_pkg_config_flags(meta->pkg_config[i], "--libs");
        if (pkg_flags) {
            append_split_flags_move_to_end(link_flags, &total_link_flags, 1024, pkg_flags);
            free(pkg_flags);
        }
    }

    // Add custom ldflags
    for (size_t i = 0; i < meta->ldflags_count; i++) {
        link_flags[total_link_flags++] = strdup(meta->ldflags[i]);
    }

    // Add macOS frameworks
    #ifdef __APPLE__
    for (size_t i = 0; i < meta->frameworks_count; i++) {
        link_flags[total_link_flags++] = strdup("-framework");
        link_flags[total_link_flags++] = strdup(meta->frameworks[i]);
    }
    #endif

    // Add system libs
    for (size_t i = 0; i < meta->system_libs_count; i++) {
        char *lib_flag = malloc(256);
        snprintf(lib_flag, 256, "-l%s", meta->system_libs[i]);
        link_flags[total_link_flags++] = lib_flag;
    }

    info->link_flags = link_flags;
    info->link_flags_count = total_link_flags;

    // Collect compile flags (include paths from pkg-config)
    size_t total_compile_flags = 0;
    char **compile_flags = calloc(1024, sizeof(char*));

    // Add pkg-config compile flags (include paths, defines)
    for (size_t i = 0; i < meta->pkg_config_count; i++) {
        char *pkg_cflags = get_pkg_config_flags(meta->pkg_config[i], "--cflags");
        if (pkg_cflags) {
            append_split_flags_move_to_end(compile_flags, &total_compile_flags, 1024, pkg_cflags);
            free(pkg_cflags);
        }
    }

    // Add custom include dirs
    for (size_t i = 0; i < meta->include_dirs_count; i++) {
        char *include_flag = malloc(256);
        snprintf(include_flag, 256, "-I%s", meta->include_dirs[i]);
        compile_flags[total_compile_flags++] = include_flag;
    }

    // Add custom cflags
    for (size_t i = 0; i < meta->cflags_count; i++) {
        compile_flags[total_compile_flags++] = strdup(meta->cflags[i]);
    }

    info->compile_flags = compile_flags;
    info->compile_flags_count = total_compile_flags;

    return info;
}

void module_build_info_free(ModuleBuildInfo *info) {
    if (!info) return;

    free(info->object_file);

    for (size_t i = 0; i < info->link_flags_count; i++) {
        free(info->link_flags[i]);
    }
    free(info->link_flags);

    for (size_t i = 0; i < info->compile_flags_count; i++) {
        free(info->compile_flags[i]);
    }
    free(info->compile_flags);

    free(info);
}

// Get all link flags from multiple modules (deduplicated)
char** module_get_link_flags(ModuleBuildInfo **modules, size_t count, size_t *out_count) {
    size_t total = 0;
    
    // Count total flags
    for (size_t i = 0; i < count; i++) {
        if (modules[i]) {
            total += modules[i]->link_flags_count;
        }
    }

    char **all_flags = calloc(total + 1, sizeof(char*));
    if (!all_flags) {
        *out_count = 0;
        return NULL;
    }

    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        if (modules[i]) {
            for (size_t j = 0; j < modules[i]->link_flags_count; j++) {
                const char *flag = modules[i]->link_flags[j];
                
                // Skip NULL or empty flags
                if (!flag || flag[0] == '\0') {
                    continue;
                }
                
                // De-duplicate by keeping the LAST occurrence (helps link order for dependent libs)
                for (size_t k = 0; k < pos; k++) {
                    if (strcmp(all_flags[k], flag) == 0) {
                        free(all_flags[k]);
                        for (size_t m = k; m + 1 < pos; m++) {
                            all_flags[m] = all_flags[m + 1];
                        }
                        pos--;
                        break;
                    }
                }

                all_flags[pos++] = strdup(flag);
            }
        }
    }

    *out_count = pos;
    return all_flags;
}

// Get all compile flags from multiple modules (with deduplication)
char** module_get_compile_flags(ModuleBuildInfo **modules, size_t count, size_t *out_count) {
    size_t total = 0;
    
    // Count total flags
    for (size_t i = 0; i < count; i++) {
        if (modules[i]) {
            total += modules[i]->compile_flags_count;
        }
    }

    char **all_flags = calloc(total + 1, sizeof(char*));
    if (!all_flags) {
        *out_count = 0;
        return NULL;
    }

    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        if (modules[i]) {
            for (size_t j = 0; j < modules[i]->compile_flags_count; j++) {
                // Skip NULL or invalid flags
                if (!modules[i]->compile_flags[j] || modules[i]->compile_flags[j][0] == '\0') {
                    continue;
                }
                
                // Check for duplicates
                bool duplicate = false;
                for (size_t k = 0; k < pos; k++) {
                    if (all_flags[k] && strcmp(all_flags[k], modules[i]->compile_flags[j]) == 0) {
                        duplicate = true;
                        break;
                    }
                }
                
                if (!duplicate) {
                    all_flags[pos++] = strdup(modules[i]->compile_flags[j]);
                }
            }
        }
    }

    *out_count = pos;
    return all_flags;
}

