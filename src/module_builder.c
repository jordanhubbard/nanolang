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

// ============================================================================
// Package Registry System - Central database of system package mappings
// ============================================================================

// Package manager enumeration
typedef enum {
    PKG_MGR_UNKNOWN = 0,
    PKG_MGR_APT,        // Debian/Ubuntu
    PKG_MGR_DNF,        // Fedora/RHEL (modern)
    PKG_MGR_YUM,        // Fedora/RHEL (legacy)
    PKG_MGR_PACMAN,     // Arch Linux
    PKG_MGR_ZYPPER,     // openSUSE
    PKG_MGR_APK,        // Alpine Linux
    PKG_MGR_PKG,        // FreeBSD pkg
    PKG_MGR_BREW,       // macOS Homebrew
    PKG_MGR_CHOCOLATEY, // Windows Chocolatey
    PKG_MGR_WINGET,     // Windows Package Manager
    PKG_MGR_SCOOP       // Windows Scoop
} PackageManager;

// Cached package registry (loaded once from packages.json)
static cJSON *package_registry = NULL;
static PackageManager detected_pkg_manager = PKG_MGR_UNKNOWN;

// Load packages.json into memory (cached)
static cJSON* load_package_registry(void) {
    if (package_registry) {
        return package_registry;
    }

    // Try to find packages.json
    const char* paths[] = {
        "packages.json",
        "../packages.json",
        "../../packages.json",
        NULL
    };

    FILE *fp = NULL;
    for (int i = 0; paths[i]; i++) {
        fp = fopen(paths[i], "r");
        if (fp) break;
    }

    if (!fp) {
        if (module_builder_verbose) {
            fprintf(stderr, "[PackageRegistry] Warning: packages.json not found, falling back to legacy package names\n");
        }
        return NULL;
    }

    // Read file
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
    cJSON *root = cJSON_Parse(content);
    free(content);

    if (!root) {
        fprintf(stderr, "[PackageRegistry] Error: Failed to parse packages.json\n");
        return NULL;
    }

    package_registry = root;
    if (module_builder_verbose) {
        printf("[PackageRegistry] Loaded packages.json\n");
    }

    return package_registry;
}

// Detect which package manager is available on this system
static PackageManager detect_package_manager(void) {
    if (detected_pkg_manager != PKG_MGR_UNKNOWN) {
        return detected_pkg_manager;
    }

    #ifdef _WIN32
    // Windows: Try chocolatey, winget, scoop
    if (access("C:\\ProgramData\\chocolatey\\bin\\choco.exe", F_OK) == 0) {
        detected_pkg_manager = PKG_MGR_CHOCOLATEY;
    } else if (system("where winget >nul 2>&1") == 0) {
        detected_pkg_manager = PKG_MGR_WINGET;
    } else if (system("where scoop >nul 2>&1") == 0) {
        detected_pkg_manager = PKG_MGR_SCOOP;
    }
    #elif defined(__APPLE__)
    // macOS: Homebrew
    if (access("/opt/homebrew/bin/brew", F_OK) == 0 || access("/usr/local/bin/brew", F_OK) == 0) {
        detected_pkg_manager = PKG_MGR_BREW;
    }
    #else
    // Unix: Try FreeBSD pkg, then Linux managers (in preference order)
    if (access("/usr/sbin/pkg", F_OK) == 0 || access("/usr/local/sbin/pkg", F_OK) == 0 ||
        system("command -v pkg >/dev/null 2>&1") == 0) {
        detected_pkg_manager = PKG_MGR_PKG;
    } else if (access("/usr/bin/apt-get", F_OK) == 0 || access("/usr/bin/apt", F_OK) == 0) {
        detected_pkg_manager = PKG_MGR_APT;
    } else if (access("/usr/bin/dnf", F_OK) == 0) {
        detected_pkg_manager = PKG_MGR_DNF;
    } else if (access("/usr/bin/yum", F_OK) == 0) {
        detected_pkg_manager = PKG_MGR_YUM;
    } else if (access("/usr/bin/pacman", F_OK) == 0) {
        detected_pkg_manager = PKG_MGR_PACMAN;
    } else if (access("/usr/bin/zypper", F_OK) == 0) {
        detected_pkg_manager = PKG_MGR_ZYPPER;
    } else if (access("/sbin/apk", F_OK) == 0) {
        detected_pkg_manager = PKG_MGR_APK;
    }
    #endif

    if (module_builder_verbose && detected_pkg_manager != PKG_MGR_UNKNOWN) {
        const char *names[] = {"unknown", "apt", "dnf", "yum", "pacman", "zypper", "apk", "pkg", "brew", "chocolatey", "winget", "scoop"};
        printf("[PackageRegistry] Detected package manager: %s\n", names[detected_pkg_manager]);
    }

    return detected_pkg_manager;
}

// Get package manager name string (for JSON lookup)
static const char* get_package_manager_name(PackageManager pm) {
    switch (pm) {
        case PKG_MGR_APT: return "apt";
        case PKG_MGR_DNF: return "dnf";
        case PKG_MGR_YUM: return "yum";
        case PKG_MGR_PACMAN: return "pacman";
        case PKG_MGR_ZYPPER: return "zypper";
        case PKG_MGR_APK: return "apk";
        case PKG_MGR_PKG: return "pkg";
        case PKG_MGR_BREW: return "brew";
        case PKG_MGR_CHOCOLATEY: return "chocolatey";
        case PKG_MGR_WINGET: return "winget";
        case PKG_MGR_SCOOP: return "scoop";
        default: return NULL;
    }
}

// Look up a package name in the registry for the current platform
// Returns NULL if not found or registry not available
static const char* lookup_package_name(const char *logical_name, PackageManager pm) {
    cJSON *registry = load_package_registry();
    if (!registry) {
        // No registry - return logical name as-is (legacy fallback)
        return logical_name;
    }

    cJSON *packages = cJSON_GetObjectItem(registry, "packages");
    if (!packages) {
        return logical_name;
    }

    cJSON *package = cJSON_GetObjectItem(packages, logical_name);
    if (!package) {
        if (module_builder_verbose) {
            fprintf(stderr, "[PackageRegistry] Warning: Package '%s' not found in registry\n", logical_name);
        }
        return logical_name;
    }

    cJSON *install = cJSON_GetObjectItem(package, "install");
    if (!install) {
        return logical_name;
    }

    const char *pm_name = get_package_manager_name(pm);
    if (!pm_name) {
        return logical_name;
    }

    cJSON *pm_entry = cJSON_GetObjectItem(install, pm_name);
    if (!pm_entry) {
        // Package not available for this package manager
        if (module_builder_verbose) {
            fprintf(stderr, "[PackageRegistry] Warning: Package '%s' not available for %s\n", logical_name, pm_name);
        }
        return NULL;
    }

    // Can be either a string or an object with "package" field
    if (cJSON_IsString(pm_entry)) {
        return pm_entry->valuestring;
    } else if (cJSON_IsObject(pm_entry)) {
        cJSON *pkg_name = cJSON_GetObjectItem(pm_entry, "package");
        if (pkg_name && cJSON_IsString(pkg_name)) {
            return pkg_name->valuestring;
        }
    }

    return logical_name;
}

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

static const char* module_builder_sudo_prefix(void) {
    static bool initialized = false;
    static const char *prefix = "sudo";

    if (!initialized) {
        bool interactive = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
        if (!interactive) {
            prefix = "sudo -n";
            printf("[Module]   Non-interactive shell detected; using sudo -n\n");
        }
        initialized = true;
    }

    return prefix;
}

// Install a single package using the detected package manager
static bool install_single_package(const char *package_name, PackageManager pm) {
    char cmd[2048];
    int result;
    const char *sudo_cmd = module_builder_sudo_prefix();

    switch (pm) {
        case PKG_MGR_APT:
            // Check if already installed
            snprintf(cmd, sizeof(cmd), "dpkg -l %s 2>/dev/null | grep -q '^ii'", package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "%s apt-get update -qq && %s apt-get install -y %s", sudo_cmd, sudo_cmd, package_name);
            printf("[Module]   Running: %s apt-get install -y %s\n", sudo_cmd, package_name);
            break;

        case PKG_MGR_DNF:
        case PKG_MGR_YUM:
            snprintf(cmd, sizeof(cmd), "rpm -q %s >/dev/null 2>&1", package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "%s %s install -y %s", 
                     sudo_cmd, pm == PKG_MGR_DNF ? "dnf" : "yum", package_name);
            break;

        case PKG_MGR_BREW:
            snprintf(cmd, sizeof(cmd), "brew list %s >/dev/null 2>&1", package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "brew install %s", package_name);
            break;

        case PKG_MGR_PKG:
            snprintf(cmd, sizeof(cmd), "pkg info -e %s >/dev/null 2>&1", package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "%s pkg install -y %s", sudo_cmd, package_name);
            printf("[Module]   Running: %s pkg install -y %s\n", sudo_cmd, package_name);
            break;

        case PKG_MGR_CHOCOLATEY:
            snprintf(cmd, sizeof(cmd), "choco list --local-only %s 2>nul | findstr /C:\"%s\" >nul", package_name, package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "choco install -y %s", package_name);
            break;

        case PKG_MGR_WINGET:
            snprintf(cmd, sizeof(cmd), "winget list %s >nul 2>&1", package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "winget install --silent %s", package_name);
            break;

        default:
            fprintf(stderr, "[Module]   ❌ Unknown package manager\n");
            return false;
    }

    printf("[Module]   Installing %s...\n", package_name);
    result = system(cmd);
    if (result == 0) {
        printf("[Module]   ✓ Successfully installed %s\n", package_name);
        return true;
    } else {
        fprintf(stderr, "[Module]   ❌ Failed to install %s\n", package_name);
        return false;
    }
}

// Install system packages from module metadata (with registry support)
static bool install_system_packages(ModuleBuildMetadata *meta) {
    PackageManager pm = detect_package_manager();
    
    if (pm == PKG_MGR_UNKNOWN) {
        if (meta->system_packages_count > 0 || meta->apt_packages_count > 0 || 
            meta->dnf_packages_count > 0 || meta->brew_packages_count > 0) {
            fprintf(stderr, "[Module] ⚠️  No supported package manager found\n");
            fprintf(stderr, "[Module]    Please install system packages manually for module '%s'\n", meta->name);
            return false;
        }
        return true;
    }

    bool all_installed = true;

    // Collect all package names (logical names for registry lookup)
    const char *pkg_names[256];
    size_t pkg_count = 0;

    // Priority 1: Use new unified system_packages format (preferred)
    if (meta->system_packages_count > 0) {
        for (size_t i = 0; i < meta->system_packages_count && pkg_count < 256; i++) {
            pkg_names[pkg_count++] = meta->system_packages[i];
        }
    } else {
        // Priority 2: Fall back to legacy platform-specific arrays (deprecated)
        for (size_t i = 0; i < meta->apt_packages_count && pkg_count < 256; i++) {
            pkg_names[pkg_count++] = meta->apt_packages[i];
        }
        for (size_t i = 0; i < meta->dnf_packages_count && pkg_count < 256; i++) {
            pkg_names[pkg_count++] = meta->dnf_packages[i];
        }
        for (size_t i = 0; i < meta->brew_packages_count && pkg_count < 256; i++) {
            pkg_names[pkg_count++] = meta->brew_packages[i];
        }
    }

    if (pkg_count > 0) {
        printf("[Module] Installing system packages for '%s'...\n", meta->name);
        
        for (size_t i = 0; i < pkg_count; i++) {
            const char *logical_name = pkg_names[i];
            
            // Look up actual package name for this platform in registry
            const char *actual_name = lookup_package_name(logical_name, pm);
            
            if (!actual_name) {
                fprintf(stderr, "[Module]   ⚠️  Package '%s' not available for this platform\n", logical_name);
                all_installed = false;
                continue;
            }

            if (!install_single_package(actual_name, pm)) {
                all_installed = false;
            }
        }
    }

    return all_installed;
}

// Track whether we've already attempted to install pkg-config
static bool pkg_config_install_attempted = false;

// Find pkg-config executable path, or NULL if not found
static const char* find_pkg_config(void) {
    // Check common locations
    if (access("/opt/homebrew/bin/pkg-config", X_OK) == 0) {
        return "/opt/homebrew/bin/pkg-config";
    }
    if (access("/usr/local/bin/pkg-config", X_OK) == 0) {
        return "/usr/local/bin/pkg-config";
    }
    if (access("/usr/bin/pkg-config", X_OK) == 0) {
        return "/usr/bin/pkg-config";
    }
    // Try PATH
    if (system("command -v pkg-config >/dev/null 2>&1") == 0) {
        return "pkg-config";
    }
    return NULL;
}

// Ensure pkg-config is installed, auto-installing if needed
static const char* ensure_pkg_config(void) {
    const char *pkg_config_path = find_pkg_config();
    if (pkg_config_path) {
        return pkg_config_path;
    }
    
    // pkg-config not found - try to auto-install it (once)
    if (!pkg_config_install_attempted) {
        pkg_config_install_attempted = true;
        
        PackageManager pm = detect_package_manager();
        if (pm != PKG_MGR_UNKNOWN) {
            const char *pkg_name = lookup_package_name("pkg-config", pm);
            if (pkg_name) {
                printf("[Module] pkg-config not found, installing...\n");
                if (install_single_package(pkg_name, pm)) {
                    // Try to find it again after installation
                    pkg_config_path = find_pkg_config();
                    if (pkg_config_path) {
                        return pkg_config_path;
                    }
                }
            }
        }
        
        fprintf(stderr, "[Module] Warning: pkg-config not available. Install it manually:\n");
        fprintf(stderr, "[Module]   macOS: brew install pkg-config\n");
        fprintf(stderr, "[Module]   Linux: sudo apt-get install pkg-config\n");
    }
    
    return NULL;
}

// Get pkg-config flags, with automatic package installation on failure
static char* get_pkg_config_flags(const char *package, const char *flag_type) {
    char cmd[512];
    
    // Ensure pkg-config is available (auto-install if needed)
    const char *pkg_config_path = ensure_pkg_config();
    if (!pkg_config_path) {
        return NULL;
    }
    
    // Run pkg-config with the found path
    snprintf(cmd, sizeof(cmd), "%s %s %s 2>/dev/null", pkg_config_path, flag_type, package);
    char *result = run_command(cmd);
    
    
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
    PARSE_STRING_ARRAY("system_packages", system_packages, system_packages_count);
    PARSE_STRING_ARRAY("apt_packages", apt_packages, apt_packages_count);
    PARSE_STRING_ARRAY("dnf_packages", dnf_packages, dnf_packages_count);
    PARSE_STRING_ARRAY("brew_packages", brew_packages, brew_packages_count);
    PARSE_STRING_ARRAY("owned_string_returns", owned_string_returns, owned_string_returns_count);

    #undef PARSE_STRING_ARRAY

    // Parse c_compiler (optional)
    cJSON *c_compiler = cJSON_GetObjectItem(json, "c_compiler");
    if (c_compiler && cJSON_IsString(c_compiler)) {
        meta->c_compiler = strdup(c_compiler->valuestring);
    }

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
    free(meta->c_compiler);

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
    FREE_STRING_ARRAY(system_packages, system_packages_count);
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
        if (meta->system_packages_count > 0 || meta->apt_packages_count > 0 || meta->dnf_packages_count > 0 || meta->brew_packages_count > 0) {
            if (!install_system_packages(meta)) {
                fprintf(stderr, "[Module] Warning: Some system packages failed to install for '%s'\n", meta->name);
                fprintf(stderr, "[Module] Continuing anyway - build may fail if dependencies are missing\n");
            }
        }

        // Get CC from environment, module.json, or use POSIX cc
        const char *cc = getenv("NANO_CC");
        if (!cc) cc = getenv("CC");
        if (!cc && meta->c_compiler) cc = meta->c_compiler;
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
        
        char shared_dir[1024];
        snprintf(shared_dir, sizeof(shared_dir), "%s/.build", meta->module_dir);
        bool shared_dir_ok = dir_exists(shared_dir) || mkdir_p(shared_dir);
        if (!shared_dir_ok) {
            fprintf(stderr, "Warning: Failed to create shared library directory for %s\n", meta->name);
        }

        if (shared_dir_ok) {
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
                bool is_framework_flag = (strcmp(flag, "-framework") == 0);
                bool is_framework_value = false;
                if (j > 0 && modules[i]->link_flags[j - 1]) {
                    is_framework_value = (strcmp(modules[i]->link_flags[j - 1], "-framework") == 0);
                }
                
                // Skip NULL or empty flags
                if (!flag || flag[0] == '\0') {
                    continue;
                }
                
                // De-duplicate by keeping the LAST occurrence (helps link order for dependent libs)
                if (!is_framework_flag && !is_framework_value) {
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

