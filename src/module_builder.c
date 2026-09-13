// Module Build System Implementation
// Handles automatic compilation of C sources, caching, and dependency tracking

#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include "module_builder.h"
#include "runtime/module_build_dir.h"
#ifdef __linux__
#include "runtime/assembler_capture.h"
#include <elf.h>
#endif
#include "utf8.h"
#include "shell_path.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>

// JSON parsing (simple, minimal implementation for module.json)
#include "cJSON.h"

bool module_builder_verbose = false;
static bool module_builder_can_prompt_sudo = false;

/* I require explicit host authority before running package-registry probes,
 * install overrides, package managers, or sudo from the module builder. */
static bool package_installation_allowed(void) {
    const char *value = getenv("NANO_ALLOW_PACKAGE_INSTALL");
    return value && strcmp(value, "1") == 0;
}

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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    fread(content, 1, size, fp);
#pragma GCC diagnostic pop
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

static bool is_linux_package_manager(PackageManager pm) {
    return pm == PKG_MGR_APT ||
           pm == PKG_MGR_DNF ||
           pm == PKG_MGR_YUM ||
           pm == PKG_MGR_PACMAN ||
           pm == PKG_MGR_ZYPPER ||
           pm == PKG_MGR_APK;
}

// Internal: locate the per-package-manager install entry in packages.json.
// Returns the cJSON node (string or object) for the requested platform, or
// NULL if the logical name / platform combination isn't represented.
static cJSON* lookup_pm_entry(const char *logical_name, PackageManager pm) {
    cJSON *registry = load_package_registry();
    if (!registry) return NULL;

    cJSON *packages = cJSON_GetObjectItem(registry, "packages");
    if (!packages) return NULL;

    cJSON *package = cJSON_GetObjectItem(packages, logical_name);
    if (!package) return NULL;

    cJSON *install = cJSON_GetObjectItem(package, "install");
    if (!install) return NULL;

    const char *pm_name = get_package_manager_name(pm);
    if (!pm_name) return NULL;

    cJSON *pm_entry = cJSON_GetObjectItem(install, pm_name);
    if (pm_entry) return pm_entry;

    if (is_linux_package_manager(pm)) {
        return cJSON_GetObjectItem(install, "linux");
    }

    return NULL;
}

// Look up a package name in the registry for the current platform.
// Returns NULL if the registry exists but the package/platform isn't listed
// (so the caller can skip), or the logical name unchanged when the registry
// itself is missing (legacy fallback).
static const char* lookup_package_name(const char *logical_name, PackageManager pm) {
    cJSON *registry = load_package_registry();
    if (!registry) return logical_name;

    cJSON *packages = cJSON_GetObjectItem(registry, "packages");
    if (!packages) return logical_name;

    cJSON *package = cJSON_GetObjectItem(packages, logical_name);
    if (!package) {
        if (module_builder_verbose) {
            fprintf(stderr, "[PackageRegistry] Warning: Package '%s' not found in registry\n", logical_name);
        }
        return logical_name;
    }

    cJSON *pm_entry = lookup_pm_entry(logical_name, pm);
    if (!pm_entry) {
        if (module_builder_verbose) {
            fprintf(stderr, "[PackageRegistry] Warning: Package '%s' not available for %s\n",
                    logical_name, get_package_manager_name(pm));
        }
        return NULL;
    }

    if (cJSON_IsString(pm_entry)) {
        return pm_entry->valuestring;
    }
    if (cJSON_IsObject(pm_entry)) {
        cJSON *pkg_name = cJSON_GetObjectItem(pm_entry, "package");
        if (pkg_name && cJSON_IsString(pkg_name)) {
            return pkg_name->valuestring;
        }
    }
    return logical_name;
}

// Optional shell command that overrides the default install incantation
// (e.g., `brew install --cask mujoco`). Returns NULL when the registry
// entry doesn't specify one.
static const char* lookup_install_command(const char *logical_name, PackageManager pm) {
    cJSON *pm_entry = lookup_pm_entry(logical_name, pm);
    if (!pm_entry || !cJSON_IsObject(pm_entry)) return NULL;
    cJSON *cmd = cJSON_GetObjectItem(pm_entry, "install_command");
    return (cmd && cJSON_IsString(cmd)) ? cmd->valuestring : NULL;
}

// Manual entries are dependencies I can detect but cannot install through the
// platform package manager. They still belong in the registry so modules get
// one clear hint instead of a bogus apt/brew/pkg failure.
static bool lookup_manual_install(const char *logical_name, PackageManager pm) {
    cJSON *pm_entry = lookup_pm_entry(logical_name, pm);
    if (!pm_entry || !cJSON_IsObject(pm_entry)) return false;
    cJSON *manual = cJSON_GetObjectItem(pm_entry, "manual");
    return cJSON_IsTrue(manual);
}

static const char* lookup_install_message(const char *logical_name, PackageManager pm) {
    cJSON *pm_entry = lookup_pm_entry(logical_name, pm);
    if (!pm_entry || !cJSON_IsObject(pm_entry)) return NULL;
    cJSON *message = cJSON_GetObjectItem(pm_entry, "install_message");
    return (message && cJSON_IsString(message)) ? message->valuestring : NULL;
}

// Optional shell command that probes whether the package is already
// installed (exit 0 = installed). Mirrors `install_command`. NULL → fall
// back to the package-manager default (e.g., `brew list <pkg>`).
static const char* lookup_test_command(const char *logical_name, PackageManager pm) {
    cJSON *pm_entry = lookup_pm_entry(logical_name, pm);
    if (!pm_entry || !cJSON_IsObject(pm_entry)) return NULL;
    cJSON *cmd = cJSON_GetObjectItem(pm_entry, "test_command");
    return (cmd && cJSON_IsString(cmd)) ? cmd->valuestring : NULL;
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

static void append_flag_fragment(char **out_flags, size_t *out_count, size_t out_cap, const char *flags) {
    if (!flags || !flags[0] || !out_flags || !out_count || *out_count >= out_cap) return;
    /* I preserve trusted shell fragments byte-for-byte and in order. Splitting
     * on whitespace corrupts quoted paths and deduplication can change linking. */
    char *copy = strdup(flags);
    if (copy) out_flags[(*out_count)++] = copy;
}

static void append_platform_ldflags(ModuleBuildMetadata *meta, char **out_flags, size_t *out_count, size_t out_cap, bool split) {
    if (!meta || !out_flags || !out_count) return;

#ifdef __APPLE__
    char **flags = meta->ldflags_macos;
    size_t count = meta->ldflags_macos_count;
#elif defined(__FreeBSD__)
    char **flags = meta->ldflags_freebsd;
    size_t count = meta->ldflags_freebsd_count;
#else
    char **flags = meta->ldflags_linux;
    size_t count = meta->ldflags_linux_count;
#endif

    for (size_t i = 0; i < count; i++) {
        if (split) {
            append_flag_fragment(out_flags, out_count, out_cap, flags[i]);
        } else if (*out_count < out_cap) {
            out_flags[(*out_count)++] = strdup(flags[i]);
        }
    }
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

/* ============================================================
 * Incremental compilation: content-hash cache
 *
 * Stores FNV-1a hashes of each C source + header in
 *   <module_build_dir>/source_hashes.json
 *   (see nano_module_build_dir / NANO_BUILD_CACHE)
 * On rebuild check: if all hashes match, skip compilation
 * even when mtime is newer (e.g. after git checkout).
 * ============================================================ */

/* FNV-1a 64-bit hash of file contents */
static uint64_t hash_file_fnv1a(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;

    uint64_t h = 14695981039346656037ULL;
    unsigned char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        for (size_t i = 0; i < n; i++) {
            h ^= (uint64_t)buf[i];
            h *= 1099511628211ULL;
        }
    }
    bool failed = ferror(fp) != 0;
    if (fclose(fp) != 0) failed = true;
    return failed ? 0 : h;
}

static const char *module_selected_compiler(const ModuleBuildMetadata *meta) {
    const char *cc = getenv("NANO_CC");
    if (!cc) cc = getenv("CC");
    if (!cc) cc = meta->c_compiler;
    return cc ? cc : "cc";
}

/* I fingerprint fields with a terminating zero, including unset versus empty
 * environment values. I persist only the digest, not environment strings. */
static void hash_context_field(uint64_t *hash, const char *value) {
    const unsigned char *p = (const unsigned char *)value;
    do {
        *hash ^= *p;
        *hash *= 1099511628211ULL;
    } while (*p++);
}

/* I resolve only a simple executable token. Shell expressions still execute
 * through the existing build path, but I cannot identify their tools safely. */
static char *module_compiler_path(const char *cc) {
    if (!cc[0] || strspn(cc, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./+-") != strlen(cc)) return NULL;
    if (strchr(cc, '/')) return access(cc, X_OK) == 0 ? realpath(cc, NULL) : NULL;
    const char *path = getenv("PATH");
    if (!path) return NULL;
    const char *part = path;
    for (;;) {
        const char *end = strchr(part, ':');
        size_t length = end ? (size_t)(end - part) : strlen(part);
        char *candidate = malloc(length + strlen(cc) + 3);
        if (!candidate) return NULL;
        if (length) {
            memcpy(candidate, part, length);
            candidate[length] = '/';
            strcpy(candidate + length + 1, cc);
        } else strcpy(candidate, cc);
        char *resolved = access(candidate, X_OK) == 0 ? realpath(candidate, NULL) : NULL;
        free(candidate);
        if (resolved) return resolved;
        if (!end) return NULL;
        part = end + 1;
    }
}

static uint64_t module_build_context(const ModuleBuildMetadata *meta) {
    /* I observed clang's Make dependency output turn literal backslashes into
     * slashes. These inputs still compile, but cannot establish cache reuse. */
    if (!meta || !meta->module_dir || strpbrk(meta->module_dir, "\\\r\n")) return 0;
    for (size_t group = 0; group < 3; group++) {
        char **paths = group == 0 ? meta->c_sources : group == 1 ? meta->shared_c_sources : meta->include_dirs;
        size_t count = group == 0 ? meta->c_sources_count : group == 1 ? meta->shared_c_sources_count : meta->include_dirs_count;
        for (size_t i = 0; i < count; i++)
            if (!paths[i] || strpbrk(paths[i], "\\\r\n")) return 0;
    }
    const char *cc = module_selected_compiler(meta);
    char *driver = module_compiler_path(cc);
    char *cwd = getcwd(NULL, 0);
    struct stat st;
    uint64_t driver_hash = driver && stat(driver, &st) == 0 && S_ISREG(st.st_mode)
        ? hash_file_fnv1a(driver) : 0;
    if (!cwd || !driver_hash) { free(driver); free(cwd); return 0; }
    uint64_t hash = 14695981039346656037ULL;
    hash_context_field(&hash, "nanolang-c-build-context-v20-gcc-objects");
    hash_context_field(&hash, cc);
    hash_context_field(&hash, driver);
    hash_context_field(&hash, cwd);
    char digest[24];
    snprintf(digest, sizeof(digest), "%llu", (unsigned long long)driver_hash);
    hash_context_field(&hash, digest);
    const char *variables[] = {
        "PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
        "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "SDKROOT", "DEVELOPER_DIR",
        "MACOSX_DEPLOYMENT_TARGET", "IPHONEOS_DEPLOYMENT_TARGET", "ARCHFLAGS",
        "CFLAGS", "CPPFLAGS", "LDFLAGS", "PKG_CONFIG", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR", "SOURCE_DATE_EPOCH", "LANG", "LC_ALL", "LC_CTYPE",
        "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
        "NANO_TOOLCHAIN_ID", "NANO_AS_CAPTURE_HELPER"
    };
    for (size_t i = 0; i < sizeof(variables) / sizeof(variables[0]); i++) {
        const char *value = getenv(variables[i]);
        hash_context_field(&hash, variables[i]);
        hash_context_field(&hash, value ? "set" : "unset");
        if (value) hash_context_field(&hash, value);
    }
    free(driver);
    free(cwd);
    return hash;
}

static char *module_get_artifact_dir(const char *module_dir) {
    char path[2048];
    return nano_module_artifact_dir(module_dir, path, sizeof(path)) ? strdup(path) : NULL;
}

/* Path to the hash cache file for a module */
static char *hash_cache_path(const char *module_dir) {
    char *build_dir = module_get_artifact_dir(module_dir);
    if (!build_dir) return NULL;
    char *out = malloc(strlen(build_dir) + 32);
    if (out) snprintf(out, strlen(build_dir) + 32, "%s/source_hashes.json", build_dir);
    free(build_dir);
    return out;
}

/* Load hash cache JSON for a module (caller must cJSON_Delete) */
static cJSON *load_hash_cache(const char *module_dir) {
    char *path = hash_cache_path(module_dir);
    if (!path) return NULL;
    FILE *fp = fopen(path, "r");
    free(path);
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) { fclose(fp); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t read_size = fread(buf, 1, (size_t)sz, fp);
    if (read_size != (size_t)sz || ferror(fp)) {
        fclose(fp);
        free(buf);
        return NULL;
    }
    buf[sz] = '\0';
    fclose(fp);
    cJSON *root = cJSON_Parse(buf);
    free(buf);
    return root;
}

/* Resolve a module system header to an absolute path by searching:
   1. -F<dir> framework dirs from cflags_macos: <dir>/<modname>.framework/Headers/<header>
   2. -I<dir> dirs from cflags, cflags_macos, cflags_linux: <dir>/<header>
   3. Standard fallback dirs.
   Returns malloc'd path or NULL. Caller must free. */
static char *find_system_header_path(const char *header, ModuleBuildMetadata *meta) {
    char probe[1024];

#ifdef __APPLE__
    /* Search -F<framework_root> flags from cflags_macos */
    static const char *framework_suffixes[] = {
        "Headers",
        "Versions/Current/Headers",
        "Versions/A/Headers",
        NULL
    };
    for (size_t fi = 0; meta && fi < meta->cflags_macos_count; fi++) {
        const char *flag = meta->cflags_macos[fi];
        if (strncmp(flag, "-F", 2) != 0) continue;
        const char *fdir = flag + 2;
        for (int si = 0; framework_suffixes[si]; si++) {
            snprintf(probe, sizeof(probe), "%s/%s.framework/%s/%s",
                     fdir, meta->name ? meta->name : "", framework_suffixes[si], header);
            if (access(probe, R_OK) == 0) return strdup(probe);
        }
    }
#endif

    /* Search -I<dir> flags across all platform cflag arrays */
    struct { char **flags; size_t count; } cflag_sets[] = {
        { meta ? meta->cflags : NULL,       meta ? meta->cflags_count : 0 },
#ifdef __APPLE__
        { meta ? meta->cflags_macos : NULL, meta ? meta->cflags_macos_count : 0 },
#elif defined(__linux__)
        { meta ? meta->cflags_linux : NULL, meta ? meta->cflags_linux_count : 0 },
#elif defined(__FreeBSD__)
        { meta ? meta->cflags_freebsd : NULL, meta ? meta->cflags_freebsd_count : 0 },
#endif
        { NULL, 0 }
    };
    for (int si = 0; cflag_sets[si].flags; si++) {
        for (size_t fi = 0; fi < cflag_sets[si].count; fi++) {
            const char *flag = cflag_sets[si].flags[fi];
            if (strncmp(flag, "-I", 2) != 0) continue;
            snprintf(probe, sizeof(probe), "%s/%s", flag + 2, header);
            if (access(probe, R_OK) == 0) return strdup(probe);
        }
    }

    /* Standard fallback locations */
    static const char *std_dirs[] = {
        "/usr/local/include", "/usr/include",
        "/opt/homebrew/include", "/opt/mujoco/include",
        NULL
    };
    for (int di = 0; std_dirs[di]; di++) {
        snprintf(probe, sizeof(probe), "%s/%s", std_dirs[di], header);
        if (access(probe, R_OK) == 0) return strdup(probe);
    }
    return NULL;
}

/* Hash all system headers declared in meta->headers and add to a cJSON object.
   Keys are prefixed with "sysheader:" to avoid colliding with c_sources entries. */
static void hash_system_headers(cJSON *root, ModuleBuildMetadata *meta) {
    if (!meta || meta->headers_count == 0) return;
    for (size_t i = 0; i < meta->headers_count; i++) {
        const char *hdr = meta->headers[i];
        char *path = find_system_header_path(hdr, meta);
        if (!path) continue;
        uint64_t h = hash_file_fnv1a(path);
        free(path);
        char hstr[24];
        snprintf(hstr, sizeof(hstr), "%llu", (unsigned long long)h);
        char key[256];
        snprintf(key, sizeof(key), "sysheader:%s", hdr);
        cJSON_AddStringToObject(root, key, hstr);
    }
}

/* Check that system header hashes in cache still match the files on disk. */
static bool system_headers_match(cJSON *cache, ModuleBuildMetadata *meta) {
    if (!meta || meta->headers_count == 0) return true;
    for (size_t i = 0; i < meta->headers_count; i++) {
        const char *hdr = meta->headers[i];
        char key[256];
        snprintf(key, sizeof(key), "sysheader:%s", hdr);
        cJSON *item = cJSON_GetObjectItemCaseSensitive(cache, key);
        /* If there's no cached entry for a sysheader, treat as clean (first run). */
        if (!item || !cJSON_IsString(item)) continue;
        char *path = find_system_header_path(hdr, meta);
        if (!path) {
            /* Header was cached but is now gone → rebuild. */
            return false;
        }
        uint64_t h = hash_file_fnv1a(path);
        free(path);
        char hstr[24];
        snprintf(hstr, sizeof(hstr), "%llu", (unsigned long long)h);
        if (h == 0 || strcmp(item->valuestring, hstr) != 0) return false;
    }
    return true;
}

/* I only cache a complete dependency record. Unknown escapes, missing files,
 * truncation and allocation failures cannot establish safe reuse. */
static bool hash_dependency(cJSON *root, const char *path) {
    uint64_t hash = hash_file_fnv1a(path);
    if (!hash) return false;
    char key[4101], value[24];
    int n = snprintf(key, sizeof(key), "dep:%s", path);
    if (n < 0 || (size_t)n >= sizeof(key)) return false;
    snprintf(value, sizeof(value), "%llu", (unsigned long long)hash);
    return cJSON_GetObjectItemCaseSensitive(root, key) ||
           cJSON_AddStringToObject(root, key, value);
}

static bool hash_depfile_into_cache(cJSON *root, const char *depfile_path) {
    FILE *fp = fopen(depfile_path, "rb");
    if (!fp) return false;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return false; }
    long size = ftell(fp);
    if (size <= 0 || (unsigned long)size >= SIZE_MAX ||
        fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return false; }
    char *buf = malloc((size_t)size + 1);
    if (!buf) { fclose(fp); return false; }
    bool ok = fread(buf, 1, (size_t)size, fp) == (size_t)size && !ferror(fp);
    if (fclose(fp) != 0) ok = false;
    buf[size] = '\0';
    static const char target[] = "nano_module_dependencies:";
    if (!ok || memchr(buf, 0, (size_t)size) ||
        strncmp(buf, target, sizeof(target) - 1) != 0) {
        free(buf);
        return false;
    }
    char *p = buf + sizeof(target) - 1;
    char token[4096];
    size_t used = 0, count = 0;
    while (ok) {
        if (!*p || *p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            if (used) {
                token[used] = '\0';
                ok = hash_dependency(root, token);
                used = 0;
                count++;
            }
            if (!*p) break;
            p++;
        } else if (*p == '\\' && p[1] == '\n') {
            p += 2;
        } else if (*p == '\\' && p[1] == '\r' && p[2] == '\n') {
            p += 3;
        } else {
            char byte = *p++;
            if (byte == '$') {
                if (*p != '$') { ok = false; break; }
                p++;
            } else if (byte == '\\' && (*p == ' ' || *p == '\t' || *p == '#' || *p == '\\')) {
                byte = *p++;
            } else if (byte == '#') {
                ok = false;
                break;
            }
            if (used == sizeof(token) - 1) { ok = false; break; }
            token[used++] = byte;
        }
    }
    free(buf);
    return ok && count > 0;
}

/* I accept literal GCC-style paths or LLVM's escaped paths. If both spellings
 * name different files, I cannot identify the compiler's input and withhold
 * reuse. The Make record remains required, but cannot erase this evidence. */
static const char module_guard_advice[] = "Multiple include guards may be useful for:";

static bool hash_include_trace(cJSON *root, const char *trace_path) {
    FILE *fp = fopen(trace_path, "rb");
    if (!fp) return false;
    char *line = NULL;
    size_t capacity = 0;
    ssize_t length;
    bool ok = true;
    bool guard_advice = false;
    while (ok && (length = getline(&line, &capacity, fp)) >= 0) {
        if (length < 3 || length > 8192 || line[length - 1] != '\n' ||
            memchr(line, 0, (size_t)length)) { ok = false; break; }
        line[length - 1] = 0;
        if (strcmp(line, module_guard_advice) == 0) { guard_advice = true; continue; }
        if (guard_advice) {
            /* GCC repeats paths here. I accept only already recorded inputs. */
            char key[4101];
            int n = snprintf(key, sizeof(key), "dep:%s", line);
            ok = n >= 0 && (size_t)n < sizeof(key) && cJSON_GetObjectItemCaseSensitive(root, key);
            continue;
        }
        char *raw = line;
        while (*raw == '.') raw++;
        if (raw == line || *raw++ != ' ' || !*raw) { ok = false; break; }
        char decoded[4096];
        size_t used = 0;
        bool decodable = true;
        for (const unsigned char *p = (unsigned char *)raw; *p; p++) {
            unsigned char byte = *p;
            if (byte < 32 || byte == 127 || used == sizeof(decoded) - 1) {
                ok = false;
                break;
            }
            if (byte == '\\') {
                p++;
                if (*p == '\\' || *p == '"') byte = *p;
                else if (*p == 'n') byte = '\n';
                else if (*p == 't') byte = '\t';
                else if (*p >= '0' && *p <= '3' && p[1] >= '0' && p[1] <= '7' &&
                         p[2] >= '0' && p[2] <= '7') {
                    byte = (unsigned char)((p[0] - '0') * 64 + (p[1] - '0') * 8 + p[2] - '0');
                    p += 2;
                    if (!byte) { decodable = false; break; }
                } else { decodable = false; break; }
            }
            decoded[used++] = (char)byte;
        }
        if (!ok) break;
        decoded[used] = 0;
        struct stat raw_stat, decoded_stat;
        bool raw_exists = stat(raw, &raw_stat) == 0 && S_ISREG(raw_stat.st_mode);
        bool decoded_exists = decodable && stat(decoded, &decoded_stat) == 0 && S_ISREG(decoded_stat.st_mode);
        if (raw_exists && decoded_exists && strcmp(raw, decoded) != 0) {
            ok = false;
        } else if (raw_exists) ok = hash_dependency(root, raw);
        else if (decoded_exists) ok = hash_dependency(root, decoded);
        else ok = false;
    }
    if (ferror(fp) || !feof(fp)) ok = false;
    free(line);
    if (fclose(fp) != 0) ok = false;
    return ok;
}

static bool hash_depfiles_in_build_dir(cJSON *root, const char *build_dir,
                                       const ModuleBuildMetadata *meta) {
    /* I require one dependency record for every requested compilation. */
    for (size_t group = 0; group < 2; group++) {
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count; i++) {
            char path[2048];
            int n;
            if (group) n = snprintf(path, sizeof(path), "%s/__shared_%zu.d", build_dir, i);
            else if (count == 1) n = snprintf(path, sizeof(path), "%s/%s.d", build_dir, meta->name);
            else n = snprintf(path, sizeof(path), "%s/%s_%zu.d", build_dir, meta->name, i);
            if (n < 0 || (size_t)n >= sizeof(path) || !hash_depfile_into_cache(root, path))
                return false;
            char trace[2060];
            int t = snprintf(trace, sizeof(trace), "%s.includes", path);
            if (t < 0 || (size_t)t >= sizeof(trace) || !hash_include_trace(root, trace)) return false;
        }
    }
    return meta->c_sources_count > 0;
}

/* Verify that every "dep:<path>" entry in the cache still matches the file on disk.
   A missing entry (no dep data yet) is treated as clean — first build sets the baseline. */
static bool dep_hashes_match(cJSON *cache) {
    cJSON *item = cache ? cache->child : NULL;
    while (item) {
        if (item->string && strncmp(item->string, "dep:", 4) == 0) {
            const char *path = item->string + 4;
            if (access(path, R_OK) != 0)
                return false; /* previously-tracked header is gone */
            uint64_t h = hash_file_fnv1a(path);
            char hstr[24];
            snprintf(hstr, sizeof(hstr), "%llu", (unsigned long long)h);
            if (h == 0 || !cJSON_IsString(item) || strcmp(item->valuestring, hstr) != 0)
                return false;
        }
        item = item->next;
    }
    return true;
}

/* Save hash cache JSON for a module */
static void save_hash_cache(const char *build_dir, cJSON *root) {
    char *path = malloc(strlen(build_dir) + 32);
    if (!path) return;
    sprintf(path, "%s/source_hashes.json", build_dir);
    char *text = cJSON_PrintUnformatted(root);
    if (!text) { free(path); return; }
    char *temporary = malloc(strlen(path) + 16);
    if (temporary) {
        sprintf(temporary, "%s.XXXXXX", path);
        int fd = mkstemp(temporary);
        if (fd >= 0) {
            FILE *fp = fdopen(fd, "w");
            bool ok = false;
            if (fp) {
                ok = fputs(text, fp) >= 0;
                if (fclose(fp) != 0) ok = false;
            } else close(fd);
            if (ok) (void)rename(temporary, path);
            (void)unlink(temporary);
        }
        free(temporary);
    }
    free(text);
    free(path);
}

typedef struct {
    char **cflags;
    char **libs;
    size_t count;
} ModulePkgFlags;

#ifdef __APPLE__
/* I read tagged NUL-terminated paths, not human-readable archive(member)
 * lines. This records selected bytes and negative searches, not a snapshot
 * of bytes during linking or a complete inventory of indirect flag inputs. */
static cJSON *module_link_inputs(const char *record_path, const char *stage,
                                 const char *output) {
    FILE *file = fopen(record_path, "rb");
    if (!file) return NULL;
    cJSON *inputs = cJSON_CreateObject();
    bool ok = inputs != NULL, version = false, emitted = false, internal = false;
    size_t records = 0, bytes = 0;
    int tag;
    while (ok && (tag = fgetc(file)) != EOF) {
        if (++records > 65536 || emitted) { ok = false; break; }
        char path[8192];
        size_t length = 0;
        int ch;
        while ((ch = fgetc(file)) != EOF && ch != 0) {
            if (length + 1 >= sizeof(path) || ++bytes > 16 * 1024 * 1024) { ok = false; break; }
            path[length++] = (char)ch;
        }
        if (!ok || ch != 0 || !length) { ok = false; break; }
        path[length] = 0;
        if (tag == 0) {
            const char *prefix = "@(#)PROGRAM:ld PROJECT:ld-";
            if (records != 1 || strncmp(path, prefix, strlen(prefix))) ok = false;
            else version = true;
            continue;
        }
        if (!version) { ok = false; break; }
        if (tag == 64) {
            emitted = strcmp(path, output) == 0;
            if (!emitted) ok = false;
            continue;
        }
        if (tag != 16 && tag != 17) { ok = false; break; }
        struct stat st;
        int status = stat(path, &st);
        char digest[24];
        if (tag == 17) {
            if (status == 0 || (errno != ENOENT && errno != ENOTDIR)) { ok = false; break; }
            strcpy(digest, "missing");
        } else {
            if (status != 0 || !S_ISREG(st.st_mode)) { ok = false; break; }
            size_t stage_length = strlen(stage);
            if (!strncmp(path, stage, stage_length) && path[stage_length] == '/') {
                internal = true;
                continue;
            }
            uint64_t hash = hash_file_fnv1a(path);
            if (!hash) { ok = false; break; }
            snprintf(digest, sizeof(digest), "%llu", (unsigned long long)hash);
        }
        cJSON *previous = cJSON_GetObjectItemCaseSensitive(inputs, path);
        if (previous) ok = cJSON_IsString(previous) && !strcmp(previous->valuestring, digest);
        else ok = cJSON_AddStringToObject(inputs, path, digest) != NULL;
    }
    ok = ok && version && emitted && internal && records > 2 && !ferror(file) && feof(file);
    if (fclose(file) != 0) ok = false;
    if (!ok) { cJSON_Delete(inputs); return NULL; }
    return inputs;
}

static bool module_link_inputs_match(const cJSON *inputs) {
    if (!cJSON_IsObject(inputs) || !inputs->child) return false;
    for (cJSON *item = inputs->child; item; item = item->next) {
        if (!item->string || !cJSON_IsString(item)) return false;
        struct stat st;
        if (!strcmp(item->valuestring, "missing")) {
            if (stat(item->string, &st) == 0 || (errno != ENOENT && errno != ENOTDIR)) return false;
        } else {
            if (stat(item->string, &st) != 0 || !S_ISREG(st.st_mode)) return false;
            uint64_t hash = hash_file_fnv1a(item->string);
            char digest[24];
            snprintf(digest, sizeof(digest), "%llu", (unsigned long long)hash);
            if (!hash || strcmp(item->valuestring, digest)) return false;
        }
    }
    return true;
}
#endif

static uint64_t module_preprocess_fingerprint(ModuleBuildMetadata *meta,
                                             const ModulePkgFlags *flags);

/* Update the on-disk hash cache after a successful build */
static void module_update_hash_cache(const char *module_dir, ModuleBuildMetadata *meta,
                                     const char *build_dir, uint64_t preprocessing,
                                     const cJSON *link_observation __attribute__((unused))) {
    if (!meta || meta->c_sources_count == 0) return;
    uint64_t context = module_build_context(meta);
    if (!context) return;
    cJSON *root = cJSON_CreateObject();
    if (!root) return;
    char context_string[24];
    snprintf(context_string, sizeof(context_string), "%llu", (unsigned long long)context);
    cJSON_AddStringToObject(root, "__build_context_v1", context_string);
    snprintf(context_string, sizeof(context_string), "%llu", (unsigned long long)preprocessing);
    if (!preprocessing || !cJSON_AddStringToObject(root, "__preprocessing_v1", context_string)) {
        cJSON_Delete(root);
        return;
    }
    for (int shared = 0; shared < 2; shared++) {
        char **sources = shared ? meta->shared_c_sources : meta->c_sources;
        size_t count = shared ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count; i++) {
            char src[1024];
            snprintf(src, sizeof(src), "%s/%s", module_dir, sources[i]);
            uint64_t h = hash_file_fnv1a(src);
            char hstr[24];
            snprintf(hstr, sizeof(hstr), "%llu", (unsigned long long)h);
            cJSON_AddStringToObject(root, sources[i], hstr);
        }
    }
    /* Also hash module.json itself */
    char mj[1024];
    snprintf(mj, sizeof(mj), "%s/module.json", module_dir);
    uint64_t mj_hash = hash_file_fnv1a(mj);
    char mj_hstr[24];
    snprintf(mj_hstr, sizeof(mj_hstr), "%llu", (unsigned long long)mj_hash);
    cJSON_AddStringToObject(root, "module.json", mj_hstr);
    /* Hash system headers declared in module.json (fast top-level check) */
    hash_system_headers(root, meta);
    /* Hash all transitively-included headers from compiler-generated .d files */
    bool complete = hash_depfiles_in_build_dir(root, build_dir, meta);
#ifdef __APPLE__
    /* I retain the hashes checked around the final link. Replacing them with
     * current hashes could label old code with bytes that were never linked. */
    cJSON *link_inputs = module_link_inputs_match(link_observation)
        ? cJSON_Duplicate(link_observation, true) : NULL;
    if (!link_inputs || !cJSON_AddItemToObject(root, "__link_inputs_v1", link_inputs)) {
        cJSON_Delete(link_inputs);
        complete = false;
    }
#endif
    if (complete)
        save_hash_cache(build_dir, root);
    else if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD"))
        fprintf(stderr, "I cannot establish complete dependency evidence; I will rebuild this module next time\n");
    cJSON_Delete(root);
}

/* Returns true if all source hashes match the cache → skip rebuild */
static bool hashes_match(const char *module_dir, ModuleBuildMetadata *meta,
                         const ModulePkgFlags *flags) {
    cJSON *cache = load_hash_cache(module_dir);
    if (!cache) return false;
    uint64_t context = module_build_context(meta);
    char context_string[24];
    snprintf(context_string, sizeof(context_string), "%llu", (unsigned long long)context);
    cJSON *stored_context = cJSON_GetObjectItemCaseSensitive(cache, "__build_context_v1");
    bool match = context && cJSON_IsString(stored_context) &&
        strcmp(stored_context->valuestring, context_string) == 0;
    for (int shared = 0; shared < 2 && match; shared++) {
        char **sources = shared ? meta->shared_c_sources : meta->c_sources;
        size_t count = shared ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count && match; i++) {
            char src[1024];
            snprintf(src, sizeof(src), "%s/%s", module_dir, sources[i]);
            uint64_t h = hash_file_fnv1a(src);
            char hstr[24];
            snprintf(hstr, sizeof(hstr), "%llu", (unsigned long long)h);
            cJSON *item = cJSON_GetObjectItemCaseSensitive(cache, sources[i]);
            if (h == 0 || !item || !cJSON_IsString(item) || strcmp(item->valuestring, hstr) != 0) {
                match = false;
            }
        }
    }
    /* Check module.json hash */
    if (match) {
        char mj[1024];
        snprintf(mj, sizeof(mj), "%s/module.json", module_dir);
        uint64_t h = hash_file_fnv1a(mj);
        char hstr[24];
        snprintf(hstr, sizeof(hstr), "%llu", (unsigned long long)h);
        cJSON *item = cJSON_GetObjectItemCaseSensitive(cache, "module.json");
        if (h == 0 || !item || !cJSON_IsString(item) || strcmp(item->valuestring, hstr) != 0)
            match = false;
    }
    if (match) match = system_headers_match(cache, meta);
    if (match) match = dep_hashes_match(cache);
#ifdef __APPLE__
    if (match) match = module_link_inputs_match(cJSON_GetObjectItemCaseSensitive(cache, "__link_inputs_v1"));
#endif
    if (match) {
        cJSON *stored = cJSON_GetObjectItemCaseSensitive(cache, "__preprocessing_v1");
        uint64_t observed = cJSON_IsString(stored) ? module_preprocess_fingerprint(meta, flags) : 0;
        char digest[24];
        snprintf(digest, sizeof(digest), "%llu", (unsigned long long)observed);
        match = observed && strcmp(stored->valuestring, digest) == 0 &&
                context == module_build_context(meta);
    }
    cJSON_Delete(cache);
    return match;
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

// Helper to detect WSL2
static bool is_wsl2(void) {
    FILE *fp = fopen("/proc/version", "r");
    if (!fp) return false;

    char buffer[256];
    bool is_wsl = false;
    if (fgets(buffer, sizeof(buffer), fp)) {
        // Check for "microsoft", "Microsoft", "WSL", or "wsl" in /proc/version
        if (strstr(buffer, "microsoft") || strstr(buffer, "Microsoft") ||
            strstr(buffer, "WSL") || strstr(buffer, "wsl")) {
            is_wsl = true;
        }
    }
    fclose(fp);
    return is_wsl;
}

// Helper to check if passwordless sudo is available
static bool has_passwordless_sudo(void) {
    // Try to run a simple command with sudo -n (non-interactive)
    int result = system("sudo -n true 2>/dev/null");
    return result == 0;
}

static const char* module_builder_sudo_prefix(void) {
    static bool initialized = false;
    static const char *prefix = "sudo";

    if (!initialized) {
        bool interactive = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
        bool wsl = is_wsl2();
        module_builder_can_prompt_sudo = interactive || wsl;

        // On WSL2, prefer interactive sudo even if shell appears non-interactive
        // since WSL2 often runs in contexts where interactive sudo works fine
        if (!interactive && !wsl) {
            // Check if passwordless sudo is available
            if (has_passwordless_sudo()) {
                prefix = "sudo -n";
                printf("[Module]   Non-interactive shell detected; using sudo -n\n");
            } else {
                prefix = "sudo";
                printf("[Module]   Non-interactive shell detected but passwordless sudo not available\n");
                printf("[Module]   Package installation may fail - consider configuring passwordless sudo\n");
            }
        } else if (!interactive && wsl) {
            // WSL2: always try interactive sudo since the environment is more forgiving
            prefix = "sudo";
            printf("[Module]   WSL2 detected; using interactive sudo\n");
        }
        initialized = true;
    }

    return prefix;
}

// dpkg pipes list output through the system pager whenever stdout is a tty, so
// a probe like `dpkg -l <pkg>` stops the build at a --More-- prompt. Every
// command I run from the registry is an unattended probe or install, so none of
// them may page.
static void disable_subcommand_pager(void) {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    initialized = true;

#ifndef _WIN32
    setenv("DPKG_PAGER", "cat", 1);
    setenv("PAGER", "cat", 1);
#endif
}

// Install a single package using the detected package manager.
// When install_cmd_override / test_cmd_override are non-NULL they win over
// the built-in defaults — used for things that don't fit a simple template
// like `brew install --cask mujoco` or `pip3 install --user <pkg>`.
static bool install_single_package_ex(const char *package_name, PackageManager pm,
                                      const char *install_cmd_override,
                                      const char *test_cmd_override) {
    if (!package_installation_allowed()) return false;
    char cmd[2048];
    int result;
    const char *sudo_cmd = module_builder_sudo_prefix();

    disable_subcommand_pager();

    // Check if sudo will work before attempting installation (for package managers that need it)
    bool needs_sudo = (pm == PKG_MGR_APT || pm == PKG_MGR_DNF || pm == PKG_MGR_YUM ||
                      pm == PKG_MGR_PKG || pm == PKG_MGR_PACMAN || pm == PKG_MGR_ZYPPER ||
                      pm == PKG_MGR_APK);

    if (test_cmd_override) {
        if (system(test_cmd_override) == 0) {
            printf("[Module]   ✓ %s already installed\n", package_name);
            return true;
        }
    }
    if (install_cmd_override) {
        printf("[Module]   Running: %s\n", install_cmd_override);
        result = system(install_cmd_override);
        if (result == 0) {
            printf("[Module]   ✓ Successfully installed %s\n", package_name);
            return true;
        }
        fprintf(stderr, "[Module]   ❌ Failed to install %s (custom command exit=%d)\n", package_name, result);
        return false;
    }

    if (needs_sudo && !module_builder_can_prompt_sudo && !has_passwordless_sudo()) {
        // Non-interactive run without passwordless sudo cannot install automatically.
        fprintf(stderr, "[Module]   ⚠️  Cannot auto-install %s: sudo requires a password\n", package_name);
        fprintf(stderr, "[Module]   Please install manually:\n");

        switch (pm) {
            case PKG_MGR_APT:
                fprintf(stderr, "[Module]     sudo apt-get install %s\n", package_name);
                break;
            case PKG_MGR_DNF:
                fprintf(stderr, "[Module]     sudo dnf install %s\n", package_name);
                break;
            case PKG_MGR_YUM:
                fprintf(stderr, "[Module]     sudo yum install %s\n", package_name);
                break;
            case PKG_MGR_PKG:
                fprintf(stderr, "[Module]     sudo pkg install %s\n", package_name);
                break;
            case PKG_MGR_PACMAN:
                fprintf(stderr, "[Module]     sudo pacman -S %s\n", package_name);
                break;
            case PKG_MGR_ZYPPER:
                fprintf(stderr, "[Module]     sudo zypper install %s\n", package_name);
                break;
            case PKG_MGR_APK:
                fprintf(stderr, "[Module]     sudo apk add %s\n", package_name);
                break;
            default:
                break;
        }
        fprintf(stderr, "[Module]   Alternatively, configure passwordless sudo for package installation\n");
        return false;
    }

    switch (pm) {
        case PKG_MGR_APT:
            // Check if already installed. Use dpkg-query rather than `dpkg -l`
            // so the probe never pipes list output through the system pager.
            snprintf(cmd, sizeof(cmd), "dpkg-query -W -f='${Status}' %s 2>/dev/null | grep -q '^install ok installed$'", package_name);
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

        case PKG_MGR_PACMAN:
            snprintf(cmd, sizeof(cmd), "pacman -Q %s >/dev/null 2>&1", package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "%s pacman -S --noconfirm %s", sudo_cmd, package_name);
            break;

        case PKG_MGR_ZYPPER:
            snprintf(cmd, sizeof(cmd), "rpm -q %s >/dev/null 2>&1", package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "%s zypper --non-interactive install -y %s", sudo_cmd, package_name);
            break;

        case PKG_MGR_APK:
            snprintf(cmd, sizeof(cmd), "apk info -e %s >/dev/null 2>&1", package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "%s apk add %s", sudo_cmd, package_name);
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

        case PKG_MGR_SCOOP:
            snprintf(cmd, sizeof(cmd), "scoop list %s >nul 2>&1", package_name);
            if (system(cmd) == 0) {
                printf("[Module]   ✓ %s already installed\n", package_name);
                return true;
            }
            snprintf(cmd, sizeof(cmd), "scoop install %s", package_name);
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

static bool install_single_package(const char *package_name, PackageManager pm) {
    return install_single_package_ex(package_name, pm, NULL, NULL);
}

static bool module_has_system_package_metadata(ModuleBuildMetadata *meta) {
    if (!meta) return false;
    return meta->system_packages_count > 0 ||
           meta->apt_packages_count > 0 ||
           meta->dnf_packages_count > 0 ||
           meta->brew_packages_count > 0;
}

#ifdef __APPLE__
// A native framework satisfies a same-named pkg-config dependency on macOS.
// Trying Homebrew first can turn an ordinary compile into a large package
// installation even though the SDK already provides the library.
static bool module_pkg_is_native_framework(const ModuleBuildMetadata *meta, const char *package) {
    if (!meta || !package) return false;
    for (size_t i = 0; i < meta->frameworks_count; i++) {
        if (strcasecmp(meta->frameworks[i], package) == 0 ||
            (strncasecmp(package, "free", 4) == 0 &&
             strcasecmp(meta->frameworks[i], package + 4) == 0)) return true;
    }
    return false;
}
#endif

// Install system packages from module metadata (with registry support)
static bool install_system_packages(ModuleBuildMetadata *meta) {
    if (!package_installation_allowed()) return false;
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

    disable_subcommand_pager();

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

#ifdef __APPLE__
            if (module_pkg_is_native_framework(meta, logical_name)) {
                printf("[Module]   ✓ %s provided by macOS framework\n", logical_name);
                continue;
            }
#endif

            // Look up actual package name for this platform in registry
            const char *actual_name = lookup_package_name(logical_name, pm);

            if (!actual_name) {
                fprintf(stderr, "[Module]   ⚠️  Package '%s' not available for this platform\n", logical_name);
                all_installed = false;
                continue;
            }

            const char *test_cmd = lookup_test_command(logical_name, pm);
            if (lookup_manual_install(logical_name, pm)) {
                if (test_cmd && system(test_cmd) == 0) {
                    printf("[Module]   ✓ %s already installed\n", actual_name);
                } else {
                    const char *message = lookup_install_message(logical_name, pm);
                    printf("[Module]   I cannot install %s with this package manager.\n", actual_name);
                    if (message && message[0]) {
                        printf("[Module]   %s\n", message);
                    }
                }
                continue;
            }

            const char *install_cmd = lookup_install_command(logical_name, pm);
            if (!install_single_package_ex(actual_name, pm, install_cmd, test_cmd)) {
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
    const char *configured = getenv("PKG_CONFIG");
    if (configured && configured[0]) return configured;
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

/* I locate pkg-config, installing it only with explicit host authority. */
static const char* ensure_pkg_config(void) {
    const char *pkg_config_path = find_pkg_config();
    if (pkg_config_path) {
        return pkg_config_path;
    }

    if (!package_installation_allowed()) {
        fprintf(stderr, "[Module] I could not find pkg-config. Install it manually, or explicitly allow package installation with NANO_ALLOW_PACKAGE_INSTALL=1.\n");
        return NULL;
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

static bool module_build_append(char *buffer, size_t capacity, const char *format, ...);

/* I preserve executable, package and search-path bytes as shell words. */
static bool pkg_config_command(char *cmd, size_t capacity, const char *package, const char *mode) {
    const char *tool = ensure_pkg_config();
    if (!tool || !package || !mode) return false;
    cmd[0] = 0;
#ifdef __APPLE__
    char search[4096];
    const char *inherited = getenv("PKG_CONFIG_PATH");
    int n = snprintf(search, sizeof(search), "/opt/homebrew/opt/%s/lib/pkgconfig:/usr/local/opt/%s/lib/pkgconfig:%s",
                     package, package, inherited ? inherited : "");
    if (n < 0 || (size_t)n >= sizeof(search) ||
        !module_append_path_flag(cmd, capacity, "PKG_CONFIG_PATH=", search)) return false;
#endif
    return module_append_path_flag(cmd, capacity, "", tool) &&
           module_append_path_flag(cmd, capacity, "", mode) &&
           module_append_path_flag(cmd, capacity, "-- ", package) &&
           module_build_append(cmd, capacity, " 2>/dev/null");
}

// Check if a pkg-config package is available (returns true if installed)
static bool check_pkg_config_package(const char *package) {
    char cmd[8192];
    return pkg_config_command(cmd, sizeof(cmd), package, "--exists") && system(cmd) == 0;
}

// Check all pkg-config dependencies for a module, return true if all available
// If missing_pkg is not NULL, it will be set to the name of first missing package
static bool check_module_pkg_dependencies(ModuleBuildMetadata *meta, const char **missing_pkg) {
    if (!meta || meta->pkg_config_count == 0) {
        return true;  // No dependencies to check
    }
    
    for (size_t i = 0; i < meta->pkg_config_count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        if (!check_pkg_config_package(meta->pkg_config[i])) {
            if (missing_pkg) {
                *missing_pkg = meta->pkg_config[i];
            }
            return false;
        }
    }
    return true;
}

/* I query compiler/linker flags through the authority-aware tool lookup. */
static char* get_pkg_config_flags(const char *package, const char *flag_type) {
    char cmd[8192];
    if (!pkg_config_command(cmd, sizeof(cmd), package, flag_type)) return NULL;
    FILE *pipe = popen(cmd, "r");
    if (!pipe) return NULL;
    enum { MAX_FLAGS = 65536 };
    char *result = malloc(MAX_FLAGS + 1);
    if (!result) { pclose(pipe); return NULL; }
    size_t count = fread(result, 1, MAX_FLAGS + 1, pipe);
    bool ok = count <= MAX_FLAGS && !ferror(pipe) && feof(pipe) && !memchr(result, 0, count);
    if (pclose(pipe) != 0) ok = false;
    if (!ok) {
        fprintf(stderr, "I could not query %s for package '%s'\n", flag_type, package);
        free(result);
        return NULL;
    }
    result[count] = 0;
    size_t first = strspn(result, " \t\r\n");
    while (count > first && strchr(" \t\r\n", result[count - 1])) count--;
    memmove(result, result + first, count - first);
    result[count - first] = 0;
    /* An allocated empty string is success; NULL is always a failed query. */
    return result;
}

static void module_pkg_flags_free(ModulePkgFlags *flags) {
    for (size_t i = 0; i < flags->count; i++) {
        free(flags->cflags ? flags->cflags[i] : NULL);
        free(flags->libs ? flags->libs[i] : NULL);
    }
    free(flags->cflags);
    free(flags->libs);
    memset(flags, 0, sizeof(*flags));
}

/* I keep one response set for all consumers in a build, including callers
 * that only need returned flags. Failed capture leaves no partial set. */
static bool module_pkg_flags_capture(ModuleBuildMetadata *meta, ModulePkgFlags *flags) {
    memset(flags, 0, sizeof(*flags));
    flags->count = meta->pkg_config_count;
    if (!flags->count) return true;
    flags->cflags = calloc(flags->count, sizeof(char *));
    flags->libs = calloc(flags->count, sizeof(char *));
    if (!flags->cflags || !flags->libs) goto failed;
    for (size_t i = 0; i < flags->count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        flags->cflags[i] = get_pkg_config_flags(meta->pkg_config[i], "--cflags");
        if (!flags->cflags[i]) goto failed;
        flags->libs[i] = get_pkg_config_flags(meta->pkg_config[i], "--libs");
        if (!flags->libs[i]) goto failed;
    }
    return true;
failed:
    module_pkg_flags_free(flags);
    return false;
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

static void append_string_array_unique(char ***arr, size_t *count, const char *value) {
    if (!arr || !count || !value || value[0] == '\0') return;

    for (size_t i = 0; i < *count; i++) {
        if ((*arr)[i] && strcmp((*arr)[i], value) == 0) {
            return;
        }
    }

    char **new_arr = realloc(*arr, sizeof(char*) * (*count + 1));
    if (!new_arr) return;

    *arr = new_arr;
    (*arr)[*count] = strdup(value);
    if ((*arr)[*count]) {
        (*count)++;
    }
}

static ModuleBuildMetadata* module_load_metadata_at_directory(const char *module_dir) {
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/module.json", module_dir);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        fprintf(stderr, "I cannot represent the module manifest path\n");
        return NULL;
    }

    if (!file_exists(path)) {
        // No module.json = pure nanolang module
        return NULL;
    }

    // Read file
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s\n", path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (size < 0) {
        fprintf(stderr, "Error: Could not read %s\n", path);
        fclose(fp);
        return NULL;
    }

    char *content = malloc((size_t)size + 1);
    if (!content) {
        fclose(fp);
        return NULL;
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    fread(content, 1, (size_t)size, fp);
#pragma GCC diagnostic pop
    content[size] = '\0';
    fclose(fp);

    if (!nl_utf8_validate(content, (size_t)size, NULL)) {
        fprintf(stderr, "Error: %s is not valid UTF-8\n", path);
        free(content);
        return NULL;
    }

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
    PARSE_STRING_ARRAY("cflags_macos", cflags_macos, cflags_macos_count);
    PARSE_STRING_ARRAY("cflags_linux", cflags_linux, cflags_linux_count);
    PARSE_STRING_ARRAY("cflags_freebsd", cflags_freebsd, cflags_freebsd_count);
    PARSE_STRING_ARRAY("ldflags", ldflags, ldflags_count);
    PARSE_STRING_ARRAY("ldflags_macos", ldflags_macos, ldflags_macos_count);
    PARSE_STRING_ARRAY("ldflags_linux", ldflags_linux, ldflags_linux_count);
    PARSE_STRING_ARRAY("ldflags_freebsd", ldflags_freebsd, ldflags_freebsd_count);
    PARSE_STRING_ARRAY("frameworks", frameworks, frameworks_count);
    PARSE_STRING_ARRAY("dependencies", dependencies, dependencies_count);
    PARSE_STRING_ARRAY("system_packages", system_packages, system_packages_count);
    PARSE_STRING_ARRAY("apt_packages", apt_packages, apt_packages_count);
    PARSE_STRING_ARRAY("dnf_packages", dnf_packages, dnf_packages_count);
    PARSE_STRING_ARRAY("brew_packages", brew_packages, brew_packages_count);
    PARSE_STRING_ARRAY("owned_string_returns", owned_string_returns, owned_string_returns_count);
    PARSE_STRING_ARRAY("shared_c_sources", shared_c_sources, shared_c_sources_count);

    #undef PARSE_STRING_ARRAY

    // Compatibility parsing for object-style dependency manifests:
    // {
    //   "dependencies": {
    //     "modules": ["std"],
    //     "system": ["sdl2"] or [{"id":"sdl2"}]
    //   }
    // }
    cJSON *dependencies_obj = cJSON_GetObjectItem(json, "dependencies");
    if (dependencies_obj && cJSON_IsObject(dependencies_obj)) {
        cJSON *module_deps = cJSON_GetObjectItem(dependencies_obj, "modules");
        if (module_deps && cJSON_IsArray(module_deps)) {
            int dep_count = cJSON_GetArraySize(module_deps);
            for (int i = 0; i < dep_count; i++) {
                cJSON *item = cJSON_GetArrayItem(module_deps, i);
                if (cJSON_IsString(item)) {
                    append_string_array_unique(&meta->dependencies, &meta->dependencies_count, item->valuestring);
                }
            }
        }

        cJSON *system_deps = cJSON_GetObjectItem(dependencies_obj, "system");
        if (system_deps && cJSON_IsArray(system_deps)) {
            int sys_count = cJSON_GetArraySize(system_deps);
            for (int i = 0; i < sys_count; i++) {
                cJSON *item = cJSON_GetArrayItem(system_deps, i);
                if (cJSON_IsString(item)) {
                    append_string_array_unique(&meta->system_packages, &meta->system_packages_count, item->valuestring);
                } else if (cJSON_IsObject(item)) {
                    cJSON *id = cJSON_GetObjectItem(item, "id");
                    if (id && cJSON_IsString(id)) {
                        append_string_array_unique(&meta->system_packages, &meta->system_packages_count, id->valuestring);
                    }
                }
            }
        }
    }

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

    // Parse install object (for dependency auto-installation)
    // Supports two formats:
    //   Nested: { "macos": { "brew": "pkg" }, "linux": { "apt": "pkg" } }
    //   Flat:   { "brew": "pkg", "apt": "pkg" }
    cJSON *install = cJSON_GetObjectItem(json, "install");
    if (install && cJSON_IsObject(install)) {
        // Try nested format first (macos/linux objects)
        cJSON *macos = cJSON_GetObjectItem(install, "macos");
        if (macos && cJSON_IsObject(macos)) {
            cJSON *brew = cJSON_GetObjectItem(macos, "brew");
            if (brew && cJSON_IsString(brew)) {
                meta->install_brew = strdup(brew->valuestring);
            }
        }
        cJSON *linux_obj = cJSON_GetObjectItem(install, "linux");
        if (linux_obj && cJSON_IsObject(linux_obj)) {
            cJSON *apt = cJSON_GetObjectItem(linux_obj, "apt");
            if (apt && cJSON_IsString(apt)) {
                meta->install_apt = strdup(apt->valuestring);
            }
        }
        // Fall back to flat format if nested not found
        if (!meta->install_brew) {
            cJSON *brew = cJSON_GetObjectItem(install, "brew");
            if (brew && cJSON_IsString(brew)) {
                meta->install_brew = strdup(brew->valuestring);
            }
        }
        if (!meta->install_apt) {
            cJSON *apt = cJSON_GetObjectItem(install, "apt");
            if (apt && cJSON_IsString(apt)) {
                meta->install_apt = strdup(apt->valuestring);
            }
        }
    }

    meta->module_dir = strdup(module_dir);

    /* Resolve relative include_dirs to absolute paths.
     * Module manifests use project-root-relative paths (e.g. "src"),
     * which break when CWD differs from project root. Walk up from
     * module_dir to find the project root and make paths absolute. */
    for (size_t i = 0; i < meta->include_dirs_count; i++) {
        if (meta->include_dirs[i][0] == '/') continue;
        if (dir_exists(meta->include_dirs[i])) continue;

        char parent[1024];
        strncpy(parent, module_dir, sizeof(parent) - 1);
        parent[sizeof(parent) - 1] = '\0';
        bool resolved = false;
        for (int depth = 0; depth < 8 && !resolved; depth++) {
            char *slash = strrchr(parent, '/');
            if (!slash) break;
            *slash = '\0';
            char candidate[2048];
            snprintf(candidate, sizeof(candidate), "%s/%s", parent, meta->include_dirs[i]);
            if (dir_exists(candidate)) {
                free(meta->include_dirs[i]);
                meta->include_dirs[i] = strdup(candidate);
                resolved = true;
            }
        }
    }

    /* Same resolution for -I flags in cflags */
    for (size_t i = 0; i < meta->cflags_count; i++) {
        if (strncmp(meta->cflags[i], "-I", 2) != 0) continue;
        const char *inc_path = meta->cflags[i] + 2;
        if (inc_path[0] == '/') continue;
        if (dir_exists(inc_path)) continue;

        char parent[1024];
        strncpy(parent, module_dir, sizeof(parent) - 1);
        parent[sizeof(parent) - 1] = '\0';
        for (int depth = 0; depth < 8; depth++) {
            char *slash = strrchr(parent, '/');
            if (!slash) break;
            *slash = '\0';
            char candidate[2048];
            snprintf(candidate, sizeof(candidate), "%s/%s", parent, inc_path);
            if (dir_exists(candidate)) {
                char resolved_flag[2060];
                snprintf(resolved_flag, sizeof(resolved_flag), "-I%s", candidate);
                free(meta->cflags[i]);
                meta->cflags[i] = strdup(resolved_flag);
                break;
            }
        }
    }

    cJSON_Delete(json);
    return meta;
}

ModuleBuildMetadata* module_load_metadata(const char *module_dir) {
    if (!module_dir || !module_dir[0]) return NULL;
    char *canonical = realpath(module_dir, NULL);
    if (!canonical) return NULL;
    /* I use the physical module directory for both cache identity and
     * module-relative include fallback, regardless of an import alias. */
    ModuleBuildMetadata *meta = module_load_metadata_at_directory(canonical);
    free(canonical);
    return meta;
}

void module_metadata_free(ModuleBuildMetadata *meta) {
    if (!meta) return;

    free(meta->name);
    free(meta->version);
    free(meta->description);
    free(meta->module_dir);
    free(meta->c_compiler);
    free(meta->install_brew);
    free(meta->install_apt);

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
    FREE_STRING_ARRAY(cflags_macos, cflags_macos_count);
    FREE_STRING_ARRAY(cflags_linux, cflags_linux_count);
    FREE_STRING_ARRAY(cflags_freebsd, cflags_freebsd_count);
    FREE_STRING_ARRAY(ldflags, ldflags_count);
    FREE_STRING_ARRAY(ldflags_macos, ldflags_macos_count);
    FREE_STRING_ARRAY(ldflags_linux, ldflags_linux_count);
    FREE_STRING_ARRAY(ldflags_freebsd, ldflags_freebsd_count);
    FREE_STRING_ARRAY(frameworks, frameworks_count);
    FREE_STRING_ARRAY(dependencies, dependencies_count);
    FREE_STRING_ARRAY(system_packages, system_packages_count);
    FREE_STRING_ARRAY(apt_packages, apt_packages_count);
    FREE_STRING_ARRAY(dnf_packages, dnf_packages_count);
    FREE_STRING_ARRAY(brew_packages, brew_packages_count);
    FREE_STRING_ARRAY(owned_string_returns, owned_string_returns_count);
    FREE_STRING_ARRAY(shared_c_sources, shared_c_sources_count);

    #undef FREE_STRING_ARRAY

    free(meta);
}

// Module build directory management

char* module_get_build_dir(const char *module_dir) {
    char *build_dir = malloc(1024);
    if (!build_dir) return NULL;
    if (!nano_module_build_dir(module_dir, build_dir, 1024)) {
        free(build_dir);
        return NULL;
    }
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

static bool module_needs_rebuild_with_flags(const char *module_dir, ModuleBuildMetadata *meta,
                                           const ModulePkgFlags *flags) {
    if (!meta || meta->c_sources_count == 0) {
        // No C sources = no rebuild needed
        return false;
    }

    char *build_dir = module_get_artifact_dir(module_dir);
    if (!build_dir) return true;

    char object_file[1024];
    snprintf(object_file, sizeof(object_file), "%s/%s.o", build_dir, meta->name);
    free(build_dir);

    struct stat object_stat;
    if (lstat(object_file, &object_stat) != 0 || !S_ISREG(object_stat.st_mode) || object_stat.st_size == 0) {
        if (module_builder_verbose) {
            printf("[Module] %s needs build: object file missing\n", meta->name);
        }
        return true;
    }

    /* If the shared library is missing, rebuild so interpreter FFI can load it */
    char *slib_dir = module_get_artifact_dir(module_dir);
    char shared_lib[1024];
    if (!slib_dir) {
        return true;
    }
    #ifdef __APPLE__
    snprintf(shared_lib, sizeof(shared_lib), "%s/lib%s.dylib", slib_dir, meta->name);
    #else
    snprintf(shared_lib, sizeof(shared_lib), "%s/lib%s.so", slib_dir, meta->name);
    #endif
    free(slib_dir);
    struct stat library_stat;
    if (lstat(shared_lib, &library_stat) != 0 || !S_ISREG(library_stat.st_mode) ||
        library_stat.st_size == 0) {
        if (module_builder_verbose) {
            printf("[Module] I must rebuild %s: shared library missing or empty\n", meta->name);
        }
        return true;
    }

    /* Fast path: compare content hashes. If all hashes match the
     * persisted cache, sources haven't changed regardless of mtime
     * (handles git checkout, rsync copies, CI environments). */
    if (hashes_match(module_dir, meta, flags)) {
        if (module_builder_verbose) {
            printf("[Module] %s up-to-date (hash cache hit)\n", meta->name);
        }
        return false;
    }

    /* I cannot turn missing or contradictory content evidence into a cache
     * hit merely because a timestamp is old. */
    return true;
}

bool module_needs_rebuild(const char *module_dir, ModuleBuildMetadata *meta) {
    return module_needs_rebuild_with_flags(module_dir, meta, NULL);
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

/* Make sure a module's declared system dependencies are present on this host.
 *
 * Whether libsdl2-ttf-dev is installed is a fact about the machine, not about
 * my object cache, so I check it on every build. Tying the check to
 * module_needs_rebuild() meant a module whose cached object was still current
 * never had its packages verified: the compile then died on a missing header
 * instead of installing the package.
 *
 * When pkg-config already resolves every declared package there is nothing to
 * do, which keeps warm builds free of package-manager probes. */
static bool ensure_module_system_deps(ModuleBuildMetadata *meta) {
    if (!meta) return false;

    const char *missing_pkg = NULL;
    bool allow_install = package_installation_allowed();
    if ((meta->pkg_config_count > 0 || !allow_install) &&
        check_module_pkg_dependencies(meta, &missing_pkg)) {
        return true;
    }

    if (!allow_install) {
        fprintf(stderr, "[Module] Package '%s' not found for module '%s'\n", missing_pkg, meta->name);
        fprintf(stderr, "[Module] I will not install system packages during compilation without NANO_ALLOW_PACKAGE_INSTALL=1. Install dependencies manually or opt in for a trusted build.\n");
        return false;
    }

    bool has_package_metadata = module_has_system_package_metadata(meta);
    if (has_package_metadata) {
        if (!install_system_packages(meta)) {
            fprintf(stderr, "[Module] Warning: Some system packages failed to install for '%s'\n", meta->name);
            fprintf(stderr, "[Module] Continuing anyway - build may fail if dependencies are missing\n");
        }
    }

    missing_pkg = NULL;
    if (check_module_pkg_dependencies(meta, &missing_pkg)) {
        return true;
    }

    fprintf(stderr, "[Module] Package '%s' not found for module '%s'\n", missing_pkg, meta->name);

    if (has_package_metadata) {
        fprintf(stderr,
                "[Module] Module '%s' declared system_packages, but '%s' is still missing after auto-install\n",
                meta->name, missing_pkg);
        return false;
    }

    // Legacy fallback for older manifests that only define install.{apt,brew}
    PackageManager pm = detect_package_manager();
    const char *legacy_pkg = NULL;
    if (pm == PKG_MGR_BREW) {
        legacy_pkg = meta->install_brew;
    } else if (pm == PKG_MGR_APT) {
        legacy_pkg = meta->install_apt;
    }

    if (legacy_pkg) {
        fprintf(stderr, "[Module] Attempting legacy install fallback: %s\n", legacy_pkg);
        if (!install_single_package(legacy_pkg, pm) || !check_pkg_config_package(missing_pkg)) {
            fprintf(stderr,
                    "[Module] Failed to install missing dependency '%s' for module '%s'\n",
                    missing_pkg, meta->name);
            return false;
        }
        return true;
    }

    if (meta->install_brew || meta->install_apt) {
        fprintf(stderr,
                "[Module] Legacy install mapping for module '%s' does not match this package manager\n",
                meta->name);
        return false;
    }

    fprintf(stderr, "[Module] Skipping module '%s' - install '%s' manually\n", meta->name, missing_pkg);
    return false;
}

/* I keep truncation inside the buffer and report it before running a command. */
static bool module_build_append(char *buffer, size_t capacity, const char *format, ...) {
    size_t used = strlen(buffer);
    if (used >= capacity) return false;
    va_list args;
    va_start(args, format);
    int n = vsnprintf(buffer + used, capacity - used, format, args);
    va_end(args);
    return n >= 0 && (size_t)n < capacity - used;
}

typedef enum { MODULE_FLAG_UNKNOWN, MODULE_FLAG_PREPROCESS, MODULE_FLAG_BOTH } ModuleFlagPhase;

/* I decode literal shell words only. Expansions, operators and globbing remain
 * on the original path. I never execute a fragment to discover its words. */
static int module_flag_word(const char **cursor, char *word, size_t capacity) {
    const unsigned char *p = (const unsigned char *)*cursor;
    size_t used = 0;
    unsigned char quote = 0;
    bool started = false;
    while (*p) {
        unsigned char c = *p++;
        if (quote == '\'') {
            if (c == '\'') { quote = 0; continue; }
        } else if (quote == '"') {
            if (c == '"') { quote = 0; continue; }
            if (c == '$' || c == '`') return -1;
            if (c == '\\') {
                if (!*p) return -1;
                if (*p == '\n') { p++; continue; }
                if (strchr("\\$`\"", *p)) c = *p++;
            }
        } else {
            if (c == ' ' || c == '\t') {
                if (started) break;
                continue;
            }
            if (c == '\\') {
                if (!*p) return -1;
                c = *p++;
                if (c == '\n') continue;
            } else if (c == '\'' || c == '"') {
                quote = c;
                started = true;
                continue;
            } else if (strchr("$`;&|()<>\n\r*?[]~#{}", c)) return -1;
            started = true;
        }
        if (used + 1 >= capacity) return -1;
        word[used++] = (char)c;
    }
    if (quote || !capacity) return -1;
    word[used] = 0;
    *cursor = (const char *)p;
    return started ? 1 : 0;
}

/* I classify decoded literal tokens, not unevaluated shell fragments. */
static ModuleFlagPhase module_snapshot_flag(const char *flag) {
    if (!flag) return MODULE_FLAG_UNKNOWN;
    const char *both[] = {
        "-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Og",
        "-g", "-g0", "-g1", "-g2", "-g3", "-fPIC", "-fpic",
        "-std=c89", "-std=c90", "-std=c99", "-std=c11", "-std=c17", "-std=c18",
        "-std=gnu89", "-std=gnu90", "-std=gnu99", "-std=gnu11", "-std=gnu17", "-std=gnu18",
        "-Wall", "-Wextra", "-Werror", "-Wpedantic",
        "-Wno-unused-parameter", "-Wno-unused-variable", "-Wno-unused-function"
    };
    for (size_t i = 0; i < sizeof(both) / sizeof(both[0]); i++)
        if (!strcmp(flag, both[i])) return MODULE_FLAG_BOTH;
    size_t length = strlen(flag);
    if (length < 3 || flag[0] != '-') return MODULE_FLAG_UNKNOWN;
    if (flag[1] == 'I' && strcmp(flag, "-I-")) return MODULE_FLAG_PREPROCESS;
    if (flag[1] != 'D' && flag[1] != 'U') return MODULE_FLAG_UNKNOWN;
    const char *name = flag + 2;
    if (!strchr("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_", *name)) return MODULE_FLAG_UNKNOWN;
    name += strspn(name, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_");
    if (!*name || (flag[1] == 'D' && *name == '=')) return MODULE_FLAG_PREPROCESS;
    return MODULE_FLAG_UNKNOWN;
}

/* NULL output validates eligibility. Otherwise I emit only the flags needed
 * after preprocessing, quoting each decoded argument independently. */
static bool module_retained_flags(const char *fragment, char *output, size_t capacity) {
    if (!fragment) return false;
    const char *cursor = fragment;
    char word[4096], argument[4096], combined[4096];
    int status;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
        if (!strcmp(word, "-D") || !strcmp(word, "-U") || !strcmp(word, "-I")) {
            if (module_flag_word(&cursor, argument, sizeof(argument)) != 1) return false;
            int n = snprintf(combined, sizeof(combined), "%s%s", word, argument);
            if (n < 0 || (size_t)n >= sizeof(combined) ||
                module_snapshot_flag(combined) != MODULE_FLAG_PREPROCESS) return false;
            continue;
        }
        ModuleFlagPhase kind = module_snapshot_flag(word);
        if (kind == MODULE_FLAG_UNKNOWN) return false;
        if (kind == MODULE_FLAG_BOTH && output &&
            !module_append_path_flag(output, capacity, "", word)) return false;
    }
    return status == 0;
}

static char **module_platform_cflags(const ModuleBuildMetadata *meta, size_t *count) {
#ifdef __APPLE__
    *count = meta->cflags_macos_count;
    return meta->cflags_macos;
#elif defined(__FreeBSD__)
    *count = meta->cflags_freebsd_count;
    return meta->cflags_freebsd;
#else
    *count = meta->cflags_linux_count;
    return meta->cflags_linux;
#endif
}

typedef enum { MODULE_C_PREPROCESS, MODULE_C_COMPILE, MODULE_C_RETAINED, MODULE_C_RETAINED_ASSEMBLY,
               MODULE_C_EMIT_ASSEMBLY, MODULE_C_ASSEMBLE } ModuleCPhase;

static bool module_compile_prefix(ModuleBuildMetadata *meta, char *prefix, size_t capacity,
                                  ModuleCPhase phase, const ModulePkgFlags *snapshot) {
    prefix[0] = 0;
    bool retained = phase == MODULE_C_RETAINED || phase == MODULE_C_RETAINED_ASSEMBLY;
    bool ok = module_build_append(prefix, capacity, "%s %s -fPIC",
                                   module_selected_compiler(meta), phase == MODULE_C_PREPROCESS ? "-E" :
                                   (phase == MODULE_C_EMIT_ASSEMBLY || phase == MODULE_C_RETAINED_ASSEMBLY) ? "-S" : "-c");
    /* I already applied C code-generation and diagnostic flags during capture. */
    if (phase == MODULE_C_ASSEMBLE) return ok;
#if !defined(__APPLE__)
    if (!retained) ok &= module_build_append(prefix, capacity, " -D_POSIX_C_SOURCE=200809L");
#endif
    for (size_t i = 0; i < meta->pkg_config_count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        const char *flags = snapshot->cflags[i];
        if (flags) {
            ok &= retained ? module_retained_flags(flags, prefix, capacity)
                           : module_build_append(prefix, capacity, " %s", flags);
        } else ok = false;
    }
    if (!retained) for (size_t i = 0; i < meta->include_dirs_count; i++)
        ok &= module_append_include(prefix, capacity, meta->include_dirs[i]);
    for (size_t group = 0; group < 2; group++) {
        size_t count = meta->cflags_count;
        char **flags = group ? module_platform_cflags(meta, &count) : meta->cflags;
        for (size_t i = 0; i < count; i++) {
            ok &= retained ? module_retained_flags(flags[i], prefix, capacity)
                           : module_build_append(prefix, capacity, " %s", flags[i]);
        }
    }
    return ok;
}

typedef enum {
    MODULE_SNAPSHOT_NONE = 0,
    MODULE_SNAPSHOT_CLANG,
    MODULE_SNAPSHOT_GCC,
    MODULE_SNAPSHOT_GCC_ASSEMBLY,
    MODULE_SNAPSHOT_GCC_REPLAY
} ModuleSnapshotMode;

static ModuleSnapshotMode module_snapshot_mode(const ModuleBuildMetadata *meta, const ModulePkgFlags *captured) {
    if (!captured || captured->count != meta->pkg_config_count) return MODULE_SNAPSHOT_NONE;
    for (size_t i = 0; i < captured->count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        if (!module_retained_flags(captured->cflags[i], NULL, 0)) return MODULE_SNAPSHOT_NONE;
    }
    for (size_t group = 0; group < 2; group++) {
        size_t count = meta->cflags_count;
        char **flags = group ? module_platform_cflags(meta, &count) : meta->cflags;
        for (size_t i = 0; i < count; i++)
            if (!module_retained_flags(flags[i], NULL, 0)) return MODULE_SNAPSHOT_NONE;
    }
    for (size_t group = 0; group < 2; group++) {
        char **sources = group ? meta->shared_c_sources : meta->c_sources;
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count; i++) {
            size_t length = strlen(sources[i]);
            if (length < 2 || strcmp(sources[i] + length - 2, ".c")) return false;
        }
    }
    if (!meta->c_sources_count) return false;
    /* I identify the supported driver family, not its authenticity. GCC's
     * capture must expose implicit PCH selection instead of ignoring it. */
    char *driver = module_compiler_path(module_selected_compiler(meta));
    if (!driver) return false;
    char command[8192] = {0};
    bool ok = module_append_path_flag(command, sizeof(command), "", driver) &&
              module_build_append(command, sizeof(command), " --version 2>/dev/null");
    free(driver);
    if (!ok) return false;
    FILE *pipe = popen(command, "r");
    if (!pipe) return false;
    char version[8192], discard[2048];
    size_t length = fread(version, 1, sizeof(version) - 1, pipe);
    version[length] = 0;
    bool complete = feof(pipe) && !memchr(version, 0, length);
    while (fread(discard, 1, sizeof(discard), pipe) > 0) {}
    ok = !ferror(pipe) && feof(pipe);
    if (pclose(pipe) != 0 || !ok || !complete) return MODULE_SNAPSHOT_NONE;
    if (strstr(version, "clang version")) return MODULE_SNAPSHOT_CLANG;
    if (strstr(version, "Free Software Foundation")) return MODULE_SNAPSHOT_GCC;
    return MODULE_SNAPSHOT_NONE;
}

static uint64_t module_snapshot_sources(ModuleBuildMetadata *meta,
                                       const ModulePkgFlags *flags, const char *directory,
                                       ModuleSnapshotMode mode, ModuleSnapshotMode *actual_mode);
static uint64_t module_gcc_validation(ModuleBuildMetadata *meta, const ModulePkgFlags *flags);

/* Other modes keep their supplemental veto and original compilation command.
 * Their include traces detect search changes even when -P hides line markers;
 * they do not acquire the retained-input guarantee from ordinary C. */
static uint64_t module_preprocess_fingerprint(ModuleBuildMetadata *meta,
                                             const ModulePkgFlags *flags) {
    if (!meta || !meta->c_sources_count) return 0;
    if (!flags) {
        ModulePkgFlags captured;
        if (!module_pkg_flags_capture(meta, &captured)) return 0;
        uint64_t result = module_preprocess_fingerprint(meta, &captured);
        module_pkg_flags_free(&captured);
        return result;
    }
    ModuleSnapshotMode mode = module_snapshot_mode(meta, flags);
    if (mode == MODULE_SNAPSHOT_GCC) return module_gcc_validation(meta, flags);
    if (mode != MODULE_SNAPSHOT_NONE) return module_snapshot_sources(meta, flags, NULL, mode, NULL);
    char prefix[4096];
    if (!module_compile_prefix(meta, prefix, sizeof(prefix), MODULE_C_PREPROCESS, flags)) return 0;
    uint64_t fingerprint = 14695981039346656037ULL;
    for (size_t i = 0; i < flags->count; i++) {
        hash_context_field(&fingerprint, meta->pkg_config[i]);
        hash_context_field(&fingerprint, flags->cflags[i] ? flags->cflags[i] : "");
        hash_context_field(&fingerprint, flags->libs[i] ? flags->libs[i] : "");
    }
    for (size_t group = 0; group < 2; group++) {
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        char **sources = group ? meta->shared_c_sources : meta->c_sources;
        hash_context_field(&fingerprint, group ? "shared" : "ordinary");
        for (size_t i = 0; i < count; i++) {
            char source[2048], command[8192] = {0};
            int n = sources[i][0] == '/' ? snprintf(source, sizeof(source), "%s", sources[i])
                : snprintf(source, sizeof(source), "%s/%s", meta->module_dir, sources[i]);
            if (n < 0 || (size_t)n >= sizeof(source) ||
                !module_build_append(command, sizeof(command), "%s%s -E -H -MD -MF /dev/null -MT nano_module_dependencies -o -",
                    prefix, group ? " -fvisibility=hidden -D_POSIX_C_SOURCE=200809L" : "") ||
                !module_append_path_flag(command, sizeof(command), "", source) ||
                !module_build_append(command, sizeof(command), " 2>&1")) return 0;
            FILE *pipe = popen(command, "r");
            if (!pipe) return 0;
            unsigned char buffer[4096];
            size_t amount;
            uint64_t hash = 14695981039346656037ULL;
            bool nonempty = false;
            while ((amount = fread(buffer, 1, sizeof(buffer), pipe)) > 0) {
                nonempty = true;
                for (size_t j = 0; j < amount; j++) { hash ^= buffer[j]; hash *= 1099511628211ULL; }
            }
            bool ok = !ferror(pipe) && feof(pipe);
            if (pclose(pipe) != 0 || !ok || !nonempty) return 0;
            char digest[24];
            snprintf(digest, sizeof(digest), "%llu", (unsigned long long)hash);
            hash_context_field(&fingerprint, source);
            hash_context_field(&fingerprint, digest);
        }
    }
    return fingerprint;
}

static bool module_source_command(char *command, size_t capacity, const char *prefix,
                                  const char *directory, const char *source,
                                  const char *object, const char *dependency, bool hidden) {
    char source_path[2048];
    int n = source[0] == '/' ? snprintf(source_path, sizeof(source_path), "%s", source)
                            : snprintf(source_path, sizeof(source_path), "%s/%s", directory, source);
    command[0] = '\0';
    if (n < 0 || (size_t)n >= sizeof(source_path)) return false;
    /* I need system headers too: an unchanged SDK label does not establish
     * unchanged transitive header contents. */
    char trace[2060];
    int t = snprintf(trace, sizeof(trace), "%s.includes", dependency);
    if (t < 0 || (size_t)t >= sizeof(trace)) return false;
    return module_build_append(command, capacity, "%s -H -MD -MT nano_module_dependencies%s", prefix,
                               hidden ? " -fvisibility=hidden -D_POSIX_C_SOURCE=200809L" : "") &&
           module_append_path_flag(command, capacity, "-MF ", dependency) &&
           module_append_path_flag(command, capacity, "", source_path) &&
           module_append_path_flag(command, capacity, "-o ", object) &&
           module_append_path_flag(command, capacity, "2>", trace);
}

static int module_run_source_command(const char *command, const char *dependency) {
    int result = system(command);
    char trace[2060];
    int n = snprintf(trace, sizeof(trace), "%s.includes", dependency);
    if (n < 0 || (size_t)n >= sizeof(trace)) return result ? result : -1;
    FILE *fp = fopen(trace, "rb");
    if (!fp) return result ? result : -1;
    char *line = NULL;
    size_t capacity = 0;
    ssize_t length;
    bool guard_advice = false;
    while ((length = getline(&line, &capacity, fp)) >= 0) {
        const char *p = line;
        while (*p == '.') p++;
        bool include_line = p != line && *p == ' ';
        if (length > 0 && line[length - 1] == '\n') {
            line[length - 1] = 0;
            if (strcmp(line, module_guard_advice) == 0) {
                guard_advice = true;
                include_line = true;
            } else if (guard_advice) {
                struct stat st;
                include_line = stat(line, &st) == 0 && S_ISREG(st.st_mode);
            }
            line[length - 1] = '\n';
        }
        /* I suppress only include-list lines on success, never diagnostics. */
        if (result || !include_line || module_builder_verbose || getenv("NANO_VERBOSE_BUILD"))
            (void)fwrite(line, 1, (size_t)length, stderr);
    }
    if ((ferror(fp) || !feof(fp)) && !result) result = -1;
    free(line);
    if (fclose(fp) != 0 && !result) result = -1;
    return result;
}

/* I capture a deliberately bounded assembler spelling, not the assembler
 * language. Literal file directives must start a line; backslash expansion and
 * alternate macro syntax fall back to retained C plus object validation. I copy
 * whole binary files so the assembler still evaluates offset/count expressions.
 * Relative paths resolve in the compiler's working directory, as in GNU as with
 * no assembler include-search flags (those flags already reject this mode). */
typedef struct {
    const char *directory;
    unsigned files;
    size_t bytes;
    uint64_t hash;
} ModuleAssemblyCapture;

static bool module_capture_assembly_file(ModuleAssemblyCapture *capture, const char *source,
                                         const char *destination, bool text, unsigned depth) {
    if (depth > 16 || ++capture->files > 256) return false;
    int fd = open(source, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    const size_t limit = 16 * 1024 * 1024;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uint64_t)st.st_size > limit) { close(fd); return false; }
    /* I reserve only the observed size plus one growth-detection byte. A file
     * that grows during capture falls back instead of expanding this buffer. */
    size_t capacity = (size_t)st.st_size;
    unsigned char *data = malloc(capacity + 1);
    if (!data) { close(fd); return false; }
    size_t size = 0;
    bool ok = true;
    for (;;) {
        ssize_t amount = read(fd, data + size, capacity + 1 - size);
        if (amount < 0 && errno == EINTR) continue;
        if (amount < 0) { ok = false; break; }
        if (!amount) break;
        size += (size_t)amount;
        if (size > capacity) { ok = false; break; }
    }
    close(fd);
    if (size > 64 * 1024 * 1024 - capture->bytes) ok = false;
    if (!ok) { free(data); return false; }
    capture->bytes += size;
    data[size] = 0;
    if (text && (memchr(data, 0, size) || memchr(data, '\\', size) ||
                 strstr((char *)data, ".altmacro") || strstr((char *)data, ".mri"))) { free(data); return false; }
    char length[24];
    snprintf(length, sizeof(length), "%zu", size);
    hash_context_field(&capture->hash, length);
    for (size_t i = 0; i < size; i++) { capture->hash ^= data[i]; capture->hash *= 1099511628211ULL; }
    FILE *output = fopen(destination, "wb");
    if (!output) { free(data); return false; }
    if (!text) ok = fwrite(data, 1, size, output) == size;
    else for (char *line = (char *)data; ok && *line;) {
        char *end = strchr(line, '\n');
        if (end) *end = 0;
        char *directive = line;
        while (*directive == ' ' || *directive == '\t') directive++;
        char *include = strstr(line, ".include"), *binary = strstr(line, ".incbin");
        if (include || binary) {
            bool is_include = include != NULL;
            char *found = is_include ? include : binary;
            size_t keyword = is_include ? 8 : 7;
            if (found != directive || (include && binary) ||
                strstr(found + keyword, ".include") || strstr(found + keyword, ".incbin") ||
                (directive[keyword] != ' ' && directive[keyword] != '\t')) { ok = false; break; }
            char *path = directive + keyword;
            while (*path == ' ' || *path == '\t') path++;
            if (*path++ != '"') { ok = false; break; }
            char *quote = strchr(path, '"');
            if (!quote || quote == path) { ok = false; break; }
            *quote = 0;
            char retained[2048] = {0};
            ok = module_build_append(retained, sizeof(retained), "%s/__assembler_%u.%s",
                    capture->directory, capture->files, is_include ? "s" : "bin");
            hash_context_field(&capture->hash, path);
            if (ok) ok = module_capture_assembly_file(capture, path, retained, is_include, depth + 1);
            if (ok) ok = fprintf(output, "%.*s.%s \"%s\"%s%s", (int)(directive - line), line,
                is_include ? "include" : "incbin", retained, quote + 1, end ? "\n" : "") >= 0;
        } else ok = fprintf(output, "%s%s", line, end ? "\n" : "") >= 0;
        if (!end) break;
        line = end + 1;
    }
    if (fclose(output)) ok = false;
    free(data);
    return ok;
}

static uint64_t module_gcc_capture_assembly(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                           const char *directory, uint64_t fingerprint) {
    /* These characters would need assembler string escaping in retained paths. */
    if (!directory || strpbrk(directory, "\"\\\n\r")) return 0;
    char prefix[4096];
    if (!module_compile_prefix(meta, prefix, sizeof(prefix), MODULE_C_RETAINED_ASSEMBLY, flags)) return 0;
    ModuleAssemblyCapture capture = {directory, 0, 0, fingerprint};
    hash_context_field(&capture.hash, "gcc-literal-assembler-files-v1");
    for (size_t group = 0; group < 2; group++) {
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count; i++) {
            char input[2048] = {0}, raw[2048] = {0}, frozen[2048] = {0}, command[8192] = {0};
            bool ok = module_build_append(input, sizeof(input), "%s/__snapshot_%zu_%zu.i", directory, group, i) &&
                module_build_append(raw, sizeof(raw), "%s/__assembly_%zu_%zu.s", directory, group, i) &&
                module_build_append(frozen, sizeof(frozen), "%s/__snapshot_%zu_%zu.s", directory, group, i) &&
                module_build_append(command, sizeof(command), "%s%s -x cpp-output", prefix,
                                    group ? " -fvisibility=hidden" : "") &&
                module_append_path_flag(command, sizeof(command), "", input) &&
                module_append_path_flag(command, sizeof(command), "-o ", raw) &&
                module_build_append(command, sizeof(command), " 2>/dev/null");
            if (!ok || system(command) || !module_capture_assembly_file(&capture, raw, frozen, true, 0)) goto failed;
        }
    }
    return capture.hash;
failed:
    /* I remove only names reserved by this capture in my private directory. */
    for (size_t group = 0; group < 2; group++) {
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count; i++) {
            char path[2048];
            snprintf(path, sizeof(path), "%s/__assembly_%zu_%zu.s", directory, group, i);
            (void)unlink(path);
            snprintf(path, sizeof(path), "%s/__snapshot_%zu_%zu.s", directory, group, i);
            (void)unlink(path);
        }
    }
    for (unsigned i = 1; i <= capture.files; i++) {
        char path[2048];
        snprintf(path, sizeof(path), "%s/__assembler_%u.s", directory, i);
        (void)unlink(path);
        snprintf(path, sizeof(path), "%s/__assembler_%u.bin", directory, i);
        (void)unlink(path);
    }
    return 0;
}

/* GCC still chooses assembler arguments; my private -B entry changes only the
 * executable receiving them. Loader configuration starts inside that wrapper,
 * never in the compiler driver, preprocessor, linker or calling process. */
static bool module_read_command(char *command, size_t capacity, const char *prefix,
                                 const char *directory, size_t group, size_t index,
                                 const char *object, bool capture) {
    char input[2048] = {0}, record[2048] = {0}, tools[2048] = {0};
    command[0] = 0;
    return module_build_append(input, sizeof(input), "%s/__snapshot_%zu_%zu.s", directory, group, index) &&
        module_build_append(record, sizeof(record), "%s/__as_read_%zu_%zu", directory, group, index) &&
        module_build_append(tools, sizeof(tools), "%s/", directory) &&
        module_build_append(command, capacity, "NANO_AS_CAPTURE_PHASE=%s", capture ? "capture" : "replay") &&
        module_append_path_flag(command, capacity, "NANO_AS_CAPTURE_PREFIX=", record) &&
        module_append_path_flag(command, capacity, "NANO_AS_CAPTURE_INPUT=", input) &&
        module_build_append(command, capacity, " %s -x assembler", prefix) &&
        module_append_path_flag(command, capacity, "-B", tools) &&
        module_append_path_flag(command, capacity, "", input) &&
        module_append_path_flag(command, capacity, "-o ", object);
}

#ifdef __linux__
static bool module_dynamic_elf(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return false;
    Elf64_Ehdr header;
    struct stat st;
    bool ok = !fstat(fileno(file), &st) && S_ISREG(st.st_mode) && st.st_size >= 0 &&
        fread(&header, 1, sizeof(header), file) == sizeof(header) &&
        !memcmp(header.e_ident, ELFMAG, SELFMAG) && header.e_ident[EI_CLASS] == ELFCLASS64 &&
        header.e_ident[EI_DATA] == ELFDATA2LSB && header.e_phentsize == sizeof(Elf64_Phdr) &&
        header.e_phnum <= 128 && header.e_phoff <= (uint64_t)st.st_size &&
        (uint64_t)header.e_phnum * sizeof(Elf64_Phdr) <= (uint64_t)st.st_size - header.e_phoff &&
        !fseeko(file, (off_t)header.e_phoff, SEEK_SET);
    bool interpreter = false;
    for (unsigned i = 0; ok && i < header.e_phnum; i++) {
        Elf64_Phdr program;
        ok = fread(&program, 1, sizeof(program), file) == sizeof(program);
        if (ok && program.p_type == PT_INTERP) interpreter = true;
    }
    if (fclose(file)) ok = false;
    return ok && interpreter;
}

static bool module_tool_output(const char *command, char *output, size_t capacity) {
    FILE *pipe = popen(command, "r");
    if (!pipe) return false;
    size_t size = fread(output, 1, capacity - 1, pipe);
    bool ok = !ferror(pipe) && feof(pipe) && !memchr(output, 0, size);
    char discard[1024];
    while (fread(discard, 1, sizeof(discard), pipe)) {}
    if (pclose(pipe)) ok = false;
    output[size] = 0;
    return ok && size;
}

static uint64_t module_gcc_read_capture(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                       const char *directory, uint64_t fingerprint) {
    if (!directory || (getenv("LD_PRELOAD") && *getenv("LD_PRELOAD")) ||
        (getenv("LD_AUDIT") && *getenv("LD_AUDIT"))) return 0;
    char helper[2048], tool[2048], command[8192] = {0}, version[8192];
    const char *configured = getenv("NANO_AS_CAPTURE_HELPER");
    if (configured) {
        int n = snprintf(helper, sizeof(helper), "%s", configured);
        if (n < 0 || (size_t)n >= sizeof(helper)) return 0;
    } else {
        ssize_t size = readlink("/proc/self/exe", helper, sizeof(helper) - 1);
        if (size <= 0 || (size_t)size >= sizeof(helper) - 1) return 0;
        helper[size] = 0;
        char *slash = strrchr(helper, '/');
        if (!slash) return 0;
        *slash = 0;
        if (!module_build_append(helper, sizeof(helper), "/nano_as_capture.so")) return 0;
    }
    if (!module_build_append(command, sizeof(command), "%s -print-prog-name=as 2>/dev/null", module_selected_compiler(meta)) ||
        !module_tool_output(command, tool, sizeof(tool))) return 0;
    size_t size = strlen(tool);
    if (size && tool[size - 1] == '\n') tool[--size] = 0;
    char *assembler = module_compiler_path(tool);
    if (!assembler) return 0;
    command[0] = 0;
    bool ok = module_append_path_flag(command, sizeof(command), "", assembler) &&
        module_build_append(command, sizeof(command), " --version 2>/dev/null") &&
        module_tool_output(command, version, sizeof(version)) &&
        strstr(version, "GNU assembler") && strstr(version, " 2.40\n");
    uint64_t assembler_hash = ok && module_dynamic_elf(assembler) ? hash_file_fnv1a(assembler) : 0;
    char copied[2048] = {0}, wrapper[2048] = {0};
    ModuleAssemblyCapture bytes = {directory, 0, 0, fingerprint};
    ok = assembler_hash && module_build_append(copied, sizeof(copied), "%s/__as_helper.so", directory) &&
        module_build_append(wrapper, sizeof(wrapper), "%s/as", directory) &&
        module_capture_assembly_file(&bytes, helper, copied, false, 0);
    char *quoted_helper = ok ? module_quote_path(copied) : NULL;
    char *quoted_as = ok ? module_quote_path(assembler) : NULL;
    FILE *script = quoted_helper && quoted_as ? fopen(wrapper, "wx") : NULL;
    if (!script) ok = false;
    else {
        ok = fprintf(script, "#!/bin/sh\nexec 3< %s || exit 125\n"
            "if [ \"$NANO_AS_CAPTURE_PHASE\" = replay ]; then\n"
            "  : > \"$NANO_AS_CAPTURE_PREFIX.replayed0\" || exit 125\nfi\n"
            "LD_PRELOAD=/proc/self/fd/3 %s \"$@\"\nnano_as_status=$?\n"
            "[ \"$nano_as_status\" -eq 0 ] || exit \"$nano_as_status\"\n"
            "if [ \"$NANO_AS_CAPTURE_PHASE\" = replay ]; then\n"
            "  nano_as_marker=\n  IFS= read -r nano_as_marker < \"$NANO_AS_CAPTURE_PREFIX.replayed0\" || :\n"
            "  [ \"$nano_as_marker\" = NACDONE1 ] || exit 125\nfi\n", quoted_helper, quoted_as) > 0;
        if (fclose(script) || chmod(wrapper, 0700)) ok = false;
    }
    free(quoted_helper); free(quoted_as);
    hash_context_field(&bytes.hash, "gcc-read-replay-v1");
    hash_context_field(&bytes.hash, assembler);
    char digest[24];
    snprintf(digest, sizeof(digest), "%llu", (unsigned long long)assembler_hash);
    hash_context_field(&bytes.hash, digest);
    free(assembler);
    char retained[4096], assemble[4096];
    if (ok) ok = module_compile_prefix(meta, retained, sizeof(retained), MODULE_C_RETAINED_ASSEMBLY, flags) &&
        module_compile_prefix(meta, assemble, sizeof(assemble), MODULE_C_ASSEMBLE, flags);
    NacRead *reads = ok ? calloc(NAC_READS, sizeof(*reads)) : NULL;
    if (!reads) ok = false;
    for (size_t group = 0; ok && group < 2; group++) {
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; ok && i < count; i++) {
            char input[2048] = {0}, assembly[2048] = {0}, object[2048] = {0}, record[2048] = {0};
            command[0] = 0;
            ok = module_build_append(input, sizeof(input), "%s/__snapshot_%zu_%zu.i", directory, group, i) &&
                module_build_append(assembly, sizeof(assembly), "%s/__snapshot_%zu_%zu.s", directory, group, i) &&
                module_build_append(object, sizeof(object), "%s/__as_capture_%zu_%zu.o", directory, group, i) &&
                module_build_append(record, sizeof(record), "%s/__as_read_%zu_%zu", directory, group, i) &&
                module_build_append(command, sizeof(command), "%s%s -x cpp-output", retained, group ? " -fvisibility=hidden" : "") &&
                module_append_path_flag(command, sizeof(command), "", input) &&
                module_append_path_flag(command, sizeof(command), "-o ", assembly) &&
                module_build_append(command, sizeof(command), " 2>/dev/null") && !system(command) &&
                module_read_command(command, sizeof(command), assemble, directory, group, i, object, true) &&
                module_build_append(command, sizeof(command), " 2>/dev/null") && !system(command);
            unsigned captured = 0;
            uint64_t hash = 0;
            if (ok) ok = nac_load(record, assembly, reads, &captured, &hash);
            if (ok) {
                snprintf(digest, sizeof(digest), "%llu", (unsigned long long)hash);
                hash_context_field(&bytes.hash, digest);
            }
        }
    }
    free(reads);
    if (ok) return bytes.hash;
    DIR *dir = opendir(directory);
    struct dirent *entry;
    while (dir && (entry = readdir(dir))) {
        size_t length = strlen(entry->d_name);
        if (!strcmp(entry->d_name, "as") || !strncmp(entry->d_name, "__as_", 5) ||
            (!strncmp(entry->d_name, "__snapshot_", 11) && length > 2 && !strcmp(entry->d_name + length - 2, ".s")))
            (void)unlinkat(dirfd(dir), entry->d_name, 0);
    }
    if (dir) closedir(dir);
    return 0;
}
#endif

/* I hash the retained input: Clang assembly or GCC preprocessed C. A later
 * capture must reproduce those bytes, not just hashes of restored source. */
static uint64_t module_snapshot_sources(ModuleBuildMetadata *meta,
                                       const ModulePkgFlags *flags, const char *directory,
                                       ModuleSnapshotMode mode, ModuleSnapshotMode *actual_mode) {
    if (actual_mode) *actual_mode = mode;
    char prefix[4096];
    bool assembly = mode == MODULE_SNAPSHOT_CLANG;
    if (!module_compile_prefix(meta, prefix, sizeof(prefix),
        assembly ? MODULE_C_EMIT_ASSEMBLY : MODULE_C_PREPROCESS, flags)) return 0;
    if (mode == MODULE_SNAPSHOT_GCC &&
        !module_build_append(prefix, sizeof(prefix), " -fpch-preprocess")) return 0;
    uint64_t fingerprint = 14695981039346656037ULL;
    hash_context_field(&fingerprint, mode == MODULE_SNAPSHOT_GCC ? "gcc-retained-v1" : "clang-assembly-v1");
    for (size_t i = 0; i < flags->count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        hash_context_field(&fingerprint, meta->pkg_config[i]);
        hash_context_field(&fingerprint, flags->cflags[i]);
        hash_context_field(&fingerprint, flags->libs[i]);
    }
    for (size_t group = 0; group < 2; group++) {
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        char **sources = group ? meta->shared_c_sources : meta->c_sources;
        hash_context_field(&fingerprint, group ? "shared" : "ordinary");
        for (size_t i = 0; i < count; i++) {
            char command[8192] = {0}, snapshot[2048] = {0}, dependency[2048] = {0};
            bool ok = true;
            if (directory) {
                ok = module_build_append(snapshot, sizeof(snapshot), "%s/__snapshot_%zu_%zu.%s", directory, group, i,
                                         assembly ? "s" : "i");
                if (group) ok &= module_build_append(dependency, sizeof(dependency), "%s/__shared_%zu.d", directory, i);
                else if (count == 1) ok &= module_build_append(dependency, sizeof(dependency), "%s/%s.d", directory, meta->name);
                else ok &= module_build_append(dependency, sizeof(dependency), "%s/%s_%zu.d", directory, meta->name, i);
                ok &= module_source_command(command, sizeof(command), prefix, meta->module_dir,
                                            sources[i], "-", dependency, group != 0);
            } else {
                char source[2048] = {0};
                ok = sources[i][0] == '/'
                    ? module_build_append(source, sizeof(source), "%s", sources[i])
                    : module_build_append(source, sizeof(source), "%s/%s", meta->module_dir, sources[i]);
                ok &= module_build_append(command, sizeof(command), "%s%s -o -", prefix,
                    group ? " -fvisibility=hidden -D_POSIX_C_SOURCE=200809L" : "");
                ok &= module_append_path_flag(command, sizeof(command), "", source);
                ok &= module_build_append(command, sizeof(command), " 2>/dev/null");
            }
            if (!ok) return 0;
            FILE *output = directory ? fopen(snapshot, "wb") : NULL;
            if (directory && !output) return 0;
            FILE *pipe = popen(command, "r");
            if (!pipe) {
                if (output) fclose(output);
                return 0;
            }
            unsigned char buffer[4096];
            size_t amount;
            uint64_t hash = 14695981039346656037ULL;
            bool nonempty = false;
            static const char pch_marker[] = "#pragma GCC pch_preprocess";
            size_t pch_matched = 0;
            bool external_pch = false;
            while ((amount = fread(buffer, 1, sizeof(buffer), pipe)) > 0) {
                nonempty = true;
                if (output && fwrite(buffer, 1, amount, output) != amount) ok = false;
                for (size_t j = 0; j < amount; j++) {
                    hash ^= buffer[j]; hash *= 1099511628211ULL;
                    /* This literal has no internal '#', so restarting at '#'
                     * preserves every possible match across read boundaries.
                     * Even a literal mention conservatively withholds reuse. */
                    if (mode == MODULE_SNAPSHOT_GCC && !external_pch) {
                        pch_matched = buffer[j] == (unsigned char)pch_marker[pch_matched]
                            ? pch_matched + 1 : (buffer[j] == '#');
                        external_pch = pch_matched == sizeof(pch_marker) - 1;
                    }
                }
            }
            ok &= !ferror(pipe) && feof(pipe);
            int status = pclose(pipe);
            if (output && fclose(output) != 0) ok = false;
            if (!ok || status != 0 || !nonempty || external_pch) return 0;
            char digest[24];
            snprintf(digest, sizeof(digest), "%llu", (unsigned long long)hash);
            hash_context_field(&fingerprint, sources[i]);
            hash_context_field(&fingerprint, digest);
        }
    }
    if (mode == MODULE_SNAPSHOT_GCC) {
        uint64_t frozen = module_gcc_capture_assembly(meta, flags, directory, fingerprint);
        if (frozen) {
            if (actual_mode) *actual_mode = MODULE_SNAPSHOT_GCC_ASSEMBLY;
            return frozen;
        }
#ifdef __linux__
        frozen = module_gcc_read_capture(meta, flags, directory, fingerprint);
        if (frozen) {
            if (actual_mode) *actual_mode = MODULE_SNAPSHOT_GCC_REPLAY;
            return frozen;
        }
#endif
    }
    return fingerprint;
}

static bool module_snapshot_command(char *command, size_t capacity, const char *prefix,
                                    const char *directory, size_t group, size_t index,
                                    const char *object, ModuleSnapshotMode mode) {
    if (mode == MODULE_SNAPSHOT_GCC_REPLAY)
        return module_read_command(command, capacity, prefix, directory, group, index, object, false);
    char snapshot[2048] = {0};
    bool assembly = mode == MODULE_SNAPSHOT_CLANG || mode == MODULE_SNAPSHOT_GCC_ASSEMBLY;
    command[0] = 0;
    return module_build_append(snapshot, sizeof(snapshot), "%s/__snapshot_%zu_%zu.%s", directory, group, index,
                               assembly ? "s" : "i") &&
        module_build_append(command, capacity, "%s%s -x %s", prefix,
                            group && !assembly ? " -fvisibility=hidden" : "", assembly ? "assembler" : "cpp-output") &&
        module_append_path_flag(command, capacity, "", snapshot) &&
        module_append_path_flag(command, capacity, "-o ", object);
}

static void module_remove_staging(const char *stage);

/* I include the actual GCC object bytes, not just the C input that preceded
 * assembler file reads. Validation builds private objects using the same recipe. */
static uint64_t module_gcc_objects(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                   const char *directory, uint64_t fingerprint, bool compile, ModuleSnapshotMode mode) {
    if (!fingerprint) return 0;
    char prefix[4096];
    if (compile && !module_compile_prefix(meta, prefix, sizeof(prefix),
        mode != MODULE_SNAPSHOT_GCC ? MODULE_C_ASSEMBLE : MODULE_C_RETAINED, flags)) return 0;
    hash_context_field(&fingerprint, "gcc-object-output-v1");
    for (size_t group = 0; group < 2; group++) {
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        hash_context_field(&fingerprint, group ? "shared-objects" : "ordinary-objects");
        for (size_t i = 0; i < count; i++) {
            char object[2048] = {0}, command[8192];
            bool ok;
            if (group) ok = module_build_append(object, sizeof(object), "%s/__shared_%zu.o", directory, i);
            else if (count == 1) ok = module_build_append(object, sizeof(object), "%s/%s.o", directory, meta->name);
            else ok = module_build_append(object, sizeof(object), "%s/%s_%zu.o", directory, meta->name, i);
            if (!ok) return 0;
            if (compile && (!module_snapshot_command(command, sizeof(command), prefix, directory,
                                group, i, object, mode) ||
                            !module_build_append(command, sizeof(command), " 2>/dev/null") || system(command))) return 0;
            uint64_t hash = hash_file_fnv1a(object);
            if (!hash) return 0;
            char digest[24];
            snprintf(digest, sizeof(digest), "%llu", (unsigned long long)hash);
            hash_context_field(&fingerprint, digest);
        }
    }
    return fingerprint;
}

static uint64_t module_gcc_validation(ModuleBuildMetadata *meta, const ModulePkgFlags *flags) {
    const char *temporary = getenv("TMPDIR");
    if (!temporary || !*temporary) temporary = "/tmp";
    char directory[2048] = {0};
    if (!module_build_append(directory, sizeof(directory), "%s/nano-gcc-check-XXXXXX", temporary) ||
        !mkdtemp(directory)) return 0;
    ModuleSnapshotMode mode = MODULE_SNAPSHOT_GCC;
    uint64_t fingerprint = module_snapshot_sources(meta, flags, directory, mode, &mode);
    fingerprint = module_gcc_objects(meta, flags, directory, fingerprint, true, mode);
    module_remove_staging(directory);
    return fingerprint;
}

/* I use one link recipe for publication and Linux warm validation. */
static bool module_shared_link_command(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                      const char *object_file, const char *shared_lib,
                                      const char *build_dir, char *lib_cmd, size_t capacity) {
    bool command_ok = true;
    const char *cc = module_selected_compiler(meta);
    lib_cmd[0] = 0;
    #ifdef __APPLE__
    /* On macOS, allow unresolved symbols so modules can reference symbols
     * provided by the host process (compiler/interpreter) at dlopen() time.
     */
    command_ok &= module_build_append(lib_cmd, capacity,
                       "%s -dynamiclib -undefined dynamic_lookup -fPIC", cc);
    #else
    command_ok &= module_build_append(lib_cmd, capacity,
                       "%s -shared -fPIC -Wl,--allow-shlib-undefined", cc);
    #endif
    /* Link the shared library from the module object (supports multi-source modules) */
    command_ok &= module_append_path_flag(lib_cmd, capacity, "-o ", shared_lib);
    command_ok &= module_append_path_flag(lib_cmd, capacity, "", object_file);
    /* I append captured package fragments without tokenizing their contents. */
    char *shared_cflags[1024] = {0};
    size_t shared_cflags_count = 0;
    for (size_t i = 0; i < meta->pkg_config_count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        char *pkg_cflags = strdup(flags->cflags[i]);
        if (!pkg_cflags) {
            for (size_t j = 0; j < shared_cflags_count; j++) free(shared_cflags[j]);
            return false;
        }
        if (pkg_cflags) {
            append_flag_fragment(shared_cflags, &shared_cflags_count, 1024, pkg_cflags);
            free(pkg_cflags);
        }
    }
    for (size_t i = 0; i < shared_cflags_count; i++) {
        command_ok &= module_build_append(lib_cmd, capacity, " %s", shared_cflags[i]);
        free(shared_cflags[i]);
    }

    char *shared_ldflags[1024] = {0};
    size_t shared_ldflags_count = 0;
    for (size_t i = 0; i < meta->pkg_config_count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        char *pkg_libs = strdup(flags->libs[i]);
        if (!pkg_libs) {
            for (size_t j = 0; j < shared_ldflags_count; j++) free(shared_ldflags[j]);
            return false;
        }
        if (pkg_libs) {
            append_flag_fragment(shared_ldflags, &shared_ldflags_count, 1024, pkg_libs);
            free(pkg_libs);
        }
    }
    for (size_t i = 0; i < meta->system_libs_count; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "-l%s", meta->system_libs[i]);
        append_flag_move_to_end(shared_ldflags, &shared_ldflags_count, 1024, buf);
    }
    for (size_t i = 0; i < meta->ldflags_count; i++) {
        append_flag_fragment(shared_ldflags, &shared_ldflags_count, 1024, meta->ldflags[i]);
    }
    append_platform_ldflags(meta, shared_ldflags, &shared_ldflags_count, 1024, true);
    #ifdef __APPLE__
    for (size_t i = 0; i < meta->frameworks_count; i++) {
        append_flag_move_to_end(shared_ldflags, &shared_ldflags_count, 1024, "-framework");
        append_flag_move_to_end(shared_ldflags, &shared_ldflags_count, 1024, meta->frameworks[i]);
    }
    #endif

    for (size_t i = 0; i < shared_ldflags_count; i++) {
        command_ok &= module_build_append(lib_cmd, capacity, " %s", shared_ldflags[i]);
        free(shared_ldflags[i]);
    }
    /* Add custom cflags (all platforms) */
    for (size_t i = 0; i < meta->cflags_count; i++) {
        command_ok &= module_build_append(lib_cmd, capacity,
                           " %s", meta->cflags[i]);
    }
    /* Add platform-specific cflags */
#ifdef __APPLE__
    for (size_t i = 0; i < meta->cflags_macos_count; i++) {
        command_ok &= module_build_append(lib_cmd, capacity,
                           " %s", meta->cflags_macos[i]);
    }
#elif defined(__FreeBSD__)
    for (size_t i = 0; i < meta->cflags_freebsd_count; i++) {
        command_ok &= module_build_append(lib_cmd, capacity,
                           " %s", meta->cflags_freebsd[i]);
    }
#else
    for (size_t i = 0; i < meta->cflags_linux_count; i++) {
        command_ok &= module_build_append(lib_cmd, capacity,
                           " %s", meta->cflags_linux[i]);
    }
#endif
    for (size_t i = 0; i < meta->shared_c_sources_count; i++) {
        char object[2048] = {0};
        command_ok &= module_build_append(object, sizeof(object), "%s/__shared_%zu.o", build_dir, i);
        command_ok &= module_append_path_flag(lib_cmd, capacity, "", object);
    }
    return command_ok;
}

#ifdef __linux__
static bool module_equal_libraries(const char *left, const char *right) {
    int a = open(left, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    int b = open(right, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    struct stat sa, sb;
    bool equal = a >= 0 && b >= 0 && fstat(a, &sa) == 0 && fstat(b, &sb) == 0 &&
        S_ISREG(sa.st_mode) && S_ISREG(sb.st_mode) && sa.st_size > 0 && sa.st_size == sb.st_size;
    FILE *fa = a >= 0 ? fdopen(a, "rb") : NULL;
    FILE *fb = b >= 0 ? fdopen(b, "rb") : NULL;
    if (!fa || !fb) equal = false;
    while (equal) {
        unsigned char ba[8192], bb[8192];
        size_t na = fread(ba, 1, sizeof(ba), fa);
        size_t nb = fread(bb, 1, sizeof(bb), fb);
        equal = na == nb && memcmp(ba, bb, na) == 0 && !ferror(fa) && !ferror(fb);
        if (!na || !nb) break;
    }
    if (fa) { if (fclose(fa) != 0) equal = false; }
    else if (a >= 0) close(a);
    if (fb) { if (fclose(fb) != 0) equal = false; }
    else if (b >= 0) close(b);
    return equal;
}

/* I validate the link's output, not an inferred inventory of GNU ld inputs.
 * A failed link is not permission to retry and hide its failure. */
static int module_linux_link_matches(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                     const char *objects, const char *staging, const char *object_file) {
    char retained[2048] = {0}, candidate[2048] = {0}, command[4096] = {0};
    bool ok = module_build_append(retained, sizeof(retained), "%s/lib%s.so", objects, meta->name) &&
        module_build_append(candidate, sizeof(candidate), "%s/lib%s.so", staging, meta->name) &&
        module_shared_link_command(meta, flags, object_file, candidate, objects, command, sizeof(command));
    if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD"))
        fprintf(stderr, "I validate the cached shared link for %s\n", meta->name);
    struct stat st;
    if (!ok || system(command) != 0 || lstat(candidate, &st) != 0 ||
        !S_ISREG(st.st_mode) || st.st_size == 0) return -1;
    return module_equal_libraries(retained, candidate) ? 1 : 0;
}
#endif

static ModuleBuildInfo* module_build_staged(ModuleBuilder *builder __attribute__((unused)),
                                           ModuleBuildMetadata *meta, const char *staging,
                                           uint64_t *preprocessing_before,
                                           const ModulePkgFlags *flags,
                                           cJSON **link_observation __attribute__((unused))) {
    if (!meta) return NULL;

    if (meta->c_sources_count == 0) {
        // No C sources = nothing to build, but still need link/compile flags
        ModuleBuildInfo *info = calloc(1, sizeof(ModuleBuildInfo));
        if (!info) return NULL;

        // Collect link flags from pkg-config and system_libs
        size_t total_link_flags = 0;
        char **link_flags = calloc(1024, sizeof(char*));

        // Add pkg-config link flags
        for (size_t i = 0; i < meta->pkg_config_count; i++) {
#ifdef __APPLE__
            if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
            char *pkg_flags = strdup(flags->libs[i]);
            if (!pkg_flags) {
                for (size_t j = 0; j < total_link_flags; j++) free(link_flags[j]);
                free(link_flags);
                module_build_info_free(info);
                return NULL;
            }
            if (pkg_flags) {
                append_flag_fragment(link_flags, &total_link_flags, 1024, pkg_flags);
                free(pkg_flags);
            }
        }

        // Add custom ldflags
        for (size_t i = 0; i < meta->ldflags_count; i++) {
            link_flags[total_link_flags++] = strdup(meta->ldflags[i]);
        }
        append_platform_ldflags(meta, link_flags, &total_link_flags, 1024, false);

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
#ifdef __APPLE__
            if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
            char *pkg_cflags = strdup(flags->cflags[i]);
            if (!pkg_cflags) {
                for (size_t j = 0; j < total_compile_flags; j++) free(compile_flags[j]);
                free(compile_flags);
                module_build_info_free(info);
                return NULL;
            }
            if (pkg_cflags) {
                append_flag_fragment(compile_flags, &total_compile_flags, 1024, pkg_cflags);
                free(pkg_cflags);
            }
        }

        // Add custom include dirs
        for (size_t i = 0; i < meta->include_dirs_count; i++) {
            char *include_flag = malloc(256);
            snprintf(include_flag, 256, "-I%s", meta->include_dirs[i]);
            compile_flags[total_compile_flags++] = include_flag;
        }

        // Add custom cflags (all platforms)
        for (size_t i = 0; i < meta->cflags_count; i++) {
            compile_flags[total_compile_flags++] = strdup(meta->cflags[i]);
        }

        // Add platform-specific cflags
#ifdef __APPLE__
        for (size_t i = 0; i < meta->cflags_macos_count; i++) {
            compile_flags[total_compile_flags++] = strdup(meta->cflags_macos[i]);
        }
#elif defined(__FreeBSD__)
        for (size_t i = 0; i < meta->cflags_freebsd_count; i++) {
            compile_flags[total_compile_flags++] = strdup(meta->cflags_freebsd[i]);
        }
#else
        for (size_t i = 0; i < meta->cflags_linux_count; i++) {
            compile_flags[total_compile_flags++] = strdup(meta->cflags_linux[i]);
        }
#endif

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
    bool needs_rebuild = module_needs_rebuild_with_flags(meta->module_dir, meta, flags);

    char *build_dir = needs_rebuild && staging ? strdup(staging) : module_get_artifact_dir(meta->module_dir);
    if (!build_dir) return false;
    char object_file[1024] = {0};
    if (!module_build_append(object_file, sizeof(object_file), "%s/%s.o",
                             build_dir, meta->name ? meta->name : "unknown")) {
        free(build_dir);
        return NULL;
    }

#ifdef __linux__
    if (!needs_rebuild && staging) {
        int matches = module_linux_link_matches(meta, flags, build_dir, staging, object_file);
        if (matches < 0) {
            fprintf(stderr, "I could not validate the cached shared link for %s\n", meta->name);
            free(build_dir);
            return NULL;
        }
        if (!matches) {
            free(build_dir);
            build_dir = strdup(staging);
            object_file[0] = 0;
            if (!build_dir || !module_build_append(object_file, sizeof(object_file), "%s/%s.o",
                                                   build_dir, meta->name)) {
                free(build_dir);
                return NULL;
            }
            needs_rebuild = true;
        }
    }
#endif

    if (needs_rebuild) {
        ModuleSnapshotMode mode = preprocessing_before ? module_snapshot_mode(meta, flags) : MODULE_SNAPSHOT_NONE;
        bool snapshots = mode != MODULE_SNAPSHOT_NONE;
        if (preprocessing_before) *preprocessing_before = snapshots
            ? module_snapshot_sources(meta, flags, build_dir, mode, &mode) : module_preprocess_fingerprint(meta, flags);
        snapshots = snapshots && *preprocessing_before;
        if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
            printf("[Module] Building %s...\n", meta->name ? meta->name : "unknown");
        }

        // Get CC from environment, module.json, or use POSIX cc
        const char *cc = module_selected_compiler(meta);

        // Build a reusable compile prefix (flags only)
        bool command_ok;
        char compile_prefix[4096] = {0};
        command_ok = module_compile_prefix(meta, compile_prefix, sizeof(compile_prefix),
            snapshots ? (mode == MODULE_SNAPSHOT_GCC ? MODULE_C_RETAINED : MODULE_C_ASSEMBLE) : MODULE_C_COMPILE, flags);

        if (meta->c_sources_count == 1) {
            // Single source can compile directly to the module object.
            char dep_path[1024] = {0};
            command_ok &= module_build_append(dep_path, sizeof(dep_path), "%s/%s.d",
                     build_dir, meta->name ? meta->name : "unknown");
            char compile_cmd[8192];
            command_ok &= module_source_command(compile_cmd, sizeof(compile_cmd), compile_prefix,
                         meta->module_dir, meta->c_sources[0], object_file, dep_path, false);
            if (snapshots) command_ok &= module_snapshot_command(compile_cmd, sizeof(compile_cmd),
                compile_prefix, build_dir, 0, 0, object_file, mode);

            if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
                printf("[Module] %s\n", compile_cmd);
            }

            int result = command_ok ? module_run_source_command(compile_cmd, dep_path) : -1;
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
                char obj_path[1024] = {0};
                command_ok &= module_build_append(obj_path, sizeof(obj_path), "%s/%s_%zu.o", build_dir, meta->name, i);
                src_objects[i] = strdup(obj_path);

                char dep_path[1024] = {0};
                command_ok &= module_build_append(dep_path, sizeof(dep_path), "%s/%s_%zu.d",
                         build_dir, meta->name ? meta->name : "unknown", i);
                char compile_cmd[8192];
                command_ok &= module_source_command(compile_cmd, sizeof(compile_cmd), compile_prefix,
                         meta->module_dir, meta->c_sources[i], obj_path, dep_path, false);
                if (snapshots) command_ok &= module_snapshot_command(compile_cmd, sizeof(compile_cmd),
                    compile_prefix, build_dir, 0, i, obj_path, mode);

                if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
                    printf("[Module] %s\n", compile_cmd);
                }

                int result = command_ok ? module_run_source_command(compile_cmd, dep_path) : -1;
                if (result != 0) {
                    fprintf(stderr, "Error: Failed to compile module %s (%s)\n", meta->name, meta->c_sources[i]);
                    for (size_t j = 0; j < meta->c_sources_count; j++) free(src_objects[j]);
                    free(src_objects);
                    free(build_dir);
                    return NULL;
                }
            }

            char combine_cmd[8192] = {0};
            command_ok &= module_build_append(combine_cmd, sizeof(combine_cmd), "%s -r", cc);
            command_ok &= module_append_path_flag(combine_cmd, sizeof(combine_cmd), "-o ", object_file);
            for (size_t i = 0; i < meta->c_sources_count; i++) {
                command_ok &= src_objects[i] && module_append_path_flag(combine_cmd, sizeof(combine_cmd), "", src_objects[i]);
            }

            if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
                printf("[Module] %s\n", combine_cmd);
            }

            int combine_result = command_ok ? system(combine_cmd) : -1;
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
        char shared_lib[1024] = {0};
        #ifdef __APPLE__
        command_ok &= module_build_append(shared_lib, sizeof(shared_lib), "%s/lib%s.dylib",
                 build_dir, meta->name);
        #else
        command_ok &= module_build_append(shared_lib, sizeof(shared_lib), "%s/lib%s.so",
                 build_dir, meta->name);
        #endif

        char shared_dir[1024];
        snprintf(shared_dir, sizeof(shared_dir), "%s", build_dir);
        bool shared_dir_ok = dir_exists(shared_dir) || mkdir_p(shared_dir);
        if (!shared_dir_ok) {
            fprintf(stderr, "I could not create the shared library directory for %s\n", meta->name);
            free(build_dir);
            return NULL;
        }

        if (shared_dir_ok) {
            char lib_cmd[4096] = {0};
            command_ok &= module_shared_link_command(meta, flags, object_file, shared_lib,
                                                      build_dir, lib_cmd, sizeof(lib_cmd));

            /* Note: ldflags/system libs/frameworks are included above via shared_ldflags */

            /* Compile shared_c_sources with hidden visibility and link into the shared lib only.
             * This embeds private dependencies (e.g. cJSON) without exporting their symbols,
             * preventing duplicate-symbol errors when the host binary also has those symbols
             * (e.g. from modules/std/json). */
            if (meta->shared_c_sources_count > 0) {
                for (size_t sci = 0; sci < meta->shared_c_sources_count; sci++) {
                    char sc_obj[2048] = {0}, sc_dep[2048] = {0};
                    command_ok &= module_build_append(sc_obj, sizeof(sc_obj), "%s/__shared_%zu.o", shared_dir, sci);
                    command_ok &= module_build_append(sc_dep, sizeof(sc_dep), "%s/__shared_%zu.d", shared_dir, sci);
                    char sc_cmd[8192];
                    command_ok &= module_source_command(sc_cmd, sizeof(sc_cmd), compile_prefix,
                                      meta->module_dir, meta->shared_c_sources[sci], sc_obj, sc_dep, true);
                    if (snapshots) command_ok &= module_snapshot_command(sc_cmd, sizeof(sc_cmd),
                        compile_prefix, build_dir, 1, sci, sc_obj, mode);

                    if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
                        printf("[Module] (shared-only) %s\n", sc_cmd);
                    }

                    if (!command_ok || module_run_source_command(sc_cmd, sc_dep) != 0) {
                        fprintf(stderr,
                                "I could not compile shared_c_source %s for %s\n",
                                meta->shared_c_sources[sci], meta->name);
                        free(build_dir);
                        return NULL;
                    }
                }
            }

            if (snapshots && (mode == MODULE_SNAPSHOT_GCC || mode == MODULE_SNAPSHOT_GCC_ASSEMBLY || mode == MODULE_SNAPSHOT_GCC_REPLAY))
                *preprocessing_before = module_gcc_objects(meta, flags, build_dir, *preprocessing_before, false, mode);

            /* Build shared library */
            if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
                printf("[Module] Building shared library: %s\n", lib_cmd);
            }
            
            int lib_result = -1;
#ifdef __APPLE__
            /* I preserve an ordinary link when dependency capture is not
             * supported. Response files can hide flag inputs from this format;
             * I do not create reuse evidence for those command lines yet. */
            char recorded_command[8192] = {0}, link_record[2048] = {0};
            bool capture = command_ok && link_observation && !strchr(lib_cmd, '@') &&
                module_build_append(link_record, sizeof(link_record), "%s/.link-dependencies", build_dir) &&
                module_build_append(recorded_command, sizeof(recorded_command), "%s -Xlinker -dependency_info", lib_cmd) &&
                module_append_path_flag(recorded_command, sizeof(recorded_command), "-Xlinker ", link_record);
            bool final_link = false;
            if (capture) {
                lib_result = system(recorded_command);
                cJSON *before = lib_result == 0 ? module_link_inputs(link_record, build_dir, shared_lib) : NULL;
                if (before) {
                    /* I discover inputs in private staging, then link again
                     * with those observed inputs checked around the final link.
                     * Both links use the same command and compilation mode. */
                    final_link = true;
                    lib_result = system(recorded_command);
                    cJSON *after = lib_result == 0 ? module_link_inputs(link_record, build_dir, shared_lib) : NULL;
                    if (after && cJSON_Compare(before, after, true)) {
                        *link_observation = before;
                        before = NULL;
                    }
                    cJSON_Delete(after);
                }
                cJSON_Delete(before);
            }
            /* A failed final link is a build failure, not an invitation to
             * publish the successful discovery link or retry past the failure. */
            if (!capture || (!final_link && lib_result != 0)) {
                if (link_record[0]) (void)unlink(link_record);
                lib_result = command_ok ? system(lib_cmd) : -1;
            }
#else
            lib_result = command_ok ? system(lib_cmd) : -1;
#endif
            struct stat library_stat;
            if (lib_result != 0 || stat(shared_lib, &library_stat) != 0 ||
                !S_ISREG(library_stat.st_mode) || library_stat.st_size == 0) {
                fprintf(stderr, "I could not build the shared library for %s\n",
                        meta->name);
                free(build_dir);
                return NULL;
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
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        char *pkg_flags = strdup(flags->libs[i]);
        if (!pkg_flags) {
            for (size_t j = 0; j < total_link_flags; j++) free(link_flags[j]);
            free(link_flags);
            module_build_info_free(info);
            return NULL;
        }
        if (pkg_flags) {
            append_flag_fragment(link_flags, &total_link_flags, 1024, pkg_flags);
            free(pkg_flags);
        }
    }

    // Add custom ldflags
    for (size_t i = 0; i < meta->ldflags_count; i++) {
        link_flags[total_link_flags++] = strdup(meta->ldflags[i]);
    }
    append_platform_ldflags(meta, link_flags, &total_link_flags, 1024, false);

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
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        char *pkg_cflags = strdup(flags->cflags[i]);
        if (!pkg_cflags) {
            for (size_t j = 0; j < total_compile_flags; j++) free(compile_flags[j]);
            free(compile_flags);
            module_build_info_free(info);
            return NULL;
        }
        if (pkg_cflags) {
            append_flag_fragment(compile_flags, &total_compile_flags, 1024, pkg_cflags);
            free(pkg_cflags);
        }
    }

    // Add custom include dirs
    for (size_t i = 0; i < meta->include_dirs_count; i++) {
        char *include_flag = malloc(256);
        snprintf(include_flag, 256, "-I%s", meta->include_dirs[i]);
        compile_flags[total_compile_flags++] = include_flag;
    }

    // Add custom cflags (all platforms)
    for (size_t i = 0; i < meta->cflags_count; i++) {
        compile_flags[total_compile_flags++] = strdup(meta->cflags[i]);
    }
    // Add platform-specific cflags
#ifdef __APPLE__
    for (size_t i = 0; i < meta->cflags_macos_count; i++) {
        compile_flags[total_compile_flags++] = strdup(meta->cflags_macos[i]);
    }
#elif defined(__FreeBSD__)
    for (size_t i = 0; i < meta->cflags_freebsd_count; i++) {
        compile_flags[total_compile_flags++] = strdup(meta->cflags_freebsd[i]);
    }
#else
    for (size_t i = 0; i < meta->cflags_linux_count; i++) {
        compile_flags[total_compile_flags++] = strdup(meta->cflags_linux[i]);
    }
#endif

    info->compile_flags = compile_flags;
    info->compile_flags_count = total_compile_flags;

    return info;
}

/* I touch only expected, complete files inside this invocation's private
 * directory. Validation precedes publication of any file. */
static bool module_validate_file(const char *stage, const char *name) {
    char source[2048];
    int s = snprintf(source, sizeof(source), "%s/%s", stage, name);
    if (s < 0 || (size_t)s >= sizeof(source)) return false;
    struct stat st;
    return lstat(source, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}

static bool module_validate_artifacts(const char *stage, ModuleBuildMetadata *meta) {
    char name[512];
    snprintf(name, sizeof(name), "%s.o", meta->name);
    if (!module_validate_file(stage, name)) return false;
#ifdef __APPLE__
    snprintf(name, sizeof(name), "lib%s.dylib", meta->name);
#else
    snprintf(name, sizeof(name), "lib%s.so", meta->name);
#endif
    if (!module_validate_file(stage, name)) return false;
    for (size_t i = 0; i < meta->c_sources_count; i++) {
        if (meta->c_sources_count == 1) snprintf(name, sizeof(name), "%s.d", meta->name);
        else snprintf(name, sizeof(name), "%s_%zu.d", meta->name, i);
        if (!module_validate_file(stage, name)) return false;
        if (meta->c_sources_count > 1) {
            snprintf(name, sizeof(name), "%s_%zu.o", meta->name, i);
            if (!module_validate_file(stage, name)) return false;
        }
    }
    for (size_t i = 0; i < meta->shared_c_sources_count; i++) {
        snprintf(name, sizeof(name), "__shared_%zu.o", i);
        if (!module_validate_file(stage, name)) return false;
        snprintf(name, sizeof(name), "__shared_%zu.d", i);
        if (!module_validate_file(stage, name)) return false;
    }
    return true;
}

static void module_remove_staging(const char *stage) {
    /* I never follow a substituted staging symlink. All entry removal stays
     * relative to this descriptor even if the directory is renamed. */
    int fd = open(stage, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        if (errno != ENOENT) fprintf(stderr, "I retained private build files in %s\n", stage);
        return;
    }
    DIR *dir = fdopendir(fd);
    if (!dir) {
        close(fd);
        fprintf(stderr, "I retained private build files in %s\n", stage);
        return;
    }
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        (void)unlinkat(fd, entry->d_name, 0);
    }
    closedir(dir);
    if (rmdir(stage) != 0) fprintf(stderr, "I retained private build files in %s\n", stage);
}

static bool module_sync_fd(int fd) {
    int result;
    do { result = fsync(fd); } while (result < 0 && errno == EINTR);
#ifdef __APPLE__
    /* I also ask the device to flush its cache. I do not silently substitute
     * the weaker host-buffer barrier if this request is unsupported. */
    if (result == 0) {
        do { result = fcntl(fd, F_FULLFSYNC); } while (result < 0 && errno == EINTR);
    }
#endif
    return result == 0;
}

static bool module_sync_directory(const char *path, bool ancestors) {
    int fd = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    bool ok = false;
    /* I walk actual directory descriptors, including on retry when mkdir
     * now reports an existing directory. I do not infer persistence from
     * existence. Mount setup itself belongs to the host. */
    for (size_t depth = 0; depth < 1024; depth++) {
        if (!module_sync_fd(fd)) break;
        if (!ancestors) { ok = true; break; }
        int parent = openat(fd, "..", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (parent < 0) break;
        struct stat current_stat, parent_stat;
        if (fstat(fd, &current_stat) != 0 || fstat(parent, &parent_stat) != 0) {
            close(parent);
            break;
        }
        bool end = current_stat.st_dev != parent_stat.st_dev ||
            current_stat.st_ino == parent_stat.st_ino;
        if (end) { ok = close(parent) == 0; break; }
        if (close(fd) != 0) { close(parent); fd = -1; break; }
        fd = parent;
    }
    if (fd >= 0 && close(fd) != 0) ok = false;
    return ok;
}

/* I flush only regular files in my private generation, never symlink targets
 * or arbitrary nested trees. A failed barrier cannot authorize publication. */
static bool module_sync_generation(const char *path) {
    int fd = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    DIR *directory = fdopendir(fd);
    if (!directory) { close(fd); return false; }
    bool ok = true;
    while (ok) {
        errno = 0;
        struct dirent *entry = readdir(directory);
        if (!entry) { if (errno) ok = false; break; }
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        int file = openat(fd, entry->d_name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
        if (file < 0) { ok = false; break; }
        struct stat st;
        ok = fstat(file, &st) == 0 && S_ISREG(st.st_mode) && module_sync_fd(file);
        if (close(file) != 0) ok = false;
    }
    if (ok) ok = module_sync_fd(fd);
    if (closedir(directory) != 0) ok = false;
    return ok;
}

static ModuleBuildInfo* module_build_with_flags(ModuleBuilder *builder, ModuleBuildMetadata *meta,
                                               const ModulePkgFlags *flags) {
    if (meta->c_sources_count == 0) return module_build_staged(builder, meta, NULL, NULL, flags, NULL);
    /* Names become artifact basenames, never paths or shell fragments. */
    if (!meta->name || !meta->name[0] || strlen(meta->name) > 255 ||
        strspn(meta->name, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != strlen(meta->name)) {
        fprintf(stderr, "I require a simple artifact name for a C module\n");
        return NULL;
    }
    if (!module_ensure_build_dir(meta->module_dir)) return NULL;
    char *cache_root = module_get_build_dir(meta->module_dir);
    char *cache = cache_root ? realpath(cache_root, NULL) : NULL;
    free(cache_root);
    if (!cache) return NULL;
    char lock_path[2048], stage[2048];
    int l = snprintf(lock_path, sizeof(lock_path), "%s/.build.lock", cache);
    int s = snprintf(stage, sizeof(stage), "%s/.nano-build-XXXXXX", cache);
    if (l < 0 || (size_t)l >= sizeof(lock_path) || s < 0 || (size_t)s >= sizeof(stage)) {
        free(cache);
        return NULL;
    }
    int fd = open(lock_path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) { free(cache); return NULL; }
    int locked = flock(fd, LOCK_EX | LOCK_NB);
    if (locked < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
        if (getenv("NANO_VERBOSE_BUILD")) fprintf(stderr, "I wait for the C-library cache lock: %s\n", cache);
        do { locked = flock(fd, LOCK_EX); } while (locked < 0 && errno == EINTR);
    }
    ModuleBuildInfo *info = NULL;
    if (locked == 0 && mkdtemp(stage)) {
        uint64_t context_before = module_build_context(meta);
        uint64_t preprocessing_before = 0;
        cJSON *link_observation = NULL;
        info = module_build_staged(builder, meta, stage, context_before ? &preprocessing_before : NULL,
                                   flags, context_before ? &link_observation : NULL);
        if (info && info->needs_rebuild) {
            char generation[2048], pointer[2048], temporary[2048], target[2048];
            int g = snprintf(generation, sizeof(generation), "%s/.nano-gen-%s", cache, stage + strlen(stage) - 6);
            int p = snprintf(pointer, sizeof(pointer), "%s/current", cache);
            int t = snprintf(temporary, sizeof(temporary), "%s.current", stage);
            bool paths_ok = g >= 0 && (size_t)g < sizeof(generation) &&
                p >= 0 && (size_t)p < sizeof(pointer) && t >= 0 && (size_t)t < sizeof(temporary);
            int n = snprintf(target, sizeof(target), "%s/%s.o", generation, meta->name);
            char *object = n >= 0 && (size_t)n < sizeof(target) ? strdup(target) : NULL;
            char *link_object = object ? strdup(object) : NULL;
            size_t object_index = 0;
            while (object_index < info->link_flags_count &&
                   (!info->object_file || !info->link_flags[object_index] ||
                    strcmp(info->link_flags[object_index], info->object_file) != 0)) object_index++;
            bool ok = paths_ok && object && link_object &&
                object_index < info->link_flags_count &&
                module_validate_artifacts(stage, meta);
            if (ok && context_before && preprocessing_before &&
                preprocessing_before == module_preprocess_fingerprint(meta, NULL) &&
                context_before == module_build_context(meta))
                module_update_hash_cache(meta->module_dir, meta, stage, preprocessing_before, link_observation);
            /* I never mutate a published generation. The old pointer remains
             * valid until the complete replacement is visible in one rename. */
            bool renamed = false, linked = false, published = false;
            struct stat st;
            if (ok) ok = module_sync_generation(stage);
            if (ok) ok = lstat(generation, &st) != 0 && errno == ENOENT;
            if (ok) ok = renamed = rename(stage, generation) == 0;
            if (ok) ok = module_sync_directory(cache, true);
            if (ok) ok = linked = symlink(strrchr(generation, '/') + 1, temporary) == 0;
            if (ok) ok = published = rename(temporary, pointer) == 0;
            if (ok) ok = module_sync_directory(cache, false);
            if (linked) (void)unlink(temporary);
            if (ok) {
                free(info->object_file);
                info->object_file = object;
                free(info->link_flags[object_index]);
                info->link_flags[object_index] = link_object;
            } else {
                /* After the pointer switch I must retain the referenced
                 * generation, even if its directory barrier reports failure. */
                if (renamed && !published) module_remove_staging(generation);
                if (published)
                    fprintf(stderr, "I published %s but could not confirm its cache-directory barrier; I retained the generation\n", meta->name);
                else
                    fprintf(stderr, "I could not publish complete C-library artifacts for %s\n", meta->name);
                free(object);
                free(link_object);
                module_build_info_free(info);
                info = NULL;
            }
        } else if (info && !module_sync_directory(cache, true)) {
            fprintf(stderr, "I could not confirm the cache-directory barrier for %s\n", meta->name);
            module_build_info_free(info);
            info = NULL;
        }
        cJSON_Delete(link_observation);
        module_remove_staging(stage);
    }
    close(fd); /* I retain the lock inode; removing it would split waiters. */
    free(cache);
    return info;
}

ModuleBuildInfo* module_build(ModuleBuilder *builder, ModuleBuildMetadata *meta) {
    if (!meta || !ensure_module_system_deps(meta)) return NULL;
    ModulePkgFlags flags;
    if (!module_pkg_flags_capture(meta, &flags)) return NULL;
    ModuleBuildInfo *info = module_build_with_flags(builder, meta, &flags);
    module_pkg_flags_free(&flags);
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
