// Module Build System Implementation
// Handles automatic compilation of C sources, caching, and dependency tracking

#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include "module_builder.h"
#include "module_link_response.h"
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
#include <sys/wait.h>
#include <spawn.h>
#include <poll.h>
#include <signal.h>

// JSON parsing (simple, minimal implementation for module.json)
#include "cJSON.h"

bool module_builder_verbose = false;
static bool module_builder_can_prompt_sudo = false;
static char *module_capture_response_fragment(const char *fragment);
static bool module_response_metadata(const ModuleBuildMetadata *meta, ModuleBuildMetadata *copy);
static void module_response_metadata_free(const ModuleBuildMetadata *meta, ModuleBuildMetadata *copy);
static bool module_response_driver(const ModuleBuildMetadata *meta);
static bool module_response_pending(const char *fragment);
static bool module_response_metadata_pending(const ModuleBuildMetadata *meta);
static bool module_flags_need_capture(char **flags, size_t count);
static bool module_coalesce_cflags(char **flags, size_t count);
static bool module_flag_operand_state(const char *fragment, bool *operand);
static char **module_response_group(const ModuleBuildMetadata *meta, size_t group, size_t *count);

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

static char **module_platform_ldflags(const ModuleBuildMetadata *meta, size_t *count) {
#ifdef __APPLE__
    *count = meta->ldflags_macos_count;
    return meta->ldflags_macos;
#elif defined(__FreeBSD__)
    *count = meta->ldflags_freebsd_count;
    return meta->ldflags_freebsd;
#else
    *count = meta->ldflags_linux_count;
    return meta->ldflags_linux;
#endif
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
    if (!cc[0] || strspn(cc, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./+-=") != strlen(cc)) return NULL;
    const char *equals = strchr(cc, '='), *slash = strchr(cc, '/');
    /* A separator before '=' makes this a path, not a shell assignment. */
    if (equals && (!slash || equals < slash)) return NULL;
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
    hash_context_field(&hash, "nanolang-c-build-context-v46-aliasing-snapshot-flags");
    hash_context_field(&hash, "retained-callback-adapters-v1");
    for (size_t i = 0; i < meta->callback_adapters_count; i++) {
        const ModuleCallbackAdapter *adapter = &meta->callback_adapters[i];
        hash_context_field(&hash, adapter->function_name);
        hash_context_field(&hash, adapter->adapter_symbol);
        hash_context_field(&hash, adapter->worker_thread ? "worker" : "owner");
    }
    const char *groups[] = {"compiler", "platform-compiler", "linker", "platform-linker"};
    for (size_t group = 0; group < 4; group++) {
        size_t count;
        char **flags = module_response_group(meta, group, &count);
        hash_context_field(&hash, groups[group]);
        for (size_t i = 0; i < count; i++) hash_context_field(&hash, flags[i]);
    }
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
        /* GCC -H reports a selected PCH with '! ', then its root source
         * with one leading space. I retain both as dependency evidence. */
        if (raw[0] == '!' && raw[1] == ' ') raw += 2;
        else if (raw[0] == ' ') raw++;
        else {
            while (*raw == '.') raw++;
            if (raw == line || *raw++ != ' ') { ok = false; break; }
        }
        if (!*raw) { ok = false; break; }
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
static bool save_hash_cache(const char *build_dir, cJSON *root) {
    char *path = malloc(strlen(build_dir) + 32);
    if (!path) return false;
    sprintf(path, "%s/source_hashes.json", build_dir);
    char *text = cJSON_PrintUnformatted(root);
    if (!text) { free(path); return false; }
    bool saved = false;
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
            if (ok) saved = rename(temporary, path) == 0;
            (void)unlink(temporary);
        }
        free(temporary);
    }
    free(text);
    free(path);
    return saved;
}

typedef struct {
    char **cflags;
    char **libs;
    size_t count;
    ModuleLinkResponseGrammar linker_grammar;
} ModulePkgFlags;

static bool module_capture_invocation(const ModuleBuildMetadata *meta, ModuleBuildMetadata *captured,
                                      ModulePkgFlags *flags);

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

/* I expose evidence decisions, not environment values or compiler commands.
 * Zero means unavailable or not observed; tracing does not add observations. */
static void module_trace_evidence(const char *phase, uint64_t expected,
                                  uint64_t observed, bool accepted) {
    if (getenv("NANO_TRACE_BUILD"))
        fprintf(stderr, "I checked build evidence: phase=%s expected=%llu observed=%llu accepted=%d\n",
                phase, (unsigned long long)expected, (unsigned long long)observed, accepted);
}

static uint64_t module_source_hash(const char *directory, const char *source) {
    char path[4096];
    int n = source[0] == '/' ? snprintf(path, sizeof(path), "%s", source)
                            : snprintf(path, sizeof(path), "%s/%s", directory, source);
    return n >= 0 && (size_t)n < sizeof(path) ? hash_file_fnv1a(path) : 0;
}

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
            uint64_t h = module_source_hash(module_dir, sources[i]);
            if (!h) { cJSON_Delete(root); return; }
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
    module_trace_evidence("record-dependencies", 1, complete, complete);
#ifdef __APPLE__
    /* I retain the hashes checked around the final link. Replacing them with
     * current hashes could label old code with bytes that were never linked. */
    cJSON *link_inputs = module_link_inputs_match(link_observation)
        ? cJSON_Duplicate(link_observation, true) : NULL;
    bool link_complete = link_inputs && cJSON_AddItemToObject(root, "__link_inputs_v1", link_inputs);
    if (!link_complete) {
        cJSON_Delete(link_inputs);
        complete = false;
    }
    module_trace_evidence("record-link-inputs", 1, link_complete, link_complete);
#endif
    if (complete) {
        bool saved = save_hash_cache(build_dir, root);
        module_trace_evidence("record-write", 1, saved, saved);
    }
    else if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD") || getenv("NANO_TRACE_BUILD"))
        fprintf(stderr, "I cannot establish complete dependency evidence; I will rebuild this module next time\n");
    cJSON_Delete(root);
}

/* Returns true if all source hashes match the cache → skip rebuild */
static bool hashes_match(const char *module_dir, ModuleBuildMetadata *meta,
                         const ModulePkgFlags *flags) {
    cJSON *cache = load_hash_cache(module_dir);
    if (!cache) {
        module_trace_evidence("reuse-record", 1, 0, false);
        return false;
    }
    uint64_t context = module_build_context(meta);
    char context_string[24];
    snprintf(context_string, sizeof(context_string), "%llu", (unsigned long long)context);
    cJSON *stored_context = cJSON_GetObjectItemCaseSensitive(cache, "__build_context_v1");
    bool match = context && cJSON_IsString(stored_context) &&
        strcmp(stored_context->valuestring, context_string) == 0;
    module_trace_evidence("reuse-initial-context", 0, context, match);
    for (int shared = 0; shared < 2 && match; shared++) {
        char **sources = shared ? meta->shared_c_sources : meta->c_sources;
        size_t count = shared ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count && match; i++) {
            uint64_t h = module_source_hash(module_dir, sources[i]);
            char hstr[24];
            snprintf(hstr, sizeof(hstr), "%llu", (unsigned long long)h);
            cJSON *item = cJSON_GetObjectItemCaseSensitive(cache, sources[i]);
            if (h == 0 || !item || !cJSON_IsString(item) || strcmp(item->valuestring, hstr) != 0) {
                match = false;
                module_trace_evidence("reuse-source", 0, h, false);
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
        if (h == 0 || !item || !cJSON_IsString(item) || strcmp(item->valuestring, hstr) != 0) {
            match = false;
            module_trace_evidence("reuse-metadata", 0, h, false);
        }
    }
    if (match) {
        match = system_headers_match(cache, meta);
        module_trace_evidence("reuse-declared-headers", 1, match, match);
    }
    if (match) {
        match = dep_hashes_match(cache);
        module_trace_evidence("reuse-dependencies", 1, match, match);
    }
#ifdef __APPLE__
    if (match) {
        match = module_link_inputs_match(cJSON_GetObjectItemCaseSensitive(cache, "__link_inputs_v1"));
        module_trace_evidence("reuse-link-inputs", 1, match, match);
    }
#endif
    if (match) {
        cJSON *stored = cJSON_GetObjectItemCaseSensitive(cache, "__preprocessing_v1");
        uint64_t observed = cJSON_IsString(stored) ? module_preprocess_fingerprint(meta, flags) : 0;
        char digest[24];
        snprintf(digest, sizeof(digest), "%llu", (unsigned long long)observed);
        match = observed && strcmp(stored->valuestring, digest) == 0;
        module_trace_evidence("reuse-preprocessing", 0, observed, match);
        uint64_t after = match ? module_build_context(meta) : 0;
        match = match && context == after;
        module_trace_evidence("reuse-context", context, after, match);
    }
    module_trace_evidence("reuse", 1, match, match);
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
    bool needed = module_flags_need_capture(flags->cflags, flags->count) ||
                  module_flags_need_capture(flags->libs, flags->count);
    for (size_t i = 0; i < flags->count; i++) {
        if (!flags->cflags[i]) continue;
        if (strstr(flags->cflags[i], "--driver-mode") || strstr(flags->libs[i], "--driver-mode")) return true;
        if (strchr(flags->cflags[i], '@')) needed = true;
    }
    if (!needed || module_response_metadata_pending(meta) || !module_response_driver(meta)) return true;
    char **expanded[2] = {calloc(flags->count, sizeof(char *)), calloc(flags->count, sizeof(char *))};
    bool complete = true, success = expanded[0] && expanded[1];
    for (size_t group = 0; group < 2 && success; group++) {
        char **source = group ? flags->libs : flags->cflags;
        for (size_t i = 0; i < flags->count && success; i++) {
            if (!source[i]) continue;
            expanded[group][i] = module_capture_response_fragment(source[i]);
            if (!expanded[group][i]) success = false;
            else if (module_response_pending(expanded[group][i])) complete = false;
        }
    }
    if (success && complete) success = module_coalesce_cflags(expanded[0], flags->count);
    if (success && complete) {
        char **original[2] = {flags->cflags, flags->libs};
        flags->cflags = expanded[0]; flags->libs = expanded[1];
        expanded[0] = original[0]; expanded[1] = original[1];
    }
    for (size_t group = 0; group < 2; group++) {
        for (size_t i = 0; expanded[group] && i < flags->count; i++) free(expanded[group][i]);
        free(expanded[group]);
    }
    if (!success) goto failed;
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

static bool module_callback_symbol_valid(const char *symbol) {
    const char *first = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";
    const char *rest = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789";
    return symbol && symbol[0] && strchr(first, symbol[0]) &&
           strspn(symbol, rest) == strlen(symbol);
}

static bool module_parse_callback_adapters(const cJSON *json, ModuleBuildMetadata *meta) {
    const cJSON *adapters = NULL;
    for (const cJSON *field = json->child; field; field = field->next) {
        if (!strcmp(field->string, "callback_adapters")) {
            if (adapters) return false;
            adapters = field;
        }
    }
    if (!adapters) return true;
    if (!cJSON_IsObject(adapters)) return false;
    size_t count = (size_t)cJSON_GetArraySize(adapters);
    meta->callback_adapters = count ? calloc(count, sizeof(*meta->callback_adapters)) : NULL;
    if (count && !meta->callback_adapters) return false;
    for (const cJSON *item = adapters->child; item; item = item->next) {
        if (!module_callback_symbol_valid(item->string) || !cJSON_IsObject(item)) return false;
        for (size_t i = 0; i < meta->callback_adapters_count; i++)
            if (!strcmp(meta->callback_adapters[i].function_name, item->string)) return false;
        const char *symbol = NULL, *abi = NULL, *execution = NULL;
        for (const cJSON *field = item->child; field; field = field->next) {
            if (!cJSON_IsString(field)) return false;
            const char **target = !strcmp(field->string, "symbol") ? &symbol :
                !strcmp(field->string, "abi") ? &abi :
                !strcmp(field->string, "execution") ? &execution : NULL;
            if (!target || *target) return false;
            *target = field->valuestring;
        }
        if (!module_callback_symbol_valid(symbol) || !abi || strcmp(abi, "retained_v1") ||
            !execution || (strcmp(execution, "worker") && strcmp(execution, "owner"))) return false;
        ModuleCallbackAdapter *adapter = &meta->callback_adapters[meta->callback_adapters_count++];
        adapter->function_name = strdup(item->string);
        adapter->adapter_symbol = strdup(symbol);
        adapter->worker_thread = !strcmp(execution, "worker");
        if (!adapter->function_name || !adapter->adapter_symbol) return false;
    }
    return true;
}

/* I reject decoded NULs before cJSON loses their length. Escaped backslashes
 * are skipped as pairs, so a literal "\\u0000" is not a decoded NUL. */
static bool module_manifest_has_nul(const char *text, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (!text[i]) return true;
        if (text[i] == '\\' && i + 1 < size) {
            if (size - i >= 6 && !memcmp(text + i + 1, "u0000", 5)) return true;
            i++;
        }
    }
    return false;
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

    size_t read_size = fread(content, 1, (size_t)size, fp);
    content[size] = '\0';
    fclose(fp);

    if (read_size != (size_t)size || module_manifest_has_nul(content, read_size)) {
        fprintf(stderr, "I require complete, NUL-free module metadata: %s\n", path);
        free(content);
        return NULL;
    }

    if (!nl_utf8_validate(content, (size_t)size, NULL)) {
        fprintf(stderr, "Error: %s is not valid UTF-8\n", path);
        free(content);
        return NULL;
    }

    // Parse JSON
    cJSON *json = cJSON_ParseWithOpts(content, NULL, true);
    free(content);

    if (!cJSON_IsObject(json)) {
        fprintf(stderr, "Error: Invalid JSON in %s\n", path);
        cJSON_Delete(json);
        return NULL;
    }

    ModuleBuildMetadata *meta = calloc(1, sizeof(ModuleBuildMetadata));
    if (!meta) {
        cJSON_Delete(json);
        return NULL;
    }

    if (!module_parse_callback_adapters(json, meta)) {
        fprintf(stderr, "I require unique callback adapters with a C symbol, retained_v1 ABI, and owner or worker execution: %s\n", path);
        module_metadata_free(meta);
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

    /* I resolve C include flags, not operands forwarded to another tool. */
    bool pending_operand = false;
    for (size_t i = 0; i < meta->cflags_count; i++) {
        bool operand = pending_operand;
        if (!module_flag_operand_state(meta->cflags[i], &pending_operand)) {
            pending_operand = false;
            continue;
        }
        if (operand) continue;
        if (strncmp(meta->cflags[i], "-I", 2) != 0) continue;
        const char *inc_path = meta->cflags[i] + 2;
        if (!inc_path[0]) continue;
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
    if (!module_dir || !module_dir[0]) {
        module_trace_evidence("build-metadata-path", 1, 0, false);
        return NULL;
    }
    char *canonical = realpath(module_dir, NULL);
    if (!canonical) {
        module_trace_evidence("build-metadata-path", 1, 0, false);
        return NULL;
    }
    /* I use the physical module directory for both cache identity and
     * module-relative include fallback, regardless of an import alias. */
    ModuleBuildMetadata *meta = module_load_metadata_at_directory(canonical);
    free(canonical);
    if (!meta) module_trace_evidence("build-metadata", 1, 0, false);
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

    for (size_t i = 0; i < meta->callback_adapters_count; i++) {
        free(meta->callback_adapters[i].function_name);
        free(meta->callback_adapters[i].adapter_symbol);
    }
    free(meta->callback_adapters);

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
    if (!meta || !meta->c_sources_count) return false;
    ModuleBuildMetadata captured;
    ModulePkgFlags flags;
    if (!module_capture_invocation(meta, &captured, &flags)) return true;
    bool result = module_needs_rebuild_with_flags(module_dir, &captured, &flags);
    module_pkg_flags_free(&flags);
    module_response_metadata_free(meta, &captured);
    return result;
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

typedef enum {
    MODULE_FLAG_UNKNOWN = 0, MODULE_FLAG_PREPROCESS = 1,
    MODULE_FLAG_C = 2, MODULE_FLAG_BOTH = 3, MODULE_FLAG_ASSEMBLER = 4,
    MODULE_FLAG_LINKER = 8, MODULE_FLAG_DEBUG = 16
} ModuleFlagPhase;

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

/* GNU response words are not shell words: dollar signs and operators are
 * literal, and a backslash quotes the next byte even inside single quotes. */
static int module_response_word(const char **cursor, char *word, size_t capacity) {
    const unsigned char *p = (const unsigned char *)*cursor;
    while (*p && strchr(" \t\r\n\v\f", *p)) p++;
    if (!*p) { *cursor = (const char *)p; return 0; }
    size_t used = 0;
    unsigned char quote = 0;
    while (*p) {
        unsigned char c = *p++;
        if (c == '\\') {
            if (!*p) { errno = EINVAL; return -1; }
            c = *p++;
        } else if (quote) {
            if (c == quote) { quote = 0; continue; }
        } else if (c == '\'' || c == '"') { quote = c; continue; }
        else if (strchr(" \t\r\n\v\f", c)) break;
        if (used + 1 >= capacity) { errno = E2BIG; return -1; }
        word[used++] = (char)c;
    }
    if (quote) { errno = EINVAL; return -1; }
    word[used] = 0;
    *cursor = (const char *)p;
    return 1;
}

static bool module_expand_response_word(const char *word, char *output, size_t capacity,
                                       unsigned depth, size_t *bytes) {
    if (word[0] != '@') {
        if (!module_append_path_flag(output, capacity, "", word)) { errno = E2BIG; return false; }
        return true;
    }
    if (depth >= 16) { errno = ELOOP; return false; }
    int fd = open(word + 1, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0) {
        close(fd); errno = EIO; return false;
    }
    if ((uint64_t)st.st_size > 65536 || (uint64_t)st.st_size > 65536 - *bytes) {
        close(fd); errno = E2BIG; return false;
    }
    size_t limit = (size_t)st.st_size, used = 0;
    char *data = malloc(limit + 1);
    if (!data) { close(fd); return false; }
    bool ok = true;
    while (used <= limit) {
        ssize_t amount = read(fd, data + used, limit + 1 - used);
        if (amount < 0 && errno == EINTR) continue;
        if (amount < 0) { ok = false; break; }
        if (!amount) break;
        used += (size_t)amount;
        if (used > limit) { ok = false; errno = E2BIG; break; }
    }
    if (close(fd)) ok = false;
    if (ok && memchr(data, 0, used)) { ok = false; errno = EINVAL; }
    if (ok) {
        data[used] = 0;
        if (strstr(data, "--driver-mode")) { ok = false; errno = EINVAL; }
    }
    if (ok) {
        *bytes += used;
        const char *cursor = data;
        char argument[4096];
        int status = 0;
        while (ok && (status = module_response_word(&cursor, argument, sizeof(argument))) > 0)
            ok = module_expand_response_word(argument, output, capacity, depth + 1, bytes);
        if (status < 0) ok = false;
    }
    int failure = errno;
    free(data);
    if (!ok) errno = failure;
    return ok;
}

static char *module_capture_response_fragment(const char *fragment) {
    if (!fragment) return NULL;
    if (!strchr(fragment, '@') || strstr(fragment, "--driver-mode")) return strdup(fragment);
    const char *cursor = fragment;
    char word[4096];
    int status;
    /* I validate the entire shell fragment before opening any response file. */
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {}
    if (status < 0) return strdup(fragment);
    char *output = calloc(65536, 1);
    if (!output) return NULL;
    cursor = fragment;
    size_t bytes = 0;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
        if (!module_expand_response_word(word, output, 65536, 0, &bytes)) {
            int failure = errno;
            free(output);
            /* Large command fragments and noncanonical response quoting keep
             * the original compiler path, not a guessed argument sequence. */
            if (failure == E2BIG || failure == EINVAL) return strdup(fragment);
            return NULL;
        }
    }
    return output;
}

static bool module_response_pending(const char *fragment) {
    if (!fragment) return false;
    if (strstr(fragment, "--driver-mode")) return true;
    const char *cursor = fragment;
    char word[4096];
    int status;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0)
        if (word[0] == '@' || strstr(word, "--driver-mode")) return true;
    return status < 0;
}

static bool module_flag_takes_operand(const char *word) {
    return !strcmp(word, "-D") || !strcmp(word, "-U") || !strcmp(word, "-I") ||
        !strcmp(word, "-Xassembler") || !strcmp(word, "-Xlinker");
}

static bool module_flag_operand_state(const char *fragment, bool *operand) {
    const char *cursor = fragment;
    char word[4096];
    int status;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0)
        *operand = !*operand && module_flag_takes_operand(word);
    return status == 0;
}

/* A metadata boundary is not an argv boundary. In particular, assembler -I
 * spans two -Xassembler pairs; a simple dangling-driver-operand check is not
 * enough. I normalize groups containing paired forms before phase selection. */
static bool module_flags_have_operands(char **flags, size_t count) {
    if (count < 2) return false;
    for (size_t i = 0; i < count; i++) {
        if (!flags[i]) continue;
        const char *cursor = flags[i];
        char word[4096];
        while (module_flag_word(&cursor, word, sizeof(word)) > 0)
            if (module_flag_takes_operand(word)) return true;
    }
    return false;
}

static bool module_flags_need_capture(char **flags, size_t count) {
    if (module_flags_have_operands(flags, count)) return true;
    size_t bytes = 0;
    for (size_t i = 0; i < count; i++) {
        if (!flags[i] || !flags[i][0]) continue;
        size_t length = strlen(flags[i]);
        if (strchr(flags[i], '@') || length >= 1024 - bytes) return true;
        bytes += length + 1;
    }
    return false;
}

/* I keep array slots (including native-framework NULLs) stable, but put a
 * literal argument sequence in its first nonempty slot for transport or
 * cross-fragment operand pairing.
 * Allocation failure leaves every original string untouched. */
static bool module_coalesce_cflags(char **flags, size_t count) {
    size_t bytes = 0, first = count;
    for (size_t i = 0; i < count; i++) {
        if (!flags[i] || !flags[i][0]) continue;
        if (module_response_pending(flags[i])) return true;
        size_t length = strlen(flags[i]);
        if (length >= 65536 - bytes) return true;
        if (first == count) first = i;
        bytes += length + 1;
    }
    if (first == count || count < 2 || (bytes <= 1024 && !module_flags_have_operands(flags, count))) return true;
    char *joined = malloc(bytes + 1);
    char **replacement = calloc(count, sizeof(char *));
    if (!joined || !replacement) { free(joined); free(replacement); return false; }
    size_t used = 0;
    for (size_t i = 0; i < count; i++) {
        if (!flags[i] || !flags[i][0]) continue;
        size_t length = strlen(flags[i]);
        if (used) joined[used++] = ' ';
        memcpy(joined + used, flags[i], length);
        used += length;
    }
    joined[used] = 0;
    replacement[first] = joined;
    bool ok = true;
    for (size_t i = 0; i < count && ok; i++) {
        if (flags[i] && i != first) {
            replacement[i] = strdup("");
            if (!replacement[i]) ok = false;
        }
    }
    for (size_t i = 0; i < count; i++) {
        if (ok) { free(flags[i]); flags[i] = replacement[i]; }
        else free(replacement[i]);
    }
    free(replacement);
    return ok;
}

/* I admit include-search and alternate-macro options from an assembler group.
 * Other assembler inputs and output options need their own phase contract. */
static bool module_wa_options(const char *word, bool search_only) {
    if (strncmp(word, "-Wa,", 4)) return false;
    const char *part = word + 4;
    if (!*part) return false;
    bool includes = false;
    while (*part) {
        const char *comma = strchr(part, ',');
        size_t length = comma ? (size_t)(comma - part) : strlen(part);
        bool alternate = length == 11 && !strncmp(part, "--alternate", 11);
        if (!alternate && (length < 2 || strncmp(part, "-I", 2))) return false;
        if (!alternate) includes = true;
        if (!alternate && length == 2) {
            if (!comma || !comma[1]) return false;
            part = comma + 1;
            comma = strchr(part, ',');
            if (comma == part) return false;
        }
        if (!comma) return !search_only || includes;
        part = comma + 1;
        if (!*part) return false;
    }
    return false;
}

/* I classify decoded literal tokens, not unevaluated shell fragments. */
static ModuleFlagPhase module_snapshot_flag(const char *flag) {
    if (!flag) return MODULE_FLAG_UNKNOWN;
    if (module_wa_options(flag, false)) return MODULE_FLAG_ASSEMBLER;
    if (!strcmp(flag, "-g") || !strcmp(flag, "-g0") || !strcmp(flag, "-g1") ||
        !strcmp(flag, "-g2") || !strcmp(flag, "-g3"))
        return MODULE_FLAG_BOTH | MODULE_FLAG_DEBUG;
    const char *both[] = {
        "-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Og",
        "-fPIC", "-fpic", "-fno-integrated-as", "-fstrict-aliasing", "-fno-strict-aliasing",
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

static bool module_assembler_argument(const char **cursor, char *argument, char *third,
                                          char *fourth, size_t capacity, bool *paired) {
    if (module_flag_word(cursor, argument, capacity) != 1) return false;
    if (!strcmp(argument, "--alternate")) { *paired = false; return true; }
    if (strncmp(argument, "-I", 2)) return false;
    *paired = !argument[2];
    return !*paired || (module_flag_word(cursor, third, capacity) == 1 &&
        !strcmp(third, "-Xassembler") && module_flag_word(cursor, fourth, capacity) == 1 && fourth[0]);
}

/* NULL output validates eligibility. Otherwise I select phases while keeping
 * every forwarded operand paired and every decoded argument literal. */
static bool module_phase_flags(const char *fragment, char *output, size_t capacity,
                               bool linker_captured, unsigned phases) {
    if (!fragment) return false;
    const char *cursor = fragment;
    char word[4096], argument[4096], combined[4096], third[4096], fourth[4096];
    int status;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
        if (linker_captured && !strcmp(word, "-Xlinker")) {
            if (module_flag_word(&cursor, argument, sizeof(argument)) != 1 || argument[0] == '@') return false;
            if (output && (phases & MODULE_FLAG_LINKER) &&
                (!module_append_path_flag(output, capacity, "", word) ||
                 !module_append_path_flag(output, capacity, "", argument))) return false;
            continue;
        }
        if (!strcmp(word, "-D") || !strcmp(word, "-U") || !strcmp(word, "-I")) {
            if (module_flag_word(&cursor, argument, sizeof(argument)) != 1) return false;
            int n = snprintf(combined, sizeof(combined), "%s%s", word, argument);
            if (n < 0 || (size_t)n >= sizeof(combined) ||
                module_snapshot_flag(combined) != MODULE_FLAG_PREPROCESS) return false;
            if (output && (phases & MODULE_FLAG_PREPROCESS) &&
                (!module_append_path_flag(output, capacity, "", word) ||
                 !module_append_path_flag(output, capacity, "", argument))) return false;
            continue;
        }
        if (!strcmp(word, "-Xassembler")) {
            bool paired;
            if (!module_assembler_argument(&cursor, argument, third, fourth, sizeof(argument), &paired)) return false;
            if (output && (phases & MODULE_FLAG_ASSEMBLER)) {
                if (!module_append_path_flag(output, capacity, "", word) ||
                    !module_append_path_flag(output, capacity, "", argument)) return false;
                if (paired && (!module_append_path_flag(output, capacity, "", third) ||
                    !module_append_path_flag(output, capacity, "", fourth))) return false;
            }
            continue;
        }
        ModuleFlagPhase kind = module_snapshot_flag(word);
        if (kind == MODULE_FLAG_UNKNOWN) return false;
        if ((kind & phases) && output &&
            !module_append_path_flag(output, capacity, "", word)) return false;
    }
    return status == 0;
}

static bool module_retained_flags(const char *fragment, char *output, size_t capacity, bool linker_captured) {
    return module_phase_flags(fragment, output, capacity, linker_captured, MODULE_FLAG_C);
}

/* I remove admitted assembler arguments from link-only jobs, including the
 * grammar query. I keep forwarded linker operands before grammar selection;
 * the query admission check still owns their safety. Compatibility fragments
 * outside this bounded grammar retain their original spelling. */
static char *module_link_cflags(const char *fragment) {
    const char *cursor = fragment;
    char word[4096];
    bool assembler = false;
    while (module_flag_word(&cursor, word, sizeof(word)) > 0) {
        if (module_wa_options(word, false) || !strcmp(word, "-Xassembler")) { assembler = true; break; }
        if (!strcmp(word, "-D") || !strcmp(word, "-U") || !strcmp(word, "-I") ||
            !strcmp(word, "-Xlinker")) {
            if (module_flag_word(&cursor, word, sizeof(word)) != 1) break;
        }
    }
    if (!assembler) return strdup(fragment);
    size_t size = strlen(fragment);
    if (size > (SIZE_MAX - 16) / 4) return NULL;
    size = size * 4 + 16;
    char *selected = calloc(size, 1);
    if (!selected) return NULL;
    if (module_phase_flags(fragment, selected, size, true, MODULE_FLAG_BOTH | MODULE_FLAG_LINKER)) return selected;
    free(selected);
    return strdup(fragment);
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

static char **module_response_group(const ModuleBuildMetadata *meta, size_t group, size_t *count) {
    if (group < 2) {
        *count = meta->cflags_count;
        return group ? module_platform_cflags(meta, count) : meta->cflags;
    }
    *count = meta->ldflags_count;
    return group == 3 ? module_platform_ldflags(meta, count) : meta->ldflags;
}

static char ***module_response_group_slot(ModuleBuildMetadata *meta, size_t group) {
    if (group == 0) return &meta->cflags;
    if (group == 2) return &meta->ldflags;
#ifdef __APPLE__
    return group == 1 ? &meta->cflags_macos : &meta->ldflags_macos;
#elif defined(__FreeBSD__)
    return group == 1 ? &meta->cflags_freebsd : &meta->ldflags_freebsd;
#else
    return group == 1 ? &meta->cflags_linux : &meta->ldflags_linux;
#endif
}

static bool module_response_metadata_pending(const ModuleBuildMetadata *meta) {
    for (size_t group = 0; group < 4; group++) {
        size_t count;
        char **flags = module_response_group(meta, group, &count);
        for (size_t i = 0; i < count; i++) if (module_response_pending(flags[i])) return true;
    }
    return false;
}

static void module_response_metadata_free(const ModuleBuildMetadata *meta, ModuleBuildMetadata *copy) {
    for (size_t group = 0; group < 4; group++) {
        size_t count, copied_count;
        char **original = module_response_group(meta, group, &count);
        char **owned = module_response_group(copy, group, &copied_count);
        if (owned && owned != original) {
            for (size_t i = 0; i < copied_count; i++) free(owned[i]);
            free(owned);
        }
    }
}

static bool module_response_metadata(const ModuleBuildMetadata *meta, ModuleBuildMetadata *copy) {
    *copy = *meta;
    bool needed = false;
    for (size_t group = 0; group < 4; group++) {
        size_t count;
        char **flags = module_response_group(meta, group, &count);
        if (module_flags_need_capture(flags, count)) needed = true;
    }
    if (!needed || !module_response_driver(meta)) return true;
    for (size_t group = 0; group < 4; group++) {
        size_t count;
        char **flags = module_response_group(meta, group, &count);
        char ***slot = module_response_group_slot(copy, group);
        *slot = count ? calloc(count, sizeof(char *)) : NULL;
        if (count && !*slot) goto failed;
        for (size_t i = 0; i < count; i++)
            if (!((*slot)[i] = module_capture_response_fragment(flags[i]))) goto failed;
    }
    if (module_response_metadata_pending(copy)) {
        module_response_metadata_free(meta, copy);
        *copy = *meta;
    } else {
        for (size_t group = 0; group < 2; group++) {
            size_t count = copy->cflags_count;
            char **flags = group ? module_platform_cflags(copy, &count) : copy->cflags;
            if (!module_coalesce_cflags(flags, count)) goto failed;
        }
    }
    return true;
failed:
    module_response_metadata_free(meta, copy);
    fprintf(stderr, "I could not capture compiler response-file arguments\n");
    return false;
}

/* I share publication and byte verification, not driver/linker token grammars.
 * A linker response includes its resolved source path in retained identity:
 * equal bytes at distinct paths must not become a repeated response input. */
static char *module_retain_response_bytes(const ModuleBuildMetadata *meta, const char *data,
                                         size_t used, const char *identity) {
    uint64_t hash = 14695981039346656037ULL;
    if (identity) {
        for (const unsigned char *p = (const unsigned char *)identity; *p; p++) {
            hash ^= *p; hash *= 1099511628211ULL;
        }
        hash ^= 0; hash *= 1099511628211ULL;
    }
    for (size_t i = 0; i < used; i++) { hash ^= (unsigned char)data[i]; hash *= 1099511628211ULL; }
    char *root = module_ensure_build_dir(meta->module_dir) ? module_get_build_dir(meta->module_dir) : NULL;
    char *directory = root ? realpath(root, NULL) : NULL;
    free(root);
    char path[2048] = {0}, temporary[2048] = {0};
    bool ok = directory && module_build_append(path, sizeof(path), "%s/.nano-%s-%016llx.rsp",
        directory, identity ? "link-response" : "args", (unsigned long long)hash) &&
        module_build_append(temporary, sizeof(temporary), "%s/.nano-args-XXXXXX", directory);
    free(directory);
    int fd = ok ? open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK) : -1;
    if (ok && fd < 0 && errno == ENOENT) {
        int out = mkstemp(temporary);
        ok = out >= 0;
        size_t offset = 0;
        while (ok && offset < used) {
            ssize_t n = write(out, data + offset, used - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) ok = false;
            else offset += (size_t)n;
        }
        if (out >= 0) {
            if (fchmod(out, 0400) || fsync(out)) ok = false;
            if (close(out)) ok = false;
            if (ok && link(temporary, path) && errno != EEXIST) ok = false;
            if (unlink(temporary)) ok = false;
        }
        if (ok) fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    }
    struct stat st;
    ok = ok && fd >= 0 && !fstat(fd, &st) && S_ISREG(st.st_mode) &&
        st.st_size >= 0 && (uint64_t)st.st_size == used;
    size_t offset = 0;
    while (ok && offset < used) {
        char buffer[4096];
        size_t want = used - offset < sizeof(buffer) ? used - offset : sizeof(buffer);
        ssize_t n = read(fd, buffer, want);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0 || memcmp(buffer, data + offset, (size_t)n)) ok = false;
        else offset += (size_t)n;
    }
    if (ok) {
        char extra;
        ssize_t n;
        do { n = read(fd, &extra, 1); } while (n < 0 && errno == EINTR);
        if (n != 0) ok = false;
    }
    if (fd >= 0 && close(fd)) ok = false;
    return ok ? strdup(path) : NULL;
}

/* I retain transport files in the module cache, not an invocation's staging
 * directory: returned native flags can outlive both metadata and build info.
 * Decoded argument strings, not these paths, remain my build identity. */
static char *module_response_transport(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                       const char *fragment) {
    if (!fragment) return NULL;
    if (strlen(fragment) <= 1024 || module_response_metadata_pending(meta) ||
        !module_response_driver(meta)) return strdup(fragment);
    for (size_t i = 0; flags && i < flags->count; i++)
        if (module_response_pending(flags->cflags[i]) || module_response_pending(flags->libs[i])) return strdup(fragment);
    if (module_response_pending(fragment)) return strdup(fragment);
    size_t length = strlen(fragment);
    if (length > 65536) return strdup(fragment);
    char *data = malloc(length * 4 + 16);
    if (!data) return NULL;
    size_t used = 0;
    const char *cursor = fragment;
    char word[4096];
    int status;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
        data[used++] = '"';
        for (const char *p = word; *p; p++) {
            if (*p == '\\' || *p == '"') data[used++] = '\\';
            data[used++] = *p;
        }
        data[used++] = '"';
        data[used++] = '\n';
    }
    if (status < 0) { free(data); return strdup(fragment); }
    char *path = module_retain_response_bytes(meta, data, used, NULL);
    free(data);
    char result[8192] = {0};
    bool ok = path && module_append_path_flag(result, sizeof(result), "@", path);
    free(path);
    return ok ? strdup(result) : NULL;
}

typedef struct {
    const ModuleBuildMetadata *meta;
    ModuleLinkResponseGrammar grammar;
    bool arguments;
    size_t count, bytes, alias_count;
    /* A result is either a retained path or an owned argument fragment. */
    struct { char *source, *result; } nodes[64];
    struct { char *spelling; size_t node; } aliases[128];
} ModuleLinkResponseGraph;

/* I locate nested-reference token spans without rewriting surrounding bytes.
 * These two explicit grammars differ in their unquoted whitespace set. Both
 * tested linkers accept an open quote at EOF and discard a trailing escape. */
static int module_link_response_word(const char **cursor, const char **begin, const char **end,
                                     char *word, size_t capacity, ModuleLinkResponseGrammar grammar) {
    const char *space = grammar == MODULE_LINK_RESPONSE_APPLE ? " \t\r\n" : " \t\r\n\v\f";
    const char *p = *cursor;
    while (*p && strchr(space, *p)) p++;
    *begin = p;
    if (!*p) { *cursor = *end = p; return 0; }
    size_t used = 0;
    char quote = 0;
    while (*p) {
        char c = *p;
        if (!quote && strchr(space, c)) break;
        p++;
        if (c == '\\') {
            if (!*p) break;
            c = *p++;
        } else if (quote) {
            if (c == quote) { quote = 0; continue; }
        } else if (c == '\'' || c == '"') { quote = c; continue; }
        if (used + 1 >= capacity) { errno = E2BIG; return -1; }
        word[used++] = c;
    }
    word[used] = 0;
    *cursor = *end = p;
    return 1;
}

static bool module_link_response_copy(char *output, size_t *used, const char *data, size_t size) {
    if (size > 65536 - *used) { errno = E2BIG; return false; }
    memcpy(output + *used, data, size);
    *used += size;
    output[*used] = 0;
    return true;
}

static const char *module_link_response_node(ModuleLinkResponseGraph *graph, const char *source,
                                              unsigned depth) {
    if (strnlen(source, 4096) == 4096) { errno = ENAMETOOLONG; return NULL; }
    for (size_t i = 0; i < graph->alias_count; i++) {
        if (strcmp(source, graph->aliases[i].spelling)) continue;
        const char *result = graph->nodes[graph->aliases[i].node].result;
        if (!result || (graph->arguments && graph->grammar == MODULE_LINK_RESPONSE_APPLE)) {
            errno = ELOOP; return NULL;
        }
        return result;
    }
    if (graph->alias_count == 128) { errno = E2BIG; return NULL; }
    char *resolved = realpath(source, NULL);
    if (!resolved) return NULL;
    size_t index = 0;
    while (index < graph->count && strcmp(resolved, graph->nodes[index].source)) index++;
    if (index == graph->count && (depth >= 16 || graph->count == 64)) {
        free(resolved); errno = E2BIG; return NULL;
    }
    char *spelling = strdup(source);
    if (!spelling) { free(resolved); return NULL; }
    graph->aliases[graph->alias_count].spelling = spelling;
    graph->aliases[graph->alias_count++].node = index;
    if (index < graph->count) {
        free(resolved);
        if (!graph->nodes[index].result || (graph->arguments && graph->grammar == MODULE_LINK_RESPONSE_APPLE)) {
            errno = ELOOP; return NULL;
        }
        return graph->nodes[index].result;
    }
    graph->count++;
    graph->nodes[index].source = resolved;
    int fd = open(resolved, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0) {
        close(fd); errno = EIO; return NULL;
    }
    if ((uint64_t)st.st_size > 65536 - graph->bytes) {
        close(fd); errno = E2BIG; return NULL;
    }
    size_t length = (size_t)st.st_size, used = 0;
    char *data = malloc(length + 1);
    if (!data) { close(fd); return NULL; }
    bool ok = true;
    while (used <= length) {
        ssize_t n = read(fd, data + used, length + 1 - used);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { ok = false; break; }
        if (!n) break;
        used += (size_t)n;
        if (used > length) { ok = false; errno = E2BIG; break; }
    }
    if (close(fd)) ok = false;
    if (ok && (used != length || memchr(data, 0, used))) { ok = false; errno = EINVAL; }
    if (!ok) { free(data); return NULL; }
    data[used] = 0;
    graph->bytes += used;
    char *rewritten = NULL;
    if (graph->arguments || strchr(data, '@')) {
        rewritten = calloc(65537, 1);
        if (!rewritten) { free(data); return NULL; }
        const char *cursor = data, *previous = data, *begin, *end;
        char word[4096];
        size_t output_size = 0;
        int status = 0;
        while (ok && (status = module_link_response_word(&cursor, &begin, &end, word,
                                                        sizeof(word), graph->grammar)) > 0) {
            if (graph->arguments) {
                if (word[0] == '@') {
                    const char *nested = module_link_response_node(graph, word + 1, depth + 1);
                    ok = nested && module_link_response_copy(rewritten, &output_size, nested, strlen(nested));
                } else {
                    char *quoted = module_quote_path(word);
                    ok = quoted && module_link_response_copy(rewritten, &output_size, " -Xlinker ", 10) &&
                        module_link_response_copy(rewritten, &output_size, quoted, strlen(quoted));
                    free(quoted);
                }
                continue;
            }
            if (word[0] != '@') continue;
            const char *nested = module_link_response_node(graph, word + 1, depth + 1);
            ok = nested && module_link_response_copy(rewritten, &output_size, previous, (size_t)(begin - previous)) &&
                module_link_response_copy(rewritten, &output_size, "\"@", 2);
            for (const char *p = nested; ok && *p; p++) {
                if (*p == '\\' || *p == '"') ok = module_link_response_copy(rewritten, &output_size, "\\", 1);
                if (ok) ok = module_link_response_copy(rewritten, &output_size, p, 1);
            }
            if (ok) ok = module_link_response_copy(rewritten, &output_size, "\"", 1);
            previous = end;
        }
        if (status < 0) ok = false;
        if (ok && !graph->arguments)
            ok = module_link_response_copy(rewritten, &output_size, previous, strlen(previous));
        used = output_size;
    }
    if (ok && graph->arguments) {
        graph->nodes[index].result = rewritten;
        rewritten = NULL;
    } else if (ok) graph->nodes[index].result = module_retain_response_bytes(graph->meta,
        rewritten ? rewritten : data, used, resolved);
    free(rewritten);
    free(data);
    return graph->nodes[index].result;
}

/* I expose this internal capture mechanism to the production probe first.
 * Invocation ownership and selected-linker admission are separate integration
 * gates: this helper alone must not authorize cache reuse. */
static char **module_capture_link_graph(const ModuleBuildMetadata *meta, const char *const *sources,
                                        size_t count, ModuleLinkResponseGrammar grammar, bool arguments) {
    if ((!arguments && (!meta || !meta->module_dir)) || !sources || !count || count > 64 ||
        (grammar != MODULE_LINK_RESPONSE_GNU && grammar != MODULE_LINK_RESPONSE_APPLE)) {
        errno = EINVAL; return NULL;
    }
    for (size_t i = 0; i < count; i++)
        if (!sources[i] || !sources[i][0]) { errno = EINVAL; return NULL; }
    char **result = calloc(count, sizeof(char *));
    if (!result) return NULL;
    ModuleLinkResponseGraph graph = {.meta = meta, .grammar = grammar, .arguments = arguments};
    bool ok = true;
    size_t expanded = 0;
    for (size_t i = 0; i < count && ok; i++) {
        const char *root = module_link_response_node(&graph, sources[i], 0);
        if (root && arguments) {
            size_t length = strlen(root);
            if (length > 65536 - expanded) { errno = E2BIG; ok = false; break; }
            expanded += length;
        }
        ok = root && (result[i] = strdup(root));
    }
    int failure = errno;
    for (size_t i = 0; i < graph.count; i++) {
        free(graph.nodes[i].source);
        free(graph.nodes[i].result);
    }
    for (size_t i = 0; i < graph.alias_count; i++) free(graph.aliases[i].spelling);
    if (!ok) {
        for (size_t i = 0; i < count; i++) free(result[i]);
        free(result);
        errno = failure;
        return NULL;
    }
    return result;
}

char **module_capture_link_responses(const ModuleBuildMetadata *meta, const char *const *sources,
                                     size_t count, ModuleLinkResponseGrammar grammar) {
    return module_capture_link_graph(meta, sources, count, grammar, false);
}

char **module_capture_link_arguments(const char *const *sources, size_t count,
                                     ModuleLinkResponseGrammar grammar) {
    return module_capture_link_graph(NULL, sources, count, grammar, true);
}

char *module_capture_link_response(const ModuleBuildMetadata *meta, const char *source,
                                   ModuleLinkResponseGrammar grammar) {
    char **paths = module_capture_link_responses(meta, &source, 1, grammar);
    if (!paths) return NULL;
    char *result = paths[0];
    free(paths);
    return result;
}

static int64_t module_link_query_clock(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) return -1;
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

/* I keep one finite allowance for the caller's entire capture scope. */
static bool module_capture_deadline(int64_t *deadline) {
    const char *value = getenv("NANO_CAPTURE_TIMEOUT_MS");
    unsigned budget = 30000;
    if (value) {
        budget = 0;
        if (!*value) goto invalid;
        for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
            if (*p < '0' || *p > '9' || budget > (300000u - (*p - '0')) / 10u)
                goto invalid;
            budget = budget * 10u + (*p - '0');
        }
        if (!budget) goto invalid;
    }
    int64_t now = module_link_query_clock();
    if (now < 0 || now > INT64_MAX - budget) {
        fprintf(stderr, "I cannot establish a capture deadline.\n");
        return false;
    }
    *deadline = now + budget;
    return true;
invalid:
    fprintf(stderr, "I require NANO_CAPTURE_TIMEOUT_MS to be decimal milliseconds in 1..300000.\n");
    return false;
}

/* I execute literal argv, not a shell, and supervise one private process group.
 * The caller supplies the shared deadline and output policy. Query requests
 * are not universally read-only; their caller owns disposable output paths. */
static bool module_process_output_options(char **args, char *output, size_t capacity, int64_t deadline,
                                          bool diagnostics, bool require_output, int input, bool capture_phase) {
    int64_t now = module_link_query_clock();
    if (now < 0 || now >= deadline) {
        module_trace_evidence("tool-deadline-before-spawn", 0, 0, false);
        return false;
    }
    int64_t started = now;
    bool trace_timing = getenv("NANO_TRACE_BUILD") != NULL;
    int64_t milestones[4] = {-1, -1, -1, -1};
    int descriptors[2];
    if (pipe(descriptors)) return false;
    bool ok = true;
    for (size_t i = 0; i < 2; i++) {
        if (descriptors[i] < 3) {
            int copied = fcntl(descriptors[i], F_DUPFD_CLOEXEC, 3);
            if (copied < 0) { ok = false; break; }
            close(descriptors[i]);
            descriptors[i] = copied;
        } else if (fcntl(descriptors[i], F_SETFD, FD_CLOEXEC)) { ok = false; break; }
    }
    if (ok && fcntl(descriptors[0], F_SETFL, O_NONBLOCK)) ok = false;
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    bool have_actions = posix_spawn_file_actions_init(&actions) == 0;
    bool have_attributes = posix_spawnattr_init(&attributes) == 0;
    ok = ok && have_actions && have_attributes;
    if (ok) ok = (input < 0 ? posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0) :
                              posix_spawn_file_actions_adddup2(&actions, input, STDIN_FILENO)) == 0 &&
        posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDOUT_FILENO) == 0 &&
        (diagnostics ? posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDERR_FILENO) :
                       posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0)) == 0 &&
        posix_spawn_file_actions_addclose(&actions, descriptors[0]) == 0 &&
        posix_spawn_file_actions_addclose(&actions, descriptors[1]) == 0 &&
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP) == 0 &&
        posix_spawnattr_setpgroup(&attributes, 0) == 0;
    pid_t child = -1;
    extern char **environ;
    char **child_environment = NULL;
    if (ok && capture_phase) {
        size_t count = 0, used = 0;
        while (environ && environ[count]) count++;
        if (count > SIZE_MAX / sizeof(char *) - 2) ok = false;
        else child_environment = calloc(count + 2, sizeof(char *));
        if (!child_environment) ok = false;
        if (ok) {
            for (size_t i = 0; i < count; i++)
                if (strncmp(environ[i], "NANO_AS_CAPTURE_PHASE=", 22)) child_environment[used++] = environ[i];
            child_environment[used] = "NANO_AS_CAPTURE_PHASE=capture";
        }
    }
    if (ok) ok = posix_spawnp(&child, args[0], &actions, &attributes, args,
                              child_environment ? child_environment : environ) == 0;
    if (trace_timing && ok) milestones[0] = module_link_query_clock();
    free(child_environment);
    if (have_actions) posix_spawn_file_actions_destroy(&actions);
    if (have_attributes) posix_spawnattr_destroy(&attributes);
    close(descriptors[1]);
    if (!ok) { close(descriptors[0]); return false; }
    bool eof = false, reaped = false;
    size_t used = 0;
    int status = 0;
    while (ok && (!eof || !reaped)) {
        now = module_link_query_clock();
        if (now < 0 || now >= deadline) { ok = false; break; }
        struct pollfd descriptor = {.fd = eof ? -1 : descriptors[0], .events = POLLIN};
        int ready = poll(&descriptor, 1, (int)(deadline - now < 25 ? deadline - now : 25));
        if (ready < 0 && errno != EINTR) { ok = false; break; }
        if (ready > 0) {
            char buffer[1024];
            ssize_t n = read(descriptors[0], buffer, sizeof(buffer));
            if (n == 0) {
                eof = true;
                if (trace_timing) milestones[2] = module_link_query_clock();
            }
            else if (n < 0) {
                if (errno != EINTR && errno != EAGAIN) ok = false;
            } else if ((size_t)n >= capacity - used || memchr(buffer, 0, (size_t)n)) ok = false;
            else {
                if (trace_timing && !used) milestones[1] = module_link_query_clock();
                memcpy(output + used, buffer, (size_t)n); used += (size_t)n;
            }
        }
        if (!reaped) {
            pid_t result = waitpid(child, &status, WNOHANG);
            if (result == child) {
                reaped = true;
                if (trace_timing) milestones[3] = module_link_query_clock();
            }
            else if (result < 0 && errno != EINTR) { ok = false; break; }
        }
    }
    close(descriptors[0]);
    /* I check completion against a fresh clock, including the last poll. */
    if (ok) {
        now = module_link_query_clock();
        ok = now >= 0 && now < deadline;
    }
    ok = ok && eof && reaped && WIFEXITED(status) && WEXITSTATUS(status) == 0 && (!require_output || used);
    /* A successful reporter can leave descendants after closing its pipe too.
     * I retain no background process from this private query group. */
    (void)kill(-child, SIGKILL);
    if (!reaped) while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    output[used] = 0;
    if (!ok) module_trace_evidence(now >= deadline ? "tool-deadline" : "tool-output", 0, used, false);
    if (trace_timing) {
        const char *milestone_names[] = {"tool-spawn-ms", "tool-first-output-ms",
                                         "tool-eof-ms", "tool-reaped-ms"};
        for (size_t i = 0; i < 4; i++)
            module_trace_evidence(milestone_names[i], 1,
                                 milestones[i] >= started ? (uint64_t)(milestones[i] - started) : 0,
                                 milestones[i] >= started);
        const char *phase = "tool-run-ms";
        for (size_t i = 1; args[i]; i++)
            if (!strcmp(args[i], "-###")) phase = "tool-query-ms";
        int64_t finished = module_link_query_clock();
        module_trace_evidence(phase, (uint64_t)(deadline - started),
                             finished >= started ? (uint64_t)(finished - started) : 0, ok);
    }
    return ok;
}

static bool module_process_output_input(char **args, char *output, size_t capacity, int64_t deadline,
                                        bool diagnostics, bool require_output, int input) {
    return module_process_output_options(args, output, capacity, deadline, diagnostics, require_output, input, false);
}

static bool module_process_output(char **args, char *output, size_t capacity, int64_t deadline,
                                  bool diagnostics, bool require_output) {
    return module_process_output_input(args, output, capacity, deadline, diagnostics, require_output, -1);
}

static bool module_link_query_output(char **args, char *output, size_t capacity, int64_t deadline) {
    return module_process_output(args, output, capacity, deadline, false, true);
}

static ModuleLinkResponseGrammar module_link_response_grammar_command(const char *command) {
    if (!command || strnlen(command, 65537) > 65536) return 0;
    char **args = calloc(2050, sizeof(char *));
    if (!args) return 0;
    const char *cursor = command;
    char word[4096];
    size_t count = 0;
    int status;
    bool ok = true;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
        if (count == 2048 || !(args[count] = strdup(word))) { ok = false; break; }
        count++;
    }
    ModuleLinkResponseGrammar grammar = 0;
    int64_t now = module_link_query_clock();
    if (ok && status == 0 && count && now >= 0) {
        int64_t deadline = now + 5000;
        char output[8193];
        args[count] = "-Wl,--version";
        if (module_link_query_output(args, output, sizeof(output), deadline) &&
            !strncmp(output, "GNU ld (", 8) && strchr(output, '\n')) grammar = MODULE_LINK_RESPONSE_GNU;
        if (!grammar) {
            args[count] = "-Wl,-version_details";
            if (module_link_query_output(args, output, sizeof(output), deadline)) {
                const char *end = NULL;
                cJSON *details = cJSON_ParseWithOpts(output, &end, true);
                cJSON *version = cJSON_GetObjectItemCaseSensitive(details, "version");
                cJSON *architectures = cJSON_GetObjectItemCaseSensitive(details, "architectures");
                cJSON *tapi = cJSON_GetObjectItemCaseSensitive(details, "tapi");
                cJSON *vendor = cJSON_GetObjectItemCaseSensitive(tapi, "version_string");
                /* I admit the installed Apple implementation covered by my
                 * identity/grammar corpus, not every JSON-speaking linker. */
                if (cJSON_IsString(version) && !strcmp(version->valuestring, "1267") &&
                    cJSON_IsArray(architectures) && cJSON_GetArraySize(architectures) > 0 &&
                    cJSON_IsString(vendor) && !strncmp(vendor->valuestring, "Apple TAPI version ", 19))
                    grammar = MODULE_LINK_RESPONSE_APPLE;
                cJSON_Delete(details);
            }
        }
    }
    for (size_t i = 0; i < count; i++) free(args[i]);
    free(args);
    return grammar;
}

static bool module_append_compiler_fragment(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                            const char *fragment, bool retained,
                                            char *output, size_t capacity) {
    char *filtered = NULL;
    if (retained) {
        size_t size = strlen(fragment);
        if (size > (SIZE_MAX - 16) / 4) return false;
        size = size * 4 + 16;
        filtered = calloc(size, 1);
        if (!filtered || !module_retained_flags(fragment, filtered, size, flags && flags->linker_grammar)) {
            free(filtered); return false;
        }
        fragment = filtered;
    }
    char *transport = module_response_transport(meta, flags, fragment);
    bool ok = transport && module_build_append(output, capacity, " %s", transport);
    free(transport);
    free(filtered);
    return ok;
}

/* I preserve the existing shared-link group order, including repeated libraries.
 * This is driver argument transport, not capture of indirect linker inputs. */
static char *module_shared_link_fragment(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags) {
    const size_t capacity = 65537;
    char *fragment = calloc(capacity, 1);
    if (!fragment) return NULL;
    size_t platform_count;
    char **platform = module_platform_ldflags(meta, &platform_count);
    size_t counts[] = {meta->pkg_config_count, meta->system_libs_count, meta->ldflags_count,
                       platform_count,
#ifdef __APPLE__
                       meta->frameworks_count
#else
                       0
#endif
    };
    bool ok = true;
    for (size_t group = 0; group < 5 && ok; group++) {
        for (size_t i = 0; i < counts[group] && ok; i++) {
#ifdef __APPLE__
            if (group == 0 && module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
            const char *value = group == 0 ? flags->libs[i] : group == 1 ? meta->system_libs[i] :
                group == 2 ? meta->ldflags[i] : group == 3 ? platform[i] : meta->frameworks[i];
            const char *prefix = group == 1 ? "-l" : group == 4 ? "-framework " : "";
            ok = value && module_build_append(fragment, capacity, " %s%s", prefix, value);
        }
    }
    if (!ok) { free(fragment); return NULL; }
    return fragment;
}

static char *module_shared_link_transport(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                          const char *fragment) {
    /* I do not hide user response inputs from Darwin's linker observation. */
    if (strchr(fragment, '@')) return strdup(fragment);
    return module_response_transport(meta, flags, fragment);
}

#ifdef __APPLE__
/* My linker observation still declines indirect user arguments. The only @
 * words I admit are exact transports of this invocation's captured flags. */
static bool module_link_response_safe(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                      const char *command) {
    if (!strchr(command, '@')) return true;
    char link_word[4096] = {0};
    char *fragment = module_shared_link_fragment(meta, flags);
    char *transport = fragment ? module_shared_link_transport(meta, flags, fragment) : NULL;
    if (!transport) { free(fragment); return false; }
    if (strcmp(fragment, transport)) {
        const char *p = transport;
        char extra[4096];
        if (module_flag_word(&p, link_word, sizeof(link_word)) != 1 || link_word[0] != '@' ||
            module_flag_word(&p, extra, sizeof(extra)) != 0) link_word[0] = 0;
    }
    free(transport);
    free(fragment);
    const char *cursor = command;
    char word[4096];
    int status;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
        if (!strchr(word, '@')) continue;
        if (word[0] != '@') return false;
        if (link_word[0] && !strcmp(word, link_word)) continue;
        bool found = false;
        for (size_t group = 0; group < 3 && !found; group++) {
            size_t count = group == 0 ? flags->count : meta->cflags_count;
            char **fragments = group == 0 ? flags->cflags :
                group == 1 ? meta->cflags : module_platform_cflags(meta, &count);
            for (size_t i = 0; i < count && !found; i++) {
                if (!fragments[i]) continue;
                char *selected = module_link_cflags(fragments[i]);
                if (!selected) return false;
                char *transport = module_response_transport(meta, flags, selected);
                if (!transport) { free(selected); return false; }
                const char *p = transport;
                char expected[4096];
                found = strcmp(transport, selected) != 0 &&
                    module_flag_word(&p, expected, sizeof(expected)) == 1 && !strcmp(expected, word) &&
                    module_flag_word(&p, expected, sizeof(expected)) == 0;
                free(transport);
                free(selected);
            }
        }
        if (!found) return false;
    }
    return status == 0;
}
#endif

typedef enum { MODULE_C_PREPROCESS, MODULE_C_COMPILE, MODULE_C_RETAINED, MODULE_C_RETAINED_ASSEMBLY,
               MODULE_C_EMIT_ASSEMBLY, MODULE_C_ASSEMBLE, MODULE_C_ASSEMBLE_UNIT,
               MODULE_C_RETAINED_INTEGRATED_ASSEMBLY, MODULE_C_INTEGRATED_PREPROCESS } ModuleCPhase;

/* The caller owns a zeroed array and frees every slot on failure. I retain
 * original include paths in metadata for dependency and cache validation. */
static bool module_include_flags(const ModuleBuildMetadata *meta, char **output) {
    for (size_t i = 0; i < meta->include_dirs_count; i++) {
        char *quoted = module_quote_path(meta->include_dirs[i]);
        if (!quoted) return false;
        size_t length = strlen(quoted);
        output[i] = length <= SIZE_MAX - 3 ? malloc(length + 3) : NULL;
        if (output[i]) { memcpy(output[i], "-I", 2); memcpy(output[i] + 2, quoted, length + 1); }
        free(quoted);
        if (!output[i]) return false;
    }
    return module_coalesce_cflags(output, meta->include_dirs_count);
}

static bool module_append_include_arguments(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                            char *output, size_t capacity) {
    size_t count = meta->include_dirs_count;
    if (!count) return true;
    char **includes = calloc(count, sizeof(char *));
    if (!includes) return false;
    bool ok = module_include_flags(meta, includes);
    for (size_t i = 0; i < count; i++) {
        if (ok && includes[i][0])
            ok = module_append_compiler_fragment(meta, flags, includes[i], false, output, capacity);
        free(includes[i]);
    }
    free(includes);
    return ok;
}

static bool module_append_source_fragment(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                          const char *fragment, bool retained, unsigned phases,
                                          char *output, size_t capacity) {
    if (phases != (MODULE_FLAG_BOTH | MODULE_FLAG_ASSEMBLER)) {
        size_t length = strlen(fragment);
        if (length > (SIZE_MAX - 16) / 4) return false;
        size_t size = length * 4 + 16;
        char *selected = calloc(size, 1);
        if (!selected) return false;
        bool admitted = module_phase_flags(fragment, selected, size, flags->linker_grammar, phases);
        bool ok = admitted && module_append_compiler_fragment(meta, flags, selected, false, output, capacity);
        free(selected);
        if (admitted || (phases & MODULE_FLAG_ASSEMBLER)) return ok;
        /* Unadmitted compatibility fragments retain their original handling. */
    }
    if (!flags->linker_grammar || retained)
        return module_append_compiler_fragment(meta, flags, fragment, retained, output, capacity);
    size_t length = strlen(fragment);
    if (length > (SIZE_MAX - 16) / 4) return false;
    size_t size = length * 4 + 16;
    char *filtered = calloc(size, 1);
    if (!filtered) return false;
    const char *cursor = fragment;
    char word[4096];
    int status;
    bool ok = true;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0 && ok) {
        if (!strcmp(word, "-Xlinker")) {
            ok = module_flag_word(&cursor, word, sizeof(word)) == 1 && word[0] != '@';
        } else ok = module_append_path_flag(filtered, size, "", word);
    }
    ok = ok && status == 0 &&
        module_append_compiler_fragment(meta, flags, filtered, false, output, capacity);
    free(filtered);
    return ok;
}

/* I preserve the assembler selector after C capture. I inspect argument words,
 * not substrings in definitions or arguments forwarded to another tool. */
static bool module_assembler_option(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags, bool search) {
    for (size_t group = 0; group < 3; group++) {
        size_t count = group == 0 ? flags->count : meta->cflags_count;
        char **fragments = group == 0 ? flags->cflags : meta->cflags;
        if (group == 2) fragments = module_platform_cflags(meta, &count);
        for (size_t i = 0; i < count; i++) {
            if (!fragments[i]) continue;
            const char *cursor = fragments[i];
            char word[4096];
            while (module_flag_word(&cursor, word, sizeof(word)) > 0) {
                if (search && module_wa_options(word, true)) return true;
                if (!strcmp(word, "-Xassembler")) {
                    if (module_flag_word(&cursor, word, sizeof(word)) != 1) break;
                    if (search && !strncmp(word, "-I", 2)) return true;
                    continue;
                }
                if (!strcmp(word, "-Xlinker") || !strcmp(word, "-Xassembler") || !strcmp(word, "-D") ||
                    !strcmp(word, "-U") || !strcmp(word, "-I")) {
                    if (module_flag_word(&cursor, word, sizeof(word)) != 1) break;
                    continue;
                }
                if (!search && !strcmp(word, "-fno-integrated-as")) return true;
            }
        }
    }
    return false;
}

static bool module_external_assembler(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags) {
    return module_assembler_option(meta, flags, false);
}

static bool module_compile_prefix(ModuleBuildMetadata *meta, char *prefix, size_t capacity,
                                  ModuleCPhase phase, const ModulePkgFlags *snapshot) {
    prefix[0] = 0;
    bool integrated_retained = phase == MODULE_C_RETAINED_INTEGRATED_ASSEMBLY;
    bool integrated_preprocess = phase == MODULE_C_INTEGRATED_PREPROCESS;
    bool retained = phase == MODULE_C_RETAINED || phase == MODULE_C_RETAINED_ASSEMBLY || integrated_retained;
    bool assembler = phase == MODULE_C_ASSEMBLE || phase == MODULE_C_ASSEMBLE_UNIT;
    unsigned phases = assembler ? MODULE_FLAG_ASSEMBLER |
        (phase == MODULE_C_ASSEMBLE_UNIT ? MODULE_FLAG_DEBUG : 0) : retained ?
        MODULE_FLAG_C | (integrated_retained ? MODULE_FLAG_ASSEMBLER : 0) :
        phase == MODULE_C_PREPROCESS ? MODULE_FLAG_BOTH : (MODULE_FLAG_BOTH | MODULE_FLAG_ASSEMBLER);
    /* Clang drops -Wa include paths from its -S and -E driver jobs. I preserve the
     * real integrated -c job's frontend search order, changing only its output
     * action to assembly or preprocessing. External assembly keeps its separate
     * search phase; standalone assembler preprocessing uses its own prefix. */
    bool search_capture = (phase == MODULE_C_EMIT_ASSEMBLY || integrated_retained || integrated_preprocess) && snapshot &&
        module_assembler_option(meta, snapshot, true);
    bool ok = module_build_append(prefix, capacity, "%s %s -fPIC",
                                   module_selected_compiler(meta), phase == MODULE_C_PREPROCESS ? "-E" :
                                   search_capture ? (integrated_preprocess ? "-c -Xclang -E" : "-c -Xclang -S") :
                                   integrated_preprocess ? "-E" :
                                   (phase == MODULE_C_EMIT_ASSEMBLY || phase == MODULE_C_RETAINED_ASSEMBLY || integrated_retained) ? "-S" : "-c");
    /* I already applied C code-generation and diagnostic flags during capture. */
    if (assembler) {
        if (snapshot && module_external_assembler(meta, snapshot))
            ok &= module_build_append(prefix, capacity, " -fno-integrated-as");
    }
#if !defined(__APPLE__)
    if (!retained && !assembler) ok &= module_build_append(prefix, capacity, " -D_POSIX_C_SOURCE=200809L");
#endif
    for (size_t i = 0; i < meta->pkg_config_count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        const char *flags = snapshot->cflags[i];
        if (flags && flags[0]) {
            ok &= module_append_source_fragment(meta, snapshot, flags, retained, phases, prefix, capacity);
        } else if (!flags) ok = false;
    }
    if (!retained && !assembler) ok &= module_append_include_arguments(meta, snapshot, prefix, capacity);
    for (size_t group = 0; group < 2; group++) {
        size_t count = meta->cflags_count;
        char **flags = group ? module_platform_cflags(meta, &count) : meta->cflags;
        for (size_t i = 0; i < count; i++) {
            if (!flags[i][0]) continue;
            ok &= module_append_source_fragment(meta, snapshot, flags[i], retained, phases, prefix, capacity);
        }
    }
    return ok;
}

typedef enum {
    MODULE_SNAPSHOT_NONE = 0,
    MODULE_SNAPSHOT_CLANG,
    MODULE_SNAPSHOT_CLANG_EXTERNAL,
    MODULE_SNAPSHOT_GCC,
    MODULE_SNAPSHOT_GCC_ASSEMBLY,
    MODULE_SNAPSHOT_GCC_REPLAY,
    MODULE_SNAPSHOT_NATIVE_UNITS,
    MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS
} ModuleSnapshotMode;

static ModuleSnapshotMode module_driver_snapshot_mode(const ModuleBuildMetadata *meta) {
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

static bool module_response_driver(const ModuleBuildMetadata *meta) {
    const char *driver = module_selected_compiler(meta);
    const char *base = strrchr(driver, '/');
    base = base ? base + 1 : driver;
    if (!strncmp(base, "clang-cl", 8)) return false;
    for (size_t group = 0; group < 4; group++) {
        size_t count;
        char **flags = module_response_group(meta, group, &count);
        for (size_t i = 0; i < count; i++)
            if (strstr(flags[i], "--driver-mode")) return false;
    }
    return module_driver_snapshot_mode(meta) != MODULE_SNAPSHOT_NONE;
}

/* Zero is unsupported, one is C, two is .s, three is .S. The selected
 * driver's platform semantics determine whether .s needs preprocessing. */
static unsigned module_source_kind(const char *source) {
    size_t length = strlen(source);
    if (length < 2 || source[length - 2] != '.') return 0;
    return source[length - 1] == 'c' ? 1 : source[length - 1] == 's' ? 2 :
           source[length - 1] == 'S' ? 3 : 0;
}

static ModuleSnapshotMode module_snapshot_mode(const ModuleBuildMetadata *meta, const ModulePkgFlags *captured) {
    if (!captured || captured->count != meta->pkg_config_count) return MODULE_SNAPSHOT_NONE;
    for (size_t i = 0; i < captured->count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        if (!module_retained_flags(captured->cflags[i], NULL, 0, captured->linker_grammar)) return MODULE_SNAPSHOT_NONE;
    }
    for (size_t group = 0; group < 2; group++) {
        size_t count = meta->cflags_count;
        char **flags = group ? module_platform_cflags(meta, &count) : meta->cflags;
        for (size_t i = 0; i < count; i++)
            if (!module_retained_flags(flags[i], NULL, 0, captured->linker_grammar)) return MODULE_SNAPSHOT_NONE;
    }
    bool assembler_sources = false;
    for (size_t group = 0; group < 2; group++) {
        char **sources = group ? meta->shared_c_sources : meta->c_sources;
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count; i++) {
            unsigned kind = module_source_kind(sources[i]);
            if (!kind) return MODULE_SNAPSHOT_NONE;
            assembler_sources |= kind != 1;
        }
    }
    if (!meta->c_sources_count) return MODULE_SNAPSHOT_NONE;
    ModuleSnapshotMode mode = module_driver_snapshot_mode(meta);
    if (assembler_sources && mode == MODULE_SNAPSHOT_CLANG && !module_external_assembler(meta, captured))
        return MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS;
    return mode == MODULE_SNAPSHOT_CLANG && module_external_assembler(meta, captured)
        ? MODULE_SNAPSHOT_CLANG_EXTERNAL : mode;
}

static uint64_t module_snapshot_sources(ModuleBuildMetadata *meta,
                                       const ModulePkgFlags *flags, const char *directory,
                                       ModuleSnapshotMode mode, ModuleSnapshotMode *actual_mode);
static uint64_t module_gcc_validation(ModuleBuildMetadata *meta, const ModulePkgFlags *flags, ModuleSnapshotMode mode);

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
    if (mode == MODULE_SNAPSHOT_GCC || mode == MODULE_SNAPSHOT_CLANG_EXTERNAL ||
        mode == MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS)
        return module_gcc_validation(meta, flags, mode);
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

static int module_source_diagnostics(int result, const char *dependency) {
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

static int module_run_source_command(const char *command, const char *dependency) {
    return module_source_diagnostics(system(command), dependency);
}

/* I capture a deliberately bounded assembler spelling, not the assembler
 * language. Literal file directives must start a line; macro expansion and
 * alternate macro syntax decline this capture path. I copy
 * whole binary files so the assembler still evaluates offset/count expressions.
 * Relative paths resolve in the compiler's working directory, as in GNU as with
 * no assembler include-search flags (those flags already reject this mode). */
typedef struct {
    const char *directory;
    unsigned files;
    size_t bytes;
    uint64_t hash;
} ModuleAssemblyCapture;

/* I leave data strings verbatim. Three octal digits denote a byte, not a
 * named macro argument. Short escapes, filename escapes and other expansion
 * syntax remain outside this copier's grammar (including MRI macros). */
static bool module_assembly_octal_data(const char *line) {
    while (*line == ' ' || *line == '\t') line++;
    size_t keyword = !strncmp(line, ".ascii", 6) || !strncmp(line, ".asciz", 6) ? 6 :
                     !strncmp(line, ".string", 7) ? 7 : 0;
    if (!keyword || (line[keyword] != ' ' && line[keyword] != '\t')) return false;
    line += keyword;
    while (*line == ' ' || *line == '\t') line++;
    if (*line++ != '"') return false;
    while (*line && *line != '"') {
        if (*line == '\\') {
            line++;
            for (unsigned i = 0; i < 3; i++) {
                if (*line < '0' || *line > (i ? '7' : '3')) return false;
                line++;
            }
        } else line++;
    }
    if (*line++ != '"') return false;
    while (*line == ' ' || *line == '\t' || *line == '\r') line++;
    if (!*line) return true;
    if (strchr(line, '\\')) return false;
    /* I do not mistake a GNU statement separator for an Apple comment. */
#ifdef __APPLE__
    if (*line == ';') return true;
#else
    if (*line == '#') return true;
#endif
    return line[0] == '/' && line[1] == '/';
}

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
    if (text && (memchr(data, 0, size) ||
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
        if (strchr(line, '\\') && !module_assembly_octal_data(line)) { ok = false; break; }
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

/* I lower retained C, but copy already-preprocessed assembler without changing
 * its grammar. The destination is always private to this capture. */
static bool module_prepare_assembly(ModuleBuildMetadata *meta, const char *prefix,
                                    const char *input, const char *output, size_t group, size_t index) {
    const char *source = group ? meta->shared_c_sources[index] : meta->c_sources[index];
    if (module_source_kind(source) != 1) {
        ModuleAssemblyCapture copy = {NULL, 0, 0, 14695981039346656037ULL};
        return module_capture_assembly_file(&copy, input, output, false, 0);
    }
    char command[8192] = {0};
    return module_build_append(command, sizeof(command), "%s%s -x cpp-output", prefix,
                               group ? " -fvisibility=hidden" : "") &&
        module_append_path_flag(command, sizeof(command), "", input) &&
        module_append_path_flag(command, sizeof(command), "-o ", output) && !system(command);
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
            char input[2048] = {0}, raw[2048] = {0}, frozen[2048] = {0};
            bool ok = module_build_append(input, sizeof(input), "%s/__snapshot_%zu_%zu.i", directory, group, i) &&
                module_build_append(raw, sizeof(raw), "%s/__assembly_%zu_%zu.s", directory, group, i) &&
                module_build_append(frozen, sizeof(frozen), "%s/__snapshot_%zu_%zu.s", directory, group, i) &&
                module_prepare_assembly(meta, prefix, input, raw, group, i);
            if (!ok || !module_capture_assembly_file(&capture, raw, frozen, true, 0)) goto failed;
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

#ifdef __linux__
/* I accept only tested version tokens on the GNU assembler banner's first
 * line. This is compatibility selection, not executable authentication. */
static bool module_assembler_version_supported(const char *version) {
    static const char prefix[] = "GNU assembler (";
    static const char *versions[] = {" 2.40", " 2.42"};
    if (!version || strncmp(version, prefix, sizeof(prefix) - 1)) return false;
    const char *end = strchr(version, '\n');
    if (!end) return false;
    for (size_t i = 0; i < sizeof(versions) / sizeof(versions[0]); i++) {
        size_t length = strlen(versions[i]);
        if ((size_t)(end - version) >= sizeof(prefix) + length &&
            *(end - length - 1) == ')' && !memcmp(end - length, versions[i], length)) return true;
    }
    return false;
}
#endif

/* My aliases live only while building or validating. Each unit gets its own
 * directory, so equal source basenames never share a retained input. */
static bool module_unit_input(ModuleBuildMetadata *meta, const char *directory,
                               size_t group, size_t index, char *input, size_t capacity,
                               char *parent, size_t parent_capacity) {
    const char *source = group ? meta->shared_c_sources[index] : meta->c_sources[index];
    const char *basename = strrchr(source, '/');
    basename = basename ? basename + 1 : source;
    char unit[80] = {0}, retained[80] = {0};
    input[0] = parent[0] = 0;
    if (!*basename || !strcmp(basename, ".") || !strcmp(basename, "..") ||
        !module_build_append(unit, sizeof(unit), "__unit_%zu_%zu", group, index) ||
        !module_build_append(retained, sizeof(retained), "__snapshot_%zu_%zu.s", group, index) ||
        !module_build_append(parent, parent_capacity, "%s/%s", directory, unit) ||
        !module_build_append(input, capacity, "%s/%s", parent, basename)) return false;
    int stage = open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (stage < 0) return false;
    bool ok = mkdirat(stage, unit, 0700) == 0 || errno == EEXIST;
    int alias = ok ? openat(stage, unit, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW) : -1;
    int from = alias >= 0 ? openat(stage, retained, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK) : -1;
    struct stat st;
    ok = from >= 0 && fstat(from, &st) == 0 && S_ISREG(st.st_mode) &&
        st.st_size >= 0 && (uint64_t)st.st_size <= 32ULL * 1024 * 1024;
    if (ok && unlinkat(alias, basename, 0) != 0 && errno != ENOENT) ok = false;
    int to = ok ? openat(alias, basename, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600) : -1;
    if (to < 0) ok = false;
    size_t total = 0;
    unsigned char bytes[8192];
    while (ok) {
        ssize_t amount = read(from, bytes, sizeof(bytes));
        if (amount < 0 && errno == EINTR) continue;
        if (amount < 0) { ok = false; break; }
        if (!amount) break;
        if ((size_t)amount > 32ULL * 1024 * 1024 - total) { ok = false; break; }
        total += (size_t)amount;
        size_t sent = 0;
        while (sent < (size_t)amount) {
            ssize_t written = write(to, bytes + sent, (size_t)amount - sent);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) { ok = false; break; }
            sent += (size_t)written;
        }
    }
    if (ok) ok = total == (uint64_t)st.st_size && fchmod(to, 0400) == 0;
    if (to >= 0 && close(to) != 0) ok = false;
    if (from >= 0 && close(from) != 0) ok = false;
    if (!ok && alias >= 0 && to >= 0) (void)unlinkat(alias, basename, 0);
    if (alias >= 0 && close(alias) != 0) ok = false;
    if (close(stage) != 0) ok = false;
    return ok;
}

/* I retain standalone debug-option ownership without replaying C diagnostics. */
static bool module_unit_assembly_prefix(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                         const char *source, const char *directory,
                                         char *prefix, size_t capacity, const char *descriptor) {
    if (!module_compile_prefix(meta, prefix, capacity, MODULE_C_ASSEMBLE_UNIT, flags)) return false;
    bool debug = false;
    for (size_t group = 0; group < 3; group++) {
        size_t count = group ? meta->cflags_count : flags->count;
        char **fragments = group ? meta->cflags : flags->cflags;
        if (group == 2) fragments = module_platform_cflags(meta, &count);
        for (size_t i = 0; i < count; i++) {
#ifdef __APPLE__
            if (!group && module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
            char selected[4096] = {0}, word[4096];
            if (!module_phase_flags(fragments[i], selected, sizeof(selected), flags->linker_grammar,
                                    MODULE_FLAG_DEBUG)) return false;
            const char *cursor = selected;
            while (module_flag_word(&cursor, word, sizeof(word)) > 0) debug = strcmp(word, "-g0") != 0;
        }
    }
    if (!debug) return true;
    char original[4096] = {0}, mapping[8192] = {0};
    bool ok = source[0] == '/' ? module_build_append(original, sizeof(original), "%s", source) :
        module_build_append(original, sizeof(original), "%s/%s", meta->module_dir, source);
    char *slash = ok ? strrchr(original, '/') : NULL;
    if (!slash) return false;
    if (slash == original) slash[1] = 0;
    else *slash = 0;
    /* I map invocation-private directories before object hashing. This keeps
     * validation reproducible; raw-source basename provenance remains separate. */
#ifdef __APPLE__
    const char *option = "-fdebug-prefix-map=";
#else
    const char *option = "--debug-prefix-map=";
#endif
    const char *forward = "-Xassembler ";
    if (descriptor) {
        /* I name the parent-held directory through its descriptor. */
        return module_build_append(mapping, sizeof(mapping), "%s%s=%s", option, descriptor, original) &&
            module_append_path_flag(prefix, capacity, forward, mapping);
    }
    /* I place the broad source alias first: the selected assemblers give
     * later mappings priority, and my staging directory can live below it. */
    char *canonical = realpath(original[0] ? original : "/", NULL);
    if (!canonical) return false;
    if (strcmp(canonical, original)) {
        ok = !strchr(canonical, '=') &&
            module_build_append(mapping, sizeof(mapping), "%s%s=%s", option, canonical, original) &&
            module_append_path_flag(prefix, capacity, forward, mapping);
    }
    free(canonical);
    if (strchr(directory, '=')) return false;
    mapping[0] = 0;
    ok = ok && module_build_append(mapping, sizeof(mapping), "%s%s=%s", option, directory, original) &&
        module_append_path_flag(prefix, capacity, forward, mapping);
    char *private_canonical = realpath(directory, NULL);
    if (!private_canonical) return false;
    if (ok && strcmp(private_canonical, directory)) {
        mapping[0] = 0;
        ok = !strchr(private_canonical, '=') &&
            module_build_append(mapping, sizeof(mapping), "%s%s=%s", option, private_canonical, original) &&
            module_append_path_flag(prefix, capacity, forward, mapping);
    }
    char *cwd = getcwd(NULL, 0);
    if (!cwd) { free(private_canonical); return false; }
    size_t cwd_length = strlen(cwd);
    if (ok && !strncmp(private_canonical, cwd, cwd_length) && private_canonical[cwd_length] == '/') {
        /* Apple's automatic assembler debug names can be cwd-relative. */
        mapping[0] = 0;
        ok = module_build_append(mapping, sizeof(mapping), "%s%s=%s", option,
                                  private_canonical + cwd_length + 1, original) &&
            module_append_path_flag(prefix, capacity, forward, mapping);
    }
    free(cwd);
    free(private_canonical);
    return ok;
}

/* GCC still chooses assembler arguments; my private -B entry changes only the
 * executable receiving them. Loader configuration starts inside that wrapper,
 * never in the compiler driver, preprocessor, linker or calling process. */
#ifdef __linux__
static int module_read_directory(const char *parent, char *name, size_t capacity) {
    name[0] = 0;
    int fd = open(parent, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return -1;
    if (fd < 3) {
        int copy = fcntl(fd, F_DUPFD_CLOEXEC, 3);
        close(fd); fd = copy;
        if (fd < 0) return -1;
    }
    if (!module_build_append(name, capacity, "/proc/%ld/fd/%d", (long)getpid(), fd)) {
        close(fd); return -1;
    }
    return fd;
}

/* I keep cwd and stdin ordinary, including for compiler wrappers that close
 * inherited descriptors before invoking their selected compiler. */
static bool module_read_execute(const char *command) {
    char report[16384] = {0}, *args[] = {"/bin/sh", "-c", (char *)command, NULL};
    int64_t deadline;
    bool ok = module_capture_deadline(&deadline) &&
        module_process_output(args, report, sizeof(report), deadline, true, false);
    if (report[0]) fputs(report, stderr);
    return ok;
}
#endif

static bool module_read_command(char *command, size_t capacity, const char *prefix,
                                 const char *directory, size_t group, size_t index,
                                 const char *object, const char *input, const char *primary, bool capture) {
    char record[2048] = {0}, tools[2048] = {0};
    command[0] = 0;
    return module_build_append(record, sizeof(record), "%s/__as_read_%zu_%zu", directory, group, index) &&
        module_build_append(tools, sizeof(tools), "%s/", directory) &&
        module_build_append(command, capacity, "NANO_AS_CAPTURE_PHASE=%s", capture ? "capture" : "replay") &&
        module_append_path_flag(command, capacity, "NANO_AS_CAPTURE_PREFIX=", record) &&
        module_append_path_flag(command, capacity, "NANO_AS_CAPTURE_INPUT=", input) &&
        module_append_path_flag(command, capacity, "NANO_AS_CAPTURE_PRIMARY=", primary) &&
        module_build_append(command, capacity, " %s -x assembler", prefix) &&
        module_append_path_flag(command, capacity, "-B", tools) &&
        module_append_path_flag(command, capacity, "", input) &&
        module_append_path_flag(command, capacity, "-o ", object);
}

/* I decode one dry-run command from a tested Clang driver's report. I never
 * execute the report as shell text, accept multiple commands, or guess a
 * backend executable from an installation directory. */
static size_t module_assembler_argv(const char *command, char **args, char *storage, size_t capacity) {
    char word[4096];
    size_t count = 0, used = 0;
    int status;
    while ((status = module_flag_word(&command, word, sizeof(word))) > 0) {
        size_t length = strlen(word) + 1;
        if (count >= 252 || length > capacity - used) return 0;
        args[count++] = storage + used;
        memcpy(storage + used, word, length);
        used += length;
    }
    args[count] = NULL;
    return status == 0 ? count : 0;
}

static size_t module_assembler_report(char *report, char **args, char *storage, size_t capacity) {
    const char *banner = !strncmp(report, "Apple clang version 21.0.0 ", 27)
        ? "Apple clang version 21.0.0 " : !strncmp(report, "Debian clang version 14.0.6\n", 28)
        ? "Debian clang version 14.0.6" : NULL;
    if (!banner) return 0;
    char *selected = NULL;
    for (char *line = report; line && *line;) {
        char *end = strchr(line, '\n');
        if (end) *end = 0;
        while (*line == ' ' || *line == '\t') line++;
        if (*line == '"') {
            if (selected) return 0;
            selected = line;
        } else if (*line && strncmp(line, banner, strlen(banner)) &&
                   strncmp(line, "Target: ", 8) && strncmp(line, "Thread model: ", 14) &&
                   strncmp(line, "InstalledDir: ", 14) &&
                   strcmp(line, "(in-process)") &&
                   strcmp(line, "clang: warning: argument unused during compilation: '-fPIC' [-Wunused-command-line-argument]")) return 0;
        line = end ? end + 1 : NULL;
    }
    return selected ? module_assembler_argv(selected, args, storage, capacity) : 0;
}

static bool module_assembler_tool_hash(uint64_t *hash, const char *tool, int64_t deadline) {
    int64_t now = module_link_query_clock();
    if (now < 0 || now >= deadline) {
        module_trace_evidence("tool-hash-deadline", 1, 0, false);
        return false;
    }
    if (tool[0] != '/') return false;
    /* I allow installed tool symlinks, but never wait for a FIFO writer or
     * hash an endless device. The descriptor determines the admitted kind.
     * Regular-file I/O still relies on the host filesystem returning. */
    int fd = open(tool, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat before, after;
    bool ok = !fstat(fd, &before) && S_ISREG(before.st_mode) && before.st_size >= 0;
    off_t remaining = ok ? before.st_size : 0;
    uint64_t bytes = 14695981039346656037ULL;
    unsigned char buffer[4096];
    while (ok && remaining) {
        now = module_link_query_clock();
        if (now < 0 || now >= deadline) { ok = false; break; }
        size_t wanted = remaining < (off_t)sizeof(buffer) ? (size_t)remaining : sizeof(buffer);
        ssize_t count = read(fd, buffer, wanted);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) { ok = false; break; }
        for (ssize_t i = 0; i < count; i++) {
            bytes ^= buffer[i];
            bytes *= 1099511628211ULL;
        }
        remaining -= count;
    }
    now = module_link_query_clock();
    ok = ok && now >= 0 && now < deadline && !fstat(fd, &after) &&
        before.st_size == after.st_size;
    if (close(fd)) ok = false;
    if (!ok || !bytes) {
        module_trace_evidence(now >= deadline ? "tool-hash-deadline" : "tool-hash-input", 1, 0, false);
        return false;
    }
    char digest[24];
    snprintf(digest, sizeof(digest), "%llu", (unsigned long long)bytes);
    hash_context_field(hash, tool);
    hash_context_field(hash, digest);
    return true;
}

/* I give the selected Clang backend retained bytes on stdin and the original
 * logical name. No private pathname needs a debug map, including paths with
 * '='. Platform preprocessing has already happened; I do not edit line data. */
static bool module_clang_native_stdin(char **args, size_t words, char *report, size_t report_size,
                                      int64_t deadline, const char *input, const char *original,
                                      const char *object, uint64_t *fingerprint, bool integrated) {
    args[words] = "-###"; args[words + 1] = NULL;
    if (!module_process_output_options(args, report, report_size, deadline, true, true, -1, true)) return false;
    char storage[16384], *job[256];
    words = module_assembler_report(report, job, storage, sizeof(storage));
    if (!words || !module_assembler_tool_hash(fingerprint, job[0], deadline)) return false;
    if (!integrated) {
        job[words] = "-###"; job[words + 1] = NULL;
        if (!module_process_output_options(job, report, report_size, deadline, true, true, -1, true)) return false;
        words = module_assembler_report(report, job, storage, sizeof(storage));
    }
    if (words < 2 || strcmp(job[1], "-cc1as") || !module_assembler_tool_hash(fingerprint, job[0], deadline)) return false;
    size_t name = 0, source = 0, format = 0, output = 0;
    for (size_t i = 2; i < words; i++) {
        if (!strcmp(job[i], "-main-file-name")) {
            if (name || i + 1 == words) return false;
            name = i + 1;
        }
        if (!strcmp(job[i], "-filetype")) {
            if (format || i + 1 == words || strcmp(job[i + 1], "obj")) return false;
            format = i + 1;
        }
        if (!strcmp(job[i], "-o")) {
            if (output || i + 1 == words || strcmp(job[i + 1], object)) return false;
            output = i + 1;
        }
        if (!strcmp(job[i], input)) {
            if (source) return false;
            source = i;
        }
    }
    if (!name || !source || !format || !output) return false;
    job[name] = (char *)original;
    job[source] = "-";
    int fd = open(input, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC);
    struct stat st;
    bool ok = fd >= 0 && !fstat(fd, &st) && S_ISREG(st.st_mode) &&
        st.st_size >= 0 && st.st_size <= 32LL * 1024 * 1024;
    if (ok && fd < 3) {
        int copy = fcntl(fd, F_DUPFD_CLOEXEC, 3);
        close(fd); fd = copy;
        ok = fd >= 0;
    }
    report[0] = 0;
    if (ok) ok = module_process_output_options(job, report, report_size, deadline, true, false, fd, true);
    if (fd >= 0 && close(fd)) ok = false;
    return ok;
}

static uint64_t module_clang_expansion(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                      const char *directory, uint64_t fingerprint, bool integrated) {
    if (!directory) return 0;
    char retained[4096], assemble[4096];
    if (!module_compile_prefix(meta, retained, sizeof(retained), integrated ? MODULE_C_RETAINED_INTEGRATED_ASSEMBLY :
                               MODULE_C_RETAINED_ASSEMBLY, flags) ||
        !module_compile_prefix(meta, assemble, sizeof(assemble), MODULE_C_ASSEMBLE, flags)) return 0;
    ModuleAssemblyCapture capture = {directory, 0, 0, fingerprint};
    hash_context_field(&capture.hash, integrated ? "clang-integrated-expanded-native-v1" :
                                                 "apple-selected-assembler-expanded-native-v2");
    for (size_t group = 0; group < 2; group++) {
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count; i++) {
            char input[2048] = {0}, raw[2048] = {0}, expanded[2048] = {0}, frozen[2048] = {0}, object[2048] = {0};
            char command[8192] = {0}, report[16384], storage[16384], *args[256];
            bool ok = module_build_append(input, sizeof(input), "%s/__snapshot_%zu_%zu.i", directory, group, i) &&
                module_build_append(raw, sizeof(raw), "%s/__assembly_%zu_%zu.s", directory, group, i) &&
                module_build_append(expanded, sizeof(expanded), "%s/__expanded_%zu_%zu.s", directory, group, i) &&
                module_build_append(frozen, sizeof(frozen), "%s/__snapshot_%zu_%zu.s", directory, group, i) &&
                module_build_append(object, sizeof(object), "%s/__as_query_%zu_%zu.o", directory, group, i) &&
                module_prepare_assembly(meta, retained, input, raw, group, i);
            command[0] = 0;
            if (ok) ok = module_build_append(command, sizeof(command), "%s -x assembler", assemble) &&
                module_append_path_flag(command, sizeof(command), "", raw) &&
                module_append_path_flag(command, sizeof(command), "-o ", object);
            size_t words = ok ? module_assembler_argv(command, args, storage, sizeof(storage)) : 0;
            int64_t deadline;
            if (!words || !module_capture_deadline(&deadline)) goto failed;
            args[words] = "-###"; args[words + 1] = NULL;
            if (!module_process_output(args, report, sizeof(report), deadline, true, true)) goto failed;
            words = module_assembler_report(report, args, storage, sizeof(storage));
            if (!words || !module_assembler_tool_hash(&capture.hash, args[0], deadline)) goto failed;
            if (!integrated) {
                args[words] = "-###"; args[words + 1] = NULL;
                if (!module_process_output(args, report, sizeof(report), deadline, true, true)) goto failed;
                words = module_assembler_report(report, args, storage, sizeof(storage));
            }
            if (words < 2 || strcmp(args[1], "-cc1as") ||
                !module_assembler_tool_hash(&capture.hash, args[0], deadline)) goto failed;
            size_t format = 0, output = 0, sources = 0;
            for (size_t j = 1; j < words; j++) {
                if (!strcmp(args[j], "-filetype")) {
                    if (format || j + 1 == words || strcmp(args[j + 1], "obj")) goto failed;
                    format = j + 1;
                }
                if (!strcmp(args[j], "-o")) {
                    if (output || j + 1 == words || strcmp(args[j + 1], object)) goto failed;
                    output = j + 1;
                }
                if (!strcmp(args[j], raw)) sources++;
                hash_context_field(&capture.hash, !strcmp(args[j], raw) ? "@retained-input" :
                    !strcmp(args[j], object) ? "@private-output" : args[j]);
            }
            if (!format || !output || sources != 1) goto failed;
            const char *source = group ? meta->shared_c_sources[i] : meta->c_sources[i];
            if (module_source_kind(source) > 1) {
                /* I retain native debug emission before text expansion loses
                 * source locations. Final unit output copies these bytes. */
                char alias[2048], parent[2048], native[2048] = {0}, unit_prefix[4096];
                char native_storage[16384], *native_args[256];
                command[0] = 0;
                if (!module_capture_assembly_file(&capture, raw, frozen, false, 0) ||
                    !module_unit_input(meta, directory, group, i, alias, sizeof(alias), parent, sizeof(parent)) ||
                    !module_compile_prefix(meta, unit_prefix, sizeof(unit_prefix), MODULE_C_ASSEMBLE_UNIT, flags) ||
                    !module_build_append(native, sizeof(native), "%s/__native_unit_%zu_%zu.o", directory, group, i) ||
                    !module_build_append(command, sizeof(command), "%s -x assembler", unit_prefix) ||
                    !module_append_path_flag(command, sizeof(command), "", alias) ||
                    !module_append_path_flag(command, sizeof(command), "-o ", native)) goto failed;
                size_t native_words = module_assembler_argv(command, native_args, native_storage, sizeof(native_storage));
                if (!native_words) goto failed;
                report[0] = 0;
                char original[4096] = {0};
                bool native_ok = source[0] == '/' ? module_build_append(original, sizeof(original), "%s", source) :
                    module_build_append(original, sizeof(original), "%s/%s", meta->module_dir, source);
                native_ok = native_ok && module_clang_native_stdin(native_args, native_words, report, sizeof(report),
                    deadline, alias, original, native, &capture.hash, integrated);
                if (!native_ok) {
                    if (report[0]) fputs(report, stderr);
                    goto failed;
                }
                uint64_t object_hash = hash_file_fnv1a(native);
                if (!object_hash) goto failed;
                char digest[24];
                snprintf(digest, sizeof(digest), "%llu", (unsigned long long)object_hash);
                hash_context_field(&capture.hash, "native-unit-object-v1");
                hash_context_field(&capture.hash, digest);
            }
            args[format] = "asm"; args[output] = expanded;
            /* I name temporary labels in text mode only. Object assembly keeps
             * the selected assembler's original symbol-retention policy. */
            args[words] = "-msave-temp-labels"; args[words + 1] = NULL;
            report[0] = 0;
            if (!module_process_output(args, report, sizeof(report), deadline, true, false)) {
                if (report[0]) fputs(report, stderr);
                goto failed;
            }
            if (!module_capture_assembly_file(&capture, expanded, frozen, false, 0)) goto failed;
        }
    }
    return capture.hash;
failed:
    for (size_t group = 0; group < 2; group++) {
        size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
        for (size_t i = 0; i < count; i++) {
            char path[2048];
            snprintf(path, sizeof(path), "%s/__expanded_%zu_%zu.s", directory, group, i);
            (void)unlink(path);
            snprintf(path, sizeof(path), "%s/__snapshot_%zu_%zu.s", directory, group, i);
            (void)unlink(path);
            snprintf(path, sizeof(path), "%s/__native_unit_%zu_%zu.o", directory, group, i);
            (void)unlink(path);
        }
    }
    return 0;
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
        module_assembler_version_supported(version);
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
            const char *source = group ? meta->shared_c_sources[i] : meta->c_sources[i];
            char unit_prefix[4096], alias[2048] = {0}, parent[2048] = {0};
            const char *selected = assemble;
            command[0] = 0;
            ok = ok && module_build_append(input, sizeof(input), "%s/__snapshot_%zu_%zu.i", directory, group, i) &&
                module_build_append(parent, sizeof(parent), "%s", directory) &&
                module_build_append(assembly, sizeof(assembly), "%s/__snapshot_%zu_%zu.s", directory, group, i) &&
                module_build_append(object, sizeof(object), "%s/__as_capture_%zu_%zu.o", directory, group, i) &&
                module_build_append(record, sizeof(record), "%s/__as_read_%zu_%zu", directory, group, i) &&
                module_prepare_assembly(meta, retained, input, assembly, group, i);
            const char *selected_input = assembly;
            if (ok && module_source_kind(source) > 1) {
                ok = module_unit_input(meta, directory, group, i, alias, sizeof(alias), parent, sizeof(parent));
                selected = unit_prefix;
                selected_input = alias;
            }
            char descriptor[128] = {0}, named_input[2048] = {0};
            int fd = -1;
            if (ok && (module_source_kind(source) > 1 || strchr(parent, '='))) {
                fd = module_read_directory(parent, descriptor, sizeof(descriptor));
                ok = fd >= 0 && module_build_append(named_input, sizeof(named_input), "%s/%s",
                                                     descriptor, strrchr(selected_input, '/') + 1);
            }
            if (ok && module_source_kind(source) > 1)
                ok = module_unit_assembly_prefix(meta, flags, source, parent, unit_prefix, sizeof(unit_prefix),
                                                  fd >= 0 ? descriptor : NULL);
            if (ok) ok = module_read_command(command, sizeof(command), selected, directory, group, i,
                                             object, fd >= 0 ? named_input : selected_input, selected_input, true) &&
                (fd >= 0 ? module_read_execute(command) : !system(command));
            if (fd >= 0 && close(fd)) ok = false;
            unsigned captured = 0;
            uint64_t hash = 0;
            if (ok) ok = nac_load(record, selected_input, reads, &captured, &hash);
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

/* I rewrite only GCC's explicit PCH pragma. Original spellings and copied
 * bytes join identity; invocation-private destination names do not. */
static uint64_t module_snapshot_pch(const char *snapshot, const char *directory,
                                    size_t group, size_t index, uint64_t hash) {
    if (!directory || strpbrk(directory, "\"\\\n\r")) return 0;
    char temporary[2048] = {0};
    if (!module_build_append(temporary, sizeof(temporary), "%s.pch", snapshot)) return 0;
    FILE *input = fopen(snapshot, "rb"), *output = input ? fopen(temporary, "wx") : NULL;
    bool ok = input && output;
    bool created = output != NULL;
    char *line = NULL;
    size_t capacity = 0;
    ssize_t length;
    unsigned count = 0;
    ModuleAssemblyCapture capture = {directory, 0, 0, hash};
    hash_context_field(&capture.hash, "gcc-retained-pch-v1");
    static const char marker[] = "#pragma GCC pch_preprocess";
    while (ok && (length = getline(&line, &capacity, input)) >= 0) {
        if (memchr(line, 0, (size_t)length)) { ok = false; break; }
        char *pragma = strstr(line, marker);
        if (!pragma) { ok = fwrite(line, 1, (size_t)length, output) == (size_t)length; continue; }
        /* I decline noncanonical or escaped paths, never guess their meaning. */
        char *path = line + sizeof(marker) - 1;
        if (pragma != line || path[0] != ' ' || path[1] != '"') { ok = false; break; }
        path += 2;
        char *end = strchr(path, '"');
        if (!end || end == path || strspn(end + 1, "\r\n") != strlen(end + 1)) { ok = false; break; }
        *end = 0;
        if (strpbrk(path, "\\\n\r")) { ok = false; break; }
        char copied[2048] = {0};
        ok = module_build_append(copied, sizeof(copied), "%s/__pch_%zu_%zu_%u.gch", directory, group, index, count++) &&
            module_capture_assembly_file(&capture, path, copied, false, 0);
        if (ok) {
            hash_context_field(&capture.hash, path);
            ok = fprintf(output, "%s \"%s\"\n", marker, copied) >= 0;
        }
    }
    free(line);
    if (input) { if (ferror(input) || !feof(input)) ok = false; if (fclose(input)) ok = false; }
    if (output && fclose(output)) ok = false;
    if (!count) ok = false;
    if (ok) ok = rename(temporary, snapshot) == 0;
    if (!ok && created) (void)unlink(temporary);
    return ok ? capture.hash : 0;
}

/* Raw assembler has no compiler-generated depfile. I record its captured root
 * explicitly; nested reads are bound separately by assembler capture/replay. */
static uint64_t module_raw_assembly(const char *source, const char *snapshot, const char *dependency) {
    if (strpbrk(source, "\r\n")) return 0;
    ModuleAssemblyCapture copy = {NULL, 0, 0, 14695981039346656037ULL};
    if (!module_capture_assembly_file(&copy, source, snapshot, false, 0)) return 0;
    FILE *file = fopen(dependency, "wb");
    if (!file) return 0;
    bool ok = fputs("nano_module_dependencies: ", file) >= 0;
    for (const char *p = source; ok && *p; p++) {
        if (strchr(" \t#\\", *p)) ok = fputc('\\', file) != EOF;
        if (*p == '$') ok = fputc('$', file) != EOF;
        if (ok) ok = fputc(*p, file) != EOF;
    }
    if (fputc('\n', file) == EOF) ok = false;
    if (fclose(file)) ok = false;
    char trace[2060];
    int n = snprintf(trace, sizeof(trace), "%s.includes", dependency);
    if (n < 0 || (size_t)n >= sizeof(trace)) return 0;
    file = fopen(trace, "wb");
    if (!file) return 0;
    if (fclose(file)) ok = false;
    return ok ? copy.hash : 0;
}

/* I hash the retained input: Clang assembly or GCC preprocessed C. A later
 * capture must reproduce those bytes, not just hashes of restored source. */
static uint64_t module_snapshot_sources(ModuleBuildMetadata *meta,
                                       const ModulePkgFlags *flags, const char *directory,
                                       ModuleSnapshotMode mode, ModuleSnapshotMode *actual_mode) {
    if (actual_mode) *actual_mode = mode;
    char prefix[4096];
    bool assembly = mode == MODULE_SNAPSHOT_CLANG;
    if (!module_compile_prefix(meta, prefix, sizeof(prefix),
        assembly ? MODULE_C_EMIT_ASSEMBLY : mode == MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS ?
        MODULE_C_INTEGRATED_PREPROCESS : MODULE_C_PREPROCESS, flags)) return 0;
    if (mode == MODULE_SNAPSHOT_GCC &&
        !module_build_append(prefix, sizeof(prefix), " -fpch-preprocess")) return 0;
    uint64_t fingerprint = 14695981039346656037ULL;
    hash_context_field(&fingerprint, mode == MODULE_SNAPSHOT_GCC ? "gcc-retained-v1" :
        mode == MODULE_SNAPSHOT_CLANG_EXTERNAL ? "clang-external-retained-v1" :
        mode == MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS ? "clang-integrated-retained-v1" : "clang-assembly-v1");
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
            unsigned kind = module_source_kind(sources[i]);
            char unit_prefix[4096];
#ifdef __APPLE__
            /* Apple Clang preprocesses lowercase .s by default too. Retained
             * replay explicitly uses -x assembler to avoid doing that twice. */
            if (kind == 2 && (mode == MODULE_SNAPSHOT_CLANG_EXTERNAL ||
                              mode == MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS)) kind = 3;
#endif
            const char *selected_prefix = prefix;
            if (kind == 3) {
                if (!module_compile_prefix(meta, unit_prefix, sizeof(unit_prefix), MODULE_C_PREPROCESS, flags) ||
                    !module_build_append(unit_prefix, sizeof(unit_prefix), " -x assembler-with-cpp")) return 0;
                selected_prefix = unit_prefix;
            }
            bool ok = true;
            if (directory) {
                ok = module_build_append(snapshot, sizeof(snapshot), "%s/__snapshot_%zu_%zu.%s", directory, group, i,
                                         assembly ? "s" : "i");
                if (group) ok &= module_build_append(dependency, sizeof(dependency), "%s/__shared_%zu.d", directory, i);
                else if (count == 1) ok &= module_build_append(dependency, sizeof(dependency), "%s/%s.d", directory, meta->name);
                else ok &= module_build_append(dependency, sizeof(dependency), "%s/%s_%zu.d", directory, meta->name, i);
                if (kind == 2) {
                    char source[2048] = {0};
                    ok &= sources[i][0] == '/'
                        ? module_build_append(source, sizeof(source), "%s", sources[i])
                        : module_build_append(source, sizeof(source), "%s/%s", meta->module_dir, sources[i]);
                    uint64_t hash = ok ? module_raw_assembly(source, snapshot, dependency) : 0;
                    if (!hash) return 0;
                    char digest[24];
                    snprintf(digest, sizeof(digest), "%llu", (unsigned long long)hash);
                    hash_context_field(&fingerprint, sources[i]);
                    hash_context_field(&fingerprint, digest);
                    continue;
                }
                ok &= module_source_command(command, sizeof(command), selected_prefix, meta->module_dir,
                                            sources[i], "-", dependency, group != 0);
            } else {
                char source[2048] = {0};
                ok = sources[i][0] == '/'
                    ? module_build_append(source, sizeof(source), "%s", sources[i])
                    : module_build_append(source, sizeof(source), "%s/%s", meta->module_dir, sources[i]);
                ok &= module_build_append(command, sizeof(command), "%s%s -o -", selected_prefix,
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
            if (directory) status = module_source_diagnostics(status, dependency);
            if (output && fclose(output) != 0) ok = false;
            if (!ok || status != 0 || !nonempty) return 0;
            if (external_pch) {
                hash = module_snapshot_pch(snapshot, directory, group, i, hash);
                if (!hash) return 0;
            }
            char digest[24];
            snprintf(digest, sizeof(digest), "%llu", (unsigned long long)hash);
            hash_context_field(&fingerprint, sources[i]);
            hash_context_field(&fingerprint, digest);
        }
    }
    if (mode == MODULE_SNAPSHOT_GCC || mode == MODULE_SNAPSHOT_CLANG_EXTERNAL ||
        mode == MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS) {
        bool native_units = mode == MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS;
        /* Literal standalone replay needs representable private debug maps.
         * Clang native stdin and GNU descriptor replay preserve input naming. */
        if (directory) {
            char *cache_root = module_get_build_dir(meta->module_dir);
            if (!cache_root) return 0;
            const char *temporary = getenv("TMPDIR");
            bool descriptor_paths = strchr(directory, '=') || strchr(cache_root, '=') ||
                (temporary && strchr(temporary, '='));
            free(cache_root);
            for (size_t group = 0; group < 2; group++) {
                char **sources = group ? meta->shared_c_sources : meta->c_sources;
                size_t count = group ? meta->shared_c_sources_count : meta->c_sources_count;
                for (size_t i = 0; i < count; i++)
                    if (module_source_kind(sources[i]) > 1 &&
                        (descriptor_paths || strchr(meta->module_dir, '=') || strchr(sources[i], '=')))
                        native_units = true;
            }
        }
        uint64_t frozen = native_units ? 0 :
            module_gcc_capture_assembly(meta, flags, directory, fingerprint);
        if (frozen) {
            if (actual_mode) *actual_mode = MODULE_SNAPSHOT_GCC_ASSEMBLY;
            return frozen;
        }
        if (mode == MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS
#ifdef __APPLE__
            || mode == MODULE_SNAPSHOT_CLANG_EXTERNAL
#endif
        ) {
            frozen = module_clang_expansion(meta, flags, directory, fingerprint,
                                            mode == MODULE_SNAPSHOT_CLANG_INTEGRATED_UNITS);
            if (frozen) {
                if (actual_mode) *actual_mode = MODULE_SNAPSHOT_NATIVE_UNITS;
                return frozen;
            }
            return 0;
        }
#ifdef __linux__
        frozen = module_gcc_read_capture(meta, flags, directory, fingerprint);
        if (frozen) {
            if (actual_mode) *actual_mode = MODULE_SNAPSHOT_GCC_REPLAY;
            return frozen;
        }
#endif
        /* Retained C alone does not retain later assembler reads. Neither
         * driver may publish that incomplete snapshot after capture fails. */
        return 0;
    }
    return fingerprint;
}

static bool module_snapshot_command(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                    char *command, size_t capacity, const char *prefix,
                                    const char *directory, size_t group, size_t index,
                                    const char *object, ModuleSnapshotMode mode, const char *descriptor) {
    /* Standalone units bypass C lowering, so their debug selector belongs to
     * final assembly. C-generated assembly already contains its debug data. */
    const char *source = group ? meta->shared_c_sources[index] : meta->c_sources[index];
    char unit_prefix[4096], snapshot[2048] = {0}, parent[2048];
    bool assembly = mode != MODULE_SNAPSHOT_GCC;
    if (!module_build_append(snapshot, sizeof(snapshot), "%s/__snapshot_%zu_%zu.%s", directory, group, index,
                              assembly ? "s" : "i")) return false;
    if (module_source_kind(source) > 1 && mode != MODULE_SNAPSHOT_GCC) {
        if (!module_unit_input(meta, directory, group, index, snapshot, sizeof(snapshot), parent, sizeof(parent)) ||
            !module_unit_assembly_prefix(meta, flags, source, parent, unit_prefix, sizeof(unit_prefix), descriptor)) return false;
        prefix = unit_prefix;
    }
    if (mode == MODULE_SNAPSHOT_GCC_REPLAY) {
        char named_input[2048] = {0};
        if (descriptor && !module_build_append(named_input, sizeof(named_input), "%s/%s", descriptor,
                                               strrchr(snapshot, '/') + 1)) return false;
        return module_read_command(command, capacity, prefix, directory, group, index, object,
                                   descriptor ? named_input : snapshot, snapshot, false);
    }
    command[0] = 0;
    return module_build_append(command, capacity, "%s%s -x %s", prefix,
                            group && !assembly ? " -fvisibility=hidden" : "", assembly ? "assembler" : "cpp-output") &&
        module_append_path_flag(command, capacity, "", snapshot) &&
        module_append_path_flag(command, capacity, "-o ", object);
}

static void module_remove_staging(const char *stage);

/* I copy one captured object within the opened private stage. I neither
 * follow substituted files nor overwrite an existing output. */
static bool module_copy_native_unit(const char *directory, size_t group, size_t index,
                                    const char *object) {
    size_t length = strlen(directory);
    if (strncmp(object, directory, length) || object[length] != '/' ||
        !object[length + 1] || strchr(object + length + 1, '/') ||
        !strcmp(object + length + 1, ".") || !strcmp(object + length + 1, "..")) return false;
    int stage = open(directory, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (stage < 0) return false;
    char source[128];
    snprintf(source, sizeof(source), "__native_unit_%zu_%zu.o", group, index);
    int from = openat(stage, source, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    struct stat st;
    bool ok = from >= 0 && !fstat(from, &st) && S_ISREG(st.st_mode) &&
        st.st_size > 0 && st.st_size <= 32LL * 1024 * 1024;
    int to = ok ? openat(stage, object + length + 1,
                         O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600) : -1;
    ok = ok && to >= 0;
    size_t total = 0;
    char bytes[8192];
    while (ok) {
        ssize_t amount = read(from, bytes, sizeof(bytes));
        if (amount < 0 && errno == EINTR) continue;
        if (amount < 0) { ok = false; break; }
        if (!amount) break;
        if ((size_t)amount > 32ULL * 1024 * 1024 - total) { ok = false; break; }
        total += (size_t)amount;
        size_t sent = 0;
        while (sent < (size_t)amount) {
            ssize_t written = write(to, bytes + sent, (size_t)amount - sent);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) { ok = false; break; }
            sent += (size_t)written;
        }
    }
    if (ok) ok = total == (uint64_t)st.st_size && !fchmod(to, 0400);
    if (from >= 0 && close(from)) ok = false;
    if (to >= 0 && close(to)) ok = false;
    if (!ok && to >= 0) (void)unlinkat(stage, object + length + 1, 0);
    if (close(stage)) ok = false;
    return ok;
}

static int module_execute_unit(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                char *command, size_t capacity, const char *prefix,
                                const char *directory, size_t group, size_t index,
                                const char *object, ModuleSnapshotMode mode, const char *dependency) {
    const char *source = group ? meta->shared_c_sources[index] : meta->c_sources[index];
    if (mode == MODULE_SNAPSHOT_NATIVE_UNITS && module_source_kind(source) > 1) {
        if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD"))
            printf("[Module] I copy captured native unit %s\n", source);
        return module_copy_native_unit(directory, group, index, object) ? 0 : -1;
    }
#ifdef __linux__
    if (mode == MODULE_SNAPSHOT_GCC_REPLAY && (module_source_kind(source) > 1 || strchr(directory, '='))) {
        char parent[2048] = {0}, descriptor[128];
        bool named = module_source_kind(source) > 1 ?
            module_build_append(parent, sizeof(parent), "%s/__unit_%zu_%zu", directory, group, index) :
            module_build_append(parent, sizeof(parent), "%s", directory);
        if (!named) return -1;
        int fd = module_read_directory(parent, descriptor, sizeof(descriptor));
        if (fd < 0) return -1;
        bool ok = module_snapshot_command(meta, flags, command, capacity, prefix, directory, group, index,
                                           object, mode, descriptor);
        if (ok && (module_builder_verbose || getenv("NANO_VERBOSE_BUILD"))) printf("[Module] %s\n", command);
        if (ok) ok = module_read_execute(command);
        if (close(fd)) ok = false;
        int status = ok ? 0 : -1;
        return dependency ? module_source_diagnostics(status, dependency) : status;
    }
#endif
    if (mode != MODULE_SNAPSHOT_NONE &&
        !module_snapshot_command(meta, flags, command, capacity, prefix, directory, group, index, object, mode, NULL)) return -1;
    if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) printf("[Module] %s\n", command);
    if (dependency) return module_run_source_command(command, dependency);
    return module_build_append(command, capacity, " 2>/dev/null") ? system(command) : -1;
}

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
            if (compile && module_execute_unit(meta, flags, command, sizeof(command), prefix, directory,
                                              group, i, object, mode, NULL)) return 0;
            uint64_t hash = hash_file_fnv1a(object);
            if (!hash) return 0;
            char digest[24];
            snprintf(digest, sizeof(digest), "%llu", (unsigned long long)hash);
            hash_context_field(&fingerprint, digest);
        }
    }
    return fingerprint;
}

static uint64_t module_gcc_validation(ModuleBuildMetadata *meta, const ModulePkgFlags *flags, ModuleSnapshotMode mode) {
    const char *temporary = getenv("TMPDIR");
    if (!temporary || !*temporary) temporary = "/tmp";
    char directory[2048] = {0};
    if (!module_build_append(directory, sizeof(directory), "%s/nano-gcc-check-XXXXXX", temporary) ||
        !mkdtemp(directory)) return 0;
    uint64_t fingerprint = module_snapshot_sources(meta, flags, directory, mode, &mode);
    fingerprint = module_gcc_objects(meta, flags, directory, fingerprint, true, mode);
    module_remove_staging(directory);
    return fingerprint;
}

/* I use one link recipe for publication and Linux warm validation. */
static bool module_append_link_cflags(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                      const char *fragment, bool query, char *output, size_t capacity) {
    char *selected = module_link_cflags(fragment);
    bool ok = selected && (query ? module_build_append(output, capacity, " %s", selected) :
        module_append_compiler_fragment(meta, flags, selected, false, output, capacity));
    free(selected);
    return ok;
}

static bool module_shared_link_command(ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                      const char *object_file, const char *shared_lib,
                                      const char *build_dir, char *lib_cmd, size_t capacity) {
    bool command_ok = true;
    bool query = shared_lib == NULL;
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
    if (shared_lib) command_ok &= module_append_path_flag(lib_cmd, capacity, "-o ", shared_lib);
    if (object_file) command_ok &= module_append_path_flag(lib_cmd, capacity, "", object_file);
    else return false;
    /* I preserve package order while selecting link-phase arguments. */
    for (size_t i = 0; i < meta->pkg_config_count; i++) {
#ifdef __APPLE__
        if (module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
        if (!flags->cflags[i]) return false;
        if (!flags->cflags[i][0]) continue;
        command_ok &= module_append_link_cflags(meta, flags, flags->cflags[i], query, lib_cmd, capacity);
    }

    char *fragment = module_shared_link_fragment(meta, flags);
    char *transport = fragment ? (query ? strdup(fragment) : module_shared_link_transport(meta, flags, fragment)) : NULL;
    command_ok &= transport && module_build_append(lib_cmd, capacity, " %s", transport);
    free(transport);
    free(fragment);
    /* Add custom cflags (all platforms) */
    for (size_t i = 0; i < meta->cflags_count; i++) {
        if (!meta->cflags[i][0]) continue;
        command_ok &= module_append_link_cflags(meta, flags, meta->cflags[i], query, lib_cmd, capacity);
    }
    /* Add platform-specific cflags */
#ifdef __APPLE__
    for (size_t i = 0; i < meta->cflags_macos_count; i++) {
        if (!meta->cflags_macos[i][0]) continue;
        command_ok &= module_append_link_cflags(meta, flags, meta->cflags_macos[i], query, lib_cmd, capacity);
    }
#elif defined(__FreeBSD__)
    for (size_t i = 0; i < meta->cflags_freebsd_count; i++) {
        if (!meta->cflags_freebsd[i][0]) continue;
        command_ok &= module_append_link_cflags(meta, flags, meta->cflags_freebsd[i], query, lib_cmd, capacity);
    }
#else
    for (size_t i = 0; i < meta->cflags_linux_count; i++) {
        if (!meta->cflags_linux[i][0]) continue;
        command_ok &= module_append_link_cflags(meta, flags, meta->cflags_linux[i], query, lib_cmd, capacity);
    }
#endif
    for (size_t i = 0; !query && i < meta->shared_c_sources_count; i++) {
        char object[2048] = {0};
        command_ok &= module_build_append(object, sizeof(object), "%s/__shared_%zu.o", build_dir, i);
        command_ok &= module_append_path_flag(lib_cmd, capacity, "", object);
    }
    return command_ok;
}

/* I pin the primary output at both layers without dropping selection flags.
 * The caller still admits auxiliary output options and indirect controls; this
 * is not a filesystem sandbox for arbitrary compiler/linker arguments. */
static ModuleLinkResponseGrammar module_query_link_response_unchecked(const char *command, const char *parent) {
    if (!command || !parent || strnlen(command, 65537) > 65536) return 0;
    const char *cursor = command;
    char word[4096];
    size_t words = 0;
    int status;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
        if (!words++ && !word[0]) return 0;
        /* End-of-options would prevent the appended output pins taking effect. */
        if (!strcmp(word, "--")) return 0;
        if (!strncmp(word, "-Wl,", 4)) {
            const char *part = word + 4;
            while (*part) {
                size_t length = strcspn(part, ",");
                if (length == 2 && !memcmp(part, "--", 2)) return 0;
                part += length;
                if (*part) part++;
            }
        }
    }
    if (status < 0 || !words) return 0;
    char *resolved = realpath(parent, NULL);
    char directory[2048] = {0}, output[2048] = {0};
    bool ok = resolved && module_build_append(directory, sizeof(directory), "%s/.nano-link-query-XXXXXX", resolved);
    free(resolved);
    if (!ok || !mkdtemp(directory)) return 0;
    char *query = calloc(65537, 1);
    ok = query && module_build_append(output, sizeof(output), "%s/probe.so", directory) &&
        module_build_append(query, 65537, "%s", command) &&
        module_append_path_flag(query, 65537, "-o ", output) &&
        module_build_append(query, 65537, " -Xlinker -o") &&
        module_append_path_flag(query, 65537, "-Xlinker ", output);
    ModuleLinkResponseGrammar grammar = ok ? module_link_response_grammar_command(query) : 0;
    free(query);
    module_remove_staging(directory);
    struct stat st;
    if (lstat(directory, &st) == 0 || errno != ENOENT) return 0;
    return grammar;
}

/* I admit option controls, not arbitrary tool or native-input side effects.
 * A positive result never substitutes for trusting the configured toolchain.
 * I use explicit forms instead of broad prefixes such as -l*, which would
 * accidentally include Apple's -lto_library plugin control. */
static bool module_link_query_option(const char *word, bool *operand) {
    if (!word[0] || word[0] == '@' || !strcmp(word, "--")) return false;
    if (*operand) {
        *operand = false;
        return word[0] != '-';
    }
    const char *paired[] = {"-L", "-l", "-o", "-rpath", "-soname", "-install_name",
        "-undefined", "-arch", "-syslibroot", "-e", "-u", "-framework", "-F"};
    for (size_t i = 0; i < sizeof(paired) / sizeof(paired[0]); i++) {
        if (!strcmp(word, paired[i])) { *operand = true; return true; }
    }
    const char *plain[] = {"-shared", "-dylib", "-static", "-Bstatic", "-Bdynamic", "-Bsymbolic",
        "-Bsymbolic-functions", "--as-needed", "--no-as-needed", "--whole-archive", "--no-whole-archive",
        "--start-group", "--end-group", "--allow-shlib-undefined", "--no-undefined", "-dead_strip",
        "--gc-sections", "-lm", "-lc", "-ldl", "-lpthread"};
    for (size_t i = 0; i < sizeof(plain) / sizeof(plain[0]); i++)
        if (!strcmp(word, plain[i])) return true;
    if ((!strncmp(word, "-L", 2) || !strncmp(word, "-F", 2)) && word[2]) return true;
    /* A positional native input can itself contain linker controls. I do not
     * certify its contents here or turn this check into a filesystem sandbox. */
    return word[0] != '-';
}

static bool module_link_query_options_admitted(const char *command) {
    if (!command || strnlen(command, 65537) > 65536) return false;
    const char *cursor = command;
    char word[4096], value[4096];
    if (module_flag_word(&cursor, word, sizeof(word)) != 1 || !word[0]) return false;
    bool linker_operand = false;
    size_t count = 1;
    int status;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
        if (++count > 2048 || !word[0] || word[0] == '@' || !strcmp(word, "--")) return false;
        if (!strcmp(word, "-Xlinker")) {
            if (++count > 2048 || module_flag_word(&cursor, value, sizeof(value)) != 1 ||
                !module_link_query_option(value, &linker_operand)) return false;
            continue;
        }
        if (!strncmp(word, "-Wl,", 4)) {
            char *part = word + 4;
            do {
                char *comma = strchr(part, ',');
                if (comma) *comma = 0;
                if (!module_link_query_option(part, &linker_operand)) return false;
                part = comma ? comma + 1 : NULL;
            } while (part);
            continue;
        }
        /* I do not guess driver reordering for a dangling linker operand. */
        if (linker_operand) return false;
        if (!strcmp(word, "-Xassembler")) {
            char third[4096], fourth[4096];
            bool paired;
            if (!module_assembler_argument(&cursor, value, third, fourth, sizeof(value), &paired)) return false;
            count += paired ? 3 : 1;
            if (count > 2048) return false;
            continue;
        }
        if (!strcmp(word, "-x")) {
            if (++count > 2048 || module_flag_word(&cursor, value, sizeof(value)) != 1 ||
                (strcmp(value, "c") && strcmp(value, "none"))) return false;
            continue;
        }
        const char *paired[] = {"-o", "-B", "-target", "--target", "-arch", "-isysroot", "--sysroot",
            "-D", "-U", "-I", "-L", "-l", "-F", "-framework", "-undefined"};
        bool consumes = false;
        for (size_t i = 0; i < sizeof(paired) / sizeof(paired[0]); i++)
            if (!strcmp(word, paired[i])) { consumes = true; break; }
        if (consumes) {
            if (++count > 2048 || module_flag_word(&cursor, value, sizeof(value)) != 1 ||
                !value[0] || value[0] == '@' || value[0] == '-') return false;
            continue;
        }
        if (module_snapshot_flag(word) != MODULE_FLAG_UNKNOWN) continue;
        const char *plain[] = {"-shared", "-dynamiclib", "-pthread", "-m32", "-m64",
            "-lm", "-lc", "-ldl", "-lpthread"};
        bool admitted = false;
        for (size_t i = 0; i < sizeof(plain) / sizeof(plain[0]); i++)
            if (!strcmp(word, plain[i])) { admitted = true; break; }
        const char *joined[] = {"-B", "-L", "-F", "-fuse-ld=", "--target=", "--sysroot="};
        for (size_t i = 0; i < sizeof(joined) / sizeof(joined[0]); i++) {
            size_t length = strlen(joined[i]);
            if (!strncmp(word, joined[i], length) && word[length]) { admitted = true; break; }
        }
        if (!admitted && word[0] == '-') return false;
    }
    return status == 0 && !linker_operand;
}

ModuleLinkResponseGrammar module_query_link_response_grammar(const char *command, const char *parent) {
    if (!module_link_query_options_admitted(command)) return 0;
    return module_query_link_response_unchecked(command, parent);
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

/* I size returned compiler flags from metadata, not a fixed pointer budget.
 * The same owner and failure path serve source-free and compiled modules. */
static bool module_collect_compile_flags(ModuleBuildInfo *info, const ModuleBuildMetadata *meta,
                                          const ModulePkgFlags *flags) {
    size_t platform_count;
    char **platform = module_platform_cflags(meta, &platform_count);
    size_t counts[] = {flags->count, meta->include_dirs_count, meta->cflags_count, platform_count};
    size_t capacity = 0;
    for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
        if (counts[i] > SIZE_MAX - capacity) return false;
        capacity += counts[i];
    }
    if (capacity > SIZE_MAX / sizeof(char *)) return false;
    char **collected = capacity ? calloc(capacity, sizeof(char *)) : NULL;
    if (capacity && !collected) return false;
    size_t count = 0;
    for (size_t group = 0; group < 4; group++) {
        if (group == 1 && counts[group]) {
            if (!module_include_flags(meta, collected + count)) goto failed;
            count += counts[group];
            continue;
        }
        char **values = group == 0 ? flags->cflags : group == 1 ? meta->include_dirs :
            group == 2 ? meta->cflags : platform;
        for (size_t i = 0; i < counts[group]; i++) {
#ifdef __APPLE__
            if (group == 0 && module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
            if (!values[i]) goto failed;
            if (group == 0 && !values[i][0]) continue;
            char *value = strdup(values[i]);
            if (!value) goto failed;
            collected[count++] = value;
        }
    }
    info->compile_flags = collected;
    info->compile_flags_count = count;
    return true;
failed:
    for (size_t i = 0; i < capacity; i++) free(collected[i]);
    free(collected);
    return false;
}

static bool module_collect_link_flags(ModuleBuildInfo *info, const ModuleBuildMetadata *meta,
                                       const ModulePkgFlags *flags) {
    size_t platform_count;
    char **platform = module_platform_ldflags(meta, &platform_count);
    size_t framework_words = 0;
#ifdef __APPLE__
    if (meta->frameworks_count > SIZE_MAX / 2) return false;
    framework_words = meta->frameworks_count * 2;
#endif
    size_t counts[] = {info->object_file ? 1 : 0, flags->count, meta->ldflags_count,
                       platform_count, framework_words, meta->system_libs_count};
    size_t capacity = 0;
    for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
        if (counts[i] > SIZE_MAX - capacity) return false;
        capacity += counts[i];
    }
    if (capacity > SIZE_MAX / sizeof(char *)) return false;
    char **collected = capacity ? calloc(capacity, sizeof(char *)) : NULL;
    if (capacity && !collected) return false;
    size_t count = 0;
    for (size_t group = 0; group < 6; group++) {
        for (size_t i = 0; i < counts[group]; i++) {
#ifdef __APPLE__
            if (group == 1 && module_pkg_is_native_framework(meta, meta->pkg_config[i])) continue;
#endif
            const char *source = group == 0 ? info->object_file : group == 1 ? flags->libs[i] :
                group == 2 ? meta->ldflags[i] : group == 3 ? platform[i] :
                group == 4 ? (i % 2 ? meta->frameworks[i / 2] : "-framework") : meta->system_libs[i];
            if (!source) goto failed;
            if (group == 1 && !source[0]) continue;
            char *value;
            if (group == 5) {
                size_t length = strlen(source);
                if (length > SIZE_MAX - 3) goto failed;
                value = malloc(length + 3);
                if (value) { memcpy(value, "-l", 2); memcpy(value + 2, source, length + 1); }
            } else value = strdup(source);
            if (!value) goto failed;
            collected[count++] = value;
        }
    }
    info->link_flags = collected;
    info->link_flags_count = count;
    return true;
failed:
    for (size_t i = 0; i < count; i++) free(collected[i]);
    free(collected);
    return false;
}

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

        if (!module_collect_link_flags(info, meta, flags) || !module_collect_compile_flags(info, meta, flags)) {
            module_build_info_free(info);
            return NULL;
        }
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
        bool external_capture = mode == MODULE_SNAPSHOT_CLANG_EXTERNAL;
        bool admitted = mode != MODULE_SNAPSHOT_NONE;
        bool snapshots = admitted;
        if (preprocessing_before) *preprocessing_before = snapshots
            ? module_snapshot_sources(meta, flags, build_dir, mode, &mode) : module_preprocess_fingerprint(meta, flags);
        snapshots = snapshots && *preprocessing_before;
        if (admitted && !snapshots) {
            fprintf(stderr, "I could not retain %sinputs for %s; I will not compile live sources after capture failure\n",
                    external_capture ? "external-assembler " : "compiler and assembler ",
                    meta->name ? meta->name : "unknown");
            free(build_dir);
            return NULL;
        }
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
            int result = command_ok ? module_execute_unit(meta, flags, compile_cmd, sizeof(compile_cmd),
                compile_prefix, build_dir, 0, 0, object_file, snapshots ? mode : MODULE_SNAPSHOT_NONE, dep_path) : -1;
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
                int result = command_ok ? module_execute_unit(meta, flags, compile_cmd, sizeof(compile_cmd),
                    compile_prefix, build_dir, 0, i, obj_path, snapshots ? mode : MODULE_SNAPSHOT_NONE, dep_path) : -1;
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
                    if (!command_ok || module_execute_unit(meta, flags, sc_cmd, sizeof(sc_cmd), compile_prefix,
                            build_dir, 1, sci, sc_obj, snapshots ? mode : MODULE_SNAPSHOT_NONE, sc_dep) != 0) {
                        fprintf(stderr,
                                "I could not compile shared_c_source %s for %s\n",
                                meta->shared_c_sources[sci], meta->name);
                        free(build_dir);
                        return NULL;
                    }
                }
            }

            if (snapshots && (mode == MODULE_SNAPSHOT_GCC || mode == MODULE_SNAPSHOT_GCC_ASSEMBLY ||
                              mode == MODULE_SNAPSHOT_GCC_REPLAY || mode == MODULE_SNAPSHOT_NATIVE_UNITS))
                *preprocessing_before = module_gcc_objects(meta, flags, build_dir, *preprocessing_before, false, mode);

            /* Build shared library */
            if (module_builder_verbose || getenv("NANO_VERBOSE_BUILD")) {
                printf("[Module] Building shared library: %s\n", lib_cmd);
            }
            
            int lib_result = -1;
#ifdef __APPLE__
            /* I preserve an ordinary link when dependency capture is not
             * supported. I admit my retained compiler argument transports,
             * but not indirect user response inputs hidden from this format. */
            char recorded_command[8192] = {0}, link_record[2048] = {0};
            bool capture = command_ok && link_observation && module_link_response_safe(meta, flags, lib_cmd) &&
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

    if (!info->object_file || !module_collect_link_flags(info, meta, flags) ||
        !module_collect_compile_flags(info, meta, flags)) {
        module_build_info_free(info);
        return NULL;
    }

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

/* I remove only my reserved one-level alias directories. I never traverse
 * substituted symlinks or make arbitrary nested trees publishable. */
static bool module_remove_unit_aliases_fd(int parent) {
    int fd = openat(parent, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    DIR *dir = fdopendir(fd);
    if (!dir) { close(fd); return false; }
    bool ok = true;
    struct dirent *entry;
    errno = 0;
    while ((entry = readdir(dir))) {
        const char *name = entry->d_name;
        if (strncmp(name, "__unit_", 7) || (name[7] != '0' && name[7] != '1') || name[8] != '_' ||
            !name[9] || strspn(name + 9, "0123456789") != strlen(name + 9)) continue;
        int child = openat(fd, name, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (child < 0) {
            if (unlinkat(fd, name, 0) != 0) ok = false;
            errno = 0;
            continue;
        }
        DIR *contents = fdopendir(child);
        if (!contents) { close(child); ok = false; errno = 0; continue; }
        struct dirent *file;
        errno = 0;
        while ((file = readdir(contents))) {
            if (!strcmp(file->d_name, ".") || !strcmp(file->d_name, "..")) continue;
            if (unlinkat(child, file->d_name, 0) != 0) ok = false;
            errno = 0;
        }
        if (errno) ok = false;
        if (closedir(contents) != 0) ok = false;
        if (unlinkat(fd, name, AT_REMOVEDIR) != 0) ok = false;
        errno = 0;
    }
    if (errno) ok = false;
    if (closedir(dir) != 0) ok = false;
    return ok;
}

static bool module_remove_unit_aliases(const char *stage) {
    int fd = open(stage, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return errno == ENOENT;
    bool ok = module_remove_unit_aliases_fd(fd);
    if (close(fd) != 0) ok = false;
    return ok;
}

static void module_remove_staging(const char *stage) {
    /* I never follow a substituted staging symlink. All entry removal stays
     * relative to this descriptor even if the directory is renamed. */
    int fd = open(stage, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        if (errno != ENOENT) fprintf(stderr, "I retained private build files in %s\n", stage);
        return;
    }
    (void)module_remove_unit_aliases_fd(fd);
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
    if (!module_ensure_build_dir(meta->module_dir)) {
        module_trace_evidence("build-cache-directory", 1, 0, false);
        return NULL;
    }
    char *cache_root = module_get_build_dir(meta->module_dir);
    char *cache = cache_root ? realpath(cache_root, NULL) : NULL;
    free(cache_root);
    if (!cache) {
        module_trace_evidence("build-cache-path", 1, 0, false);
        return NULL;
    }
    char lock_path[2048], stage[2048];
    int l = snprintf(lock_path, sizeof(lock_path), "%s/.build.lock", cache);
    int s = snprintf(stage, sizeof(stage), "%s/.nano-build-XXXXXX", cache);
    if (l < 0 || (size_t)l >= sizeof(lock_path) || s < 0 || (size_t)s >= sizeof(stage)) {
        module_trace_evidence("build-cache-path-capacity", 1, 0, false);
        free(cache);
        return NULL;
    }
    int fd = open(lock_path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) {
        module_trace_evidence("build-lock-open", 1, 0, false);
        free(cache); return NULL;
    }
    int locked = flock(fd, LOCK_EX | LOCK_NB);
    if (locked < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
        if (getenv("NANO_VERBOSE_BUILD")) fprintf(stderr, "I wait for the C-library cache lock: %s\n", cache);
        do { locked = flock(fd, LOCK_EX); } while (locked < 0 && errno == EINTR);
    }
    ModuleBuildInfo *info = NULL;
    bool staged = locked == 0 && mkdtemp(stage);
    if (!staged)
        module_trace_evidence(locked != 0 ? "build-lock-acquire" : "build-stage-create", 1, 0, false);
    if (staged) {
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
                module_remove_unit_aliases(stage) &&
                module_validate_artifacts(stage, meta);
            uint64_t preprocessing_after = ok && context_before && preprocessing_before
                ? module_preprocess_fingerprint(meta, flags->linker_grammar ? flags : NULL) : 0;
            bool same_inputs = ok && context_before && preprocessing_before &&
                preprocessing_before == preprocessing_after;
            module_trace_evidence("publish-preprocessing", preprocessing_before, preprocessing_after, same_inputs);
            uint64_t context_after = same_inputs ? module_build_context(meta) : 0;
            module_trace_evidence("publish-context", context_before, context_after,
                                 same_inputs && context_before == context_after);
            if (same_inputs && context_before == context_after)
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

typedef struct {
    char *sources[64];
    char **arguments;
    size_t count, next;
} ModuleForwardedRoots;

static char **module_forwarded_group(const ModuleBuildMetadata *meta, const ModulePkgFlags *flags,
                                     size_t group, size_t *count) {
    if (group < 4) return module_response_group(meta, group, count);
    *count = flags->count;
    return group == 4 ? flags->cflags : flags->libs;
}

/* A NULL output collects roots without opening them. The second pass installs
 * their complete owned argument fragments, preserving every field's position. */
static bool module_forwarded_fragment(const char *fragment, ModuleForwardedRoots *roots, char *output) {
    const char *cursor = fragment;
    char word[4096];
    int status;
    while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
        if (word[0] == '@' || strstr(word, "--driver-mode")) { errno = EINVAL; return false; }
        if (!strncmp(word, "-Wl,", 4)) {
            char *part = word + 4;
            do {
                char *comma = strchr(part, ',');
                if (comma) *comma = 0;
                if (!part[0]) { errno = EINVAL; return false; }
                if (part[0] == '@') {
                    if (!part[1]) { errno = EINVAL; return false; }
                    if (!output) {
                        if (roots->count == 64) { errno = E2BIG; return false; }
                        char *source = strdup(part + 1);
                        if (!source) return false;
                        roots->sources[roots->count++] = source;
                    } else if (roots->next >= roots->count ||
                        !module_build_append(output, 65537, "%s", roots->arguments[roots->next++])) {
                        errno = E2BIG; return false;
                    }
                } else if (output && !module_append_path_flag(output, 65537, "-Xlinker ", part)) return false;
                part = comma ? comma + 1 : NULL;
            } while (part);
        } else if (output && !module_append_path_flag(output, 65537, "", word)) return false;
    }
    if (status < 0) errno = EINVAL;
    return status == 0;
}

/* This is the shared link's contribution order, not the metadata struct order. */
static const size_t module_forwarded_order[] = {4, 5, 2, 3, 0, 1};

static bool module_forwarded_candidate(const ModuleBuildMetadata *base, const ModulePkgFlags *original,
                                       ModuleBuildMetadata *meta, ModulePkgFlags *flags, ModuleForwardedRoots *roots) {
    *meta = *base;
    memset(flags, 0, sizeof(*flags));
    flags->count = original->count;
    for (size_t group = 0; group < 4; group++) {
        size_t count;
        (void)module_response_group(base, group, &count);
        char ***slot = module_response_group_slot(meta, group);
        *slot = count ? calloc(count, sizeof(char *)) : NULL;
        if (count && !*slot) return false;
    }
    if (flags->count) {
        flags->cflags = calloc(flags->count, sizeof(char *));
        flags->libs = calloc(flags->count, sizeof(char *));
        if (!flags->cflags || !flags->libs) return false;
    }
    char *output = calloc(65537, 1);
    if (!output) return false;
    size_t bytes = 0;
    bool ok = true;
    roots->next = 0;
    for (size_t index = 0; index < 6 && ok; index++) {
        size_t count, ignored, group = module_forwarded_order[index];
        char **source = module_forwarded_group(base, original, group, &count);
        char **target = module_forwarded_group(meta, flags, group, &ignored);
        for (size_t i = 0; i < count && ok; i++) {
            if (!source[i]) continue;
            output[0] = 0;
            ok = module_forwarded_fragment(source[i], roots, output);
            size_t length = strlen(output);
            if (length > 65536 - bytes) { errno = E2BIG; ok = false; }
            if (ok) { bytes += length; ok = (target[i] = strdup(output)) != NULL; }
        }
    }
    free(output);
    return ok && roots->next == roots->count;
}

static bool module_install_forwarded(const ModuleBuildMetadata *original, ModuleBuildMetadata *meta,
                                     ModulePkgFlags *flags) {
    ModuleForwardedRoots roots = {0};
    size_t bytes = 0, slots = 0;
    bool literal = true;
    for (size_t index = 0; index < 6 && literal; index++) {
        size_t count;
        char **source = module_forwarded_group(meta, flags, module_forwarded_order[index], &count);
        if (count > 2048 - slots) { literal = false; break; }
        slots += count;
        for (size_t i = 0; i < count && literal; i++) {
            if (!source[i]) continue;
            size_t length = strnlen(source[i], 65537);
            if (length > 65536 - bytes) { literal = false; break; }
            bytes += length;
            literal = module_forwarded_fragment(source[i], &roots, NULL);
        }
    }
    if (!literal || !roots.count || !module_response_driver(meta)) {
        for (size_t i = 0; i < roots.count; i++) free(roots.sources[i]);
        return true; /* I leave the complete previous path intact on decline. */
    }
    ModuleBuildMetadata candidate[2] = {*meta, *meta};
    ModulePkgFlags packages[2] = {{0}, {0}};
    bool ready[2] = {false, false};
    int capture_error[2] = {0, 0};
    /* Both candidates own their selected inputs before either tool query can
     * mutate an original response. I do not claim one filesystem-wide epoch. */
    for (size_t index = 0; index < 2; index++) {
        roots.arguments = module_capture_link_arguments((const char *const *)roots.sources,
            roots.count, index == 0 ? MODULE_LINK_RESPONSE_GNU : MODULE_LINK_RESPONSE_APPLE);
        if (!roots.arguments) { capture_error[index] = errno; continue; }
        ready[index] = module_forwarded_candidate(meta, flags, &candidate[index], &packages[index], &roots);
        for (size_t i = 0; i < roots.count; i++) free(roots.arguments[i]);
        free(roots.arguments);
    }
    for (size_t i = 0; i < roots.count; i++) free(roots.sources[i]);
    int chosen = -1;
    bool ok = true;
    char *parent = NULL, *command = NULL;
    char source_dir[2048] = {0}, source_file[2048] = {0};
    if ((ready[0] || ready[1]) && module_ensure_build_dir(meta->module_dir)) {
        parent = module_get_build_dir(meta->module_dir);
        command = calloc(65537, 1);
    }
    /* An empty .c input needs no -x reset. Clang rejects a trailing -x none
     * under -Werror; I do not weaken the user's warning policy to query ld. */
    bool source_owned = parent && command &&
        module_build_append(source_dir, sizeof(source_dir), "%s/.nano-link-source-XXXXXX", parent) &&
        mkdtemp(source_dir);
    bool source_ready = source_owned &&
        module_build_append(source_file, sizeof(source_file), "%s/probe.c", source_dir);
    if (source_ready) {
        int fd = open(source_file, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        source_ready = fd >= 0;
        if (fd >= 0 && close(fd)) source_ready = false;
    }
    for (size_t index = 0; source_ready && index < 2; index++) {
        if (!ready[index]) continue;
        if (!module_shared_link_command(&candidate[index], &packages[index], source_file, NULL, NULL, command, 65537)) continue;
        ModuleLinkResponseGrammar grammar = module_query_link_response_grammar(command, parent);
        if (grammar == (index == 0 ? MODULE_LINK_RESPONSE_GNU : MODULE_LINK_RESPONSE_APPLE)) {
            chosen = (int)index;
            packages[index].linker_grammar = grammar;
            break;
        }
        if (grammar == MODULE_LINK_RESPONSE_APPLE && capture_error[1] == ELOOP) {
            fprintf(stderr, "I reject repeated or cyclic response identities for the selected Apple linker\n");
            ok = false;
            break;
        }
    }
    if (source_owned) module_remove_staging(source_dir);
    free(parent);
    free(command);
    for (size_t index = 0; index < 2; index++) {
        if ((int)index == chosen) continue;
        module_response_metadata_free(meta, &candidate[index]);
        module_pkg_flags_free(&packages[index]);
    }
    if (chosen >= 0) {
        module_response_metadata_free(original, meta);
        *meta = candidate[chosen];
        module_pkg_flags_free(flags);
        *flags = packages[chosen];
    }
    return ok;
}

static bool module_capture_invocation(const ModuleBuildMetadata *meta, ModuleBuildMetadata *captured,
                                      ModulePkgFlags *flags) {
    if (!module_response_metadata(meta, captured)) return false;
    if (!module_pkg_flags_capture(captured, flags)) {
        module_response_metadata_free(meta, captured);
        return false;
    }
    for (size_t i = 0; i < flags->count; i++) {
        if (module_response_pending(flags->cflags[i]) || module_response_pending(flags->libs[i])) {
            module_response_metadata_free(meta, captured);
            *captured = *meta;
            break;
        }
    }
    if (!module_install_forwarded(meta, captured, flags)) {
        module_response_metadata_free(meta, captured);
        module_pkg_flags_free(flags);
        return false;
    }
    return true;
}

ModuleBuildInfo* module_build(ModuleBuilder *builder, ModuleBuildMetadata *meta) {
    if (!meta || !ensure_module_system_deps(meta)) {
        module_trace_evidence("build-system-dependencies", 1, 0, false);
        return NULL;
    }
    ModuleBuildMetadata captured;
    ModulePkgFlags flags;
    if (!module_capture_invocation(meta, &captured, &flags)) {
        module_trace_evidence("build-invocation", 1, 0, false);
        return NULL;
    }
    ModuleBuildInfo *info = module_build_with_flags(builder, &captured, &flags);
    if (info) {
        info->module_dir = realpath(meta->module_dir, NULL);
        if (!info->module_dir) { module_build_info_free(info); info = NULL; }
    }
    for (size_t i = 0; info && i < info->compile_flags_count; i++) {
        char *transport = module_response_transport(&captured, &flags, info->compile_flags[i]);
        if (!transport) { module_build_info_free(info); info = NULL; break; }
        free(info->compile_flags[i]);
        info->compile_flags[i] = transport;
    }
    module_pkg_flags_free(&flags);
    module_response_metadata_free(meta, &captured);
    if (!info) module_trace_evidence("build-result", 1, 0, false);
    return info;
}

void module_build_info_free(ModuleBuildInfo *info) {
    if (!info) return;

    free(info->object_file);

    free(info->module_dir);

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
