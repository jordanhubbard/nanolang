/* I share this implementation with the existing module-cache owning TU. */
#include "runtime/dyn_array.h"
#include <stdint.h>
#include <limits.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <openssl/evp.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include "native_sdk_inventory.inc"

#define SDK_PATH_CAPACITY 4096
#define SDK_FILE_LIMIT 8192
#define SDK_BYTE_LIMIT ((uint64_t)1024 * 1024 * 1024)
static const char sdk_header[] = "NANOLANG_SDK_ABI=2\n";

static bool sdk_regular(const char *path, struct stat *st) {
    return lstat(path, st) == 0 && S_ISREG(st->st_mode);
}

static bool sdk_path(char *dest, size_t capacity, const char *root, const char *relative) {
    if (!relative || !relative[0] || relative[0] == '/') return false;
    const char *part = relative;
    for (const char *p = relative;; ++p) {
        if (*p && ((unsigned char)*p < 32 || *p == '\\' || *p == 127)) return false;
        if (!*p || *p == '/') {
            size_t n = (size_t)(p - part);
            if (!n || (n == 1 && part[0] == '.') ||
                (n == 2 && part[0] == '.' && part[1] == '.')) return false;
            if (!*p) break;
            part = p + 1;
        }
    }
    int length = snprintf(dest, capacity, "%s/%s", root, relative);
    if (length < 0 || (size_t)length >= capacity) return false;
    /* I reject symlink descendants rather than treating them as package files. */
    for (char *p = dest + strlen(root) + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = 0;
        struct stat st;
        bool ok = lstat(dest, &st) == 0 && S_ISDIR(st.st_mode);
        *p = '/';
        if (!ok) return false;
    }
    return true;
}

static bool sdk_hex(const unsigned char *digest, const char *expected) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < 32; ++i)
        if (expected[2*i] != hex[digest[i] >> 4] ||
            expected[2*i+1] != hex[digest[i] & 15]) return false;
    return expected[64] == 0;
}

static bool sdk_file_digest(const char *path, const char *expected) {
    FILE *file = fopen(path, "rb");
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    bool ok = file && ctx && EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1;
    unsigned char buffer[8192], value[32];
    unsigned int size = 0;
    size_t amount;
    while (ok && (amount = fread(buffer, 1, sizeof(buffer), file)) != 0)
        ok = EVP_DigestUpdate(ctx, buffer, amount) == 1;
    if (file) {
        if (ferror(file)) ok = false;
        if (fclose(file)) ok = false;
    }
    if (ok) ok = EVP_DigestFinal_ex(ctx, value, &size) == 1 && size == 32 && sdk_hex(value, expected);
    EVP_MD_CTX_free(ctx);
    return ok;
}

static NanoSdkStatus sdk_check_header_abi(const char *root) {
    char path[SDK_PATH_CAPACITY];
    struct stat st;
    if (!sdk_path(path, sizeof(path), root, "src/runtime/dyn_array.h") ||
        !sdk_regular(path, &st)) return NANO_SDK_INVALID;
    FILE *header = fopen(path, "rb");
    if (!header) return NANO_SDK_IO;
    char line[256], expected[96];
    const char prefix[] = "#define NANO_DYN_ARRAY_ABI_VERSION ";
    bool matched = false;
    size_t declarations = 0;
    snprintf(expected, sizeof(expected), "%s%uu", prefix, NANO_DYN_ARRAY_ABI_VERSION);
    while (fgets(line, sizeof(line), header)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!strncmp(line, prefix, sizeof(prefix)-1)) {
            if (++declarations > 1) break;
            matched = !strcmp(line, expected);
        }
    }
    bool clean = !ferror(header);
    if (fclose(header)) clean = false;
    return clean && matched && declarations == 1 ? NANO_SDK_OK : NANO_SDK_INVALID;
}

static NanoSdkStatus sdk_installed(const char *root, const char *manifest) {
    FILE *file = fopen(manifest, "rb");
    if (!file) return NANO_SDK_IO;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) { fclose(file); return NANO_SDK_MEMORY; }
    char line[4096], previous[2049] = "", path[SDK_PATH_CAPACITY];
    bool ok = fgets(line, sizeof(line), file) && !strcmp(line, sdk_header) &&
              EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1 &&
              EVP_DigestUpdate(ctx, line, strlen(line)) == 1;
    uint64_t total = 0;
    size_t count = 0;
    const size_t required_count = sizeof(sdk_required_inputs)/sizeof(sdk_required_inputs[0]);
    while (ok && fgets(line, sizeof(line), file)) {
        size_t length = strlen(line);
        if (!length || line[length-1] != '\n' || ++count > SDK_FILE_LIMIT || count > required_count ||
            EVP_DigestUpdate(ctx, line, length) != 1) { ok = false; break; }
        line[length-1] = 0;
        char *mode_end = strchr(line, '\t');
        char *size_end = mode_end ? strchr(mode_end+1, '\t') : NULL;
        char *hash_end = size_end ? strchr(size_end+1, '\t') : NULL;
        if (!mode_end || !size_end || !hash_end) { ok = false; break; }
        *mode_end++ = 0; *size_end++ = 0; *hash_end++ = 0;
        if (strlen(line) != 4 || strspn(line, "01234567") != 4 ||
            !mode_end[0] || strspn(mode_end, "0123456789") != strlen(mode_end) ||
            (mode_end[0] == '0' && mode_end[1]) || strlen(size_end) != 64 ||
            strspn(size_end, "0123456789abcdef") != 64 ||
            strlen(hash_end) > 2048 || strcmp(previous, hash_end) >= 0 ||
            strcmp(hash_end, sdk_required_inputs[count-1])) { ok = false; break; }
        errno = 0;
        unsigned long long size = strtoull(mode_end, NULL, 10);
        unsigned long mode = strtoul(line, NULL, 8);
        if (errno || mode > 0777 || size > SDK_BYTE_LIMIT - total ||
            ((!strncmp(hash_end, "bin/", 4) || !strcmp(hash_end, "scripts/generate_list.sh")) && !(mode & 0111)) ||
            !sdk_path(path, sizeof(path), root, hash_end)) { ok = false; break; }
        total += size;
        struct stat st;
        if (!sdk_regular(path, &st) || st.st_size < 0 || (uint64_t)st.st_size != size ||
            ((unsigned long)st.st_mode & 0111) != (mode & 0111) ||
            !sdk_file_digest(path, size_end)) { ok = false; break; }
        memcpy(previous, hash_end, strlen(hash_end)+1);
    }
    if (ferror(file)) ok = false;
    if (fclose(file)) ok = false;
    const char *identity = strrchr(root, '/');
    unsigned char value[32]; unsigned int size = 0;
    if (ok) ok = count == required_count && identity && strlen(++identity) == 64 &&
        EVP_DigestFinal_ex(ctx, value, &size) == 1 && size == 32 && sdk_hex(value, identity);
    EVP_MD_CTX_free(ctx);
    return ok ? sdk_check_header_abi(root) : NANO_SDK_INVALID;
}

NanoSdkStatus nano_native_sdk_root(char *dest, size_t capacity, bool *installed) {
    if (!dest || !capacity || !installed) return NANO_SDK_INVALID;
    char root[SDK_PATH_CAPACITY], path[SDK_PATH_CAPACITY], executable[SDK_PATH_CAPACITY];
    const char *override = getenv("NANOLANG_SDK_ROOT");
    if (override) {
        if (!override[0] || !realpath(override, root)) return NANO_SDK_INVALID;
    } else {
#ifdef __APPLE__
        uint32_t size = sizeof(executable);
        if (_NSGetExecutablePath(executable, &size)) return NANO_SDK_LIMIT;
#else
        ssize_t size = readlink("/proc/self/exe", executable, sizeof(executable)-1);
        if (size < 0) return NANO_SDK_IO;
        if ((size_t)size >= sizeof(executable)-1) return NANO_SDK_LIMIT;
        executable[size] = 0;
#endif
        if (!realpath(executable, root)) return NANO_SDK_IO;
        char *slash = strrchr(root, '/');
        if (!slash) return NANO_SDK_INVALID;
        *slash = 0;
        slash = strrchr(root, '/');
        if (!slash || strcmp(slash, "/bin")) return NANO_SDK_INVALID;
        *slash = 0;
    }
    struct stat st;
    if (!sdk_path(path, sizeof(path), root, "sdk.inputs")) return NANO_SDK_INVALID;
    bool package = lstat(path, &st) == 0;
    NanoSdkStatus status = NANO_SDK_OK;
    if (package) {
        if (!S_ISREG(st.st_mode) || st.st_size > 4 * 1024 * 1024) return NANO_SDK_INVALID;
        status = sdk_installed(root, path);
    } else {
        const char *markers[] = {"Makefile.gnu", "src/nanolang.h", "src_nano/nanoc_v06.nano",
                                 "src/runtime/dyn_array.h", "scripts/generate_list.sh"};
        for (size_t i = 0; i < sizeof(markers)/sizeof(markers[0]); ++i)
            if (!sdk_path(path, sizeof(path), root, markers[i]) || !sdk_regular(path, &st))
                return NANO_SDK_INVALID;
        status = sdk_check_header_abi(root);
    }
    if (status != NANO_SDK_OK) return status;
    size_t length = strlen(root);
    if (length >= capacity) return NANO_SDK_LIMIT;
    memcpy(dest, root, length+1);
    *installed = package;
    return NANO_SDK_OK;
}

static char sdk_private_cache[SDK_PATH_CAPACITY];
static pid_t sdk_cache_owner;
static bool sdk_retain_work;

void nano_native_retain_private_work(void) { sdk_retain_work = true; }

static bool sdk_remove_tree(const char *path, unsigned depth) {
    if (depth > 16) return false;
    struct stat st;
    if (lstat(path, &st)) return errno == ENOENT;
    if (!S_ISDIR(st.st_mode)) return unlink(path) == 0;
    DIR *dir = opendir(path);
    if (!dir) return false;
    bool ok = true;
    struct dirent *entry;
    for (;;) {
        errno = 0;
        entry = readdir(dir);
        if (!entry) { if (errno) ok = false; break; }
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char child[SDK_PATH_CAPACITY];
        int n = snprintf(child, sizeof(child), "%s/%s", path, entry->d_name);
        if (n >= 0 && (size_t)n < sizeof(child)) ok = sdk_remove_tree(child, depth+1) && ok;
        else ok = false;
    }
    if (closedir(dir)) ok = false;
    if (rmdir(path)) ok = false;
    return ok;
}

void nano_native_remove_private_tree(const char *directory) {
    if (directory && directory[0] && !sdk_remove_tree(directory, 0))
        fprintf(stderr, "I could not completely remove private compiler products at: %s\n", directory);
}

bool nano_native_generate_list(const char *root, const char *directory,
                               const char *name, const char *c_type) {
    const char *values[] = {name, c_type};
    for (size_t i = 0; i < 2; ++i) {
        const char *value = values[i];
        if (!value || !*value || strlen(value) > 127 ||
            !(value[0] == '_' || (value[0] >= 'a' && value[0] <= 'z') ||
              (value[0] >= 'A' && value[0] <= 'Z')) ||
            strspn(value, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != strlen(value))
            return false;
    }
    char script[SDK_PATH_CAPACITY]; struct stat st;
    if (!root || !directory || !sdk_path(script, sizeof(script), root, "scripts/generate_list.sh") ||
        !sdk_regular(script, &st) || !(st.st_mode & 0111) ||
        lstat(directory, &st) || !S_ISDIR(st.st_mode)) return false;
    pid_t child = fork();
    if (child < 0) return false;
    if (!child) {
        execl(script, script, name, directory, c_type, (char *)NULL);
        _exit(127);
    }
    int status;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    return waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

#ifdef __APPLE__
extern void ffi_loader_shutdown(void) __attribute__((weak_import));
#else
extern void ffi_loader_shutdown(void) __attribute__((weak));
#endif
static void sdk_cache_shutdown(void) {
    if (getpid() != sdk_cache_owner) return;
    if (ffi_loader_shutdown) ffi_loader_shutdown();
    if (sdk_private_cache[0]) {
        if (sdk_retain_work)
            fprintf(stderr, "I retain generated compiler products at: %s\n", sdk_private_cache);
        else nano_native_remove_private_tree(sdk_private_cache);
    }
}

static NanoSdkStatus sdk_make_work(void) {
    if (sdk_private_cache[0]) return NANO_SDK_OK;
    const char *temporary = getenv("TMPDIR");
    if (!temporary || !*temporary) temporary = "/tmp";
    char candidate[SDK_PATH_CAPACITY];
    int n = snprintf(candidate, sizeof(candidate), "%s/nanolang-native-work-XXXXXX", temporary);
    if (n < 0 || (size_t)n >= sizeof(candidate)) return NANO_SDK_LIMIT;
    if (!mkdtemp(candidate)) return NANO_SDK_IO;
    memcpy(sdk_private_cache, candidate, strlen(candidate)+1);
    sdk_cache_owner = getpid();
    if (atexit(sdk_cache_shutdown)) {
        rmdir(candidate); sdk_private_cache[0] = 0;
        return NANO_SDK_MEMORY;
    }
    return NANO_SDK_OK;
}

NanoSdkStatus nano_native_sdk_prepare(void) {
    char root[SDK_PATH_CAPACITY]; bool installed;
    NanoSdkStatus status = nano_native_sdk_root(root, sizeof(root), &installed);
    if (status != NANO_SDK_OK || !installed) return status;
    status = sdk_make_work();
    if (status != NANO_SDK_OK) return status;
    const char *cache = getenv("NANO_BUILD_CACHE");
    if ((!cache || !*cache) && setenv("NANO_BUILD_CACHE", sdk_private_cache, 1))
        return NANO_SDK_MEMORY;
    return NANO_SDK_OK;
}

NanoSdkStatus nano_native_module_objects_dir(char *dest, size_t capacity) {
    if (!dest || !capacity) return NANO_SDK_INVALID;
    NanoSdkStatus status = sdk_make_work();
    if (status != NANO_SDK_OK) return status;
    char path[SDK_PATH_CAPACITY];
    int n = snprintf(path, sizeof(path), "%s/objects-XXXXXX", sdk_private_cache);
    if (n < 0 || (size_t)n >= sizeof(path) || (size_t)n >= capacity) return NANO_SDK_LIMIT;
    if (!mkdtemp(path)) return NANO_SDK_IO;
    memcpy(dest, path, (size_t)n+1);
    return NANO_SDK_OK;
}
