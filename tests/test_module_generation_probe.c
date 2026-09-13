/* I expose production cache paths so tests can hold them across publication. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#include "module_builder.h"
#include "runtime/module_build_dir.h"
#include "runtime/ffi_loader.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <signal.h>
#include <fcntl.h>

static const char *generation_sync_event;

static void generation_test_event(const char *event) {
    const char *path = getenv("NANO_TEST_SYNC_EVENTS");
    if (path) {
        FILE *file = fopen(path, "a");
        if (file) { fprintf(file, "%s\n", event); fclose(file); }
    }
    /* I terminate only this test probe, after its compiler children exit. */
    const char *crash = getenv("NANO_TEST_CRASH_EVENT");
    if (crash && !strcmp(crash, event)) (void)kill(getpid(), SIGKILL);
}

static int generation_test_fsync(int fd) {
    static unsigned cache_calls = 0;
    static bool interrupted = false;
    struct stat st, cache;
    const char *root = getenv("NANO_TEST_SYNC_CACHE");
    const char *failure = getenv("NANO_TEST_SYNC_FAILURE");
    const char *event = "file";
    if (fstat(fd, &st) != 0) return -1;
    if (S_ISDIR(st.st_mode)) {
        if (root && stat(root, &cache) == 0 && cache.st_ino == st.st_ino && cache.st_dev == st.st_dev)
            event = ++cache_calls == 1 ? "cache-1" : "cache-2";
        else {
            event = root ? "ancestor" : "stage";
            DIR *directory = root ? opendir(root) : NULL;
            struct dirent *entry;
            while (directory && (entry = readdir(directory))) {
                if (strncmp(entry->d_name, ".nano-build-", 12)) continue;
                char path[4096];
                struct stat candidate;
                int length = snprintf(path, sizeof(path), "%s/%s", root, entry->d_name);
                if (length > 0 && (size_t)length < sizeof(path) && stat(path, &candidate) == 0 &&
                    candidate.st_dev == st.st_dev && candidate.st_ino == st.st_ino) event = "stage";
            }
            if (directory) closedir(directory);
        }
    }
    generation_test_event(event);
    generation_sync_event = event;
    const char *identities = getenv("NANO_TEST_SYNC_IDENTITIES");
    if (identities && S_ISDIR(st.st_mode)) {
        FILE *file = fopen(identities, "a");
        if (file) {
            fprintf(file, "%llu:%llu\n", (unsigned long long)st.st_dev, (unsigned long long)st.st_ino);
            fclose(file);
        }
    }
    const char *fail_path = getenv("NANO_TEST_SYNC_FAIL_PATH");
    if (fail_path && stat(fail_path, &cache) == 0 && cache.st_dev == st.st_dev && cache.st_ino == st.st_ino) {
        errno = EIO;
        return -1;
    }
    if (getenv("NANO_TEST_FULL_SYNC")) return fsync(fd);
    if (failure && !strcmp(failure, "eintr") && !interrupted) {
        interrupted = true;
        errno = EINTR;
        return -1;
    }
    if (failure && !strcmp(failure, event)) { errno = EIO; return -1; }
    return fsync(fd);
}

#ifdef __APPLE__
static int generation_test_fcntl(int fd, int command) {
    static bool interrupted = false;
    if (command != F_FULLFSYNC) { errno = EINVAL; return -1; }
    const char *failure = getenv("NANO_TEST_SYNC_FAILURE");
    if (getenv("NANO_TEST_FULL_SYNC") && failure) {
        if (!strcmp(failure, "unsupported")) { errno = ENOTSUP; return -1; }
        if (!strcmp(failure, "eintr") && !interrupted) {
            interrupted = true;
            errno = EINTR;
            return -1;
        }
        if (generation_sync_event && !strcmp(failure, generation_sync_event)) {
            errno = EIO;
            return -1;
        }
    }
    return fcntl(fd, command);
}
#define fcntl generation_test_fcntl
#endif

/* I inject a final-pointer rename failure without damaging the old pointer.
 * This hook is confined to the test translation unit. */
static int generation_test_rename(const char *source, const char *target) {
    const char *leaf = strrchr(target, '/');
    if (leaf && strcmp(leaf, "/current") == 0 && getenv("NANO_TEST_POINTER_FAILURE")) {
        errno = EIO;
        return -1;
    }
    int result = rename(source, target);
    if (!result && leaf) {
        if (!strncmp(leaf, "/.nano-gen-", 11)) generation_test_event("generation");
        if (!strcmp(leaf, "/current")) generation_test_event("pointer");
    }
    return result;
}

static int generation_test_unlinkat(int fd, const char *name, int flags) {
    static bool swapped = false;
    const char *stage = getenv("NANO_TEST_CLEANUP_STAGE");
    const char *target = getenv("NANO_TEST_CLEANUP_TARGET");
    if (!swapped && stage && target) {
        char moved[4096];
        int length = snprintf(moved, sizeof(moved), "%s.moved", stage);
        if (length < 0 || (size_t)length >= sizeof(moved)) { errno = ENAMETOOLONG; return -1; }
        if (rename(stage, moved) != 0 || symlink(target, stage) != 0) return -1;
        swapped = true;
    }
    return unlinkat(fd, name, flags);
}
#define unlinkat generation_test_unlinkat
#define rename generation_test_rename
#define fsync generation_test_fsync
static long generation_allocation_limit = -1;
static bool generation_allocation_fails(void) {
    if (generation_allocation_limit < 0) return false;
    if (generation_allocation_limit == 0) return true;
    generation_allocation_limit--;
    return false;
}
static void *generation_test_malloc(size_t size) {
    return generation_allocation_fails() ? NULL : malloc(size);
}
static void *generation_test_calloc(size_t count, size_t size) {
    return generation_allocation_fails() ? NULL : calloc(count, size);
}
static char *generation_test_strdup(const char *value) {
    return generation_allocation_fails() ? NULL : strdup(value);
}
#define malloc generation_test_malloc
#define calloc generation_test_calloc
#define strdup generation_test_strdup
#include "../src/module_builder.c"
#undef malloc
#undef calloc
#undef strdup
#undef rename
#undef fsync
#undef unlinkat
#ifdef __APPLE__
#undef fcntl
#endif

int main(int argc, char **argv) {
#ifdef __APPLE__
    if (argc == 5 && strcmp(argv[1], "link-inputs") == 0) {
        cJSON *inputs = module_link_inputs(argv[2], argv[3], argv[4]);
        if (!inputs) return 1;
        char *json = cJSON_PrintUnformatted(inputs);
        if (json) puts(json);
        int status = json ? 0 : 1;
        free(json);
        cJSON_Delete(inputs);
        return status;
    }
#endif
    if (argc != 3 && argc != 4) return 2;
    if (argc == 3 && !strcmp(argv[1], "compile-flags-allocation")) {
        char *packages[] = {"-DPACKAGE=1", ""}, *includes[] = {"/include"};
        char *common[] = {"-DCOMMON=1"}, *platform[] = {"-DPLATFORM=1"};
        ModuleBuildMetadata meta = {0};
        meta.cflags = common; meta.cflags_count = 1;
        meta.include_dirs = includes; meta.include_dirs_count = 1;
#ifdef __APPLE__
        char *names[] = {"fixture-a", "fixture-b"};
        meta.pkg_config = names; meta.pkg_config_count = 2;
        meta.cflags_macos = platform; meta.cflags_macos_count = 1;
#elif defined(__FreeBSD__)
        meta.cflags_freebsd = platform; meta.cflags_freebsd_count = 1;
#else
        meta.cflags_linux = platform; meta.cflags_linux_count = 1;
#endif
        ModulePkgFlags flags = {.count = 2, .cflags = packages};
        ModuleBuildInfo *info = calloc(1, sizeof(*info));
        if (!info) return 1;
        bool overflow = !strcmp(argv[2], "overflow");
        if (overflow) meta.cflags_count = SIZE_MAX;
        generation_allocation_limit = overflow ? -1 : strtol(argv[2], NULL, 10);
        bool result = module_collect_compile_flags(info, &meta, &flags);
        generation_allocation_limit = -1;
        bool ok = !result && !info->compile_flags && !info->compile_flags_count;
        meta.cflags_count = 1;
        ok = ok && module_collect_compile_flags(info, &meta, &flags) &&
            info->compile_flags_count == 4 && !strcmp(info->compile_flags[0], "-DPACKAGE=1") &&
            !strcmp(info->compile_flags[1], "-I/include") &&
            !strcmp(info->compile_flags[2], "-DCOMMON=1") &&
            !strcmp(info->compile_flags[3], "-DPLATFORM=1");
        module_build_info_free(info);
        return ok ? 0 : 1;
    }
    if (argc == 3 && !strcmp(argv[1], "capture-response")) {
        char *captured = module_capture_response_fragment(argv[2]);
        if (!captured) return 1;
        puts(captured);
        free(captured);
        return 0;
    }
    if (argc == 4 && !strcmp(argv[1], "capture-pch"))
        return module_snapshot_pch(argv[2], argv[3], 0, 0, 14695981039346656037ULL) ? 0 : 1;
#ifdef __linux__
    if (argc == 3 && !strcmp(argv[1], "assembler-version"))
        return module_assembler_version_supported(argv[2]) ? 0 : 1;
#endif
    if (argc == 4 && strcmp(argv[1], "capture-assembly") == 0) {
        char *directory = strdup(argv[3]);
        if (!directory) return 1;
        char *slash = strrchr(directory, '/');
        if (!slash) { free(directory); return 2; }
        *slash = 0;
        ModuleAssemblyCapture capture = {directory, 0, 0, 14695981039346656037ULL};
        bool ok = module_capture_assembly_file(&capture, argv[2], argv[3], true, 0);
        free(directory);
        return ok ? 0 : 1;
    }
    if (strcmp(argv[1], "flag-words") == 0) {
        cJSON *words = cJSON_CreateArray();
        if (!words) return 1;
        const char *cursor = argv[2];
        char word[4096];
        int status;
        while ((status = module_flag_word(&cursor, word, sizeof(word))) > 0) {
            cJSON *item = cJSON_CreateString(word);
            if (!item || !cJSON_AddItemToArray(words, item)) {
                cJSON_Delete(item);
                cJSON_Delete(words);
                return 1;
            }
        }
        char *json = status == 0 ? cJSON_PrintUnformatted(words) : NULL;
        bool ok = json != NULL;
        if (json) puts(json);
        free(json);
        cJSON_Delete(words);
        return ok ? 0 : 1;
    }
#ifdef __linux__
    if (argc == 4 && strcmp(argv[1], "equal-libraries") == 0)
        return module_equal_libraries(argv[2], argv[3]) ? 0 : 1;
#endif
    if (strcmp(argv[1], "remove-staging") == 0) {
        module_remove_staging(argv[2]);
        return 0;
    }
    if (strcmp(argv[1], "sync-generation") == 0)
        return module_sync_generation(argv[2]) ? 0 : 1;
    if (strcmp(argv[1], "pkgflags") == 0) {
        char *flags = get_pkg_config_flags(argv[2], argc == 4 ? argv[3] : "--cflags");
        if (!flags) return 1;
        puts(flags);
        free(flags);
        return 0;
    }
    if (strcmp(argv[1], "deps") == 0 || strcmp(argv[1], "includes") == 0) {
        cJSON *root = cJSON_CreateObject();
        if (!root) return 1;
        bool ok = strcmp(argv[1], "deps") == 0 ? hash_depfile_into_cache(root, argv[2])
                                               : hash_include_trace(root, argv[2]);
        char *json = cJSON_PrintUnformatted(root);
        if (json) puts(json);
        free(json);
        cJSON_Delete(root);
        return ok ? 0 : 1;
    }
    if (strcmp(argv[1], "root") == 0) {
        char path[2048];
        size_t size = argc == 4 ? (size_t)strtoul(argv[3], NULL, 10) : sizeof(path);
        if (size > sizeof(path) || !nano_module_build_dir(argv[2], path, size)) return 1;
        puts(path);
        return 0;
    }
    if (strcmp(argv[1], "directory") == 0) {
        char path[2048];
        if (!nano_module_artifact_dir(argv[2], path, sizeof(path))) return 1;
        puts(path);
        return 0;
    }
    ModuleBuildMetadata *meta = module_load_metadata(argv[2]);
    if (!meta) return 1;
    int status = 1;
    if (strcmp(argv[1], "build") == 0 || strcmp(argv[1], "build-info") == 0) {
        ModuleBuildInfo *info = module_build(NULL, meta);
        if (info && (info->object_file || strcmp(argv[1], "build-info") == 0)) {
            puts(info->object_file ? info->object_file : "no object");
            if (strcmp(argv[1], "build-info") == 0) {
                for (size_t i = 0; i < info->compile_flags_count; i++)
                    printf("compile:%s\n", info->compile_flags[i]);
                for (size_t i = 0; i < info->link_flags_count; i++)
                    printf("link:%s\n", info->link_flags[i]);
            }
            status = 0;
        }
        module_build_info_free(info);
    } else if (strcmp(argv[1], "needs-rebuild") == 0) {
        puts(module_needs_rebuild(argv[2], meta) ? "1" : "0");
        status = 0;
    } else if (strcmp(argv[1], "library") == 0) {
        char path[2048];
        if (ffi_loader_find_library(meta->name, argv[2], path, sizeof(path))) {
            puts(path);
            status = 0;
        }
    }
    module_metadata_free(meta);
    return status;
}
