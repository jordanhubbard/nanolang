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
#include <stdarg.h>

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

static int generation_mutate_response(void) {
    static bool response_mutated = false;
    const char *response = getenv("NANO_TEST_RESPONSE_MUTATE");
    if (response && !response_mutated) {
        response_mutated = true;
        const char *action = getenv("NANO_TEST_RESPONSE_ACTION");
        if (action && !strcmp(action, "remove")) {
            if (unlink(response)) return -1;
        } else if (action && !strcmp(action, "retarget")) {
            const char *target = getenv("NANO_TEST_RESPONSE_TARGET");
            if (!target || unlink(response) || symlink(target, response)) return -1;
        } else {
            FILE *changed = fopen(response, "wb");
            if (!changed) return -1;
            bool ok = fputs("-lc\n", changed) >= 0;
            if (fclose(changed)) ok = false;
            if (!ok) return -1;
        }
    }
    return 0;
}

/* I mutate only the named fixture, after its bytes have reached the reader.
 * No production read path contains this test hook. */
static ssize_t generation_test_read(int fd, void *buffer, size_t capacity) {
    ssize_t amount = read(fd, buffer, capacity);
    const char *response = getenv("NANO_TEST_RESPONSE_MUTATE");
    if (amount > 0 && response && getenv("NANO_TEST_RESPONSE_ON_READ")) {
        struct stat source, observed;
        if (!stat(response, &source) && !fstat(fd, &observed) &&
            source.st_dev == observed.st_dev && source.st_ino == observed.st_ino &&
            generation_mutate_response()) return -1;
    }
    return amount;
}

static int generation_test_fsync(int fd) {
    static unsigned cache_calls = 0;
    static bool interrupted = false;
    struct stat st, cache;
    const char *root = getenv("NANO_TEST_SYNC_CACHE");
    const char *failure = getenv("NANO_TEST_SYNC_FAILURE");
    const char *event = "file";
    if (fstat(fd, &st) != 0) return -1;
    if (S_ISREG(st.st_mode) && !getenv("NANO_TEST_RESPONSE_ON_READ") && generation_mutate_response()) return -1;
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
static int generation_test_fcntl(int fd, int command, ...) {
    static bool interrupted = false;
    if (command == F_DUPFD_CLOEXEC || command == F_SETFD || command == F_SETFL) {
        va_list arguments;
        va_start(arguments, command);
        int value = va_arg(arguments, int);
        va_end(arguments);
        return fcntl(fd, command, value);
    }
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
#define read generation_test_read
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
#undef read
#undef unlinkat
#ifdef __APPLE__
#undef fcntl
#endif

int main(int argc, char **argv) {
    if ((argc >= 4 && !strcmp(argv[1], "capture-link-arguments")) ||
        (argc >= 5 && !strcmp(argv[1], "capture-link-arguments-allocation"))) {
        bool allocation = !strcmp(argv[1], "capture-link-arguments-allocation");
        size_t first = allocation ? 4 : 3, count = (size_t)argc - first;
        ModuleLinkResponseGrammar grammar = !strcmp(argv[2], "gnu") ? MODULE_LINK_RESPONSE_GNU :
            !strcmp(argv[2], "apple") ? MODULE_LINK_RESPONSE_APPLE : 0;
        const char *const *sources = (const char *const *)(argv + first);
        if (allocation) generation_allocation_limit = strtol(argv[3], NULL, 10);
        char **arguments = module_capture_link_arguments(sources, count, grammar);
        int failure = errno;
        generation_allocation_limit = -1;
        if (allocation) puts(arguments ? "captured" : "failed");
        else if (!arguments) {
            fprintf(stderr, "I could not capture linker arguments: errno=%d\n", failure);
            return 1;
        }
        cJSON *array = allocation ? NULL : cJSON_CreateArray();
        bool ok = allocation || array;
        for (size_t i = 0; arguments && i < count; i++) {
            if (!allocation && ok) {
                cJSON *value = cJSON_CreateString(arguments[i]);
                ok = value && cJSON_AddItemToArray(array, value);
                if (!ok) cJSON_Delete(value);
            }
            free(arguments[i]);
        }
        free(arguments);
        char *json = ok && !allocation ? cJSON_PrintUnformatted(array) : NULL;
        if (!allocation) {
            ok = json != NULL;
            if (json) puts(json);
        }
        free(json);
        cJSON_Delete(array);
        if (allocation) {
            arguments = module_capture_link_arguments(sources, count, grammar);
            if (!arguments) return 1;
            for (size_t i = 0; i < count; i++) free(arguments[i]);
            free(arguments);
        }
        return ok ? 0 : 1;
    }
    if (argc == 4 && !strcmp(argv[1], "link-response-words")) {
        ModuleLinkResponseGrammar grammar = !strcmp(argv[2], "gnu") ? MODULE_LINK_RESPONSE_GNU :
            !strcmp(argv[2], "apple") ? MODULE_LINK_RESPONSE_APPLE : 0;
        if (!grammar) return 2;
        const char *cursor = argv[3], *begin, *end;
        char word[4096];
        int status;
        cJSON *words = cJSON_CreateArray();
        if (!words) return 1;
        while ((status = module_link_response_word(&cursor, &begin, &end, word, sizeof(word), grammar)) > 0) {
            cJSON *value = cJSON_CreateString(word);
            if (!value || !cJSON_AddItemToArray(words, value)) {
                cJSON_Delete(value);
                status = -1;
                break;
            }
        }
        char *json = status < 0 ? NULL : cJSON_PrintUnformatted(words);
        if (json) puts(json);
        int result = json ? 0 : 1;
        free(json);
        cJSON_Delete(words);
        return result;
    }
    if ((argc >= 5 && !strcmp(argv[1], "capture-link-responses")) ||
        (argc >= 6 && !strcmp(argv[1], "capture-link-responses-allocation"))) {
        bool allocation = !strcmp(argv[1], "capture-link-responses-allocation");
        size_t first = allocation ? 5 : 4, count = (size_t)argc - first;
        ModuleBuildMetadata meta = {.module_dir = argv[3]};
        ModuleLinkResponseGrammar grammar = !strcmp(argv[2], "gnu") ? MODULE_LINK_RESPONSE_GNU :
            !strcmp(argv[2], "apple") ? MODULE_LINK_RESPONSE_APPLE : 0;
        const char *const *sources = (const char *const *)(argv + first);
        if (allocation) generation_allocation_limit = strtol(argv[4], NULL, 10);
        char **paths = module_capture_link_responses(&meta, sources, count, grammar);
        generation_allocation_limit = -1;
        if (allocation) puts(paths ? "captured" : "failed");
        if (paths) {
            for (size_t i = 0; i < count; i++) {
                if (!allocation) puts(paths[i]);
                free(paths[i]);
            }
            free(paths);
        } else if (!allocation) return 1;
        if (allocation) {
            paths = module_capture_link_responses(&meta, sources, count, grammar);
            if (!paths) return 1;
            for (size_t i = 0; i < count; i++) free(paths[i]);
            free(paths);
        }
        return 0;
    }
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
    if (argc != 3 && argc != 4 &&
        !(argc == 5 && !strcmp(argv[1], "capture-link-response")) &&
        !(argc == 5 && !strcmp(argv[1], "private-link-response-allocation")) &&
        !(argc == 6 && !strcmp(argv[1], "capture-link-response-allocation"))) return 2;
    if (argc == 3 && !strcmp(argv[1], "link-response-allocation")) {
        char large[1300];
        memset(large, 'x', sizeof(large) - 1); large[sizeof(large) - 1] = 0;
        memcpy(large, "-L/", 3);
        char *common[] = {"-O2"}, *link[] = {large};
        ModuleBuildMetadata meta = {0}, copy;
        meta.cflags = common; meta.cflags_count = 1;
        meta.ldflags = link; meta.ldflags_count = 1;
        unsigned failures = 0, captures = 0;
        for (long limit = 0; limit < 32; limit++) {
            generation_allocation_limit = limit;
            bool ok = module_response_metadata(&meta, &copy);
            generation_allocation_limit = -1;
            if (meta.cflags != common || meta.ldflags != link || strcmp(common[0], "-O2") || link[0] != large) return 1;
            if (ok) {
                bool owned = copy.ldflags != link;
                if ((copy.cflags != common) != owned || strcmp(copy.ldflags[0], large)) return 1;
                captures += owned;
                module_response_metadata_free(&meta, &copy);
            } else failures++;
            if (!module_response_metadata(&meta, &copy) || copy.ldflags == link || copy.cflags == common) return 1;
            module_response_metadata_free(&meta, &copy);
        }
        return failures && captures ? 0 : 1;
    }
    if (argc == 3 && !strcmp(argv[1], "link-fragment-allocation")) {
        char *libs[] = {"-lpkg"}, *system[] = {"m", "c", "m"}, *common[] = {"-L/common"};
        ModuleBuildMetadata meta = {0};
        meta.pkg_config_count = 1;
        char *packages[] = {"fixture"};
        meta.pkg_config = packages;
        meta.system_libs = system; meta.system_libs_count = 3;
        meta.ldflags = common; meta.ldflags_count = 1;
        ModulePkgFlags flags = {.libs = libs, .count = 1};
        generation_allocation_limit = 0;
        char *failed = module_shared_link_fragment(&meta, &flags);
        generation_allocation_limit = -1;
        if (failed) { free(failed); return 1; }
        char *retry = module_shared_link_fragment(&meta, &flags);
        bool ok = retry && !strcmp(retry, " -lpkg -lm -lc -lm -L/common");
        free(retry);
        return ok ? 0 : 1;
    }
    if (argc == 3 && !strcmp(argv[1], "coalesce-allocation")) {
        char left[700], right[700];
        memset(left, 'x', sizeof(left) - 1); left[sizeof(left) - 1] = 0;
        memset(right, 'y', sizeof(right) - 1); right[sizeof(right) - 1] = 0;
        memcpy(left, "-DLEFT=", 7); memcpy(right, "-DRIGHT=", 8);
        for (long failure = 0; failure < 4; failure++) {
            char *flags[] = {NULL, strdup(""), strdup(left), strdup(right)};
            if (!flags[1] || !flags[2] || !flags[3]) {
                for (size_t i = 0; i < 4; i++) free(flags[i]);
                return 1;
            }
            char *original[] = {flags[0], flags[1], flags[2], flags[3]};
            generation_allocation_limit = failure;
            bool result = module_coalesce_cflags(flags, 4);
            generation_allocation_limit = -1;
            bool ok = !result;
            for (size_t i = 0; i < 4; i++) ok = ok && flags[i] == original[i];
            ok = ok && !strcmp(flags[1], "") && !strcmp(flags[2], left) && !strcmp(flags[3], right);
            ok = ok && module_coalesce_cflags(flags, 4) && flags[0] == NULL && !flags[1][0] &&
                !strncmp(flags[2], left, strlen(left)) && strstr(flags[2], " -DRIGHT=") && !flags[3][0];
            for (size_t i = 0; i < 4; i++) free(flags[i]);
            if (!ok) return 1;
        }
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "link-flags-allocation")) {
        char *packages[] = {"-lpackage", ""}, *common[] = {"-Lcommon"};
        char *platform[] = {"-Lplatform"}, *libraries[] = {"fixture"};
        ModuleBuildMetadata meta = {0};
        meta.ldflags = common; meta.ldflags_count = 1;
        meta.system_libs = libraries; meta.system_libs_count = 1;
        size_t allocations = 6, expected = 5;
#ifdef __APPLE__
        char *names[] = {"fixture-a", "fixture-b"}, *frameworks[] = {"Foundation"};
        meta.pkg_config = names; meta.pkg_config_count = 2;
        meta.frameworks = frameworks; meta.frameworks_count = 1;
        meta.ldflags_macos = platform; meta.ldflags_macos_count = 1;
        allocations += 2; expected += 2;
#elif defined(__FreeBSD__)
        meta.ldflags_freebsd = platform; meta.ldflags_freebsd_count = 1;
#else
        meta.ldflags_linux = platform; meta.ldflags_linux_count = 1;
#endif
        ModulePkgFlags flags = {.count = 2, .libs = packages};
        for (size_t failure = 0; failure < allocations + 2; failure++) {
            ModuleBuildInfo *info = calloc(1, sizeof(*info));
            if (!info) return 1;
            info->object_file = strdup("fixture.o");
            if (!info->object_file) { module_build_info_free(info); return 1; }
            meta.ldflags_count = failure == allocations ? SIZE_MAX :
                failure == allocations + 1 ? SIZE_MAX / sizeof(char *) : 1;
            generation_allocation_limit = failure < allocations ? (long)failure : -1;
            bool result = module_collect_link_flags(info, &meta, &flags);
            generation_allocation_limit = -1;
            bool ok = !result && !info->link_flags && !info->link_flags_count;
            meta.ldflags_count = 1;
            ok = ok && module_collect_link_flags(info, &meta, &flags) &&
                info->link_flags_count == expected && !strcmp(info->link_flags[0], "fixture.o") &&
                !strcmp(info->link_flags[1], "-lpackage") && !strcmp(info->link_flags[2], "-Lcommon") &&
                !strcmp(info->link_flags[3], "-Lplatform") && !strcmp(info->link_flags[expected - 1], "-lfixture");
#ifdef __APPLE__
            ok = ok && !strcmp(info->link_flags[4], "-framework") && !strcmp(info->link_flags[5], "Foundation");
#endif
            module_build_info_free(info);
            if (!ok) return 1;
        }
        return 0;
    }
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
            !strcmp(info->compile_flags[1], "-I'/include'") &&
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
    if (argc == 3 && !strcmp(argv[1], "link-response-grammar")) {
        printf("%d\n", module_link_response_grammar_command(argv[2]));
        return 0;
    }
    if (argc == 4 && !strcmp(argv[1], "private-link-response-grammar")) {
        printf("%d\n", module_query_link_response_grammar(argv[2], argv[3]));
        return 0;
    }
    if (argc == 5 && !strcmp(argv[1], "private-link-response-allocation")) {
        generation_allocation_limit = strtol(argv[4], NULL, 10);
        ModuleLinkResponseGrammar grammar = module_query_link_response_grammar(argv[2], argv[3]);
        generation_allocation_limit = -1;
        printf("%d\n", grammar);
        return module_query_link_response_grammar(argv[2], argv[3]) ? 0 : 1;
    }
    if (argc == 4 && !strcmp(argv[1], "link-response-query-allocation")) {
        generation_allocation_limit = strtol(argv[3], NULL, 10);
        ModuleLinkResponseGrammar grammar = module_link_response_grammar_command(argv[2]);
        generation_allocation_limit = -1;
        printf("%d\n", grammar);
        return module_link_response_grammar_command(argv[2]) ? 0 : 1;
    }
    if (argc == 5 && !strcmp(argv[1], "capture-link-response")) {
        ModuleBuildMetadata meta = {.module_dir = argv[3]};
        ModuleLinkResponseGrammar grammar = !strcmp(argv[2], "gnu") ? MODULE_LINK_RESPONSE_GNU :
            !strcmp(argv[2], "apple") ? MODULE_LINK_RESPONSE_APPLE : 0;
        char *captured = module_capture_link_response(&meta, argv[4], grammar);
        if (!captured) return 1;
        puts(captured);
        free(captured);
        return 0;
    }
    if (argc == 6 && !strcmp(argv[1], "capture-link-response-allocation")) {
        ModuleBuildMetadata meta = {.module_dir = argv[3]};
        ModuleLinkResponseGrammar grammar = !strcmp(argv[2], "gnu") ? MODULE_LINK_RESPONSE_GNU :
            !strcmp(argv[2], "apple") ? MODULE_LINK_RESPONSE_APPLE : 0;
        generation_allocation_limit = strtol(argv[5], NULL, 10);
        char *captured = module_capture_link_response(&meta, argv[4], grammar);
        generation_allocation_limit = -1;
        puts(captured ? "captured" : "failed");
        free(captured);
        captured = module_capture_link_response(&meta, argv[4], grammar);
        if (!captured) return 1;
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
    if (!strcmp(argv[1], "shared-link-command")) {
        size_t capacity = argc == 4 ? (size_t)strtoul(argv[3], NULL, 10) : 131072;
        ModulePkgFlags flags;
        char *command = capacity && capacity <= 131072 ? malloc(capacity) : NULL;
        if (command && module_pkg_flags_capture(meta, &flags)) {
            if (module_shared_link_command(meta, &flags, "fixture.o", "fixture.so", argv[2], command, capacity)) {
                puts(command);
                status = 0;
            }
            module_pkg_flags_free(&flags);
        }
        free(command);
        module_metadata_free(meta);
        return status;
    }
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
