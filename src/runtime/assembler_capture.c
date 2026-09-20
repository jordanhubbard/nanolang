/* I interpose the tested GNU-as stdio input boundary. I am not a sandbox.
 * The builder must separately identify the assembler and validate its exit. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "assembler_capture.h"
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/syscall.h>

static FILE *(*native_fopen)(const char *, const char *);
static const char *capture_prefix;
static FILE *journal, *completion;
static NacRead *retained;
static unsigned read_count, replay_at;
static uint64_t total_bytes, record_hash = NAC_SEED;
static int replay, active, failed;
static pid_t owner;
static const char *primary_input;

static FILE *nac_fail(void) { failed = 1; errno = EIO; return NULL; }

__attribute__((constructor)) static void nac_start(void) {
    native_fopen = dlsym(RTLD_NEXT, "fopen");
    capture_prefix = getenv("NANO_AS_CAPTURE_PREFIX");
    const char *phase = getenv("NANO_AS_CAPTURE_PHASE");
    if (!capture_prefix && !phase) return;
    active = 1; owner = getpid();
    const char *input = getenv("NANO_AS_CAPTURE_INPUT");
    if (!capture_prefix || !phase || !input || !*input) { nac_fail(); return; }
    primary_input = getenv("NANO_AS_CAPTURE_PRIMARY");
    if (!primary_input) primary_input = input;
    if (!*primary_input || strlen(primary_input) >= NAC_PATH) { nac_fail(); return; }
    replay = !strcmp(phase, "replay");
    char path[NAC_PATH];
    if (!native_fopen || (!replay && strcmp(phase, "capture"))) { nac_fail(); return; }
    if (replay) {
        retained = calloc(NAC_READS, sizeof(*retained));
        if (!retained || !nac_name(path, capture_prefix, "replayed", 0)) { nac_fail(); return; }
        completion = nac_file(path, O_WRONLY | O_CREAT, "wb");
        if (!completion || ftruncate(fileno(completion), 0) ||
            !nac_load(capture_prefix, primary_input, retained, &read_count, &record_hash)) nac_fail();
    } else {
        if (!nac_name(path, capture_prefix, "partial", 0)) { nac_fail(); return; }
        journal = nac_file(path, O_WRONLY | O_CREAT | O_EXCL, "wb");
        if (!journal || fwrite("NASCAP01", 1, 8, journal) != 8 || fflush(journal)) nac_fail();
    }
}

__attribute__((destructor)) static void nac_finish(void) {
    if (active && getpid() != owner) _exit(125);
    char partial[NAC_PATH], manifest[NAC_PATH];
    if (active && getpid() == owner && !failed && read_count) {
        if (replay) {
            if (replay_at != read_count || !completion ||
                fwrite("NACDONE1", 1, 8, completion) != 8) failed = 1;
        } else if (journal) {
            unsigned char seal[32] = {0};
            nac_put(seal + 4, read_count, 4); nac_put(seal + 16, total_bytes, 8); nac_put(seal + 24, record_hash, 8);
            if (fwrite(seal, 1, sizeof(seal), journal) != sizeof(seal) || fflush(journal)) failed = 1;
        }
    } else if (active) failed = 1;
    if (journal && fclose(journal)) failed = 1;
    if (completion && fclose(completion)) failed = 1;
    if (active && !replay && !failed) {
        if (!nac_name(partial, capture_prefix, "partial", 0) ||
            !nac_name(manifest, capture_prefix, "manifest", 0) || link(partial, manifest) || unlink(partial)) failed = 1;
    }
    free(retained);
    /* An ignored open failure or unread tail must not report assembly success. */
    if (active && failed) {
        static const char message[] = "I could not complete assembler input capture/replay.\n";
        size_t sent = 0;
        while (sent < sizeof(message) - 1) {
            ssize_t amount = write(STDERR_FILENO, message + sent, sizeof(message) - 1 - sent);
            if (amount < 0 && errno == EINTR) continue;
            if (amount <= 0) break;
            sent += (size_t)amount;
        }
        _exit(125);
    }
}

static FILE *nac_open_input(const char *path, const char *mode) {
    if (!active) return native_fopen ? native_fopen(path, mode) : NULL;
    if (failed || getpid() != owner || syscall(SYS_gettid) != owner) return nac_fail();
    if (mode[0] != 'r') return native_fopen(path, mode);
    if (strspn(mode + 1, "be") != strlen(mode + 1) || strlen(path) >= NAC_PATH || !*path) return nac_fail();
    if (!replay && !read_count && strcmp(path, getenv("NANO_AS_CAPTURE_INPUT"))) return nac_fail();
    char copy[NAC_PATH];
    if (replay) {
        if (replay_at >= read_count || strcmp(path, replay_at ? retained[replay_at].path :
                                              getenv("NANO_AS_CAPTURE_INPUT"))) return nac_fail();
        NacRead *entry = &retained[replay_at];
        unsigned index = replay_at++;
        if (entry->error) { errno = (int)entry->error; return NULL; }
        FILE *result = nac_name(copy, capture_prefix, "input", index) ? nac_file(copy, O_RDONLY, "rb") : NULL;
        return result ? result : nac_fail();
    }
    if (read_count >= NAC_READS) return nac_fail();
    errno = 0;
    /* I follow source symlinks, but do not block on a substituted FIFO. */
    int input_fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    FILE *input = input_fd >= 0 ? fdopen(input_fd, "rb") : NULL, *result = NULL;
    if (input_fd >= 0 && !input) { close(input_fd); return nac_fail(); }
    unsigned error = input ? 0 : (unsigned)errno;
    uint64_t size = 0, hash = 0;
    if (input) {
        struct stat st;
        if (fstat(fileno(input), &st) || !S_ISREG(st.st_mode) || st.st_size < 0 ||
            (uint64_t)st.st_size > NAC_FILE_LIMIT || (uint64_t)st.st_size > NAC_TOTAL_LIMIT - total_bytes) {
            fclose(input); return nac_fail();
        }
        size_t capacity = (size_t)st.st_size;
        unsigned char *bytes = malloc(capacity + 1);
        if (!bytes) { fclose(input); return nac_fail(); }
        size = fread(bytes, 1, capacity + 1, input);
        int ok = size <= capacity && feof(input) && !ferror(input);
        if (fclose(input)) ok = 0;
        FILE *output = ok && nac_name(copy, capture_prefix, "input", read_count)
            ? nac_file(copy, O_WRONLY | O_CREAT | O_EXCL, "wb") : NULL;
        if (!output) ok = 0;
        else {
            hash = nac_hash(NAC_SEED, bytes, (size_t)size);
            if (fwrite(bytes, 1, (size_t)size, output) != size || fflush(output) || fchmod(fileno(output), 0400)) ok = 0;
            if (fclose(output)) ok = 0;
        }
        free(bytes);
        if (!ok) return nac_fail();
        result = nac_file(copy, O_RDONLY, "rb");
        if (!result) return nac_fail();
    }
    unsigned char header[32] = {0};
    /* The first source has a stable logical identity and an invocation-private
     * descriptor spelling. All later ordered opens retain their exact names. */
    const char *recorded = read_count ? path : primary_input;
    size_t length = strlen(recorded);
    nac_put(header, 1, 4); nac_put(header + 4, length, 4); nac_put(header + 8, error, 4);
    nac_put(header + 16, size, 8); nac_put(header + 24, hash, 8);
    if (!journal || (!input && (!error || error > 4095)) ||
        fwrite(header, 1, 32, journal) != 32 || fwrite(recorded, 1, length, journal) != length || fflush(journal)) {
        if (result) fclose(result);
        return nac_fail();
    }
    record_hash = nac_record_hash(record_hash, header, recorded, read_count++);
    total_bytes += size;
    errno = (int)error;
    return result;
}

FILE *fopen(const char *path, const char *mode) { return nac_open_input(path, mode); }
FILE *fopen64(const char *path, const char *mode) { return nac_open_input(path, mode); }
