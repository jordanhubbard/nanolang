/* I test a GNU-as file-open boundary, not a general-purpose preload runtime.
 * My trial is single-process/single-threaded, with a private trusted directory.
 * Native-endian records contain uint32 path length, uint32 copy id, then path
 * bytes. Names are not line-delimited. I never fall through to live replay reads.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static FILE *(*native_fopen)(const char *, const char *);
static unsigned next_copy;
static unsigned next_read;

static FILE *capture_open(const char *path, const char *mode) {
    if (!native_fopen) native_fopen = dlsym(RTLD_NEXT, "fopen");
    if (!native_fopen) { errno = ENOSYS; return NULL; }
    const char *directory = getenv("NANO_AS_TRIAL_DIRECTORY");
    const char *phase = getenv("NANO_AS_TRIAL_PHASE");
    if (!directory || !phase || (strcmp(mode, "r") && strcmp(mode, "rb")))
        return native_fopen(path, mode);
    char manifest[4096], copy[4096];
    int length = snprintf(manifest, sizeof(manifest), "%s/reads", directory);
    if (length < 0 || (size_t)length >= sizeof(manifest)) { errno = ENAMETOOLONG; return NULL; }
    size_t path_length = strlen(path);
    if (path_length > 4095) { errno = ENAMETOOLONG; return NULL; }
    if (!strcmp(phase, "replay")) {
        FILE *record = native_fopen(manifest, "rb");
        if (!record) return NULL;
        uint32_t header[2];
        char saved[4096];
        FILE *result = NULL;
        while (fread(header, sizeof(header), 1, record) == 1) {
            if (header[0] >= sizeof(saved) || fread(saved, 1, header[0], record) != header[0]) break;
            if (header[1] != next_read) continue;
            if (header[0] != path_length || memcmp(saved, path, path_length)) break;
            length = snprintf(copy, sizeof(copy), "%s/input-%u", directory, header[1]);
            if (length >= 0 && (size_t)length < sizeof(copy)) result = native_fopen(copy, mode);
            if (result) next_read++;
            break;
        }
        fclose(record);
        if (!result) errno = ENOENT;
        return result;
    }
    if (strcmp(phase, "capture")) { errno = EINVAL; return NULL; }
    FILE *input = native_fopen(path, mode);
    if (!input) return NULL;
    struct stat st;
    if (fstat(fileno(input), &st) || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        st.st_size > 32 * 1024 * 1024 || next_copy >= 256) {
        fclose(input); errno = EFBIG; return NULL;
    }
    size_t capacity = (size_t)st.st_size;
    unsigned char *bytes = malloc(capacity + 1);
    if (!bytes) { fclose(input); return NULL; }
    size_t size = fread(bytes, 1, capacity + 1, input);
    int ok = !ferror(input) && feof(input) && size <= capacity;
    fclose(input);
    unsigned id = next_copy++;
    length = snprintf(copy, sizeof(copy), "%s/input-%u", directory, id);
    if (length < 0 || (size_t)length >= sizeof(copy)) ok = 0;
    FILE *output = NULL;
    int fd = ok ? open(copy, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600) : -1;
    if (fd >= 0) {
        output = fdopen(fd, "wb");
        if (!output) close(fd);
    }
    if (!output) ok = 0;
    else {
        if (fwrite(bytes, 1, size, output) != size) ok = 0;
        if (fclose(output)) ok = 0;
    }
    free(bytes);
    if (ok) {
        FILE *record = native_fopen(manifest, "ab");
        uint32_t header[2] = {(uint32_t)path_length, id};
        if (!record) ok = 0;
        else {
            if (fwrite(header, sizeof(header), 1, record) != 1 ||
                fwrite(path, 1, path_length, record) != path_length) ok = 0;
            if (fclose(record)) ok = 0;
        }
    }
    if (!ok) { errno = EIO; return NULL; }
    /* I expose a deterministic test barrier after retaining a selected read. */
    const char *pause_path = getenv("NANO_AS_TRIAL_PAUSE_PATH");
    static int paused;
    if (!paused && pause_path && !strcmp(path, pause_path)) {
        const char *notify = getenv("NANO_AS_TRIAL_NOTIFY_FD");
        const char *release = getenv("NANO_AS_TRIAL_RELEASE_FD");
        char byte;
        paused = 1;
        if (!notify || !release || write(atoi(notify), "1", 1) != 1 ||
            read(atoi(release), &byte, 1) != 1) { errno = EIO; return NULL; }
    }
    return native_fopen(copy, mode);
}

FILE *fopen(const char *path, const char *mode) { return capture_open(path, mode); }
FILE *fopen64(const char *path, const char *mode) { return capture_open(path, mode); }
