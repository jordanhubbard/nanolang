#ifndef NANO_ASSEMBLER_CAPTURE_H
#define NANO_ASSEMBLER_CAPTURE_H

/* I describe ordered file opens, not an atomic snapshot of a source tree.
 * All integers are little-endian. A seal binds every record and retained copy.
 * My first pathname is the logical retained input, not a transient descriptor
 * spelling. Its bytes bind through its content hash and callers check its
 * logical spelling. Every later pathname describes the actual ordered open. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define NAC_READS 256
#define NAC_PATH 4096
#define NAC_FILE_LIMIT (32ULL * 1024 * 1024)
#define NAC_TOTAL_LIMIT (64ULL * 1024 * 1024)
#define NAC_SEED UINT64_C(14695981039346656037)

typedef struct {
    char path[NAC_PATH];
    uint32_t error;
    uint64_t size, hash;
} NacRead;

static inline uint64_t nac_hash(uint64_t hash, const void *bytes, size_t length) {
    const unsigned char *p = bytes;
    while (length--) { hash ^= *p++; hash *= UINT64_C(1099511628211); }
    return hash;
}

static inline void nac_put(unsigned char *p, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; i++) { p[i] = value & 255; value >>= 8; }
}

static inline uint64_t nac_get(const unsigned char *p, unsigned width) {
    uint64_t value = 0;
    for (unsigned i = width; i; i--) value = (value << 8) | p[i - 1];
    return value;
}

static inline int nac_name(char *path, const char *prefix, const char *suffix, unsigned index) {
    int n = snprintf(path, NAC_PATH, "%s.%s%u", prefix, suffix, index);
    return n > 0 && n < NAC_PATH;
}

static inline FILE *nac_file(const char *path, int flags, const char *mode) {
    int fd = open(path, flags | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0600);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode)) { close(fd); return NULL; }
    FILE *file = fdopen(fd, mode);
    if (!file) close(fd);
    return file;
}

static inline uint64_t nac_record_hash(uint64_t hash, const unsigned char *header,
                                      const char *path, unsigned index) {
    unsigned char normalized[32];
    memcpy(normalized, header, sizeof(normalized));
    if (!index) nac_put(normalized + 4, 0, 4);
    hash = nac_hash(hash, normalized, sizeof(normalized));
    return index ? nac_hash(hash, path, (size_t)nac_get(header + 4, 4)) : hash;
}

static inline int nac_load(const char *prefix, const char *first, NacRead *reads,
                            unsigned *count, uint64_t *digest) {
    char path[NAC_PATH];
    if (!nac_name(path, prefix, "manifest", 0)) return 0;
    FILE *file = nac_file(path, O_RDONLY, "rb");
    if (!file) return 0;
    unsigned char header[32], magic[8];
    int ok = fread(magic, 1, 8, file) == 8 && !memcmp(magic, "NASCAP01", 8);
    unsigned index = 0;
    uint64_t total = 0, hash = NAC_SEED;
    while (ok) {
        if (fread(header, 1, 32, file) != 32) { ok = 0; break; }
        uint64_t kind = nac_get(header, 4), length = nac_get(header + 4, 4);
        uint64_t error = nac_get(header + 8, 4), reserved = nac_get(header + 12, 4);
        uint64_t size = nac_get(header + 16, 8), expected = nac_get(header + 24, 8);
        if (!kind) {
            ok = index && length == index && !error && !reserved && size == total &&
                 expected == hash && fgetc(file) == EOF && !ferror(file);
            break;
        }
        if (kind != 1 || reserved || index >= NAC_READS || !length || length >= NAC_PATH ||
            error > 4095 || size > NAC_FILE_LIMIT || size > NAC_TOTAL_LIMIT - total ||
            (error && (size || expected))) { ok = 0; break; }
        NacRead *entry = &reads[index];
        if (fread(entry->path, 1, (size_t)length, file) != length || memchr(entry->path, 0, (size_t)length)) {
            ok = 0; break;
        }
        entry->path[length] = 0;
        entry->error = (uint32_t)error; entry->size = size; entry->hash = expected;
        if (!index && (error || (first && strcmp(first, entry->path)))) { ok = 0; break; }
        if (!error) {
            FILE *copy = nac_name(path, prefix, "input", index) ? nac_file(path, O_RDONLY, "rb") : NULL;
            if (!copy) { ok = 0; break; }
            unsigned char buffer[8192];
            size_t amount;
            uint64_t actual = NAC_SEED, observed = 0;
            while ((amount = fread(buffer, 1, sizeof(buffer), copy))) {
                observed += amount;
                if (observed > size) break;
                actual = nac_hash(actual, buffer, amount);
            }
            ok = observed == size && actual == expected && feof(copy) && !ferror(copy);
            if (fclose(copy)) ok = 0;
            if (!ok) break;
        }
        total += size;
        hash = nac_record_hash(hash, header, entry->path, index++);
    }
    if (fclose(file)) ok = 0;
    if (ok) { *count = index; *digest = hash; }
    return ok;
}
#endif
