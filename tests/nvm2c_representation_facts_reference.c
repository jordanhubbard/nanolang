#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include "nvm2c_representation_declarations.h"
typedef struct {
    char *err; size_t err_len; int failed;
    size_t local_width, record_width; uint8_t *array_results;
    uint32_t classify_function_index; size_t classify_offset;
    Nvm2cFieldBlock *field_blocks;
} Nvm2cBuf;
/* I compile the unchanged helper block extracted verbatim by my test builder. */
#include "nvm2c_representation_actual.h"

static uint32_t word(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static unsigned char *read_packet(const char *name, size_t *length) {
    int fd = open(name, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat info;
    if (fstat(fd, &info) || !S_ISREG(info.st_mode) || info.st_size < 0 || info.st_size > 1048576) {
        close(fd); return NULL;
    }
    *length = (size_t)info.st_size;
    unsigned char *bytes = malloc(*length ? *length : 1);
    if (!bytes) { close(fd); return NULL; }
    size_t done = 0;
    while (done < *length) {
        ssize_t amount = read(fd, bytes + done, *length - done);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) { free(bytes); close(fd); return NULL; }
        done += (size_t)amount;
    }
    unsigned char extra;
    ssize_t amount;
    do { amount = read(fd, &extra, 1); } while (amount < 0 && errno == EINTR);
    int closed = close(fd);
    if (amount || closed) { free(bytes); return NULL; }
    return bytes;
}

static unsigned half(const unsigned char *p) { return (unsigned)p[0] | (unsigned)p[1] << 8; }
static int location_ok(unsigned policy, unsigned role, uint32_t ordinal, uint32_t local, uint32_t record, size_t span) {
    if (role > 4 || local < 1 || local > 66559 || record < 1 || record > 65535) return 0;
    if (2 * (uint64_t)local * (1 + (uint64_t)record) + 4 * (uint64_t)record + 4 > 1048576) return 0;
    if (policy >= 4) return ordinal < 2 && (policy != 6 || record >= 2);
    if (!role) return 1;
    uint64_t size = role == 1 ? 2 * (uint64_t)local : role == 2 ? 2 * (uint64_t)local * record : role == 3 ? 2 * (uint64_t)record : 2;
    return span <= size && ordinal <= size - span;
}
static int valid_packet(const unsigned char *p, size_t length) {
    if (length < 4 || length > 1048576) return 0;
    uint32_t count = word(p); if (count > 262144 || count > (length - 4) / 5) return 0;
    size_t pos = 4;
    for (uint32_t i = 0; i < count; ++i) {
        if (pos >= length) return 0;
        unsigned op = p[pos]; size_t width;
        if (op == 0 || op == 1) {
            width = 5; if (length - pos < width) return 0;
            if (op == 0 && (p[pos + 1] > 6 || p[pos + 4] > 7)) return 0;
        } else if (op == 2) {
            width = 26; if (length - pos < width) return 0;
            unsigned policy = p[pos + 1]; uint32_t local = word(p + pos + 18), record = word(p + pos + 22);
            if (policy > 6 || p[pos + 4] > 7 || !location_ok(policy, p[pos + 5], word(p + pos + 6), local, record, policy == 3 ? record : 1)) return 0;
        } else if (op == 3) {
            if (length - pos < 10) return 0;
            unsigned policy = p[pos + 1], record = half(p + pos + 8);
            if (policy != 0 && policy != 3 && policy != 5 && policy != 6) return 0;
            if (p[pos + 2] > 7 || !location_ok(policy, p[pos + 3], word(p + pos + 4), 3, record, record)) return 0;
            width = 10 + 2 * (size_t)record;
        } else if (op == 4) {
            if (length - pos < 6 || p[pos + 1] > 7 || p[pos + 3] > 1) return 0;
            width = 6 + (p[pos + 3] ? half(p + pos + 4) : 0);
        } else if (op == 5) {
            if (length - pos < 5 || p[pos + 1] > 7) return 0;
            unsigned n = half(p + pos + 3); width = 5 + 2 * (size_t)n;
            if (width > length - pos) return 0;
            for (unsigned j = 0; j < n; ++j) if (p[pos + 5 + 2 * j] > 6) return 0;
        } else return 0;
        if (width > length - pos) return 0;
        pos += width;
    }
    return pos == length;
}
typedef struct { Nvm2cBuf b; Nvm2cFacts facts; uint8_t *arena, *external; uint8_t stored[2]; char error[2048]; } Frame;
static int frame_init(Frame *f, size_t local, size_t record, uint32_t function, uint32_t offset, unsigned flags) {
    memset(f, 0, sizeof *f); size_t size = 2 * local * (1 + record) + 4 * record + 4;
    f->arena = malloc(size); f->external = malloc(record + 1);
    if (!f->arena || !f->external) { free(f->arena); free(f->external); return 0; }
    memset(f->arena, NVM2C_VK_UNK, size); memset(f->external, NVM2C_VK_UNK, record + 1);
    f->b.local_width = local; f->b.record_width = record; f->b.classify_function_index = function; f->b.classify_offset = offset;
    f->b.err = f->error; f->b.err_len = sizeof f->error; f->b.failed = (flags >> 1) & 1;
    if (flags & 4) strcpy(f->error, "prior");
    f->facts.changed = flags & 1; f->facts.parameters = f->arena; f->facts.fields = f->facts.parameters + 2 * local;
    f->facts.results = f->facts.fields + 2 * local * record; f->b.array_results = f->facts.results + 2 * record;
    f->facts.global_kinds = f->b.array_results + 2; f->facts.global_fields = f->facts.global_kinds + 2; f->facts.global_stored = f->stored;
    return 1;
}
static void frame_free(Frame *f) {
    while (f->b.field_blocks) { Nvm2cFieldBlock *p = f->b.field_blocks; f->b.field_blocks = p->next; free(p); }
    free(f->arena); free(f->external);
}
static uint8_t *destination(Frame *f, unsigned policy, unsigned role, uint32_t ordinal) {
    if (policy >= 4) return policy == 4 ? f->facts.global_kinds + ordinal : f->facts.global_fields + ordinal * f->b.record_width;
    if (role == 1) return f->facts.parameters + ordinal;
    if (role == 2) return f->facts.fields + ordinal;
    if (role == 3) return f->facts.results + ordinal;
    if (role == 4) return f->b.array_results + ordinal;
    return f->external;
}
static int merge(Frame *f, unsigned policy, uint32_t ordinal, uint8_t *dest, const uint8_t *fields) {
    switch (policy) {
    case 0: return merge_fact(&f->b, &f->facts, dest, fields[0]);
    case 1: return merge_parameter(&f->b, &f->facts, dest, fields[0]);
    case 2: return merge_call_parameter(&f->b, &f->facts, dest, fields[0]);
    case 3: return merge_record_results(&f->b, &f->facts, dest, fields);
    case 4: return merge_global_kind(&f->b, &f->facts, ordinal, fields[0]);
    case 5: return merge_global_fields(&f->b, &f->facts, ordinal, fields);
    default: merge_global_map_fields(&f->b, &f->facts, ordinal, fields); return 1;
    }
}
static void values(const uint8_t *p, size_t count) {
    for (size_t i = 0; i < count; ++i) printf(i ? ",%u" : "%u", (unsigned)p[i]);
}
int main(int argc, char **argv) {
    for (int arg = 1; arg < argc; ++arg) {
        size_t length = 0; unsigned char *p = read_packet(argv[arg], &length);
        if (!p || !valid_packet(p, length)) { puts("P"); free(p); continue; }
        uint32_t count = word(p); printf("Z|%u\n", count); size_t pos = 4;
        for (uint32_t i = 0; i < count; ++i) {
            unsigned op = p[pos], a = p[pos + 1], b = p[pos + 2], d = p[pos + 3]; size_t width = 5;
            unsigned policy = a, flags = p[pos + 4], role = policy >= 4 ? 0 : policy == 3 ? 3 : 1;
            uint32_t ordinal = policy >= 4 ? 1 : policy == 3 ? 4 : 5, function = 17, offset = 29, local = 3, record = 4;
            if (op == 2) { width = 26; role = p[pos + 5]; ordinal = word(p + pos + 6); function = word(p + pos + 10); offset = word(p + pos + 14); local = word(p + pos + 18); record = word(p + pos + 22); }
            if (op == 3) { flags = b; role = d; ordinal = word(p + pos + 4); record = half(p + pos + 8); width = 10 + 2 * record; }
            if (op == 4) { flags = a; record = half(p + pos + 4); width = 6 + (d ? record : 0); }
            if (op == 5) { flags = a; width = 5 + 2 * half(p + pos + 3); }
            Frame f; if (!frame_init(&f, local, record, function, offset, flags)) { free(p); return 2; }
            if (op == 0 || op == 2) {
                uint8_t *dest = destination(&f, policy, role, ordinal); *dest = (uint8_t)b;
                uint8_t *incoming = malloc(record + 1); if (!incoming) { frame_free(&f); free(p); return 2; }
                memset(incoming, NVM2C_VK_UNK, record + 1); incoming[0] = (uint8_t)d;
                int accepted = merge(&f, policy, ordinal, dest, incoming);
                printf("C|%u|%u|%u|%u|1|%d|%u|%d|%d|%s\n", policy, flags, b, d, accepted, (unsigned)*dest, f.facts.changed, f.b.failed, f.error); free(incoming);
            } else if (op == 1) {
                f.stored[1] = (uint8_t)a; f.facts.global_kinds[1] = (uint8_t)b;
                f.facts.global_fields[4] = (uint8_t)d; f.facts.global_fields[5] = p[pos + 4];
                printf("M|%u|%u|%u|%u|%d\n", a, b, d, (unsigned)p[pos + 4], exact_global_map_pair(&f.b, &f.facts, 1));
            } else if (op == 3) {
                uint8_t *dest = destination(&f, policy, role, ordinal); memcpy(dest, p + pos + 10, record);
                int accepted = policy == 0 ? merge_fields(&f.b, &f.facts, dest, p + pos + 10 + record) : merge(&f, policy, ordinal, dest, p + pos + 10 + record);
                printf("V|%u|%u|%u|%u|1|%d|%d|%d|%s|", a, b, d, ordinal, accepted, f.facts.changed, f.b.failed, f.error); values(dest, record); putchar('|'); values(p + pos + 10, record); putchar('\n');
            } else if (op == 4) {
                const uint8_t *source = d ? p + pos + 6 : NULL; uint8_t *copy = sim_fields(&f.b, source, (uint8_t)b);
                if (!copy) { frame_free(&f); free(p); return 2; }
                int detached = 1;
                if (d && record) { uint8_t before = source[0]; copy[0] = (uint8_t)(copy[0] + 1); detached = source[0] == before; }
                printf("D|%u|%u|%u|%u|1|1|%d|%d|%d|%s|", a, b, d, record, detached, f.facts.changed, f.b.failed, f.error); values(copy, record); putchar('\n');
            } else {
                unsigned n = half(p + pos + 3), current = b;
                printf("Q|%u|%u|%u|", a, b, n);
                for (unsigned j = 0; j < n; ++j) {
                    policy = p[pos + 5 + 2 * j]; unsigned incoming_value = p[pos + 6 + 2 * j];
                    role = policy >= 4 ? 0 : policy == 3 ? 3 : 1; ordinal = policy >= 4 ? 1 : policy == 3 ? 4 : 5;
                    memset(f.arena, NVM2C_VK_UNK, 2 * local * (1 + record) + 4 * record + 4); memset(f.external, NVM2C_VK_UNK, record + 1);
                    uint8_t *dest = destination(&f, policy, role, ordinal); *dest = (uint8_t)current;
                    uint8_t incoming[5]; memset(incoming, NVM2C_VK_UNK, sizeof incoming); incoming[0] = (uint8_t)incoming_value;
                    int accepted = merge(&f, policy, ordinal, dest, incoming); current = *dest;
                    printf("%s%u^%u^%d^%u^%d^%d^%s", j ? "~" : "", policy, incoming_value, accepted, current, f.facts.changed, f.b.failed, f.error);
                }
                putchar('\n');
            }
            frame_free(&f); pos += width;
        }
        free(p);
    }
    return 0;
}
