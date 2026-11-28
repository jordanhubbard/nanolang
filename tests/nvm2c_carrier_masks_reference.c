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
#include "../src/nanoisa/isa.h"
/* I compile the unchanged helper block extracted verbatim by my test builder. */
#include "nvm2c_carrier_actual.h"

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

static int valid_packet(const unsigned char *bytes, size_t length) {
    if (length < 4 || length > 1048576) return 0;
    uint32_t count = word(bytes);
    if (count > 262144 || count > (length - 4) / 2) return 0;
    size_t position = 4;
    for (uint32_t i = 0; i < count; i++) {
        if (position >= length || bytes[position] > 1) return 0;
        size_t width = bytes[position] == 1 ? 3 : 2;
        if (width > length - position) return 0;
        position += width;
    }
    return position == length;
}

int main(int argc, char **argv) {
    for (int arg = 1; arg < argc; arg++) {
        size_t length = 0;
        unsigned char *bytes = read_packet(argv[arg], &length);
        if (!bytes || !valid_packet(bytes, length)) { puts("P"); free(bytes); continue; }
        uint32_t count = word(bytes);
        printf("Z|%u\n", count);
        size_t position = 4;
        for (uint32_t i = 0; i < count; i++) {
            uint8_t op = bytes[position], left = bytes[position + 1];
            if (op == 0) {
                printf("K|%u|%u\n", (unsigned)left, (unsigned)scalar_kind_tags(left));
            } else {
                uint16_t mask = (uint16_t)left | (uint16_t)bytes[position + 2] << 8;
                printf("T|%u|%d|%d\n", (unsigned)mask, boxed_carrier_tags(mask), optional_scalar_tags(mask));
            }
            position += op == 1 ? 3 : 2;
        }
        free(bytes);
    }
    return 0;
}
