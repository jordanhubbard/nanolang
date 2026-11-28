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
#include "../src/nanoisa/nvm_format.h"
typedef struct { uint8_t *array_results; } Nvm2cBuf;
/* I compile the unchanged helper block extracted verbatim by my test builder. */
#include "nvm2c_c_abi_actual.h"

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
        if (position >= length || bytes[position] > 2) return 0;
        size_t width = bytes[position] == 0 ? 4 : 2;
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
            uint8_t op = bytes[position], value = bytes[position + 1];
            if (op == 0) {
                uint8_t tag = bytes[position + 2], kind = bytes[position + 3];
                NvmFunctionEntry fn = {0}; fn.result_count = value; fn.result_tag = tag;
                uint8_t array_results[] = {kind}; Nvm2cBuf owner = {array_results};
                const char *text = c_result_type(&owner, &fn, 0);
                printf("R|%u|%u|%u|%d|%d|%s\n", (unsigned)value, (unsigned)tag,
                       (unsigned)kind, text != NULL, result_is_i64(&fn), text ? text : "");
            } else if (op == 1) {
                printf("L|%u|%s\n", (unsigned)value, c_local_type(value));
            } else {
                printf("A|%u|%d|%u\n", (unsigned)value, aggregate_value_tag(value),
                       (unsigned)aggregate_kind_for_tag(value));
            }
            position += op == 0 ? 4 : 2;
        }
        free(bytes);
    }
    return 0;
}
