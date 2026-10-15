/* I keep both exact catalog APIs live in one process without shared authority. */
#include "nsi_file_binding.h"
#include "nsi_socket_binding.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *read_input(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb"); assert(f);
    assert(fseek(f, 0, SEEK_END) == 0);
    long n = ftell(f); assert(n > 0);
    rewind(f);
    unsigned char *bytes = malloc((size_t)n); assert(bytes);
    assert(fread(bytes, 1, (size_t)n, f) == (size_t)n);
    assert(fclose(f) == 0);
    *size = (size_t)n;
    return bytes;
}
int main(int argc, char **argv) {
    assert(argc == 3);
    size_t file_size, socket_size, count;
    unsigned char *file = read_input(argv[1], &file_size);
    unsigned char *socket = read_input(argv[2], &socket_size);
    NlFileBindingPlan *f = NULL;
    NlSocketBindingPlan *s = NULL;
    assert(nl_file_binding_prepare(file, file_size, &f) == NL_FILE_BINDING_OK);
    assert(nl_socket_binding_prepare(socket, socket_size, &s) == NL_SOCKET_BINDING_OK);
    NlFileBindingPlan *keep_f = f;
    NlSocketBindingPlan *keep_s = s;
    assert(nl_file_binding_prepare(socket, socket_size, &f) == NL_FILE_BINDING_INVALID);
    assert(nl_socket_binding_prepare(file, file_size, &s) == NL_SOCKET_BINDING_INVALID);
    assert(f == keep_f && s == keep_s);
    memset(file, 0, file_size); memset(socket, 0, socket_size);
    free(file); free(socket);
    const unsigned char *fs = nl_file_binding_source_bytes(f, &count);
    assert(fs && count && strstr((const char *)fs, "nsi:nanolang/filesystem"));
    assert(strstr((const char *)fs, "shadow read_byte"));
    const unsigned char *ss = nl_socket_binding_source_bytes(s, &count);
    assert(ss && count && strstr((const char *)ss, "nsi:nanolang/net"));
    assert(!strstr((const char *)ss, "filesystem"));
    unsigned char *saved = malloc(count); assert(saved); memcpy(saved, ss, count);
    nl_file_binding_free(f);
    size_t after;
    assert(nl_socket_binding_source_bytes(s, &after) == ss);
    assert(after == count && memcmp(saved, ss, count) == 0);
    free(saved); nl_socket_binding_free(s);
    puts("PASS simultaneous File/Socket plans, cross-catalog refusal and independent lifetimes");
    return 0;
}
