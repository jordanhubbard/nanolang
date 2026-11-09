#ifndef __APPLE__
#define _GNU_SOURCE
#endif
#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include TRACE_HELPERS

typedef struct {
    unsigned char bytes[4096];
    size_t used, threshold, calls, attempted, errors;
    int mode, closed;
    bool injected;
} Sink;

static size_t accept(Sink *sink, const char *bytes, size_t count, bool *failed) {
    sink->calls++;
    sink->attempted += count;
    *failed = false;
    if ((sink->mode == 2 || sink->mode == 3) &&
        sink->used >= sink->threshold && (!sink->injected || sink->mode == 3)) {
        sink->injected = true;
        sink->errors++;
        errno = EIO;
        *failed = true;
        return 0;
    }
    if ((sink->mode == 2 || sink->mode == 3) && !sink->injected &&
        sink->used < sink->threshold && count > sink->threshold - sink->used)
        count = sink->threshold - sink->used;
    if (sink->mode == 1 && count > 3) count = 3;
    assert(count <= sizeof(sink->bytes) - sink->used);
    memcpy(sink->bytes + sink->used, bytes, count);
    sink->used += count;
    return count;
}

#ifdef __APPLE__
static int sink_write(void *cookie, const char *bytes, int count) {
    assert(count >= 0);
    bool failed;
    size_t written = accept(cookie, bytes, (size_t)count, &failed);
    return failed ? -1 : (int)written;
}
#else
static ssize_t sink_write(void *cookie, const char *bytes, size_t count) {
    bool failed;
    size_t written = accept(cookie, bytes, count, &failed);
    /* I follow the GNU cookie hook's zero-on-error convention. */
    return failed ? 0 : (ssize_t)written;
}
#endif

static int sink_close(void *cookie) {
    Sink *sink = cookie;
    sink->closed++;
    return 0;
}

static void measure(const char *name, const char *input, int mode,
                    size_t threshold, bool buffered) {
    Sink sink = {.threshold = threshold, .mode = mode};
#ifdef __APPLE__
    FILE *out = funopen(&sink, NULL, sink_write, NULL, sink_close);
#else
    cookie_io_functions_t hooks = {.write = sink_write, .close = sink_close};
    FILE *out = fopencookie(&sink, "w", hooks);
#endif
    assert(out);
    char buffer[64];
    assert(setvbuf(out, buffered ? buffer : NULL, buffered ? _IOFBF : _IONBF,
                   buffered ? sizeof(buffer) : 0) == 0);
    errno = E2BIG;
    shadow_json_escape(out, input);
    int helper_errno = errno, helper_error = !!ferror(out);
    errno = 0;
    int flush = fflush(out), flush_errno = errno, flush_error = !!ferror(out);
    errno = 0;
    int close_result = fclose(out), close_errno = errno;
    assert(sink.closed == 1);
    printf("{\"input\":\"%s\",\"mode\":%d,\"threshold\":%zu,\"buffered\":%s,"
           "\"helper_errno\":%d,\"helper_error\":%d,\"flush\":%d,\"flush_errno\":%d,"
           "\"flush_error\":%d,\"close\":%d,\"close_errno\":%d,\"write_calls\":%zu,"
           "\"attempted_bytes\":%zu,\"errors\":%zu,\"delivered_hex\":\"",
           name, mode, threshold, buffered ? "true" : "false", helper_errno,
           helper_error, flush, flush_errno, flush_error, close_result, close_errno,
           sink.calls, sink.attempted, sink.errors);
    for (size_t i = 0; i < sink.used; i++) printf("%02x", sink.bytes[i]);
    puts("\"}");
}

int main(void) {
    /* I exercise the escaper here; the unchanged full event has prior controls. */
    (void)shadow_trace_event;
    static const size_t thresholds[] = {0, 1, 2, 5, 12, 25};
    static const char *names[] = {"ordinary", "escapes", "utf8"};
    static const char *inputs[] = {
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmno",
        "AB\\CD\"EF\nGH\tIJ\rKL\x01" "MNabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "AB\xc3\xa9" "CD\xf0\x9d\x95\x90" "EFabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    };
    for (size_t i = 0; i < sizeof(inputs)/sizeof(inputs[0]); i++)
        for (int mode = 0; mode < 4; mode++)
            for (int buffered = 0; buffered <= 1; buffered++)
                for (size_t j = 0; j < sizeof(thresholds)/sizeof(thresholds[0]); j++)
                    measure(names[i], inputs[i], mode, thresholds[j], buffered != 0);
    return 0;
}
