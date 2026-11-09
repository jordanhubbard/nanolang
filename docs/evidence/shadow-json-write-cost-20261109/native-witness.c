#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

static int saved_error = -1;
#define CHECK(c) do { if (!(c)) { if (saved_error >= 0) { fflush(stderr); dup2(saved_error, STDERR_FILENO); close(saved_error); saved_error = -1; } fprintf(stderr, "I failed trace control at line %d: %s\n", __LINE__, #c); fflush(stderr); exit(1); } } while (0)

#ifdef COUNT_STDIO
static uint64_t ordinary_calls, block_calls;
static int count_character(int c, FILE *out) { ordinary_calls++; return fputc(c, out); }
#define fputc count_character
#endif
#include "actual_trace_helpers.inc"
#ifdef COUNT_STDIO
#undef fputc
#endif

#include "trace-events.h"

static void escaped(const char *input, const unsigned char *expected, size_t length) {
    FILE *out = tmpfile();
    CHECK(out != NULL);
    shadow_json_escape(out, input);
    CHECK(fflush(out) == 0);
    CHECK(ftell(out) == (long)length);
    rewind(out);
    for (size_t i = 0; i < length; i++) CHECK(fgetc(out) == expected[i]);
    CHECK(fgetc(out) == EOF && !ferror(out));
    CHECK(fclose(out) == 0);
}

static void controls(void) {
    static const char *control[] = {
        "", "\\u0001", "\\u0002", "\\u0003", "\\u0004", "\\u0005", "\\u0006", "\\u0007",
        "\\u0008", "\\t", "\\n", "\\u000b", "\\u000c", "\\r", "\\u000e", "\\u000f",
        "\\u0010", "\\u0011", "\\u0012", "\\u0013", "\\u0014", "\\u0015", "\\u0016", "\\u0017",
        "\\u0018", "\\u0019", "\\u001a", "\\u001b", "\\u001c", "\\u001d", "\\u001e", "\\u001f"
    };
    escaped(NULL, (const unsigned char *)"", 0);
    escaped("", (const unsigned char *)"", 0);
    for (unsigned int c = 1; c <= 255; c++) {
        char input[] = {(char)c, 0};
        const char *expected = c < 32 ? control[c] : c == '"' ? "\\\"" : c == '\\' ? "\\\\" : input;
        escaped(input, (const unsigned char *)expected, strlen(expected));
    }
    escaped("ab\0ignored", (const unsigned char *)"ab", 2);
    escaped("\xc3\xa9\xf0\x9d\x95\x90", (const unsigned char *)"\xc3\xa9\xf0\x9d\x95\x90", 6);
    char long_name[4097];
    memset(long_name, 'A', sizeof(long_name)-1); long_name[sizeof(long_name)-1] = 0;
    escaped(long_name, (const unsigned char *)long_name, sizeof(long_name)-1);

    FILE *out = tmpfile(); CHECK(out != NULL);
    int saved = dup(STDERR_FILENO); CHECK(saved >= 0); saved_error = saved;
    CHECK(fflush(stderr) == 0 && dup2(fileno(out), STDERR_FILENO) >= 0);
    shadow_trace_event("shadow-start", "a\"\\\n", "path\t\xc3\xa9", 23, 1.25, 2);
    const char *expected = "{\"event\":\"shadow-start\",\"name\":\"a\\\"\\\\\\n\",\"source\":\"path\\t\xc3\xa9\",\"line\":23,\"elapsed_seconds\":1.250000,\"failures\":2}\n";
    struct stat status; CHECK(fstat(fileno(out), &status) == 0);
    CHECK(status.st_size == (off_t)strlen(expected));
    CHECK(dup2(saved, STDERR_FILENO) >= 0 && close(saved) == 0); saved_error = -1;
    rewind(out);
    for (size_t i = 0; i < strlen(expected); i++) CHECK(fgetc(out) == (unsigned char)expected[i]);
    CHECK(fgetc(out) == EOF && !ferror(out)); CHECK(fclose(out) == 0);

    out = tmpfile(); CHECK(out != NULL); CHECK(close(fileno(out)) == 0);
    shadow_json_escape(out, "I retain a stream refusal.\n");
    CHECK(fflush(out) == EOF && ferror(out)); CHECK(fclose(out) == EOF);
    puts("I preserve all 255 nonzero byte escapes, null/empty/NUL/UTF-8/long inputs, exact event bytes, boundary flush and stream-error visibility.");
}

int main(int argc, char **argv) {
    static char buffered_stderr[8192];
    if (argc == 1) {
        CHECK(setvbuf(stderr, buffered_stderr, _IOFBF, sizeof(buffered_stderr)) == 0);
        controls(); return 0;
    }
    CHECK(argc == 2 && strcmp(argv[1], "--replay") == 0);
    CHECK(setvbuf(stderr, NULL, _IONBF, 0) == 0);
    struct timespec before, after;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &before) == 0);
    for (size_t i = 0; i < sizeof(events)/sizeof(events[0]); i++) {
        const TraceEvent *event = &events[i];
        shadow_trace_event(event->event, event->name, event->source, event->line, event->elapsed, event->failures);
    }
    CHECK(clock_gettime(CLOCK_MONOTONIC, &after) == 0 && !ferror(stderr));
    double elapsed = (double)(after.tv_sec-before.tv_sec) + (double)(after.tv_nsec-before.tv_nsec)/1e9;
    printf("{\"wall_seconds\":%.9f,\"events\":%zu", elapsed, sizeof(events)/sizeof(events[0]));
#ifdef COUNT_STDIO
    printf(",\"counted\":true,\"fputc_calls\":%" PRIu64 ",\"fwrite_calls\":%" PRIu64, ordinary_calls, block_calls);
#else
    printf(",\"counted\":false");
#endif
    puts("}");
    return 0;
}
