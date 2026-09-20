#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static size_t live, attempts, fail, hits, replacement_faults;
static FILE *output;
static int inject(void) {
    if (++attempts != fail) return 0;
    hits++;
    if (ftell(output) == 3) replacement_faults++;
    return 1;
}
static void *tracked_malloc(size_t bytes) {
    if (bytes && inject()) return NULL;
    void *pointer = malloc(bytes); if (pointer) live++; return pointer;
}
static void *tracked_calloc(size_t count, size_t bytes) {
    if (count && bytes && inject()) return NULL;
    void *pointer = calloc(count, bytes); if (pointer) live++; return pointer;
}
static void tracked_free(void *pointer) {if (pointer) {assert(live); live--; free(pointer);}}
#define malloc tracked_malloc
#define calloc tracked_calloc
#define free tracked_free
#undef stdout
#define stdout output
#define NVM2C_NO_MAIN
#include "GENERATED"
#undef malloc
#undef calloc
#undef free
static int run(size_t fault) {
    output = tmpfile(); assert(output);
    attempts = hits = 0; fail = fault;
    int64_t value = -91; int status = nvm_owned_entry(&value); fail = 0;
    assert(!live);
    assert(!fflush(output)); long count = ftell(output); rewind(output);
    char text[32] = {0}; assert(count >= 0 && count < (long)sizeof text);
    assert(fread(text, 1, (size_t)count, output) == (size_t)count);
    assert(!fclose(output)); output = NULL;
    assert(hits ? (hits == 1 && status == 1 && value == -91) :
           (status == (ASSERTION_CASE ? 2 : 0) && value == (ASSERTION_CASE ? -91 : 0)));
    assert(hits ? (!strcmp(text, "") || !strcmp(text, "11\n") || !strcmp(text, "11\n12\n")) :
           !strcmp(text, ASSERTION_CASE ? "11\n12\n" : "11\n12\n13\n"));
    fprintf(stderr, "native overwrite assertion=%d fault=%zu hits=%zu status=%d allocations=%zu prefix=%ld live=%zu\n",
            ASSERTION_CASE, fault, hits, status, attempts, count, live);
    return hits != 0;
}
int main(void) {
    assert(!run(0)); assert(!run(0)); int done = 0;
    for (size_t fault = 1; fault < 256; fault++) {
        if (!run(fault)) {done = 1; break;}
        assert(!run(0));
    }
    assert(done && replacement_faults > 0);
    return 0;
}
