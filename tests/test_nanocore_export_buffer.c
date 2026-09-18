/* I retain buffer ownership through checked allocation and formatting failure. */
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* I supply the ordinary CLI globals when linked with the common runtime. */
int g_argc = 0;
char **g_argv = NULL;

static int allocation_budget = -1;
static int format_budget = -1;
static int allow_allocation(void) {
    if (allocation_budget < 0) return 1;
    if (!allocation_budget) return 0;
    allocation_budget--;
    return 1;
}
static void *buffer_malloc(size_t size) {
    return allow_allocation() ? malloc(size) : NULL;
}
static void *buffer_realloc(void *old, size_t size) {
    return allow_allocation() ? realloc(old, size) : NULL;
}
static int buffer_vsnprintf(char *out, size_t size, const char *format, va_list args) {
    if (format_budget == 0) return -1;
    if (format_budget > 0) format_budget--;
    return vsnprintf(out, size, format, args);
}
#define malloc buffer_malloc
#define realloc buffer_realloc
#define vsnprintf buffer_vsnprintf
#include "../src/nanocore_export.c"
#undef malloc
#undef realloc
#undef vsnprintf

int main(void) {
    char text[2048];
    memset(text, 'a', sizeof(text) - 1);
    text[sizeof(text) - 1] = 0;
    SBuf normal = sbuf_new();
    sbuf_append(&normal, "prefix:");
    sbuf_appendf(&normal, "%s:%d", text, 42);
    char *result = sbuf_finish(&normal);
    assert(result && strlen(result) == strlen(text) + 10);
    assert(!memcmp(result, "prefix:", 7));
    assert(!strcmp(result + strlen(result) - 3, ":42"));
    free(result);

    allocation_budget = 0;
    SBuf initial = sbuf_new();
    sbuf_append(&initial, "ordinary");
    assert(initial.failed && sbuf_finish(&initial) == NULL);

    allocation_budget = 1;
    SBuf growth = sbuf_new();
    char *original = growth.data;
    sbuf_append(&growth, "kept");
    sbuf_append(&growth, text);
    assert(growth.failed && growth.data == original);
    assert(!strcmp(growth.data, "kept"));
    sbuf_appendf(&growth, "%d", 42);
    assert(sbuf_finish(&growth) == NULL);

    allocation_budget = -1;
    SBuf size = sbuf_new();
    assert(!sbuf_ensure(&size, SIZE_MAX));
    assert(size.failed && sbuf_finish(&size) == NULL);

    for (int budget = 0; budget <= 1; budget++) {
        format_budget = budget;
        SBuf format = sbuf_new();
        sbuf_appendf(&format, "%d", 42);
        assert(format.failed && sbuf_finish(&format) == NULL);
    }
    puts("I pass ordinary growth and five checked failure paths.");
    return 0;
}
