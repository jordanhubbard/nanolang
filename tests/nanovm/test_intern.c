/*
 * test_intern.c — unit tests for the NanoVM string interning table.
 *
 * The intern table is a chained hash-bucket structure (heap.c). These tests
 * exercise its observable contract: identical content is deduplicated to a
 * single VmString, reference counts track intern hits, freed strings are
 * unlinked so identical content re-interns to a fresh object, and the table
 * stays correct as it grows past its initial bucket capacity.
 */

#include "nanovm/heap.h"
#include "nanovm/value.h"
#include "nanoisa/isa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Required by runtime/cli.c (linked via NANOVM_OBJECTS chain) */
int g_argc = 0;
char **g_argv = NULL;

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

static NanoValue str_val(VmString *s) {
    NanoValue v;
    v.tag = TAG_STRING;
    v.as.string = s;
    return v;
}

/* Identical content is deduplicated: same pointer, bumped ref count. */
static void test_dedup_same_pointer(void) {
    const char *test_name = "intern: identical content dedups to one object";
    VmHeap heap;
    vm_heap_init(&heap);

    VmString *a = vm_string_new(&heap, "hello", 5);
    VmString *b = vm_string_new(&heap, "hello", 5);
    ASSERT(a != NULL && b != NULL, "allocation failed");
    ASSERT(a == b, "identical content should intern to the same object");
    ASSERT(a->header.ref_count == 2, "second intern should bump ref_count to 2");
    ASSERT(heap.intern_count == 1, "only one string should be interned");

    vm_release(&heap, str_val(a));
    vm_release(&heap, str_val(b));
    vm_heap_destroy(&heap);
    PASS(test_name);
}

/* Distinct content produces distinct interned objects. */
static void test_distinct_content(void) {
    const char *test_name = "intern: distinct content produces distinct objects";
    VmHeap heap;
    vm_heap_init(&heap);

    VmString *a = vm_string_new(&heap, "foo", 3);
    VmString *b = vm_string_new(&heap, "bar", 3);
    ASSERT(a != b, "different content must not dedup");
    ASSERT(heap.intern_count == 2, "two distinct strings expected");

    vm_release(&heap, str_val(a));
    vm_release(&heap, str_val(b));
    vm_heap_destroy(&heap);
    PASS(test_name);
}

/* Freeing an interned string unlinks it; re-interning yields a fresh object. */
static void test_unlink_on_free(void) {
    const char *test_name = "intern: freed string is unlinked and re-interns fresh";
    VmHeap heap;
    vm_heap_init(&heap);

    VmString *a = vm_string_new(&heap, "transient", 9);
    ASSERT(heap.intern_count == 1, "one interned string expected");
    vm_release(&heap, str_val(a)); /* ref_count 1 -> 0, frees + unlinks */
    ASSERT(heap.intern_count == 0, "freed string must be removed from table");

    VmString *b = vm_string_new(&heap, "transient", 9);
    ASSERT(b != NULL, "re-intern allocation failed");
    ASSERT(heap.intern_count == 1, "re-interned string should be present");
    ASSERT(b->header.ref_count == 1, "fresh object starts at ref_count 1");

    vm_release(&heap, str_val(b));
    vm_heap_destroy(&heap);
    PASS(test_name);
}

/* Table stays correct while growing past the initial bucket capacity, and
 * dedup still works for content seen earlier. */
static void test_growth_and_dedup(void) {
    const char *test_name = "intern: dedup survives bucket growth";
    VmHeap heap;
    vm_heap_init(&heap);

    enum { N = 2000 };
    VmString *first[16];
    for (int i = 0; i < N; i++) {
        char buf[32];
        int len = snprintf(buf, sizeof(buf), "key-%d", i);
        VmString *s = vm_string_new(&heap, buf, (uint32_t)len);
        if (i < 16) first[i] = s;
    }
    ASSERT(heap.intern_count == N, "each distinct key should intern once");
    ASSERT(heap.intern_bucket_count > 256, "table should have grown");

    /* Re-request early keys: must dedup to the original object after growth. */
    for (int i = 0; i < 16; i++) {
        char buf[32];
        int len = snprintf(buf, sizeof(buf), "key-%d", i);
        VmString *s = vm_string_new(&heap, buf, (uint32_t)len);
        ASSERT(s == first[i], "dedup must survive rehash");
        vm_release(&heap, str_val(s)); /* drop the extra ref we just took */
    }
    ASSERT(heap.intern_count == N, "dedup must not create new entries");

    vm_heap_destroy(&heap);
    PASS(test_name);
}

/* Embedded NUL bytes are compared by length, not C-string terminator. */
static void test_embedded_nul(void) {
    const char *test_name = "intern: embedded NUL bytes disambiguate content";
    VmHeap heap;
    vm_heap_init(&heap);

    VmString *a = vm_string_new(&heap, "a\0b", 3);
    VmString *b = vm_string_new(&heap, "a\0c", 3);
    ASSERT(a != b, "content differing after a NUL must not dedup");
    ASSERT(heap.intern_count == 2, "two distinct strings expected");

    vm_release(&heap, str_val(a));
    vm_release(&heap, str_val(b));
    vm_heap_destroy(&heap);
    PASS(test_name);
}

int main(void) {
    printf("[String interning]\n");
    test_dedup_same_pointer();
    test_distinct_content();
    test_unlink_on_free();
    test_growth_and_dedup();
    test_embedded_nul();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
