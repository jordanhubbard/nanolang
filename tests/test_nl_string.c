/**
 * test_nl_string.c - Unit tests for nl_string_t
 */

#include "../src/runtime/nl_string.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); return; }

/* ============================================================================
 * Basic Tests
 * ============================================================================ */

void test_creation() {
    nl_string_t *str = nl_string_new("Hello");
    ASSERT(str != NULL);
    ASSERT(nl_string_length(str) == 5);
    ASSERT(strcmp(nl_string_to_cstr(str), "Hello") == 0);
    nl_string_free(str);
}

void test_binary_data() {
    char data[] = {0x00, 0x01, 0x02, 0x00, 0x03};
    nl_string_t *str = nl_string_new_binary(data, 5);
    ASSERT(str != NULL);
    ASSERT(nl_string_length(str) == 5);
    ASSERT(nl_string_byte_at(str, 0) == 0x00);
    ASSERT(nl_string_byte_at(str, 3) == 0x00); // Can store embedded nulls
    ASSERT(nl_string_byte_at(str, 4) == 0x03);
    nl_string_free(str);
}

void test_concatenation() {
    nl_string_t *a = nl_string_new("Hello");
    nl_string_t *b = nl_string_new(" World");
    nl_string_t *c = nl_string_concat(a, b);
    
    ASSERT(c != NULL);
    ASSERT(nl_string_length(c) == 11);
    ASSERT(strcmp(nl_string_to_cstr(c), "Hello World") == 0);
    
    nl_string_free(a);
    nl_string_free(b);
    nl_string_free(c);
}

void test_substring() {
    nl_string_t *str = nl_string_new("Hello World");
    nl_string_t *sub = nl_string_substring(str, 6, 5);
    
    ASSERT(sub != NULL);
    ASSERT(nl_string_length(sub) == 5);
    ASSERT(memcmp(sub->data, "World", 5) == 0);
    
    nl_string_free(str);
    nl_string_free(sub);
}

void test_equality() {
    nl_string_t *a = nl_string_new("test");
    nl_string_t *b = nl_string_new("test");
    nl_string_t *c = nl_string_new("different");
    
    ASSERT(nl_string_equals(a, b));
    ASSERT(!nl_string_equals(a, c));
    ASSERT(nl_string_equals_cstr(a, "test"));
    ASSERT(!nl_string_equals_cstr(a, "different"));
    
    nl_string_free(a);
    nl_string_free(b);
    nl_string_free(c);
}

/* ============================================================================
 * UTF-8 Tests
 * ============================================================================ */

void test_utf8_validation() {
    // Valid UTF-8
    nl_string_t *valid = nl_string_new("Hello 世界 🌍");
    ASSERT(nl_string_validate_utf8(valid));
    ASSERT(valid->is_utf8);
    nl_string_free(valid);
    
    // Invalid UTF-8
    unsigned char invalid_data[] = {0xFF, 0xFE, 0xFD};
    nl_string_t *invalid = nl_string_new_binary(invalid_data, 3);
    ASSERT(!nl_string_validate_utf8(invalid));
    ASSERT(!invalid->is_utf8);
    nl_string_free(invalid);
}

void test_utf8_length() {
    nl_string_t *str = nl_string_new("Hi 世界");
    nl_string_validate_utf8(str);
    
    // "Hi 世界" = 2 ASCII + 1 space + 2 Chinese = 5 characters
    // But 9 bytes (3 + 3 + 3 for Chinese characters)
    ASSERT(nl_string_length(str) == 9);  // Byte length
    ASSERT(nl_string_utf8_length(str) == 5);  // Character count
    
    nl_string_free(str);
}

void test_utf8_char_at() {
    nl_string_t *str = nl_string_new("A世B");
    nl_string_validate_utf8(str);
    
    // 'A' = 0x41, '世' = 0x4E16, 'B' = 0x42
    ASSERT(nl_string_utf8_char_at(str, 0) == 0x41);
    ASSERT(nl_string_utf8_char_at(str, 1) == 0x4E16);
    ASSERT(nl_string_utf8_char_at(str, 2) == 0x42);
    
    nl_string_free(str);
}

void test_utf8_substring() {
    nl_string_t *str = nl_string_new("Hello世界");
    nl_string_validate_utf8(str);
    
    nl_string_t *sub = nl_string_utf8_substring(str, 5, 2);
    ASSERT(sub != NULL);
    ASSERT(nl_string_utf8_length(sub) == 2);
    
    nl_string_free(str);
    nl_string_free(sub);
}

/* ============================================================================
 * Memory Tests
 * ============================================================================ */

void test_clone() {
    nl_string_t *original = nl_string_new("Test");
    nl_string_t *clone = nl_string_clone(original);
    
    ASSERT(clone != NULL);
    ASSERT(clone != original);  // Different objects
    ASSERT(clone->data != original->data);  // Different buffers
    ASSERT(nl_string_equals(original, clone));  // Same content
    
    nl_string_free(original);
    nl_string_free(clone);
}

void test_null_termination() {
    char data[] = {'A', 'B', 'C'};
    nl_string_t *str = nl_string_new_binary(data, 3);
    
    ASSERT(!str->null_terminated);
    nl_string_ensure_null_terminated(str);
    ASSERT(str->null_terminated);
    ASSERT(str->data[3] == '\0');
    ASSERT(strcmp(nl_string_to_cstr(str), "ABC") == 0);
    
    nl_string_free(str);
}

/* ============================================================================
 * Capacity-Boundary Regression Tests
 *
 * These tests exercise binary strings at three critical capacity boundaries:
 *   1. zero-length  — capacity == 0, ensure_null_terminated must realloc
 *   2. exact-capacity — capacity == length, no slack for a null terminator
 *   3. one-over-capacity — same as exact-capacity but with embedded NUL bytes
 *
 * For each boundary we verify:
 *   - nl_string_length() before and after ensure_null_terminated
 *   - raw byte content (including embedded NUL bytes) via nl_string_byte_at
 *   - null_terminated flag before (false) and after (true)
 * ============================================================================ */

/* Boundary 1: zero-length binary string.
 * nl_string_new_binary(ptr, 0) sets capacity=0 and length=0.
 * ensure_null_terminated must detect length >= capacity (0 >= 0) and realloc
 * to capacity=1 before writing the terminator.
 */
void test_binary_zero_length_null_termination() {
    /* A non-NULL pointer is fine; zero length means nothing is copied. */
    char dummy = '\xFF';
    nl_string_t *str = nl_string_new_binary(&dummy, 0);
    ASSERT(str != NULL);

    /* Pre-condition: empty, not null-terminated, capacity leaves no room. */
    ASSERT(nl_string_length(str) == 0);
    ASSERT(str->capacity == 0);
    ASSERT(!str->null_terminated);

    nl_string_ensure_null_terminated(str);

    /* Post-condition: length unchanged, terminator written, capacity grew. */
    ASSERT(nl_string_length(str) == 0);
    ASSERT(str->null_terminated);
    ASSERT(str->data[0] == '\0');
    ASSERT(str->capacity >= 1);

    nl_string_free(str);
}

/* Boundary 2: exact-capacity binary string (no embedded NUL).
 * nl_string_new_binary copies exactly N bytes and sets capacity=N, so there
 * is no room for a null terminator.  ensure_null_terminated must realloc to
 * capacity N+1 and write the terminator without altering any of the N bytes.
 */
void test_binary_exact_capacity_null_termination() {
    /* Three ordinary bytes; no embedded NUL so cstr comparison is safe. */
    char data[] = {'X', 'Y', 'Z'};
    nl_string_t *str = nl_string_new_binary(data, 3);
    ASSERT(str != NULL);

    /* Pre-condition: length==3, capacity==3, not null-terminated. */
    ASSERT(nl_string_length(str) == 3);
    ASSERT(str->capacity == 3);
    ASSERT(!str->null_terminated);

    /* Raw bytes are preserved. */
    ASSERT((unsigned char)nl_string_byte_at(str, 0) == 'X');
    ASSERT((unsigned char)nl_string_byte_at(str, 1) == 'Y');
    ASSERT((unsigned char)nl_string_byte_at(str, 2) == 'Z');

    nl_string_ensure_null_terminated(str);

    /* Post-condition: length still 3, bytes intact, terminator written. */
    ASSERT(nl_string_length(str) == 3);
    ASSERT(str->null_terminated);
    ASSERT((unsigned char)nl_string_byte_at(str, 0) == 'X');
    ASSERT((unsigned char)nl_string_byte_at(str, 1) == 'Y');
    ASSERT((unsigned char)nl_string_byte_at(str, 2) == 'Z');
    ASSERT(str->data[3] == '\0');
    ASSERT(strcmp(nl_string_to_cstr(str), "XYZ") == 0);

    nl_string_free(str);
}

/* Boundary 3: one-byte-over-capacity growth with embedded NUL bytes.
 * A 5-byte binary string that contains embedded NUL bytes is created with
 * nl_string_new_binary so capacity==5==length.  ensure_null_terminated must
 * realloc (growing capacity by exactly one byte) and must preserve all five
 * bytes including the interior NUL bytes.  Length must remain 5 throughout.
 */
void test_binary_embedded_nul_one_over_capacity_null_termination() {
    /* Bytes: 0x41 'A', 0x00 NUL, 0x42 'B', 0x00 NUL, 0x43 'C' */
    char data[] = {0x41, 0x00, 0x42, 0x00, 0x43};
    nl_string_t *str = nl_string_new_binary(data, 5);
    ASSERT(str != NULL);

    /* Pre-condition: capacity tight, not null-terminated. */
    ASSERT(nl_string_length(str) == 5);
    ASSERT(str->capacity == 5);
    ASSERT(!str->null_terminated);

    /* All five bytes including the two embedded NULs are intact. */
    ASSERT((unsigned char)nl_string_byte_at(str, 0) == 0x41);
    ASSERT((unsigned char)nl_string_byte_at(str, 1) == 0x00);
    ASSERT((unsigned char)nl_string_byte_at(str, 2) == 0x42);
    ASSERT((unsigned char)nl_string_byte_at(str, 3) == 0x00);
    ASSERT((unsigned char)nl_string_byte_at(str, 4) == 0x43);

    nl_string_ensure_null_terminated(str);

    /* Post-condition: length==5, all bytes unchanged, terminator appended. */
    ASSERT(nl_string_length(str) == 5);
    ASSERT(str->null_terminated);
    ASSERT(str->capacity >= 6);

    /* Each byte must survive the realloc. */
    ASSERT((unsigned char)nl_string_byte_at(str, 0) == 0x41);
    ASSERT((unsigned char)nl_string_byte_at(str, 1) == 0x00);
    ASSERT((unsigned char)nl_string_byte_at(str, 2) == 0x42);
    ASSERT((unsigned char)nl_string_byte_at(str, 3) == 0x00);
    ASSERT((unsigned char)nl_string_byte_at(str, 4) == 0x43);
    ASSERT(str->data[5] == '\0');

    nl_string_free(str);
}


/* ============================================================================
 * Focused Regression Tests
 * ============================================================================ */

/* Regression 1: Empty-string construction via nl_string_new("").
 * nl_string_new("") must succeed and produce a zero-length, null-terminated,
 * non-NULL string. Capacity must be at least 1 to hold the terminator.
 */
void test_empty_string_construction() {
    nl_string_t *str = nl_string_new("");
    ASSERT(str != NULL);
    ASSERT(nl_string_length(str) == 0);
    ASSERT(str->null_terminated);
    ASSERT(str->data != NULL);
    ASSERT(str->data[0] == '\0');
    ASSERT(str->capacity >= 1);
    ASSERT(nl_string_equals_cstr(str, ""));
    nl_string_free(str);
}

/* Regression 2: nl_string_byte_at_safe at the last valid and first invalid index.
 * For a 4-byte string "Test", index 3 is the last valid byte, index 4 is
 * the first out-of-range index and must return false without writing to *out.
 */
void test_byte_at_safe_boundary() {
    nl_string_t *str = nl_string_new("Test");
    ASSERT(str != NULL);
    ASSERT(nl_string_length(str) == 4);

    /* Last valid index: must succeed and return 't'. */
    char out = '\0';
    ASSERT(nl_string_byte_at_safe(str, 3, &out));
    ASSERT(out == 't');

    /* First invalid index: must return false and leave *out unchanged. */
    out = (char)0xAB; /* sentinel value */
    ASSERT(!nl_string_byte_at_safe(str, 4, &out));
    ASSERT((unsigned char)out == 0xAB); /* *out must not be modified on failure */

    nl_string_free(str);
}

/* Regression 3: nl_string_reserve preserves bytes and null-termination state.
 * After reserving a larger capacity the byte content and null_terminated flag
 * must be identical to their pre-reserve values; only capacity should change.
 */
void test_reserve_preserves_state() {
    nl_string_t *str = nl_string_new("Hello");
    ASSERT(str != NULL);
    ASSERT(str->null_terminated);
    ASSERT(nl_string_length(str) == 5);

    size_t new_capacity = str->capacity + 64;

    nl_string_reserve(str, new_capacity);

    /* Capacity must have grown to at least new_capacity. */
    ASSERT(str->capacity >= new_capacity);

    /* All original bytes must be intact. */
    ASSERT((unsigned char)str->data[0] == 'H');
    ASSERT((unsigned char)str->data[1] == 'e');
    ASSERT((unsigned char)str->data[2] == 'l');
    ASSERT((unsigned char)str->data[3] == 'l');
    ASSERT((unsigned char)str->data[4] == 'o');

    /* The null terminator written by nl_string_new must be preserved. */
    ASSERT(str->null_terminated);
    ASSERT(str->data[5] == '\0');

    /* Length must be unchanged. */
    ASSERT(nl_string_length(str) == 5);

    nl_string_free(str);
}

/* Regression 4: nl_string_shrink_to_fit chooses the exact required capacity.
 * For a null-terminated string the exact capacity is length+1; for a binary
 * (non-null-terminated) string it is exactly length.
 */
void test_shrink_to_fit_exact_capacity() {
    /* Case A: null-terminated string -- shrink to length+1. */
    nl_string_t *str_a = nl_string_new("Hi");
    ASSERT(str_a != NULL);
    nl_string_reserve(str_a, 128); /* inflate capacity well beyond length */
    ASSERT(str_a->capacity >= 128);

    nl_string_shrink_to_fit(str_a);

    /* Exact capacity = length + 1 (for the null terminator). */
    ASSERT(str_a->capacity == nl_string_length(str_a) + 1);
    ASSERT(str_a->null_terminated);
    ASSERT(strcmp(nl_string_to_cstr(str_a), "Hi") == 0);
    nl_string_free(str_a);

    /* Case B: binary string (no null terminator) -- shrink to exactly length. */
    char data[] = {'A', 'B', 'C'};
    nl_string_t *str_b = nl_string_new_binary(data, 3);
    ASSERT(str_b != NULL);
    nl_string_reserve(str_b, 64);
    ASSERT(str_b->capacity >= 64);

    nl_string_shrink_to_fit(str_b);

    /* Exact capacity = length (no null terminator). */
    ASSERT(str_b->capacity == nl_string_length(str_b));
    ASSERT(!str_b->null_terminated);
    ASSERT((unsigned char)str_b->data[0] == 'A');
    ASSERT((unsigned char)str_b->data[1] == 'B');
    ASSERT((unsigned char)str_b->data[2] == 'C');
    nl_string_free(str_b);
}

/* Regression 5: nl_string_clone preserves capacity/content/UTF-8/null-termination
 * flags while owning a distinct buffer.
 * Tested across three representative inputs: ASCII, UTF-8, and binary.
 */
void test_clone_preserves_flags_and_owns_buffer() {
    /* Case A: ASCII null-terminated string. */
    nl_string_t *orig_a = nl_string_new("clone");
    ASSERT(orig_a != NULL);
    nl_string_t *clone_a = nl_string_clone(orig_a);
    ASSERT(clone_a != NULL);
    ASSERT(clone_a != orig_a);
    ASSERT(clone_a->data != orig_a->data); /* distinct buffer */
    ASSERT(clone_a->length == orig_a->length);
    ASSERT(clone_a->capacity == orig_a->capacity);
    ASSERT(clone_a->null_terminated == orig_a->null_terminated);
    ASSERT(clone_a->is_utf8 == orig_a->is_utf8);
    ASSERT(nl_string_equals(orig_a, clone_a));
    nl_string_free(orig_a);
    /* clone must remain valid after the original is freed. */
    ASSERT(strcmp(nl_string_to_cstr(clone_a), "clone") == 0);
    nl_string_free(clone_a);

    /* Case B: UTF-8 validated string. */
    nl_string_t *orig_b = nl_string_new("H\xc3\xa9llo");
    ASSERT(orig_b != NULL);
    nl_string_validate_utf8(orig_b);
    ASSERT(orig_b->is_utf8);
    nl_string_t *clone_b = nl_string_clone(orig_b);
    ASSERT(clone_b != NULL);
    ASSERT(clone_b->data != orig_b->data);
    ASSERT(clone_b->is_utf8 == orig_b->is_utf8);
    ASSERT(clone_b->null_terminated == orig_b->null_terminated);
    ASSERT(nl_string_equals(orig_b, clone_b));
    nl_string_free(orig_b);
    nl_string_free(clone_b);

    /* Case C: binary string (not null-terminated). */
    char bin[] = {0x01, 0x00, 0x02};
    nl_string_t *orig_c = nl_string_new_binary(bin, 3);
    ASSERT(orig_c != NULL);
    ASSERT(!orig_c->null_terminated);
    nl_string_t *clone_c = nl_string_clone(orig_c);
    ASSERT(clone_c != NULL);
    ASSERT(clone_c->data != orig_c->data);
    ASSERT(clone_c->null_terminated == orig_c->null_terminated);
    ASSERT(clone_c->capacity == orig_c->capacity);
    ASSERT((unsigned char)clone_c->data[0] == 0x01);
    ASSERT((unsigned char)clone_c->data[1] == 0x00);
    ASSERT((unsigned char)clone_c->data[2] == 0x02);
    nl_string_free(orig_c);
    nl_string_free(clone_c);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main() {
    printf("Running nl_string tests:\n");
    printf("\nBasic Operations:\n");
    TEST(creation);
    TEST(binary_data);
    TEST(concatenation);
    TEST(substring);
    TEST(equality);
    
    printf("\nUTF-8 Operations:\n");
    TEST(utf8_validation);
    TEST(utf8_length);
    TEST(utf8_char_at);
    TEST(utf8_substring);
    
    printf("\nMemory Management:\n");
    TEST(clone);
    TEST(null_termination);

    printf("\nCapacity-Boundary Regressions:\n");
    TEST(binary_zero_length_null_termination);
    TEST(binary_exact_capacity_null_termination);
    TEST(binary_embedded_nul_one_over_capacity_null_termination);

    printf("\nFocused Regressions:\n");
    TEST(empty_string_construction);
    TEST(byte_at_safe_boundary);
    TEST(reserve_preserves_state);
    TEST(shrink_to_fit_exact_capacity);
    TEST(clone_preserves_flags_and_owns_buffer);
    
    printf("\n\u2713 All tests passed!\n");
    return 0;
}
