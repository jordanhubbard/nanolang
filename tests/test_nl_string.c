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
 * Focused Regression Tests
 *
 * Five targeted cases covering: empty-string construction, byte_at_safe
 * boundary indices, reserve preserving state, shrink_to_fit exact capacity,
 * and clone owning a distinct buffer with all flags preserved.
 * ============================================================================ */

/* Regression 1: empty-string construction via nl_string_new("").
 * The API documents that nl_string_new copies from a C string.  An empty C
 * string must produce a non-NULL nl_string_t with length==0, capacity>=1 to
 * hold the null terminator, null_terminated==true, and the data byte
 * accessible as data[0]=='\0'.
 */
void test_empty_string_construction() {
    nl_string_t *str = nl_string_new("");
    ASSERT(str != NULL);
    ASSERT(nl_string_length(str) == 0);
    ASSERT(str->capacity >= 1);
    ASSERT(str->null_terminated);
    ASSERT(str->data[0] == '\0');
    ASSERT(strcmp(nl_string_to_cstr(str), "") == 0);
    nl_string_free(str);
}

/* Regression 2: nl_string_byte_at_safe at last valid and first invalid index.
 * For a 4-byte string "abcd" the last valid index is 3; the first invalid
 * index is 4.  byte_at_safe must return true and the correct byte for index 3,
 * and false (leaving *out unchanged) for index 4.
 */
void test_byte_at_safe_boundary() {
    nl_string_t *str = nl_string_new("abcd");
    ASSERT(str != NULL);
    ASSERT(nl_string_length(str) == 4);

    /* Last valid index: 3 */
    char result = '\0';
    ASSERT(nl_string_byte_at_safe(str, 3, &result));
    ASSERT(result == 'd');

    /* First invalid index: 4 */
    char sentinel = '\xAB';
    ASSERT(!nl_string_byte_at_safe(str, 4, &sentinel));
    ASSERT(sentinel == '\xAB'); /* out must be unchanged on failure */

    nl_string_free(str);
}

/* Regression 3: nl_string_reserve preserves bytes and null-termination state.
 * Start with a null-terminated "Hello" (length 5, capacity 6).  Reserve a
 * larger capacity; verify that the bytes are identical and null_terminated
 * remains true.  A reserve smaller than the current capacity is a no-op and
 * must not alter any state.
 */
void test_reserve_preserves_state() {
    nl_string_t *str = nl_string_new("Hello");
    ASSERT(str != NULL);
    size_t orig_capacity = str->capacity; /* 6 */
    ASSERT(str->null_terminated);

    /* Grow the capacity. */
    nl_string_reserve(str, orig_capacity + 32);
    ASSERT(str->capacity >= orig_capacity + 32);
    ASSERT(nl_string_length(str) == 5);
    ASSERT(str->null_terminated); /* flag must survive realloc */
    ASSERT(strcmp(nl_string_to_cstr(str), "Hello") == 0);

    /* A no-op reserve (new_capacity <= current) must not change anything. */
    size_t capacity_before = str->capacity;
    nl_string_reserve(str, 1); /* smaller than current */
    ASSERT(str->capacity == capacity_before);
    ASSERT(strcmp(nl_string_to_cstr(str), "Hello") == 0);

    nl_string_free(str);
}

/* Regression 4: nl_string_shrink_to_fit chooses the exact required capacity.
 * Build a string with excess capacity (via reserve), then shrink it.  For a
 * null-terminated string the required capacity is length+1; for a binary
 * string without a terminator it is exactly length.  Verify both cases.
 */
void test_shrink_to_fit_exact_capacity() {
    /* Case A: null-terminated string */
    nl_string_t *str = nl_string_new("Hi");   /* length=2, capacity=3 */
    ASSERT(str != NULL);
    nl_string_reserve(str, 64);               /* bloat capacity */
    ASSERT(str->capacity >= 64);

    nl_string_shrink_to_fit(str);

    /* Expected capacity == length + 1 (room for the null terminator). */
    ASSERT(str->capacity == nl_string_length(str) + 1);
    ASSERT(str->null_terminated);
    ASSERT(strcmp(nl_string_to_cstr(str), "Hi") == 0);
    nl_string_free(str);

    /* Case B: binary string (no null terminator) */
    char data[] = {'P', 'Q', 'R'};
    nl_string_t *bin = nl_string_new_binary(data, 3);
    ASSERT(bin != NULL);
    /* Manually bloat capacity for test. */
    nl_string_reserve(bin, 64);
    ASSERT(!bin->null_terminated);

    nl_string_shrink_to_fit(bin);

    /* Expected capacity == length (no terminator byte needed). */
    ASSERT(bin->capacity == nl_string_length(bin));
    ASSERT(!bin->null_terminated);
    ASSERT((unsigned char)nl_string_byte_at(bin, 0) == 'P');
    ASSERT((unsigned char)nl_string_byte_at(bin, 1) == 'Q');
    ASSERT((unsigned char)nl_string_byte_at(bin, 2) == 'R');
    nl_string_free(bin);
}

/* Regression 5: nl_string_clone preserves capacity, content, UTF-8 flag, and
 * null-termination flag while owning a distinct buffer.
 * Verify for (a) a plain ASCII null-terminated string, (b) a validated UTF-8
 * string, and (c) a binary string that is not null-terminated.
 */
void test_clone_preserves_flags_and_owns_buffer() {
    /* Case A: ASCII null-terminated */
    nl_string_t *orig = nl_string_new("World");
    ASSERT(orig != NULL);
    nl_string_t *clone = nl_string_clone(orig);
    ASSERT(clone != NULL);
    ASSERT(clone != orig);
    ASSERT(clone->data != orig->data);          /* distinct buffer */
    ASSERT(nl_string_length(clone) == nl_string_length(orig));
    ASSERT(clone->capacity == orig->capacity);
    ASSERT(clone->null_terminated == orig->null_terminated);
    ASSERT(clone->is_utf8 == orig->is_utf8);
    ASSERT(nl_string_equals(orig, clone));
    nl_string_free(orig);
    nl_string_free(clone);

    /* Case B: validated UTF-8 string */
    nl_string_t *utf8 = nl_string_new("Héllo");
    ASSERT(utf8 != NULL);
    nl_string_validate_utf8(utf8);
    ASSERT(utf8->is_utf8);
    nl_string_t *utf8_clone = nl_string_clone(utf8);
    ASSERT(utf8_clone != NULL);
    ASSERT(utf8_clone->data != utf8->data);
    ASSERT(utf8_clone->is_utf8 == utf8->is_utf8);
    ASSERT(utf8_clone->null_terminated == utf8->null_terminated);
    ASSERT(nl_string_equals(utf8, utf8_clone));
    nl_string_free(utf8);
    nl_string_free(utf8_clone);

    /* Case C: binary string, not null-terminated */
    char data[] = {0x01, 0x02, 0x03};
    nl_string_t *bin = nl_string_new_binary(data, 3);
    ASSERT(bin != NULL);
    ASSERT(!bin->null_terminated);
    nl_string_t *bin_clone = nl_string_clone(bin);
    ASSERT(bin_clone != NULL);
    ASSERT(bin_clone->data != bin->data);
    ASSERT(!bin_clone->null_terminated);
    ASSERT(bin_clone->capacity == bin->capacity);
    ASSERT((unsigned char)nl_string_byte_at(bin_clone, 0) == 0x01);
    ASSERT((unsigned char)nl_string_byte_at(bin_clone, 1) == 0x02);
    ASSERT((unsigned char)nl_string_byte_at(bin_clone, 2) == 0x03);
    nl_string_free(bin);
    nl_string_free(bin_clone);
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


    printf("\nFocused Regressions:\n");
    TEST(empty_string_construction);
    TEST(byte_at_safe_boundary);
    TEST(reserve_preserves_state);
    TEST(shrink_to_fit_exact_capacity);
    TEST(clone_preserves_flags_and_owns_buffer);

    printf("\nCapacity-Boundary Regressions:\n");
    TEST(binary_zero_length_null_termination);
    TEST(binary_exact_capacity_null_termination);
    TEST(binary_embedded_nul_one_over_capacity_null_termination);
    
    printf("\n✓ All tests passed!\n");
    return 0;
}
