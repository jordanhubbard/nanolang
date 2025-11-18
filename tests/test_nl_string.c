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
    
    printf("\n✓ All tests passed!\n");
    return 0;
}
