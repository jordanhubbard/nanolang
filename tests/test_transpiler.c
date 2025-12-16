/**
 * test_transpiler.c - Unit tests for transpiler components
 * 
 * Tests StringBuilder, registries, and error handling.
 */

#include "../src/nanolang.h"
#include "../src/stdlib_runtime.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_EQ(a, b) if ((a) != (b)) { printf("\n    FAILED: %s == %s at line %d (got %d, expected %d)\n", #a, #b, __LINE__, (int)(a), (int)(b)); exit(1); }
#define ASSERT_STR_EQ(a, b) if (strcmp((a), (b)) != 0) { printf("\n    FAILED: %s == %s at line %d\n    got: \"%s\"\n    expected: \"%s\"\n", #a, #b, __LINE__, (a), (b)); exit(1); }

/* g_argc and g_argv are required by runtime/cli.c but not used in tests */
int g_argc = 0;
char **g_argv = NULL;

/* ============================================================================
 * StringBuilder Tests
 * ============================================================================ */

void test_stringbuilder_create() {
    StringBuilder *sb = sb_create();
    ASSERT(sb != NULL);
    ASSERT(sb->buffer != NULL);
    ASSERT(sb->length == 0);
    ASSERT(sb->capacity >= 1024);
    ASSERT(sb->buffer[0] == '\0');
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_append_simple() {
    StringBuilder *sb = sb_create();
    
    sb_append(sb, "Hello");
    ASSERT_EQ(sb->length, 5);
    ASSERT_STR_EQ(sb->buffer, "Hello");
    
    sb_append(sb, " World");
    ASSERT_EQ(sb->length, 11);
    ASSERT_STR_EQ(sb->buffer, "Hello World");
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_append_empty() {
    StringBuilder *sb = sb_create();
    
    sb_append(sb, "");
    ASSERT_EQ(sb->length, 0);
    ASSERT_STR_EQ(sb->buffer, "");
    
    sb_append(sb, "Test");
    ASSERT_EQ(sb->length, 4);
    ASSERT_STR_EQ(sb->buffer, "Test");
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_append_multiple() {
    StringBuilder *sb = sb_create();
    
    for (int i = 0; i < 10; i++) {
        sb_append(sb, "X");
    }
    
    ASSERT_EQ(sb->length, 10);
    ASSERT_STR_EQ(sb->buffer, "XXXXXXXXXX");
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_large_append() {
    StringBuilder *sb = sb_create();
    
    // Create a string larger than initial capacity (1024)
    char large_str[2000];
    for (int i = 0; i < 1999; i++) {
        large_str[i] = 'A';
    }
    large_str[1999] = '\0';
    
    sb_append(sb, large_str);
    ASSERT_EQ(sb->length, 1999);
    ASSERT(sb->capacity >= 1999);
    ASSERT_STR_EQ(sb->buffer, large_str);
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_capacity_growth() {
    StringBuilder *sb = sb_create();
    int initial_capacity = sb->capacity;
    
    // Append enough to trigger reallocation
    char buffer[2048];
    for (int i = 0; i < 2047; i++) {
        buffer[i] = 'B';
    }
    buffer[2047] = '\0';
    
    sb_append(sb, buffer);
    
    // Capacity should have grown
    ASSERT(sb->capacity > initial_capacity);
    ASSERT(sb->capacity >= 2047);
    ASSERT_EQ(sb->length, 2047);
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_null_termination() {
    StringBuilder *sb = sb_create();
    
    sb_append(sb, "Test");
    ASSERT(sb->buffer[sb->length] == '\0');
    
    sb_append(sb, "123");
    ASSERT(sb->buffer[sb->length] == '\0');
    
    free(sb->buffer);
    free(sb);
}

/* ============================================================================
 * Stdlib Runtime Generation Tests
 * ============================================================================ */

void test_generate_string_operations() {
    StringBuilder *sb = sb_create();
    
    generate_string_operations(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "char_at") != NULL);
    ASSERT(strstr(sb->buffer, "string_from_char") != NULL);
    ASSERT(strstr(sb->buffer, "is_digit") != NULL);
    ASSERT(strstr(sb->buffer, "is_alpha") != NULL);
    ASSERT(strstr(sb->buffer, "int_to_string") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_generate_file_operations() {
    StringBuilder *sb = sb_create();
    
    generate_file_operations(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "nl_os_file_read") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_file_write") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_file_exists") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_file_read_bytes") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_generate_dir_operations() {
    StringBuilder *sb = sb_create();
    
    generate_dir_operations(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "nl_os_dir_create") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_dir_remove") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_dir_list") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_getcwd") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_generate_path_operations() {
    StringBuilder *sb = sb_create();
    
    generate_path_operations(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "nl_os_path_isfile") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_isdir") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_join") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_basename") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_dirname") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_generate_math_utility_builtins() {
    StringBuilder *sb = sb_create();
    
    generate_math_utility_builtins(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "nl_abs") != NULL);
    ASSERT(strstr(sb->buffer, "nl_min") != NULL);
    ASSERT(strstr(sb->buffer, "nl_max") != NULL);
    ASSERT(strstr(sb->buffer, "nl_cast_int") != NULL);
    ASSERT(strstr(sb->buffer, "nl_println") != NULL);
    ASSERT(strstr(sb->buffer, "nl_array_length") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_stdlib_runtime_complete() {
    StringBuilder *sb = sb_create();
    
    generate_stdlib_runtime(sb);
    
    // Should include all components
    ASSERT(sb->length > 5000);  // Substantial amount of code
    ASSERT(strstr(sb->buffer, "OS Standard Library") != NULL);
    ASSERT(strstr(sb->buffer, "Advanced String Operations") != NULL);
    ASSERT(strstr(sb->buffer, "Math and Utility Built-in Functions") != NULL);
    
    free(sb->buffer);
    free(sb);
}

/* ============================================================================
 * Memory Safety Tests
 * ============================================================================ */

void test_stringbuilder_no_buffer_overflow() {
    StringBuilder *sb = sb_create();
    
    // Append many small strings to test growth
    for (int i = 0; i < 1000; i++) {
        sb_append(sb, "test");
    }
    
    ASSERT_EQ(sb->length, 4000);
    ASSERT(sb->capacity >= 4000);
    
    // Verify no corruption
    for (int i = 0; i < 4000; i += 4) {
        ASSERT(sb->buffer[i] == 't');
        ASSERT(sb->buffer[i+1] == 'e');
        ASSERT(sb->buffer[i+2] == 's');
        ASSERT(sb->buffer[i+3] == 't');
    }
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_empty_appends() {
    StringBuilder *sb = sb_create();
    
    // Many empty appends shouldn't break anything
    for (int i = 0; i < 100; i++) {
        sb_append(sb, "");
    }
    
    ASSERT_EQ(sb->length, 0);
    ASSERT_STR_EQ(sb->buffer, "");
    
    free(sb->buffer);
    free(sb);
}

/* ============================================================================
 * Integration Tests
 * ============================================================================ */

void test_generate_all_stdlib_functions() {
    StringBuilder *sb = sb_create();
    
    // Generate all stdlib components
    generate_math_utility_builtins(sb);
    generate_string_operations(sb);
    generate_file_operations(sb);
    generate_dir_operations(sb);
    generate_path_operations(sb);
    
    // Should have substantial content
    ASSERT(sb->length > 10000);
    
    // Verify key functions are present
    ASSERT(strstr(sb->buffer, "nl_abs") != NULL);
    ASSERT(strstr(sb->buffer, "char_at") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_file_read") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_dir_create") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_join") != NULL);
    
    free(sb->buffer);
    free(sb);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main() {
    printf("Running transpiler component tests:\n");
    
    printf("\nStringBuilder Tests:\n");
    TEST(stringbuilder_create);
    TEST(stringbuilder_append_simple);
    TEST(stringbuilder_append_empty);
    TEST(stringbuilder_append_multiple);
    TEST(stringbuilder_large_append);
    TEST(stringbuilder_capacity_growth);
    TEST(stringbuilder_null_termination);
    
    printf("\nStdlib Runtime Generation Tests:\n");
    TEST(generate_string_operations);
    TEST(generate_file_operations);
    TEST(generate_dir_operations);
    TEST(generate_path_operations);
    TEST(generate_math_utility_builtins);
    TEST(stdlib_runtime_complete);
    
    printf("\nMemory Safety Tests:\n");
    TEST(stringbuilder_no_buffer_overflow);
    TEST(stringbuilder_empty_appends);
    
    printf("\nIntegration Tests:\n");
    TEST(generate_all_stdlib_functions);
    
    printf("\n✓ All %d tests passed!\n", 16);
    return 0;
}

