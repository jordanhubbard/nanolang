/* llm_pattern_suggestions.c - Pattern-based suggestions for LLM code generation */

#include "llm_pattern_suggestions.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Pattern library entries */
static PatternEntry pattern_library[] = {
    /* Missing shadow test - most common error for LLMs */
    {
        .error_code = "E0001",
        .error_pattern = "missing_shadow_test",
        .pattern_name = "simple_shadow_test",
        .description = "Every function I compile must have a shadow test block that checks at least one claim",
        .code_template =
            "shadow function_name {\n"
            "    assert (== (function_name test_input) expected_output)\n"
            "}",
        .memory_reference = "MEMORY.md#test_patterns_basic",
        .stdlib_examples = "stdlib/array.nano, stdlib/string.nano, stdlib/math.nano",
        .estimated_fix_time = 30
    },

    /* Accumulator pattern for loops */
    {
        .error_code = "W0001",
        .error_pattern = "inefficient_loop_allocation",
        .pattern_name = "accumulator_loop",
        .description = "When a loop builds a result, keep the result as an accumulator",
        .code_template =
            "# BEFORE (inefficient):\n"
            "let results: array<int> = []\n"
            "while (< i 100) { \n"
            "    let temp: array<int> = []\n"
            "    (array_push results value)\n"
            "}\n"
            "\n"
            "# AFTER (efficient):\n"
            "let mut results: array<int> = []\n"
            "while (< i 100) {\n"
            "    (array_push results value)\n"
            "}",
        .memory_reference = "MEMORY.md#accumulator_loop",
        .stdlib_examples = "examples/language/nl_map_array.nano, examples/language/nl_filter.nano",
        .estimated_fix_time = 60
    },

    /* Test coverage patterns */
    {
        .error_code = "W0002",
        .error_pattern = "weak_shadow_test",
        .pattern_name = "comprehensive_edge_case_test",
        .description = "A useful shadow test checks the common path and the edge cases",
        .code_template =
            "shadow array_operation {\n"
            "    # Happy path\n"
            "    assert (== (array_operation [1,2,3]) expected_value)\n"
            "    # Empty case\n"
            "    assert (== (array_operation []) expected_empty)\n"
            "    # Single element\n"
            "    assert (== (array_operation [42]) expected_single)\n"
            "    # Negative/boundary cases\n"
            "    assert (== (array_operation [-1,-2,-3]) expected_negative)\n"
            "}",
        .memory_reference = "MEMORY.md#test_patterns_edge_cases",
        .stdlib_examples = "stdlib/array.nano, stdlib/math.nano",
        .estimated_fix_time = 90
    },

    /* Function composition pattern */
    {
        .error_code = "W0003",
        .error_pattern = "function_not_found",
        .pattern_name = "module_function_reference",
        .description = "When I call an external function, the module must be imported",
        .code_template =
            "import \"stdlib/array.nano\" as array_mod\n"
            "\n"
            "fn my_function(arr: array<int>) -> int {\n"
            "    return (array_mod.array_length arr)\n"
            "}",
        .memory_reference = "MEMORY.md#module_imports",
        .stdlib_examples = "stdlib/array.nano, stdlib/string.nano",
        .estimated_fix_time = 45
    },

    /* Type annotation pattern */
    {
        .error_code = "E0002",
        .error_pattern = "type_mismatch",
        .pattern_name = "explicit_type_annotation",
        .description = "I require explicit types; add the missing annotation",
        .code_template =
            "# Variables must have explicit types\n"
            "let x: int = 42               # Correct\n"
            "let x = 42                    # Error - missing type\n"
            "\n"
            "# Function parameters must have types\n"
            "fn add(a: int, b: int) -> int {  # All types explicit\n"
            "    return (+ a b)\n"
            "}",
        .memory_reference = "MEMORY.md#explicit_types",
        .stdlib_examples = "examples/language/nl_types.nano",
        .estimated_fix_time = 60
    },

    /* Early return pattern */
    {
        .error_code = "I0001",
        .error_pattern = "deep_nesting",
        .pattern_name = "early_return",
        .description = "Use early returns when they say the same thing with less nesting",
        .code_template =
            "# BEFORE (nested):\n"
            "fn validate(input: string) -> bool {\n"
            "    if (!= input \"\") {\n"
            "        if (< (str_length input) 10) {\n"
            "            return true\n"
            "        }\n"
            "    }\n"
            "    return false\n"
            "}\n"
            "\n"
            "# AFTER (early return):\n"
            "fn validate(input: string) -> bool {\n"
            "    if (== input \"\") { return false }\n"
            "    if (>= (str_length input) 10) { return false }\n"
            "    return true\n"
            "}",
        .memory_reference = "MEMORY.md#early_return_pattern",
        .stdlib_examples = "examples/language/nl_search.nano",
        .estimated_fix_time = 45
    }
};

static int library_size = sizeof(pattern_library) / sizeof(PatternEntry);

void llm_pattern_init(void) {
    /* Pattern library is static; initialization is a no-op */
}

bool llm_is_missing_shadow_test_error(const char *message) {
    if (!message) return false;
    return strstr(message, "shadow") != NULL &&
           strstr(message, "test") != NULL;
}

bool llm_is_type_mismatch_error(const char *message) {
    if (!message) return false;
    return strstr(message, "type mismatch") != NULL ||
           strstr(message, "Type mismatch") != NULL ||
           strstr(message, "expected") != NULL;
}

bool llm_is_undefined_function_error(const char *message) {
    if (!message) return false;
    return strstr(message, "undefined") != NULL ||
           strstr(message, "not found") != NULL ||
           strstr(message, "not defined") != NULL;
}

PatternSuggestion *llm_get_pattern_suggestion(const char *error_code,
                                              const char *error_message,
                                              const char *function_name) {
    /* function_name is reserved for future use (e.g. fuzzy-matching similar
       function names when suggesting fixes for "function not found" errors) */
    (void)function_name;

    if (!error_code || !error_message) {
        return NULL;
    }

    PatternSuggestion *result = NULL;

    /* Detect error pattern from message */
    const char *detected_pattern = NULL;

    if (llm_is_missing_shadow_test_error(error_message)) {
        detected_pattern = "missing_shadow_test";
    } else if (llm_is_type_mismatch_error(error_message)) {
        detected_pattern = "type_mismatch";
    } else if (llm_is_undefined_function_error(error_message)) {
        detected_pattern = "function_not_found";
    }

    /* Find matching pattern in library */
    for (int i = 0; i < library_size; i++) {
        PatternEntry *entry = &pattern_library[i];

        if (detected_pattern && strcmp(entry->error_pattern, detected_pattern) == 0) {
            result = (PatternSuggestion *)malloc(sizeof(PatternSuggestion));
            if (result) {
                result->pattern_name = (char *)malloc(strlen(entry->pattern_name) + 1);
                strcpy(result->pattern_name, entry->pattern_name);

                result->pattern_description = (char *)malloc(strlen(entry->description) + 1);
                strcpy(result->pattern_description, entry->description);

                result->code_template = (char *)malloc(strlen(entry->code_template) + 1);
                strcpy(result->code_template, entry->code_template);

                result->memory_reference = (char *)malloc(strlen(entry->memory_reference) + 1);
                strcpy(result->memory_reference, entry->memory_reference);

                result->similar_functions = (char *)malloc(strlen(entry->stdlib_examples) + 1);
                strcpy(result->similar_functions, entry->stdlib_examples);

                result->estimated_fix_time_sec = entry->estimated_fix_time;
            }
            break;
        }
    }

    return result;
}

void llm_free_pattern_suggestion(PatternSuggestion *ps) {
    if (!ps) return;
    free(ps->pattern_name);
    free(ps->pattern_description);
    free(ps->code_template);
    free(ps->memory_reference);
    free(ps->similar_functions);
    free(ps);
}

void llm_pattern_cleanup(void) {
    /* No dynamic allocations in library itself */
}
