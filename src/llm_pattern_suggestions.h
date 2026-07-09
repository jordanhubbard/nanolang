/* llm_pattern_suggestions.h - Pattern-based suggestions for LLM code generation */

#ifndef LLM_PATTERN_SUGGESTIONS_H
#define LLM_PATTERN_SUGGESTIONS_H

#include <stdbool.h>

/* Pattern suggestion structure */
typedef struct {
    char *pattern_name;           /* e.g., "simple_accumulator_test" */
    char *pattern_description;    /* User-friendly description */
    char *code_template;          /* Example code to fix the issue */
    char *memory_reference;       /* e.g., "MEMORY.md#test_patterns_accumulators" */
    char *similar_functions;      /* Comma-separated list of similar functions in stdlib */
    int estimated_fix_time_sec;   /* How long to fix (for UX) */
} PatternSuggestion;

/* Pattern library */
typedef struct {
    const char *error_code;           /* E.g., "E0001" */
    const char *error_pattern;        /* What to match: "missing_shadow_test" */
    const char *pattern_name;
    const char *description;
    const char *code_template;
    const char *memory_reference;
    const char *stdlib_examples;      /* Similar functions in stdlib */
    int estimated_fix_time;
} PatternEntry;

/* Initialize pattern library */
void llm_pattern_init(void);

/* Get pattern suggestion for an error code/message */
PatternSuggestion *llm_get_pattern_suggestion(const char *error_code,
                                              const char *error_message,
                                              const char *function_name);

/* Detect if this is a "missing shadow test" error */
bool llm_is_missing_shadow_test_error(const char *message);

/* Detect if this is a "type mismatch" error */
bool llm_is_type_mismatch_error(const char *message);

/* Detect if this is a "undefined function" error */
bool llm_is_undefined_function_error(const char *message);

/* Free pattern suggestion */
void llm_free_pattern_suggestion(PatternSuggestion *ps);

/* Cleanup pattern library */
void llm_pattern_cleanup(void);

#endif /* LLM_PATTERN_SUGGESTIONS_H */
