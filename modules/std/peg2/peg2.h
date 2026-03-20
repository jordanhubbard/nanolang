#ifndef NANOLANG_STD_PEG2_H
#define NANOLANG_STD_PEG2_H

#include <stdint.h>
#include <stddef.h>

/* Opaque grammar handle */
typedef struct Peg2Grammar Peg2Grammar;

/* Parse tree node (public) */
typedef struct Peg2Node {
    const char *rule_name;     /* rule name or "" for anonymous */
    const char *capture_name;  /* named capture label, or NULL */
    int start;                 /* byte offset in input */
    int end;                   /* byte offset (exclusive) */
    struct Peg2Node **children;
    int child_count;
} Peg2Node;

/* Parse result */
typedef struct {
    int ok;             /* 1 = success, 0 = failure */
    Peg2Node *tree;
    int error_pos;      /* byte offset of furthest failure */
    char *error_msg;    /* heap-allocated; caller must free or use peg2_result_free */
} Peg2Result;

/* ---- Grammar lifecycle ---- */

/* Compile a grammar string ("rule <- expr\nrule2 <- expr2\n...\n").
 * Returns NULL on parse error; *error_out is set to a malloc'd error string. */
Peg2Grammar *nl_peg2_compile(const char *grammar_src, char **error_out);

void nl_peg2_free(Peg2Grammar *g);

/* ---- Matching ---- */

/* Full match: entire input must match the first (start) rule. */
Peg2Result nl_peg2_parse(Peg2Grammar *g, const char *input, int input_len);

/* Partial match: find first occurrence of start rule anywhere in input. */
Peg2Result nl_peg2_find(Peg2Grammar *g, const char *input, int input_len);

/* All non-overlapping occurrences (partial match, advances past each match). */
Peg2Node **nl_peg2_find_all(Peg2Grammar *g, const char *input, int input_len, int *count_out);

/* Quick boolean: does the grammar match this input? */
int nl_peg2_matches(Peg2Grammar *g, const char *input, int input_len);

/* ---- Tree navigation ---- */

/* Get the text matched by a node (returns malloc'd string; caller frees). */
char *nl_peg2_node_text(const Peg2Node *node, const char *input);

/* Find first child with given rule name (or NULL). */
Peg2Node *nl_peg2_node_get(const Peg2Node *node, const char *rule_name);

/* Find first child with given capture name (or NULL). */
Peg2Node *nl_peg2_node_capture(const Peg2Node *node, const char *capture_name);

/* ---- Result/node cleanup ---- */

void nl_peg2_result_free(Peg2Result *r);
void nl_peg2_node_free(Peg2Node *node);
void nl_peg2_free_all(Peg2Node **nodes, int count);

/* ---- NanoLang FFI entry points (use void* for opaque passing) ---- */

void *nl_peg2_ffi_compile(const char *grammar_src);
int64_t nl_peg2_ffi_parse(void *g, const char *input);
int64_t nl_peg2_ffi_find(void *g, const char *input);
int64_t nl_peg2_ffi_matches(void *g, const char *input);
void nl_peg2_ffi_free(void *g);

/* Tree accessors (tree stored thread-locally after parse/find) */
const char *nl_peg2_ffi_tree_rule(void);
const char *nl_peg2_ffi_tree_capture(void);
int64_t nl_peg2_ffi_tree_start(void);
int64_t nl_peg2_ffi_tree_end(void);
int64_t nl_peg2_ffi_tree_child_count(void);
void *nl_peg2_ffi_tree_child(int64_t idx);
const char *nl_peg2_ffi_tree_text(const char *input);
void *nl_peg2_ffi_tree_get(const char *rule_name);
void *nl_peg2_ffi_tree_capture_get(const char *cap_name);
const char *nl_peg2_ffi_last_error(void);
int64_t nl_peg2_ffi_last_error_pos(void);

#endif /* NANOLANG_STD_PEG2_H */
