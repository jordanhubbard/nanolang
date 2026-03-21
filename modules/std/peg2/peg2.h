#ifndef NANOLANG_STD_PEG2_H
#define NANOLANG_STD_PEG2_H

#include <stdint.h>
#include <stddef.h>

/* Opaque grammar handle */
typedef struct Peg2Grammar Peg2Grammar;

/* Parse tree node (public — used by FFI result accessors) */
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

/* ---- NanoLang FFI entry points (use void* for opaque passing) ---- */
/* These are the only externally-visible symbols. All other nl_peg2_*
 * functions are static within peg2.c to avoid name conflicts with the
 * NanoLang transpiler's generated C names (e.g., nl_peg2_compile). */

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
