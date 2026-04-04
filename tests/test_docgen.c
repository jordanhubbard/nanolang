/*
 * test_docgen.c — unit tests for the nanolang HTML documentation generator
 *
 * Parses a small .nano fixture, runs emit_doc_html(), then checks that the
 * generated HTML contains the expected function names, type signatures, and
 * doc-comment text.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "nanolang.h"
#include "docgen.h"
#include "docgen_md.h"

/* cli.c requires these globals */
int g_argc = 0;
char **g_argv = NULL;

/* ── Helpers ──────────────────────────────────────────────────────────────── */

static int tests_run    = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg)); \
        tests_failed++; \
    } \
} while (0)

/* Read the entire contents of a file into a malloc'd buffer. */
static char *read_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    fread(buf, 1, (size_t)sz, f);
#pragma GCC diagnostic pop
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* ── Nano source used for tests ───────────────────────────────────────────── */

static const char *NANO_SRC =
    "/// Add two integers.\n"
    "/// @param a first operand\n"
    "/// @param b second operand\n"
    "/// @returns the sum\n"
    "pub fn add(a: int, b: int) -> int {\n"
    "  return (+ a b)\n"
    "}\n"
    "\n"
    "/// Multiply two integers.\n"
    "pub fn multiply(a: int, b: int) -> int {\n"
    "  return (* a b)\n"
    "}\n"
    "\n"
    "// Private helper -- should NOT appear in docs.\n"
    "fn internal_helper(x: int) -> int {\n"
    "  return x\n"
    "}\n"
    "\n"
    "/// A 2D point.\n"
    "pub struct Point {\n"
    "  x: int,\n"
    "  y: int\n"
    "}\n"
    "\n"
    "/// Primary colors.\n"
    "pub enum Color {\n"
    "  Red,\n"
    "  Green,\n"
    "  Blue\n"
    "}\n"
    "\n"
    "/// Optional integer.\n"
    "pub union Maybe {\n"
    "  Some { value: int },\n"
    "  None {}\n"
    "}\n";

/* ── Tests ────────────────────────────────────────────────────────────────── */

static void test_basic_generation(void) {
    /* Parse the source */
    int token_count = 0;
    Token *tokens = tokenize(NANO_SRC, &token_count);
    ASSERT(tokens != NULL, "tokenize should succeed");
    if (!tokens) return;

    ASTNode *program = parse_program(tokens, token_count);
    ASSERT(program != NULL, "parse_program should succeed");
    if (!program) { free_tokens(tokens, token_count); return; }

    /* Emit HTML to a temp file */
    const char *out_path = "/tmp/test_docgen_out.html";
    bool ok = emit_doc_html(out_path, program, NANO_SRC, "testmod");
    ASSERT(ok, "emit_doc_html should return true");

    /* Read output and verify contents */
    char *html = read_file(out_path);
    ASSERT(html != NULL, "output HTML file should be readable");
    if (!html) {
        free_ast(program);
        free_tokens(tokens, token_count);
        return;
    }

    /* Page title / module name */
    ASSERT(strstr(html, "testmod") != NULL,
           "HTML should contain module name 'testmod'");

    /* Exported functions appear */
    ASSERT(strstr(html, "add") != NULL,
           "HTML should contain exported function 'add'");
    ASSERT(strstr(html, "multiply") != NULL,
           "HTML should contain exported function 'multiply'");

    /* Private function must NOT appear in a <section id="..."> */
    ASSERT(strstr(html, "id=\"internal_helper\"") == NULL,
           "Private function 'internal_helper' must not appear as a section");

    /* Doc comment text extracted */
    ASSERT(strstr(html, "Add two integers") != NULL,
           "Doc comment 'Add two integers' should appear in HTML");
    ASSERT(strstr(html, "@param") != NULL,
           "HTML should contain @param tag");
    ASSERT(strstr(html, "@returns") != NULL,
           "HTML should contain @returns tag");

    /* Type signatures */
    ASSERT(strstr(html, "int") != NULL,
           "HTML should contain type 'int'");

    /* Struct */
    ASSERT(strstr(html, "Point") != NULL,
           "HTML should contain struct 'Point'");
    ASSERT(strstr(html, "id=\"Point\"") != NULL,
           "HTML should have anchor for struct 'Point'");

    /* Enum */
    ASSERT(strstr(html, "Color") != NULL,
           "HTML should contain enum 'Color'");
    ASSERT(strstr(html, "Red") != NULL,
           "HTML should contain enum variant 'Red'");

    /* Union */
    ASSERT(strstr(html, "Maybe") != NULL,
           "HTML should contain union 'Maybe'");
    ASSERT(strstr(html, "Some") != NULL,
           "HTML should contain union variant 'Some'");

    /* Sidebar TOC links */
    ASSERT(strstr(html, "href=\"#add\"") != NULL,
           "Sidebar should link to #add");
    ASSERT(strstr(html, "href=\"#Point\"") != NULL,
           "Sidebar should link to #Point");

    /* Self-contained: no external resource references */
    ASSERT(strstr(html, "<link rel=\"stylesheet\"") == NULL,
           "HTML should not reference an external stylesheet");
    ASSERT(strstr(html, "<script src=") == NULL,
           "HTML should not reference an external script");

    free(html);
    free_ast(program);
    free_tokens(tokens, token_count);
}

static void test_no_exported_items(void) {
    const char *src = "fn private_fn(x: int) -> int { return x }\n";
    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    ASSERT(tokens != NULL, "tokenize should succeed for private-only source");
    if (!tokens) return;

    ASTNode *program = parse_program(tokens, token_count);
    ASSERT(program != NULL, "parse_program should succeed for private-only source");
    if (!program) { free_tokens(tokens, token_count); return; }

    const char *out_path = "/tmp/test_docgen_empty.html";
    bool ok = emit_doc_html(out_path, program, src, "private_mod");
    ASSERT(ok, "emit_doc_html should succeed even with no exported items");

    char *html = read_file(out_path);
    ASSERT(html != NULL, "output HTML should be readable for private-only module");
    if (html) {
        ASSERT(strstr(html, "<!DOCTYPE html>") != NULL,
               "Output should be valid HTML even for empty module");
        ASSERT(strstr(html, "private_fn") == NULL,
               "Private function should not appear in HTML output");
        free(html);
    }

    free_ast(program);
    free_tokens(tokens, token_count);
}

static void test_doc_comment_association(void) {
    /* Verify that a doc comment directly preceding a declaration is attached */
    const char *src =
        "/// Compute the square of n.\n"
        "pub fn square(n: int) -> int {\n"
        "  return (* n n)\n"
        "}\n";

    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    ASSERT(tokens != NULL, "tokenize should succeed");
    if (!tokens) return;

    ASTNode *program = parse_program(tokens, token_count);
    ASSERT(program != NULL, "parse_program should succeed");
    if (!program) { free_tokens(tokens, token_count); return; }

    const char *out_path = "/tmp/test_docgen_comment.html";
    bool ok = emit_doc_html(out_path, program, src, "sq");
    ASSERT(ok, "emit_doc_html should succeed");

    char *html = read_file(out_path);
    ASSERT(html != NULL, "output should be readable");
    if (html) {
        ASSERT(strstr(html, "Compute the square of n") != NULL,
               "Doc comment text should appear in HTML");
        ASSERT(strstr(html, "square") != NULL,
               "Function name 'square' should appear in HTML");
        free(html);
    }

    free_ast(program);
    free_tokens(tokens, token_count);
}

/* ── Markdown doc generator tests ────────────────────────────────────────── */

static void test_md_basic_generation(void) {
    int token_count = 0;
    Token *tokens = tokenize(NANO_SRC, &token_count);
    ASSERT(tokens != NULL, "tokenize should succeed");
    if (!tokens) return;

    ASTNode *program = parse_program(tokens, token_count);
    ASSERT(program != NULL, "parse_program should succeed");
    if (!program) { free_tokens(tokens, token_count); return; }

    const char *out_path = "/tmp/test_docgen_out.md";
    bool ok = emit_doc_md(out_path, program, NANO_SRC, "testmod");
    ASSERT(ok, "emit_doc_md should return true");

    char *md = read_file(out_path);
    ASSERT(md != NULL, "output MD file should be readable");
    if (md) {
        ASSERT(strstr(md, "testmod") != NULL, "MD should contain module name");
        ASSERT(strstr(md, "add") != NULL, "MD should contain exported function 'add'");
        ASSERT(strstr(md, "multiply") != NULL, "MD should contain exported function 'multiply'");
        ASSERT(strstr(md, "Point") != NULL, "MD should contain struct 'Point'");
        ASSERT(strstr(md, "Color") != NULL, "MD should contain enum 'Color'");
        free(md);
    }

    free_ast(program);
    free_tokens(tokens, token_count);
}

static void test_md_no_exported_items(void) {
    const char *src = "fn private_fn(x: int) -> int { return x }\n";
    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    ASSERT(tokens != NULL, "tokenize should succeed");
    if (!tokens) return;

    ASTNode *program = parse_program(tokens, token_count);
    ASSERT(program != NULL, "parse_program should succeed");
    if (!program) { free_tokens(tokens, token_count); return; }

    const char *out_path = "/tmp/test_docgen_empty.md";
    bool ok = emit_doc_md(out_path, program, src, "private_mod");
    ASSERT(ok, "emit_doc_md should succeed even with no exported items");

    char *md = read_file(out_path);
    ASSERT(md != NULL, "output MD should be readable");
    if (md) {
        ASSERT(strstr(md, "private_fn") == NULL, "Private function should not appear");
        free(md);
    }

    free_ast(program);
    free_tokens(tokens, token_count);
}

static void test_md_doc_comment(void) {
    const char *src =
        "/// Compute the square of n.\n"
        "pub fn square(n: int) -> int {\n"
        "  return (* n n)\n"
        "}\n";

    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    ASSERT(tokens != NULL, "tokenize should succeed");
    if (!tokens) return;

    ASTNode *program = parse_program(tokens, token_count);
    ASSERT(program != NULL, "parse_program should succeed");
    if (!program) { free_tokens(tokens, token_count); return; }

    const char *out_path = "/tmp/test_docgen_comment.md";
    bool ok = emit_doc_md(out_path, program, src, "sq");
    ASSERT(ok, "emit_doc_md should succeed");

    char *md = read_file(out_path);
    ASSERT(md != NULL, "output should be readable");
    if (md) {
        ASSERT(strstr(md, "square") != NULL, "MD should contain function name");
        ASSERT(strstr(md, "Compute the square of n") != NULL, "MD should contain doc comment");
        free(md);
    }

    free_ast(program);
    free_tokens(tokens, token_count);
}

/* ── Entry point ──────────────────────────────────────────────────────────── */

int main(void) {
    printf("Running docgen unit tests...\n");

    test_basic_generation();
    test_no_exported_items();
    test_doc_comment_association();

    printf("\nRunning docgen_md unit tests...\n");
    test_md_basic_generation();
    test_md_no_exported_items();
    test_md_doc_comment();

    if (tests_failed == 0) {
        printf("docgen tests: %d/%d passed\n", tests_run, tests_run);
        return 0;
    } else {
        printf("docgen tests: %d/%d FAILED\n", tests_failed, tests_run);
        return 1;
    }
}
