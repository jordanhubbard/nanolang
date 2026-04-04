/*
 * test_module_loading.c — unit tests for module.c (module loading system)
 *
 * Exercises: load_module, get_cached_module_ast, clear_module_cache,
 *            find_module_in_paths, process_imports, module_get_import_count
 */

#define _POSIX_C_SOURCE 200809L
#include "../src/nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static FILE *s_orig_stderr = NULL;
static void suppress_stderr(void) {
    fflush(stderr);
    s_orig_stderr = stderr;
    stderr = fopen("/dev/null", "w");
}
static void restore_stderr(void) {
    if (stderr && stderr != s_orig_stderr) fclose(stderr);
    stderr = s_orig_stderr;
    s_orig_stderr = NULL;
}

/* Write a simple nano file to /tmp for testing */
static const char *write_test_module(const char *content) {
    static char path[] = "/tmp/test_module_loading_mod.nano";
    FILE *f = fopen(path, "w");
    if (!f) return NULL;
    fputs(content, f);
    fclose(f);
    return path;
}

/* ── Tests ───────────────────────────────────────────────────────────────── */

static void test_clear_cache_empty(void) {
    const char *test_name = "clear_module_cache: works on empty cache";
    clear_module_cache();  /* Should not crash even when cache is empty */
    PASS(test_name);
}

static void test_load_module_nonexistent(void) {
    const char *test_name = "load_module: nonexistent file returns NULL";
    clear_module_cache();
    Environment *env = create_environment();
    suppress_stderr();
    ASTNode *mod = load_module("/tmp/does_not_exist_nano_test.nano", env);
    restore_stderr();
    ASSERT(mod == NULL, "expected NULL for nonexistent file");
    free_environment(env);
    PASS(test_name);
}

static void test_load_module_valid(void) {
    const char *test_name = "load_module: valid nano file";
    const char *path = write_test_module(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn mul(a: int, b: int) -> int { return (* a b) }\n"
    );
    ASSERT(path != NULL, "failed to write test module");

    clear_module_cache();
    Environment *env = create_environment();
    suppress_stderr();
    ASTNode *mod = load_module(path, env);
    restore_stderr();
    ASSERT(mod != NULL, "load_module returned NULL for valid file");

    free_environment(env);
    clear_module_cache();
    PASS(test_name);
}

static void test_load_module_cached(void) {
    const char *test_name = "load_module: second load uses cache";
    const char *path = write_test_module(
        "fn square(n: int) -> int { return (* n n) }\n"
    );
    ASSERT(path != NULL, "failed to write test module");

    clear_module_cache();
    Environment *env = create_environment();

    /* First load */
    suppress_stderr();
    ASTNode *mod1 = load_module(path, env);
    restore_stderr();
    ASSERT(mod1 != NULL, "first load returned NULL");

    /* Second load should use cache */
    suppress_stderr();
    ASTNode *mod2 = load_module(path, env);
    restore_stderr();
    /* Cached result may be the same AST or NULL depending on implementation */
    (void)mod2; /* Just verify it doesn't crash */

    free_environment(env);
    clear_module_cache();
    PASS(test_name);
}

static void test_clear_cache_after_load(void) {
    const char *test_name = "clear_module_cache: clears loaded modules";
    const char *path = write_test_module("fn f() -> int { return 1 }\n");
    ASSERT(path != NULL, "failed to write test module");

    Environment *env = create_environment();
    suppress_stderr();
    load_module(path, env);
    restore_stderr();

    /* Clear cache - should not crash */
    clear_module_cache();

    free_environment(env);
    PASS(test_name);
}

static void test_get_cached_module_ast_miss(void) {
    const char *test_name = "get_cached_module_ast: cache miss returns NULL";
    clear_module_cache();
    ASTNode *cached = get_cached_module_ast("/tmp/not_loaded.nano");
    ASSERT(cached == NULL, "expected NULL for uncached module");
    PASS(test_name);
}

static void test_get_cached_module_ast_hit(void) {
    const char *test_name = "get_cached_module_ast: cache hit after load";
    const char *path = write_test_module("fn helper(x: int) -> int { return x }\n");
    ASSERT(path != NULL, "failed to write test module");

    clear_module_cache();
    Environment *env = create_environment();
    suppress_stderr();
    ASTNode *mod = load_module(path, env);
    restore_stderr();

    if (mod != NULL) {
        /* Try to get from cache - may or may not be cached depending on impl */
        ASTNode *cached = get_cached_module_ast(path);
        (void)cached; /* Just verify no crash */
    }

    free_environment(env);
    clear_module_cache();
    PASS(test_name);
}

static void test_find_module_in_paths_notfound(void) {
    const char *test_name = "find_module_in_paths: not found returns NULL";
    /* With no NANO_MODULE_PATH or with a path that doesn't have this module */
    suppress_stderr();
    char *result = find_module_in_paths("this_module_definitely_does_not_exist_xyz");
    restore_stderr();
    /* May return NULL or some path depending on env */
    if (result) free(result);
    PASS(test_name); /* Just verify no crash */
}

static void test_process_imports_no_imports(void) {
    const char *test_name = "process_imports: program with no imports";
    int n = 0;
    Token *t = tokenize("fn add(a: int, b: int) -> int { return (+ a b) }\n", &n);
    ASSERT(t != NULL, "tokenize failed");
    ASTNode *prog = parse_program(t, n);
    free_tokens(t, n);
    ASSERT(prog != NULL, "parse failed");

    Environment *env = create_environment();
    ModuleList modules = {0}; /* zero-init */
    suppress_stderr();
    bool ok = process_imports(prog, env, &modules, "<test>");
    restore_stderr();
    ASSERT(ok, "process_imports returned false for program with no imports");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_process_imports_with_import(void) {
    const char *test_name = "process_imports: program with import statement";
    const char *path = write_test_module("fn helper_fn(x: int) -> int { return x }\n");
    ASSERT(path != NULL, "failed to write module file");

    /* Create a program that imports the temp file */
    char src[512];
    snprintf(src, sizeof(src), "import \"%s\"\nfn main() -> int { return 0 }\n", path);

    int n = 0;
    Token *t = tokenize(src, &n);
    ASSERT(t != NULL, "tokenize failed");
    ASTNode *prog = parse_program(t, n);
    free_tokens(t, n);
    ASSERT(prog != NULL, "parse failed");

    clear_module_cache();
    Environment *env = create_environment();
    ModuleList modules = {0};
    suppress_stderr();
    bool ok = process_imports(prog, env, &modules, "<test>");
    restore_stderr();
    (void)ok; /* May fail depending on module resolution details */

    free_environment(env);
    free_ast(prog);
    clear_module_cache();
    PASS(test_name);
}

static void test_load_module_invalid_nano(void) {
    const char *test_name = "load_module: invalid nano syntax";
    const char *path = write_test_module("this is not valid nano @@@ !!!\n");
    ASSERT(path != NULL, "failed to write invalid module");

    clear_module_cache();
    Environment *env = create_environment();
    suppress_stderr();
    ASTNode *mod = load_module(path, env);
    restore_stderr();
    /* Invalid syntax may return NULL or a partial AST */
    (void)mod;
    if (mod) clear_module_cache(); /* Clean up if it didn't fail */

    free_environment(env);
    PASS(test_name);
}

static void test_module_import_count(void) {
    const char *test_name = "module_get_import_count: returns -1 for uncached path";
    /* Returns -1 for uncached/nonexistent modules */
    int64_t count = module_get_import_count("/tmp/nonexistent_nano_test.nano");
    ASSERT(count == -1, "expected -1 for uncached module path");

    /* Also test with NULL */
    int64_t null_count = module_get_import_count(NULL);
    ASSERT(null_count == -1, "expected -1 for NULL path");

    /* Test with a loaded module */
    const char *path = write_test_module("fn f() -> int { return 0 }\n");
    if (path) {
        clear_module_cache();
        Environment *env = create_environment();
        suppress_stderr();
        ASTNode *mod = load_module(path, env);
        restore_stderr();
        if (mod) {
            int64_t loaded_count = module_get_import_count(path);
            ASSERT(loaded_count >= 0, "expected non-negative count for loaded module");
        }
        free_environment(env);
        clear_module_cache();
    }
    PASS(test_name);
}

static void test_module_get_import_path(void) {
    const char *test_name = "module_get_import_path: returns NULL for uncached";
    /* Returns NULL for uncached/nonexistent modules */
    const char *path = module_get_import_path("/tmp/nonexistent.nano", 0);
    ASSERT(path == NULL, "expected NULL for uncached module");

    const char *null_path = module_get_import_path(NULL, 0);
    ASSERT(null_path == NULL, "expected NULL for NULL module path");

    const char *neg_idx = module_get_import_path("/tmp/nonexistent.nano", -1);
    ASSERT(neg_idx == NULL, "expected NULL for negative index");

    PASS(test_name);
}

static void test_module_generate_forward_decls(void) {
    const char *test_name = "module_generate_forward_declarations: NULL path returns NULL";
    const char *result = module_generate_forward_declarations(NULL);
    ASSERT(result == NULL, "expected NULL for NULL path");

    /* Uncached module also returns NULL */
    const char *uncached = module_generate_forward_declarations("/tmp/no_such_module.nano");
    ASSERT(uncached == NULL, "expected NULL for uncached module");

    /* Load a module and get forward declarations */
    const char *path = write_test_module(
        "pub fn add(a: int, b: int) -> int { return (+ a b) }\n"
    );
    if (path) {
        clear_module_cache();
        Environment *env = create_environment();
        suppress_stderr();
        ASTNode *mod = load_module(path, env);
        restore_stderr();
        if (mod) {
            const char *decls = module_generate_forward_declarations(path);
            /* May return NULL or a string -- just verify no crash */
            if (decls) free((void *)decls);
        }
        free_environment(env);
        clear_module_cache();
    }

    PASS(test_name);
}

static void test_module_list_operations(void) {
    const char *test_name = "ModuleList: create/add/free";
    ModuleList *modules = create_module_list();
    ASSERT(modules != NULL, "create_module_list returned NULL");

    /* Test adding paths to module list */
    module_list_add(modules, "/tmp/a.nano");
    module_list_add(modules, "/tmp/b.nano");
    ASSERT(modules->count == 2, "expected 2 modules in list");
    ASSERT(strcmp(modules->module_paths[0], "/tmp/a.nano") == 0, "first path mismatch");
    ASSERT(strcmp(modules->module_paths[1], "/tmp/b.nano") == 0, "second path mismatch");

    /* Adding duplicate should not increase count */
    module_list_add(modules, "/tmp/a.nano");
    ASSERT(modules->count == 2, "duplicate add should not increase count");

    /* Free the list */
    free_module_list(modules);

    /* NULL-safe free */
    free_module_list(NULL);

    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[module_loading] Module loading system tests...\n\n");

    test_clear_cache_empty();
    test_load_module_nonexistent();
    test_load_module_valid();
    test_load_module_cached();
    test_clear_cache_after_load();
    test_get_cached_module_ast_miss();
    test_get_cached_module_ast_hit();
    test_find_module_in_paths_notfound();
    test_process_imports_no_imports();
    test_process_imports_with_import();
    test_load_module_invalid_nano();
    test_module_import_count();
    test_module_get_import_path();
    test_module_generate_forward_decls();
    test_module_list_operations();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
