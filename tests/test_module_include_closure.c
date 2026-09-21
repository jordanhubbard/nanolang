/* I inspect the actual cache-to-argument helper without executing a compiler. */
#include "../src/module.c"
int g_argc;
char **g_argv;
static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "I failed %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static void write_manifest(const char *path, const char *text) {
    FILE *file = fopen(path, "wb"); CHECK(file);
    CHECK(fwrite(text, 1, strlen(text), file) == strlen(text));
    CHECK(fclose(file) == 0);
}
int main(void) {
    char directory[] = "/tmp/nano-header-unit-XXXXXX";
    CHECK(mkdtemp(directory));
    char path[4096], manifest[4096];
    CHECK(snprintf(path, sizeof(path), "%s/main.nano", directory) > 0);
    CHECK(snprintf(manifest, sizeof(manifest), "%s/module.json", directory) > 0);
    char *paths[MODULE_INCLUDE_MAX_PATHS];
    for (size_t i = 0; i < MODULE_INCLUDE_MAX_PATHS; ++i) paths[i] = path;
    ModuleCache cache = {paths, NULL, 1, MODULE_INCLUDE_MAX_PATHS};
    module_cache = &cache;
    ModuleIncludeClosure *closure = module_include_closure();
    CHECK(closure); CHECK(closure->count == 1);
    char canonical[4096]; CHECK(realpath(directory, canonical));
    char *quoted = module_quote_path(canonical); CHECK(quoted);
    char expected[8192]; CHECK(snprintf(expected, sizeof(expected), " -I%s", quoted) > 0);
    CHECK(strcmp(expected, closure->flags) == 0); free(quoted);
    module_include_closure_free(closure);
    /* I unit-test the exact request cap with borrowed repeated cache entries;
     * this does not claim the parser normally stores duplicate canonical rows. */
    cache.count = MODULE_INCLUDE_MAX_PATHS;
    closure = module_include_closure(); CHECK(closure); CHECK(closure->count == 1);
    module_include_closure_free(closure);
    cache.count = MODULE_INCLUDE_MAX_PATHS + 1;
    CHECK(!module_include_closure());
    cache.count = 1;
    write_manifest(manifest, "{}");
    closure = module_include_closure(); CHECK(closure); module_include_closure_free(closure);
    char oversized[2200];
    memcpy(oversized, "{\"cflags\":[\"-I", 14);
    memset(oversized + 14, 'x', 2100);
    memcpy(oversized + 2114, "\"]}", 4);
    write_manifest(manifest, oversized);
    CHECK(!module_load_metadata(directory));
    write_manifest(manifest, "{invalid"); CHECK(!module_include_closure());
    write_manifest(manifest, "{\"include_dirs\":[\"/this-header-fixture-directory-does-not-exist\"]}");
    CHECK(!module_include_closure()); CHECK(unlink(manifest) == 0);
    CHECK(symlink("missing-header-manifest", manifest) == 0);
    CHECK(!module_include_closure()); CHECK(unlink(manifest) == 0);
    CHECK(mkfifo(manifest, 0600) == 0);
    CHECK(!module_include_closure()); CHECK(unlink(manifest) == 0);
    CHECK(mkdir(manifest, 0700) == 0);
    CHECK(!module_include_closure()); CHECK(rmdir(manifest) == 0);
    /* A valid manifest symlink retains the ordinary metadata origin. */
    char target[4096]; CHECK(snprintf(target, sizeof(target), "%s/metadata.json", directory) > 0);
    write_manifest(target, "{}"); CHECK(symlink("metadata.json", manifest) == 0);
    closure = module_include_closure(); CHECK(closure); module_include_closure_free(closure);
    CHECK(unlink(manifest) == 0); CHECK(unlink(target) == 0);
    /* Exact quoted-argument capacity, not a silently truncated success. */
    char small[6] = "";
    CHECK(module_append_include(small, sizeof(small), ""));
    CHECK(strcmp(small, " -I''") == 0);
    CHECK(!module_append_include(small, sizeof(small), directory));
    module_cache = NULL; CHECK(rmdir(directory) == 0);
    printf("I passed %u actual module include closure checks.\n", checks);
    return 0;
}
