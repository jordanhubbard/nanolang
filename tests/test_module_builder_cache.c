/* I pin source identity without exposing mutable cache internals.
 * The generation suite separately checks artifact reuse. */
#include "../src/module_builder.c"
#include <assert.h>

int main(void) {
    char dir[] = "/tmp/nanolang-source-cache-XXXXXX";
    char source[1024];
    assert(mkdtemp(dir));
    assert(snprintf(source, sizeof(source), "%s/native.c", dir) > 0);
    FILE *file = fopen(source, "wb");
    assert(file);
    assert(fputs("int native_value(void) { return 1; }\n", file) >= 0);
    assert(fclose(file) == 0);
    uint64_t initial = module_source_hash(dir, "native.c");
    assert(initial != 0);
    assert(module_source_hash(dir, source) == initial);
    file = fopen(source, "wb");
    assert(file);
    assert(fputs("int native_value(void) { return 2; }\n", file) >= 0);
    assert(fclose(file) == 0);
    uint64_t changed = module_source_hash(dir, source);
    assert(changed != 0 && changed != initial);
    assert(unlink(source) == 0);
    assert(module_source_hash(dir, source) == 0);
    assert(module_source_hash(dir, "native.c") == 0);
    char long_path[8192];
    memset(long_path, 'x', sizeof(long_path) - 1);
    long_path[0] = '/';
    long_path[sizeof(long_path) - 1] = '\0';
    assert(module_source_hash(dir, long_path) == 0);
    assert(module_source_hash(dir, long_path + 1) == 0);
    assert(rmdir(dir) == 0);
    puts("I passed absolute, relative, changed, missing and oversized source checks.");
    return 0;
}
