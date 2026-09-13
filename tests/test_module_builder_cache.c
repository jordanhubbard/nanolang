#include "../src/module_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

static void require(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(1);
    }
}

static void write_file(const char *path, const char *contents) {
    FILE *fp = fopen(path, "wb");
    require(fp != NULL, "open fixture");
    require(fwrite(contents, 1, strlen(contents), fp) == strlen(contents), "write fixture");
    require(fclose(fp) == 0, "close fixture");
}

int main(void) {
    char dir[] = "/tmp/nanolang-source-cache-XXXXXX";
    char source[1024];
    char module_json[1024];
    char *relative_sources[] = { "native.c" };
    char *absolute_sources[] = { source };
    ModuleBuildMetadata relative = {0};
    ModuleBuildMetadata absolute = {0};
    uint64_t relative_hash;
    uint64_t absolute_hash;

    require(mkdtemp(dir) != NULL, "create fixture directory");
    snprintf(source, sizeof(source), "%s/native.c", dir);
    snprintf(module_json, sizeof(module_json), "%s/module.json", dir);
    write_file(source, "int native_value(void) { return 1; }\n");
    write_file(module_json, "{}\n");

    relative.c_sources = relative_sources;
    relative.c_sources_count = 1;
    absolute.c_sources = absolute_sources;
    absolute.c_sources_count = 1;

    require(module_ensure_build_dir(dir), "create module build directory");
    module_update_hash_cache(dir, &relative);
    require(module_source_hashes_match(dir, &relative), "relative source reuses cache");
    require(module_hash_native_source(dir, relative_sources[0], &relative_hash),
            "hash relative source");
    require(module_hash_native_source(dir, absolute_sources[0], &absolute_hash),
            "hash absolute source");
    require(relative_hash == absolute_hash, "relative and absolute digests are equal");

    module_update_hash_cache(dir, &absolute);
    require(module_source_hashes_match(dir, &absolute), "absolute source reuses cache");
    write_file(source, "int native_value(void) { return 2; }\n");
    require(!module_source_hashes_match(dir, &absolute), "changed absolute source invalidates cache");

    absolute_sources[0] = "/tmp/this-native-source-does-not-exist.c";
    module_update_hash_cache(dir, &absolute);
    require(!module_source_hashes_match(dir, &absolute), "failed source hash is rejected");

    absolute_sources[0] = malloc(2048);
    require(absolute_sources[0] != NULL, "allocate long path");
    absolute_sources[0][0] = '/';
    memset(absolute_sources[0] + 1, 'x', 2046);
    absolute_sources[0][2047] = '\0';
    require(!module_source_hashes_match(dir, &absolute), "overflowing source path is rejected");
    free(absolute_sources[0]);

    printf("module builder source cache tests passed\n");
    return 0;
}
