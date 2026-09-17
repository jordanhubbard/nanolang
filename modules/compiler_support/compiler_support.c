#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "compiler_support.h"
#include "../../src/module_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

const char *nlc_module_artifact(const char *source_path) {
    static _Thread_local char *snapshot;
    char *source = NULL, *generation = NULL, *library = NULL;
    ModuleBuildMetadata *metadata = NULL;
    ModuleBuildInfo *build = NULL;
    struct stat st;
    free(snapshot);
    snapshot = NULL;
    if (!source_path || !source_path[0] || !(source = realpath(source_path, NULL)) ||
        stat(source, &st) != 0 || !S_ISREG(st.st_mode)) goto done;
    char *slash = strrchr(source, '/');
    if (!slash) goto done;
    if (slash == source) slash[1] = '\0';
    else *slash = '\0';
    metadata = module_load_metadata(source);
    if (!metadata || !metadata->name || !metadata->name[0] ||
        metadata->c_sources_count == 0) goto done;
    build = module_build(NULL, metadata);
    if (!build || !build->object_file || !(generation = strdup(build->object_file))) goto done;
    slash = strrchr(generation, '/');
    if (!slash) goto done;
    *slash = '\0';
#ifdef __APPLE__
    const char *extension = "dylib";
#else
    const char *extension = "so";
#endif
    /* I bind the immutable generation returned by this build, not .build/current. */
    int library_length = snprintf(NULL, 0, "%s/lib%s.%s",
                                  generation, metadata->name, extension);
    if (library_length < 0) goto done;
    library = malloc((size_t)library_length + 1U);
    if (!library || snprintf(library, (size_t)library_length + 1U,
                             "%s/lib%s.%s", generation, metadata->name,
                             extension) != library_length) {
        free(library);
        library = NULL;
        goto done;
    }
    snapshot = realpath(library, NULL);
    if (snapshot && (stat(snapshot, &st) != 0 || !S_ISREG(st.st_mode))) {
        free(snapshot);
        snapshot = NULL;
    }
done:
    free(source);
    free(generation);
    free(library);
    module_build_info_free(build);
    module_metadata_free(metadata);
    return snapshot ? snapshot : "";
}
