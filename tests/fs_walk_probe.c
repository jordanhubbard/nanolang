#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#ifdef WALK_HEADER
#include "runtime/directory_walk.h"
#define walk nl_fs_walkdir
#else
#include "../modules/std/fs.h"
#define walk fs_walkdir
#endif

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) return 3;
    if (limit.rlim_cur > 32) limit.rlim_cur = 32;
    if (setrlimit(RLIMIT_NOFILE, &limit) != 0) return 3;
    for (int repeat = 0; repeat < 40; ++repeat) {
        DynArray *paths = walk(argv[1]);
        if (!paths) return 4;
        if (repeat == 39) {
            for (int64_t i = 0; i < dyn_array_length(paths); ++i) {
                const char *path = dyn_array_get_string(paths, i);
                if (fwrite(path, 1, strlen(path) + 1, stdout) != strlen(path) + 1) return 5;
            }
        }
        gc_release(paths);
    }
    gc_shutdown();
    return 0;
}
