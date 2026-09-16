#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>

/* I isolate fixture loading from VM dispatch and callback state. */
int main(int argc, char **argv) {
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (int i = 1; i < argc; ++i) {
            fprintf(stderr, "I am checking native image reload %d: %s\n", repeat + 1, argv[i]);
            void *image = dlopen(argv[i], RTLD_NOW | RTLD_LOCAL);
            if (!image) { fprintf(stderr, "%s\n", dlerror()); return 1; }
            if (dlclose(image)) return 2;
        }
    }
    /* I also keep both images alive in the same global lookup namespace.
     * I resolve their shared scalar name through each specific image handle. */
    if (argc != 3) return 3;
    void *first = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    void *second = dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL);
    if (!first || !second) return 4;
    int64_t (*first_answer)(void) = (int64_t (*)(void))dlsym(first, "nano_artifact_answer");
    int64_t (*second_answer)(void) = (int64_t (*)(void))dlsym(second, "nano_artifact_answer");
    if (!first_answer || !second_answer || first_answer() != 42 || second_answer() != 43) return 5;
    if (dlclose(second) || dlclose(first)) return 6;
    return 0;
}
