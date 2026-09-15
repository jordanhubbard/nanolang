#include <dlfcn.h>
#include <stdio.h>

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
    return 0;
}
