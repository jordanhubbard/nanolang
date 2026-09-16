#include <dlfcn.h>
#include <stdio.h>

/* I test the loader without linking SDL2's dialog-producing constructor. */
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        fprintf(stderr, "I could not load %s: %s\n", argv[1], dlerror());
        return 1;
    }
    dlclose(library);
    return 0;
}
