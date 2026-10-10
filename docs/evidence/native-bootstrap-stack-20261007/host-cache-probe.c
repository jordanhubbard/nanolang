#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "%s\n", dlerror()); return 3; }
    const char *(*artifact)(const char *) = (const char *(*)(const char *))dlsym(library, "nlc_module_artifact");
    if (!artifact) return 4;
    for (int i = 0; i < 2; ++i) {
        const char *path = artifact(argv[2]);
        if (!path || !path[0]) return 5;
        printf("artifact=%s\n", path); fflush(stdout);
    }
    return 0;
}
