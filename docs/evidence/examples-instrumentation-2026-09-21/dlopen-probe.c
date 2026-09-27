#include <dlfcn.h>
#include <stdio.h>
int main(void) { void *p = dlopen("/opt/homebrew/lib/libSDL3.dylib", RTLD_NOW | RTLD_LOCAL); if (!p) { fprintf(stderr,"%s\n",dlerror()); return 1; } puts("SDL3 loaded"); dlclose(p); return 0; }
