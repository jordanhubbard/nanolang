#ifndef NANOLANG_NATIVE_ARRAY_ABI_H
#define NANOLANG_NATIVE_ARRAY_ABI_H

#include <dlfcn.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* I check linked references before calling or exposing a foreign function.
 * Missing declarations denote version 1, including in a future v2 host. */
static inline void nano_require_native_array_abi(void *function,
                                                const char *declaration_name,
                                                uint32_t expected,
                                                const char *name) {
    uint32_t actual = 1;
    const uint32_t *declaration = dlsym(RTLD_DEFAULT, declaration_name);
    bool valid = function != NULL;
    if (declaration) {
        Dl_info function_image, declaration_image;
        valid = valid && dladdr(function, &function_image) &&
            dladdr(declaration, &declaration_image) &&
            function_image.dli_fbase == declaration_image.dli_fbase;
        memcpy(&actual, declaration, sizeof actual);
    }
    if (!valid || actual != expected) {
        fprintf(stderr, "I require native array ABI %u for %s; its declaration is incompatible or belongs to another image\n",
                expected, name);
        abort();
    }
}
#endif
