/* I share exact array ABI declarations with standalone native adapters. */
#ifndef NANOLANG_DYN_ARRAY_ABI_H
#define NANOLANG_DYN_ARRAY_ABI_H
#include <stdint.h>

/* I expand these same tokens as C declarations and as generated C source. */
#define NANO_DYN_ARRAY_ABI_DECLARATIONS \
    typedef enum { \
        ELEM_INT = 1, ELEM_U8 = 8, ELEM_FLOAT = 2, ELEM_STRING = 3, \
        ELEM_BOOL = 4, ELEM_ARRAY = 5, ELEM_STRUCT = 6, ELEM_POINTER = 7 \
    } ElementType; \
    typedef struct { \
        int64_t length; \
        int64_t capacity; \
        ElementType elem_type; \
        uint8_t elem_size; \
        void *data; \
    } DynArray;

NANO_DYN_ARRAY_ABI_DECLARATIONS

#define NANO_DYN_ARRAY_ABI_STRINGIFY(...) #__VA_ARGS__
#define NANO_DYN_ARRAY_ABI_EXPAND(...) NANO_DYN_ARRAY_ABI_STRINGIFY(__VA_ARGS__)
#define NANO_DYN_ARRAY_ABI_SOURCE \
    "#ifndef NANOLANG_DYN_ARRAY_ABI_H\n#define NANOLANG_DYN_ARRAY_ABI_H\n" \
    "#include <stdint.h>\n" \
    NANO_DYN_ARRAY_ABI_EXPAND(NANO_DYN_ARRAY_ABI_DECLARATIONS) "\n#endif\n"
#endif
