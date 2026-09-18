/* I copy binary64 representations without numeric conversion or NaN quieting. */
#ifndef NANOLANG_BINARY64_BITS_H
#define NANOLANG_BINARY64_BITS_H
#include <stdint.h>
#include <string.h>

#define NL_BINARY64_FROM_BITS \
static inline double nl_float_from_bits(int64_t value) { \
    uint64_t bits = (uint64_t)value; double result; \
    _Static_assert(sizeof(result) == sizeof(bits), "I require binary64 storage."); \
    memcpy(&result, &bits, sizeof(result)); return result; \
}
#define NL_BINARY64_TO_BITS \
static inline int64_t nl_float_to_bits(double value) { \
    uint64_t bits; \
    _Static_assert(sizeof(value) == sizeof(bits), "I require binary64 storage."); \
    memcpy(&bits, &value, sizeof(bits)); \
    return bits <= INT64_MAX ? (int64_t)bits : -1 - (int64_t)(UINT64_MAX - bits); \
}
NL_BINARY64_FROM_BITS
NL_BINARY64_TO_BITS

#define NL_BITS_STRINGIFY_INNER(...) #__VA_ARGS__
#define NL_BITS_STRINGIFY(...) NL_BITS_STRINGIFY_INNER(__VA_ARGS__)
/* Standalone generated C receives the same helper bodies as my interpreter. */
#define NL_BINARY64_BITS_SOURCE \
    "#include <string.h>\n" \
    NL_BITS_STRINGIFY(NL_BINARY64_FROM_BITS) "\n" \
    NL_BITS_STRINGIFY(NL_BINARY64_TO_BITS) "\n"
#endif
