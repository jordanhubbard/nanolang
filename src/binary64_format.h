/* I preserve nonfinite signs independently of host libc spelling. */
#ifndef NANOLANG_BINARY64_FORMAT_H
#define NANOLANG_BINARY64_FORMAT_H
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || DBL_MIN_EXP != -1021
#error "I require binary64 formatting storage."
#endif
#define NL_BINARY64_FORMAT_HELPERS \
typedef char nano_rt_f64_format_storage[(sizeof(double) == 8 && sizeof(uint64_t) == 8) ? 1 : -1]; \
static inline const char *nano_rt_f64_nonfinite(double value) { \
    uint64_t bits; memcpy(&bits, &value, sizeof bits); \
    if ((bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000)) return NULL; \
    if (bits & UINT64_C(0x000fffffffffffff)) return bits >> 63 ? "-nan" : "nan"; \
    return bits >> 63 ? "-inf" : "inf"; \
} \
static inline int nano_rt_f64_format(char *out, size_t size, double value) { \
    const char *special = nano_rt_f64_nonfinite(value); \
    return special ? snprintf(out, size, "%s", special) : snprintf(out, size, "%g", value); \
}
NL_BINARY64_FORMAT_HELPERS
#define NL_FORMAT_STRINGIFY_INNER(...) #__VA_ARGS__
#define NL_FORMAT_STRINGIFY(...) NL_FORMAT_STRINGIFY_INNER(__VA_ARGS__)
#define NL_BINARY64_FORMAT_SOURCE \
    "#include <float.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n" \
    "#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || DBL_MIN_EXP != -1021\n" \
    "#error \"I require binary64 formatting storage.\"\n#endif\n" \
    NL_FORMAT_STRINGIFY(NL_BINARY64_FORMAT_HELPERS) "\n"
#endif
