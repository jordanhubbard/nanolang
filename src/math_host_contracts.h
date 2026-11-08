#ifndef NANO_MATH_HOST_CONTRACTS_H
#define NANO_MATH_HOST_CONTRACTS_H
#include <string.h>

static inline unsigned nano_math_host_arity(const char *name) {
    if (!name) return 0;
#define NANO_MATH_HOST(symbol, count) if (strcmp(name, #symbol) == 0) return count;
#include "math_host_contracts.def"
#undef NANO_MATH_HOST
    return 0;
}
#endif
