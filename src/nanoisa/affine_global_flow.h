#ifndef NANOISA_AFFINE_GLOBAL_FLOW_H
#define NANOISA_AFFINE_GLOBAL_FLOW_H
#include <stdint.h>

/* I encode a relation with bit (2 * input + output), where zero means
 * uninitialized. Preconditions name permitted input states. A lexical return
 * leaves the current continuation; its owning activation is tracked separately. */
typedef struct {
    uint8_t allowed;
    uint8_t normal;
    uint8_t lexical_return;
} NvmAffineGlobalFlow;

static inline uint8_t nvm_affine_global_require(uint8_t relation, uint8_t permitted) {
    uint8_t allowed = 3;
    for (unsigned input = 0; input < 2; input++)
        if (((relation >> (input * 2)) & 3u) & ~permitted)
            allowed &= (uint8_t)~(1u << input);
    return allowed;
}

static inline uint8_t nvm_affine_global_compose(uint8_t before, uint8_t after) {
    uint8_t result = 0;
    for (unsigned input = 0; input < 2; input++)
        for (unsigned middle = 0; middle < 2; middle++)
            if (before & (1u << (input * 2 + middle)))
                result |= (uint8_t)(((after >> (middle * 2)) & 3u) << (input * 2));
    return result;
}

static inline NvmAffineGlobalFlow nvm_affine_global_flow(uint8_t allowed,
        uint8_t normal, uint8_t lexical_return) {
    allowed &= 3u;
    uint8_t rows = (uint8_t)(((allowed & 1u) ? 3u : 0u) |
                            ((allowed & 2u) ? 12u : 0u));
    return (NvmAffineGlobalFlow){allowed, normal & rows, lexical_return & rows};
}

/* I execute the second transfer only on resumed/ordinary paths. An earlier
 * lexical return bypasses it, including its initialization preconditions. */
static inline NvmAffineGlobalFlow nvm_affine_global_then(
        NvmAffineGlobalFlow first, NvmAffineGlobalFlow second) {
    uint8_t allowed = first.allowed &
        nvm_affine_global_require(first.normal, second.allowed);
    return nvm_affine_global_flow(allowed,
        nvm_affine_global_compose(first.normal, second.normal),
        first.lexical_return |
        nvm_affine_global_compose(first.normal, second.lexical_return));
}

/* I require both feasible alternatives to be safe, retaining every outcome. */
static inline NvmAffineGlobalFlow nvm_affine_global_join(
        NvmAffineGlobalFlow left, NvmAffineGlobalFlow right) {
    return nvm_affine_global_flow(left.allowed & right.allowed,
        left.normal | right.normal, left.lexical_return | right.lexical_return);
}
#endif
