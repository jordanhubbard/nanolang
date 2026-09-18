#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
int64_t nlr_f0_main(void);
double nlr_f1_relay(double nlr_a0);
/* I canonicalize only scalar binary arithmetic results, never transported bits. */
#ifndef NANOLANG_BINARY64_ARITHMETIC_H
#define NANOLANG_BINARY64_ARITHMETIC_H
#include <float.h>
#include <stdint.h>
#include <string.h>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "I require ordinary IEEE arithmetic without fast-math."
#endif
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || DBL_MIN_EXP != -1021
#error "I require binary64 double arithmetic."
#endif
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "I require operations evaluated in their binary64 type."
#endif
/* I retain a compile-time storage check in both C99 and C11 output. */
typedef char nano_rt_binary64_storage_guard[
    sizeof(double) == 8 && sizeof(uint64_t) == 8 ? 1 : -1];

/* I inspect a rounded result with integer operations, not another FP operation. */
static inline double nano_rt_f64_arithmetic_result(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) &&
        (bits & UINT64_C(0x000fffffffffffff)) != 0) {
        bits = UINT64_C(0x7ff8000000000000);
        memcpy(&value, &bits, sizeof(value));
    }
    return value;
}

/* Each volatile store/load is a binary64 rounding and noncontraction boundary. */
static inline double nano_rt_f64_add(double a, double b) {
    volatile double rounded = a + b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_sub(double a, double b) {
    volatile double rounded = a - b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_mul(double a, double b) {
    volatile double rounded = a * b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_div(double a, double b) {
    uint64_t divisor;
    memcpy(&divisor, &b, sizeof(divisor));
    /* Either signed zero takes precedence even over a signaling NaN numerator. */
    if ((divisor & UINT64_C(0x7fffffffffffffff)) == 0) return 0.0;
    volatile double rounded = a / b;
    return nano_rt_f64_arithmetic_result(rounded);
}
#endif

#include <string.h>
static double nlr_f64_from_bits(uint64_t bits) {
    double value;
    _Static_assert(sizeof(value) == sizeof(bits), "I require binary64 storage.");
    memcpy(&value, &bits, sizeof(value));
    return value;
}
static int64_t nlr_f64_to_bits(double value) {
    uint64_t bits;
    _Static_assert(sizeof(value) == sizeof(bits), "I require binary64 storage.");
    memcpy(&bits, &value, sizeof(bits));
    return bits <= INT64_MAX ? (int64_t)bits : -1 - (int64_t)(UINT64_MAX - bits);
}
int64_t nlr_f0_main(void) {
    double nlr_t0 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000000));
    (void)nlr_t0;
    double nlr_t9 = nlr_f1_relay(nlr_t0);
    (void)nlr_t9;
    double nlr_t14 = nlr_f64_from_bits(UINT64_C(0x4000000000000000));
    (void)nlr_t14;
    double nlr_t23 = nlr_f1_relay(nlr_t14);
    (void)nlr_t23;
    double nlr_t28 = nano_rt_f64_add(nlr_t9, nlr_t23);
    (void)nlr_t28;
    int64_t nlr_t30 = INT64_C(0);
    (void)nlr_t30;
    return nlr_t30;
}
double nlr_f1_relay(double nlr_a0) {
    double nlr_l0 = nlr_a0;
    (void)nlr_l0;
    double nlr_t0 = nlr_l0;
    (void)nlr_t0;
    return nlr_t0;
}
int main(void) {
    (void)nlr_f64_from_bits; (void)nlr_f64_to_bits;
    return (int)nlr_f0_main();
}
