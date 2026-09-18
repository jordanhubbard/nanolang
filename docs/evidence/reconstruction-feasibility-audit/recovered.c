#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
double nlr_f0_combine(double nlr_a0, double nlr_a1);
int64_t nlr_f1_main(void);
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
double nlr_f0_combine(double nlr_a0, double nlr_a1) {
    double nlr_l0_left = nlr_a0;
    (void)nlr_l0_left;
    double nlr_l1_right = nlr_a1;
    (void)nlr_l1_right;
    double nlr_t0 = nlr_l0_left;
    (void)nlr_t0;
    double nlr_t3 = nlr_l1_right;
    (void)nlr_t3;
    double nlr_t6 = nano_rt_f64_add(nlr_t0, nlr_t3);
    (void)nlr_t6;
    return nlr_t6;
}
int64_t nlr_f1_main(void) {
    double nlr_l0_input = nlr_f64_from_bits(UINT64_C(0x0000000000000000));
    (void)nlr_l0_input;
    double nlr_l1_result = nlr_f64_from_bits(UINT64_C(0x0000000000000000));
    (void)nlr_l1_result;
    int64_t nlr_t0 = INT64_C(9221120237041090602);
    (void)nlr_t0;
    double nlr_t9 = nlr_f64_from_bits((uint64_t)nlr_t0);
    (void)nlr_t9;
    nlr_l0_input = nlr_t9;
    double nlr_t13 = nlr_l0_input;
    (void)nlr_t13;
    double nlr_t16 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000000));
    (void)nlr_t16;
    double nlr_t25 = nlr_f0_combine(nlr_t13, nlr_t16);
    (void)nlr_t25;
    nlr_l1_result = nlr_t25;
    double nlr_t33 = nlr_l1_result;
    (void)nlr_t33;
    int64_t nlr_t36 = nlr_f64_to_bits(nlr_t33);
    (void)nlr_t36;
    int64_t nlr_t37 = INT64_C(9221120237041090560);
    (void)nlr_t37;
    bool nlr_t46 = (nlr_t36 != nlr_t37);
    (void)nlr_t46;
    if (nlr_t46) {
        int64_t nlr_t52 = INT64_C(1);
        (void)nlr_t52;
        return nlr_t52;
    }
    double nlr_t62 = nlr_l0_input;
    (void)nlr_t62;
    int64_t nlr_t65 = nlr_f64_to_bits(nlr_t62);
    (void)nlr_t65;
    int64_t nlr_t66 = INT64_C(9221120237041090602);
    (void)nlr_t66;
    bool nlr_t75 = (nlr_t65 != nlr_t66);
    (void)nlr_t75;
    if (nlr_t75) {
        int64_t nlr_t81 = INT64_C(2);
        (void)nlr_t81;
        return nlr_t81;
    }
    int64_t nlr_t91 = INT64_C(0);
    (void)nlr_t91;
    return nlr_t91;
}
int main(void) {
    (void)nlr_f64_from_bits; (void)nlr_f64_to_bits;
    (void)nano_rt_f64_add; (void)nano_rt_f64_sub; (void)nano_rt_f64_mul; (void)nano_rt_f64_div;
    return (int)nlr_f1_main();
}
