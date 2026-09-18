#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
int64_t nlr_f0_main(void);
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
    double nlr_l0 = nlr_f64_from_bits(UINT64_C(0x0000000000000000));
    (void)nlr_l0;
    double nlr_l1 = nlr_f64_from_bits(UINT64_C(0x0000000000000000));
    (void)nlr_l1;
    double nlr_t0 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000000));
    (void)nlr_t0;
    nlr_l0 = nlr_t0;
    double nlr_t12 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000000));
    (void)nlr_t12;
    nlr_l1 = nlr_t12;
    double nlr_t24 = nlr_l0;
    (void)nlr_t24;
    double nlr_t27 = nlr_l1;
    (void)nlr_t27;
    double nlr_t30 = nano_rt_f64_add(nlr_t24, nlr_t27);
    (void)nlr_t30;
    int64_t nlr_t31 = nlr_f64_to_bits(nlr_t30);
    (void)nlr_t31;
    int64_t nlr_t32 = INT64_C(4611686018427387904);
    (void)nlr_t32;
    bool nlr_t41 = (nlr_t31 == nlr_t32);
    (void)nlr_t41;
    if ((!nlr_t41)) {
        int64_t nlr_t47 = INT64_C(1);
        (void)nlr_t47;
        return nlr_t47;
    }
    double nlr_t57 = nlr_l0;
    (void)nlr_t57;
    int64_t nlr_t60 = nlr_f64_to_bits(nlr_t57);
    (void)nlr_t60;
    int64_t nlr_t61 = INT64_C(4607182418800017408);
    (void)nlr_t61;
    bool nlr_t70 = (nlr_t60 == nlr_t61);
    (void)nlr_t70;
    if ((!nlr_t70)) {
        int64_t nlr_t76 = INT64_C(2);
        (void)nlr_t76;
        return nlr_t76;
    }
    double nlr_t86 = nlr_l1;
    (void)nlr_t86;
    int64_t nlr_t89 = nlr_f64_to_bits(nlr_t86);
    (void)nlr_t89;
    int64_t nlr_t90 = INT64_C(4607182418800017408);
    (void)nlr_t90;
    bool nlr_t99 = (nlr_t89 == nlr_t90);
    (void)nlr_t99;
    if ((!nlr_t99)) {
        int64_t nlr_t105 = INT64_C(3);
        (void)nlr_t105;
        return nlr_t105;
    }
    double nlr_t115 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000000));
    (void)nlr_t115;
    nlr_l0 = nlr_t115;
    double nlr_t127 = nlr_f64_from_bits(UINT64_C(0x3ca0000000000000));
    (void)nlr_t127;
    nlr_l1 = nlr_t127;
    double nlr_t139 = nlr_l0;
    (void)nlr_t139;
    double nlr_t142 = nlr_l1;
    (void)nlr_t142;
    double nlr_t145 = nano_rt_f64_add(nlr_t139, nlr_t142);
    (void)nlr_t145;
    int64_t nlr_t146 = nlr_f64_to_bits(nlr_t145);
    (void)nlr_t146;
    int64_t nlr_t147 = INT64_C(4607182418800017408);
    (void)nlr_t147;
    bool nlr_t156 = (nlr_t146 == nlr_t147);
    (void)nlr_t156;
    if ((!nlr_t156)) {
        int64_t nlr_t162 = INT64_C(4);
        (void)nlr_t162;
        return nlr_t162;
    }
    double nlr_t172 = nlr_l0;
    (void)nlr_t172;
    int64_t nlr_t175 = nlr_f64_to_bits(nlr_t172);
    (void)nlr_t175;
    int64_t nlr_t176 = INT64_C(4607182418800017408);
    (void)nlr_t176;
    bool nlr_t185 = (nlr_t175 == nlr_t176);
    (void)nlr_t185;
    if ((!nlr_t185)) {
        int64_t nlr_t191 = INT64_C(5);
        (void)nlr_t191;
        return nlr_t191;
    }
    double nlr_t201 = nlr_l1;
    (void)nlr_t201;
    int64_t nlr_t204 = nlr_f64_to_bits(nlr_t201);
    (void)nlr_t204;
    int64_t nlr_t205 = INT64_C(4368491638549381120);
    (void)nlr_t205;
    bool nlr_t214 = (nlr_t204 == nlr_t205);
    (void)nlr_t214;
    if ((!nlr_t214)) {
        int64_t nlr_t220 = INT64_C(6);
        (void)nlr_t220;
        return nlr_t220;
    }
    double nlr_t230 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000001));
    (void)nlr_t230;
    nlr_l0 = nlr_t230;
    double nlr_t242 = nlr_f64_from_bits(UINT64_C(0x3ca0000000000000));
    (void)nlr_t242;
    nlr_l1 = nlr_t242;
    double nlr_t254 = nlr_l0;
    (void)nlr_t254;
    double nlr_t257 = nlr_l1;
    (void)nlr_t257;
    double nlr_t260 = nano_rt_f64_add(nlr_t254, nlr_t257);
    (void)nlr_t260;
    int64_t nlr_t261 = nlr_f64_to_bits(nlr_t260);
    (void)nlr_t261;
    int64_t nlr_t262 = INT64_C(4607182418800017410);
    (void)nlr_t262;
    bool nlr_t271 = (nlr_t261 == nlr_t262);
    (void)nlr_t271;
    if ((!nlr_t271)) {
        int64_t nlr_t277 = INT64_C(7);
        (void)nlr_t277;
        return nlr_t277;
    }
    double nlr_t287 = nlr_l0;
    (void)nlr_t287;
    int64_t nlr_t290 = nlr_f64_to_bits(nlr_t287);
    (void)nlr_t290;
    int64_t nlr_t291 = INT64_C(4607182418800017409);
    (void)nlr_t291;
    bool nlr_t300 = (nlr_t290 == nlr_t291);
    (void)nlr_t300;
    if ((!nlr_t300)) {
        int64_t nlr_t306 = INT64_C(8);
        (void)nlr_t306;
        return nlr_t306;
    }
    double nlr_t316 = nlr_l1;
    (void)nlr_t316;
    int64_t nlr_t319 = nlr_f64_to_bits(nlr_t316);
    (void)nlr_t319;
    int64_t nlr_t320 = INT64_C(4368491638549381120);
    (void)nlr_t320;
    bool nlr_t329 = (nlr_t319 == nlr_t320);
    (void)nlr_t329;
    if ((!nlr_t329)) {
        int64_t nlr_t335 = INT64_C(9);
        (void)nlr_t335;
        return nlr_t335;
    }
    double nlr_t345 = nlr_f64_from_bits(UINT64_C(0x7fefffffffffffff));
    (void)nlr_t345;
    nlr_l0 = nlr_t345;
    double nlr_t357 = nlr_f64_from_bits(UINT64_C(0x7fefffffffffffff));
    (void)nlr_t357;
    nlr_l1 = nlr_t357;
    double nlr_t369 = nlr_l0;
    (void)nlr_t369;
    double nlr_t372 = nlr_l1;
    (void)nlr_t372;
    double nlr_t375 = nano_rt_f64_add(nlr_t369, nlr_t372);
    (void)nlr_t375;
    int64_t nlr_t376 = nlr_f64_to_bits(nlr_t375);
    (void)nlr_t376;
    int64_t nlr_t377 = INT64_C(9218868437227405312);
    (void)nlr_t377;
    bool nlr_t386 = (nlr_t376 == nlr_t377);
    (void)nlr_t386;
    if ((!nlr_t386)) {
        int64_t nlr_t392 = INT64_C(10);
        (void)nlr_t392;
        return nlr_t392;
    }
    double nlr_t402 = nlr_l0;
    (void)nlr_t402;
    int64_t nlr_t405 = nlr_f64_to_bits(nlr_t402);
    (void)nlr_t405;
    int64_t nlr_t406 = INT64_C(9218868437227405311);
    (void)nlr_t406;
    bool nlr_t415 = (nlr_t405 == nlr_t406);
    (void)nlr_t415;
    if ((!nlr_t415)) {
        int64_t nlr_t421 = INT64_C(11);
        (void)nlr_t421;
        return nlr_t421;
    }
    double nlr_t431 = nlr_l1;
    (void)nlr_t431;
    int64_t nlr_t434 = nlr_f64_to_bits(nlr_t431);
    (void)nlr_t434;
    int64_t nlr_t435 = INT64_C(9218868437227405311);
    (void)nlr_t435;
    bool nlr_t444 = (nlr_t434 == nlr_t435);
    (void)nlr_t444;
    if ((!nlr_t444)) {
        int64_t nlr_t450 = INT64_C(12);
        (void)nlr_t450;
        return nlr_t450;
    }
    double nlr_t460 = nlr_f64_from_bits(UINT64_C(0x7ff0000000000000));
    (void)nlr_t460;
    nlr_l0 = nlr_t460;
    double nlr_t472 = nlr_f64_from_bits(UINT64_C(0xfff0000000000000));
    (void)nlr_t472;
    nlr_l1 = nlr_t472;
    double nlr_t484 = nlr_l0;
    (void)nlr_t484;
    double nlr_t487 = nlr_l1;
    (void)nlr_t487;
    double nlr_t490 = nano_rt_f64_add(nlr_t484, nlr_t487);
    (void)nlr_t490;
    int64_t nlr_t491 = nlr_f64_to_bits(nlr_t490);
    (void)nlr_t491;
    int64_t nlr_t492 = INT64_C(9221120237041090560);
    (void)nlr_t492;
    bool nlr_t501 = (nlr_t491 == nlr_t492);
    (void)nlr_t501;
    if ((!nlr_t501)) {
        int64_t nlr_t507 = INT64_C(13);
        (void)nlr_t507;
        return nlr_t507;
    }
    double nlr_t517 = nlr_l0;
    (void)nlr_t517;
    int64_t nlr_t520 = nlr_f64_to_bits(nlr_t517);
    (void)nlr_t520;
    int64_t nlr_t521 = INT64_C(9218868437227405312);
    (void)nlr_t521;
    bool nlr_t530 = (nlr_t520 == nlr_t521);
    (void)nlr_t530;
    if ((!nlr_t530)) {
        int64_t nlr_t536 = INT64_C(14);
        (void)nlr_t536;
        return nlr_t536;
    }
    double nlr_t546 = nlr_l1;
    (void)nlr_t546;
    int64_t nlr_t549 = nlr_f64_to_bits(nlr_t546);
    (void)nlr_t549;
    int64_t nlr_t550 = INT64_C(-4503599627370496);
    (void)nlr_t550;
    bool nlr_t559 = (nlr_t549 == nlr_t550);
    (void)nlr_t559;
    if ((!nlr_t559)) {
        int64_t nlr_t565 = INT64_C(15);
        (void)nlr_t565;
        return nlr_t565;
    }
    double nlr_t575 = nlr_f64_from_bits(UINT64_C(0x8000000000000000));
    (void)nlr_t575;
    nlr_l0 = nlr_t575;
    double nlr_t587 = nlr_f64_from_bits(UINT64_C(0x8000000000000000));
    (void)nlr_t587;
    nlr_l1 = nlr_t587;
    double nlr_t599 = nlr_l0;
    (void)nlr_t599;
    double nlr_t602 = nlr_l1;
    (void)nlr_t602;
    double nlr_t605 = nano_rt_f64_add(nlr_t599, nlr_t602);
    (void)nlr_t605;
    int64_t nlr_t606 = nlr_f64_to_bits(nlr_t605);
    (void)nlr_t606;
    int64_t nlr_t607 = INT64_MIN;
    (void)nlr_t607;
    bool nlr_t616 = (nlr_t606 == nlr_t607);
    (void)nlr_t616;
    if ((!nlr_t616)) {
        int64_t nlr_t622 = INT64_C(16);
        (void)nlr_t622;
        return nlr_t622;
    }
    double nlr_t632 = nlr_l0;
    (void)nlr_t632;
    int64_t nlr_t635 = nlr_f64_to_bits(nlr_t632);
    (void)nlr_t635;
    int64_t nlr_t636 = INT64_MIN;
    (void)nlr_t636;
    bool nlr_t645 = (nlr_t635 == nlr_t636);
    (void)nlr_t645;
    if ((!nlr_t645)) {
        int64_t nlr_t651 = INT64_C(17);
        (void)nlr_t651;
        return nlr_t651;
    }
    double nlr_t661 = nlr_l1;
    (void)nlr_t661;
    int64_t nlr_t664 = nlr_f64_to_bits(nlr_t661);
    (void)nlr_t664;
    int64_t nlr_t665 = INT64_MIN;
    (void)nlr_t665;
    bool nlr_t674 = (nlr_t664 == nlr_t665);
    (void)nlr_t674;
    if ((!nlr_t674)) {
        int64_t nlr_t680 = INT64_C(18);
        (void)nlr_t680;
        return nlr_t680;
    }
    double nlr_t690 = nlr_f64_from_bits(UINT64_C(0x000fffffffffffff));
    (void)nlr_t690;
    nlr_l0 = nlr_t690;
    double nlr_t702 = nlr_f64_from_bits(UINT64_C(0x0000000000000001));
    (void)nlr_t702;
    nlr_l1 = nlr_t702;
    double nlr_t714 = nlr_l0;
    (void)nlr_t714;
    double nlr_t717 = nlr_l1;
    (void)nlr_t717;
    double nlr_t720 = nano_rt_f64_add(nlr_t714, nlr_t717);
    (void)nlr_t720;
    int64_t nlr_t721 = nlr_f64_to_bits(nlr_t720);
    (void)nlr_t721;
    int64_t nlr_t722 = INT64_C(4503599627370496);
    (void)nlr_t722;
    bool nlr_t731 = (nlr_t721 == nlr_t722);
    (void)nlr_t731;
    if ((!nlr_t731)) {
        int64_t nlr_t737 = INT64_C(19);
        (void)nlr_t737;
        return nlr_t737;
    }
    double nlr_t747 = nlr_l0;
    (void)nlr_t747;
    int64_t nlr_t750 = nlr_f64_to_bits(nlr_t747);
    (void)nlr_t750;
    int64_t nlr_t751 = INT64_C(4503599627370495);
    (void)nlr_t751;
    bool nlr_t760 = (nlr_t750 == nlr_t751);
    (void)nlr_t760;
    if ((!nlr_t760)) {
        int64_t nlr_t766 = INT64_C(20);
        (void)nlr_t766;
        return nlr_t766;
    }
    double nlr_t776 = nlr_l1;
    (void)nlr_t776;
    int64_t nlr_t779 = nlr_f64_to_bits(nlr_t776);
    (void)nlr_t779;
    int64_t nlr_t780 = INT64_C(1);
    (void)nlr_t780;
    bool nlr_t789 = (nlr_t779 == nlr_t780);
    (void)nlr_t789;
    if ((!nlr_t789)) {
        int64_t nlr_t795 = INT64_C(21);
        (void)nlr_t795;
        return nlr_t795;
    }
    double nlr_t805 = nlr_f64_from_bits(UINT64_C(0x7ff8000000000042));
    (void)nlr_t805;
    nlr_l0 = nlr_t805;
    double nlr_t817 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000000));
    (void)nlr_t817;
    nlr_l1 = nlr_t817;
    double nlr_t829 = nlr_l0;
    (void)nlr_t829;
    double nlr_t832 = nlr_l1;
    (void)nlr_t832;
    double nlr_t835 = nano_rt_f64_add(nlr_t829, nlr_t832);
    (void)nlr_t835;
    int64_t nlr_t836 = nlr_f64_to_bits(nlr_t835);
    (void)nlr_t836;
    int64_t nlr_t837 = INT64_C(9221120237041090560);
    (void)nlr_t837;
    bool nlr_t846 = (nlr_t836 == nlr_t837);
    (void)nlr_t846;
    if ((!nlr_t846)) {
        int64_t nlr_t852 = INT64_C(22);
        (void)nlr_t852;
        return nlr_t852;
    }
    double nlr_t862 = nlr_l0;
    (void)nlr_t862;
    int64_t nlr_t865 = nlr_f64_to_bits(nlr_t862);
    (void)nlr_t865;
    int64_t nlr_t866 = INT64_C(9221120237041090626);
    (void)nlr_t866;
    bool nlr_t875 = (nlr_t865 == nlr_t866);
    (void)nlr_t875;
    if ((!nlr_t875)) {
        int64_t nlr_t881 = INT64_C(23);
        (void)nlr_t881;
        return nlr_t881;
    }
    double nlr_t891 = nlr_l1;
    (void)nlr_t891;
    int64_t nlr_t894 = nlr_f64_to_bits(nlr_t891);
    (void)nlr_t894;
    int64_t nlr_t895 = INT64_C(4607182418800017408);
    (void)nlr_t895;
    bool nlr_t904 = (nlr_t894 == nlr_t895);
    (void)nlr_t904;
    if ((!nlr_t904)) {
        int64_t nlr_t910 = INT64_C(24);
        (void)nlr_t910;
        return nlr_t910;
    }
    double nlr_t920 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000000));
    (void)nlr_t920;
    nlr_l0 = nlr_t920;
    double nlr_t932 = nlr_f64_from_bits(UINT64_C(0x7ff8000000000042));
    (void)nlr_t932;
    nlr_l1 = nlr_t932;
    double nlr_t944 = nlr_l0;
    (void)nlr_t944;
    double nlr_t947 = nlr_l1;
    (void)nlr_t947;
    double nlr_t950 = nano_rt_f64_add(nlr_t944, nlr_t947);
    (void)nlr_t950;
    int64_t nlr_t951 = nlr_f64_to_bits(nlr_t950);
    (void)nlr_t951;
    int64_t nlr_t952 = INT64_C(9221120237041090560);
    (void)nlr_t952;
    bool nlr_t961 = (nlr_t951 == nlr_t952);
    (void)nlr_t961;
    if ((!nlr_t961)) {
        int64_t nlr_t967 = INT64_C(25);
        (void)nlr_t967;
        return nlr_t967;
    }
    double nlr_t977 = nlr_l0;
    (void)nlr_t977;
    int64_t nlr_t980 = nlr_f64_to_bits(nlr_t977);
    (void)nlr_t980;
    int64_t nlr_t981 = INT64_C(4607182418800017408);
    (void)nlr_t981;
    bool nlr_t990 = (nlr_t980 == nlr_t981);
    (void)nlr_t990;
    if ((!nlr_t990)) {
        int64_t nlr_t996 = INT64_C(26);
        (void)nlr_t996;
        return nlr_t996;
    }
    double nlr_t1006 = nlr_l1;
    (void)nlr_t1006;
    int64_t nlr_t1009 = nlr_f64_to_bits(nlr_t1006);
    (void)nlr_t1009;
    int64_t nlr_t1010 = INT64_C(9221120237041090626);
    (void)nlr_t1010;
    bool nlr_t1019 = (nlr_t1009 == nlr_t1010);
    (void)nlr_t1019;
    if ((!nlr_t1019)) {
        int64_t nlr_t1025 = INT64_C(27);
        (void)nlr_t1025;
        return nlr_t1025;
    }
    double nlr_t1035 = nlr_f64_from_bits(UINT64_C(0xfff0000000000043));
    (void)nlr_t1035;
    nlr_l0 = nlr_t1035;
    double nlr_t1047 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000000));
    (void)nlr_t1047;
    nlr_l1 = nlr_t1047;
    double nlr_t1059 = nlr_l0;
    (void)nlr_t1059;
    double nlr_t1062 = nlr_l1;
    (void)nlr_t1062;
    double nlr_t1065 = nano_rt_f64_add(nlr_t1059, nlr_t1062);
    (void)nlr_t1065;
    int64_t nlr_t1066 = nlr_f64_to_bits(nlr_t1065);
    (void)nlr_t1066;
    int64_t nlr_t1067 = INT64_C(9221120237041090560);
    (void)nlr_t1067;
    bool nlr_t1076 = (nlr_t1066 == nlr_t1067);
    (void)nlr_t1076;
    if ((!nlr_t1076)) {
        int64_t nlr_t1082 = INT64_C(28);
        (void)nlr_t1082;
        return nlr_t1082;
    }
    double nlr_t1092 = nlr_l0;
    (void)nlr_t1092;
    int64_t nlr_t1095 = nlr_f64_to_bits(nlr_t1092);
    (void)nlr_t1095;
    int64_t nlr_t1096 = INT64_C(-4503599627370429);
    (void)nlr_t1096;
    bool nlr_t1105 = (nlr_t1095 == nlr_t1096);
    (void)nlr_t1105;
    if ((!nlr_t1105)) {
        int64_t nlr_t1111 = INT64_C(29);
        (void)nlr_t1111;
        return nlr_t1111;
    }
    double nlr_t1121 = nlr_l1;
    (void)nlr_t1121;
    int64_t nlr_t1124 = nlr_f64_to_bits(nlr_t1121);
    (void)nlr_t1124;
    int64_t nlr_t1125 = INT64_C(4607182418800017408);
    (void)nlr_t1125;
    bool nlr_t1134 = (nlr_t1124 == nlr_t1125);
    (void)nlr_t1134;
    if ((!nlr_t1134)) {
        int64_t nlr_t1140 = INT64_C(30);
        (void)nlr_t1140;
        return nlr_t1140;
    }
    double nlr_t1150 = nlr_f64_from_bits(UINT64_C(0x3ff0000000000000));
    (void)nlr_t1150;
    nlr_l0 = nlr_t1150;
    double nlr_t1162 = nlr_f64_from_bits(UINT64_C(0xfff0000000000043));
    (void)nlr_t1162;
    nlr_l1 = nlr_t1162;
    double nlr_t1174 = nlr_l0;
    (void)nlr_t1174;
    double nlr_t1177 = nlr_l1;
    (void)nlr_t1177;
    double nlr_t1180 = nano_rt_f64_add(nlr_t1174, nlr_t1177);
    (void)nlr_t1180;
    int64_t nlr_t1181 = nlr_f64_to_bits(nlr_t1180);
    (void)nlr_t1181;
    int64_t nlr_t1182 = INT64_C(9221120237041090560);
    (void)nlr_t1182;
    bool nlr_t1191 = (nlr_t1181 == nlr_t1182);
    (void)nlr_t1191;
    if ((!nlr_t1191)) {
        int64_t nlr_t1197 = INT64_C(31);
        (void)nlr_t1197;
        return nlr_t1197;
    }
    double nlr_t1207 = nlr_l0;
    (void)nlr_t1207;
    int64_t nlr_t1210 = nlr_f64_to_bits(nlr_t1207);
    (void)nlr_t1210;
    int64_t nlr_t1211 = INT64_C(4607182418800017408);
    (void)nlr_t1211;
    bool nlr_t1220 = (nlr_t1210 == nlr_t1211);
    (void)nlr_t1220;
    if ((!nlr_t1220)) {
        int64_t nlr_t1226 = INT64_C(32);
        (void)nlr_t1226;
        return nlr_t1226;
    }
    double nlr_t1236 = nlr_l1;
    (void)nlr_t1236;
    int64_t nlr_t1239 = nlr_f64_to_bits(nlr_t1236);
    (void)nlr_t1239;
    int64_t nlr_t1240 = INT64_C(-4503599627370429);
    (void)nlr_t1240;
    bool nlr_t1249 = (nlr_t1239 == nlr_t1240);
    (void)nlr_t1249;
    if ((!nlr_t1249)) {
        int64_t nlr_t1255 = INT64_C(33);
        (void)nlr_t1255;
        return nlr_t1255;
    }
    int64_t nlr_t1265 = INT64_C(0);
    (void)nlr_t1265;
    return nlr_t1265;
}
int main(void) {
    (void)nlr_f64_from_bits; (void)nlr_f64_to_bits;
    return (int)nlr_f0_main();
}
