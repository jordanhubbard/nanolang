#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { int64_t f0; bool f1; } NlrOwned0;
typedef struct { NlrOwned0 f0; NlrOwned0 f1; } NlrOwned1;
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
int64_t nlr_f0_owned_0(void);
int64_t nlr_f1_owned_1(NlrOwned0 * nlr_a0, const NlrOwned0 * nlr_a1);
static int64_t nlr_i64_bits(uint64_t bits) {
    if (bits <= (uint64_t)INT64_MAX) return (int64_t)bits;
    return -INT64_C(1) - (int64_t)(UINT64_MAX - bits);
}
static int64_t nlr_i64_add(int64_t a, int64_t b) { return nlr_i64_bits((uint64_t)a + (uint64_t)b); }
int64_t nlr_f0_owned_0(void) {
    int64_t nlr_l0 = 0;
    (void)nlr_l0;
    bool nlr_l1 = false;
    (void)nlr_l1;
    int64_t nlr_l3 = 0;
    (void)nlr_l3;
    bool nlr_l4 = false;
    (void)nlr_l4;
    int64_t nlr_l7 = 0;
    (void)nlr_l7;
    bool nlr_l8 = false;
    (void)nlr_l8;
    int64_t nlr_l10 = 0;
    (void)nlr_l10;
    bool nlr_l11 = false;
    (void)nlr_l11;
    nlr_l0 = INT64_C(3);
    nlr_l1 = true;
    int64_t nlr_t17 = nlr_l0;
    (void)nlr_t17;
    bool nlr_t20 = nlr_l1;
    (void)nlr_t20;
    NlrOwned0 nlr_t23 = (NlrOwned0){.f0 = nlr_t17, .f1 = nlr_t20};
    NlrOwned0 nlr_l2 = nlr_t23;
    nlr_l3 = INT64_C(4);
    nlr_l4 = false;
    int64_t nlr_t48 = nlr_l3;
    (void)nlr_t48;
    bool nlr_t51 = nlr_l4;
    (void)nlr_t51;
    NlrOwned0 nlr_t54 = (NlrOwned0){.f0 = nlr_t48, .f1 = nlr_t51};
    NlrOwned0 nlr_l5 = nlr_t54;
    int64_t nlr_t73 = nlr_f1_owned_1(&nlr_l2, &nlr_l5);
    (void)nlr_t73;
    bool nlr_t90 = (nlr_t73 == INT64_C(7));
    (void)nlr_t90;
    if (!(nlr_t90)) { fputs("Assertion failed\n", stderr); abort(); }
    int64_t nlr_t103 = nlr_f1_owned_1(&nlr_l2, &nlr_l5);
    (void)nlr_t103;
    bool nlr_t120 = (nlr_t103 == INT64_C(11));
    (void)nlr_t120;
    if (!(nlr_t120)) { fputs("Assertion failed\n", stderr); abort(); }
    NlrOwned0 nlr_t122 = nlr_l2;
    NlrOwned0 nlr_l6 = nlr_t122;
    int64_t nlr_t134 = nlr_l6.f0;
    (void)nlr_t134;
    nlr_l7 = nlr_t134;
    bool nlr_t149 = nlr_l6.f1;
    (void)nlr_t149;
    nlr_l8 = nlr_t149;
    int64_t nlr_t158 = nlr_l7;
    (void)nlr_t158;
    bool nlr_t170 = (nlr_t158 == INT64_C(11));
    (void)nlr_t170;
    if (!(nlr_t170)) { fputs("Assertion failed\n", stderr); abort(); }
    bool nlr_t172 = nlr_l8;
    (void)nlr_t172;
    if (!(nlr_t172)) { fputs("Assertion failed\n", stderr); abort(); }
    NlrOwned0 nlr_t176 = nlr_l5;
    NlrOwned0 nlr_l9 = nlr_t176;
    int64_t nlr_t188 = nlr_l9.f0;
    (void)nlr_t188;
    nlr_l10 = nlr_t188;
    bool nlr_t203 = nlr_l9.f1;
    (void)nlr_t203;
    nlr_l11 = nlr_t203;
    int64_t nlr_t212 = nlr_l10;
    (void)nlr_t212;
    bool nlr_t224 = (nlr_t212 == INT64_C(4));
    (void)nlr_t224;
    if (!(nlr_t224)) { fputs("Assertion failed\n", stderr); abort(); }
    bool nlr_t226 = nlr_l11;
    (void)nlr_t226;
    bool nlr_t229 = (!nlr_t226);
    (void)nlr_t229;
    if (!(nlr_t229)) { fputs("Assertion failed\n", stderr); abort(); }
    int64_t nlr_t240_0 = nlr_l6.f0;
    (void)nlr_t240_0;
    bool nlr_t240_1 = nlr_l6.f1;
    (void)nlr_t240_1;
    int64_t nlr_t245_0 = nlr_l9.f0;
    (void)nlr_t245_0;
    bool nlr_t245_1 = nlr_l9.f1;
    (void)nlr_t245_1;
    return INT64_C(0);
}
int64_t nlr_f1_owned_1(NlrOwned0 * nlr_a0, const NlrOwned0 * nlr_a1) {
    int64_t nlr_l2 = 0;
    (void)nlr_l2;
    bool nlr_l3 = false;
    (void)nlr_l3;
    int64_t nlr_l5 = 0;
    (void)nlr_l5;
    bool nlr_l6 = false;
    (void)nlr_l6;
    int64_t nlr_l16 = 0;
    (void)nlr_l16;
    bool nlr_l17 = false;
    (void)nlr_l17;
    int64_t nlr_l20 = 0;
    (void)nlr_l20;
    bool nlr_l21 = false;
    (void)nlr_l21;
    int64_t nlr_l22 = 0;
    (void)nlr_l22;
    bool nlr_l23 = false;
    (void)nlr_l23;
    int64_t nlr_l26 = 0;
    (void)nlr_l26;
    bool nlr_l27 = false;
    (void)nlr_l27;
    int64_t nlr_l28 = 0;
    (void)nlr_l28;
    int64_t nlr_l29 = 0;
    (void)nlr_l29;
    bool nlr_l30 = false;
    (void)nlr_l30;
    int64_t nlr_l33 = 0;
    (void)nlr_l33;
    bool nlr_l34 = false;
    (void)nlr_l34;
    int64_t nlr_l36 = 0;
    (void)nlr_l36;
    bool nlr_l37 = false;
    (void)nlr_l37;
    int64_t nlr_l39 = 0;
    (void)nlr_l39;
    bool nlr_l40 = false;
    (void)nlr_l40;
    nlr_l2 = INT64_C(100);
    nlr_l3 = true;
    int64_t nlr_t17 = nlr_l2;
    (void)nlr_t17;
    bool nlr_t20 = nlr_l3;
    (void)nlr_t20;
    NlrOwned0 nlr_t23 = (NlrOwned0){.f0 = nlr_t17, .f1 = nlr_t20};
    NlrOwned0 nlr_l4 = nlr_t23;
    int64_t nlr_t37 = nlr_l4.f0;
    (void)nlr_t37;
    bool nlr_t52 = (nlr_t37 == INT64_C(100));
    (void)nlr_t52;
    if (!(nlr_t52)) { fputs("Assertion failed\n", stderr); abort(); }
    int64_t nlr_t54 = nlr_a1->f0;
    (void)nlr_t54;
    bool nlr_t68 = (nlr_t54 == INT64_C(4));
    (void)nlr_t68;
    if (!(nlr_t68)) { fputs("Assertion failed\n", stderr); abort(); }
    int64_t nlr_t70 = nlr_a0->f0;
    (void)nlr_t70;
    int64_t nlr_t75 = nlr_a1->f0;
    (void)nlr_t75;
    int64_t nlr_t80 = nlr_i64_add(nlr_t70, nlr_t75);
    (void)nlr_t80;
    nlr_a0->f0 = nlr_t80;
    int64_t nlr_t92 = nlr_l4.f0;
    (void)nlr_t92;
    bool nlr_t107 = (nlr_t92 == INT64_C(100));
    (void)nlr_t107;
    if (!(nlr_t107)) { fputs("Assertion failed\n", stderr); abort(); }
    nlr_l5 = INT64_C(8);
    nlr_l6 = false;
    int64_t nlr_t126 = nlr_l5;
    (void)nlr_t126;
    bool nlr_t129 = nlr_l6;
    (void)nlr_t129;
    NlrOwned0 nlr_t132 = (NlrOwned0){.f0 = nlr_t126, .f1 = nlr_t129};
    NlrOwned0 nlr_l7 = nlr_t132;
    NlrOwned0 nlr_t140 = nlr_l7;
    NlrOwned0 nlr_l8 = nlr_t140;
    NlrOwned0 nlr_t146 = nlr_l4;
    NlrOwned0 nlr_l9 = nlr_t146;
    NlrOwned0 nlr_t152 = nlr_l9;
    NlrOwned0 nlr_t155 = nlr_l8;
    NlrOwned1 nlr_t158 = (NlrOwned1){.f0 = nlr_t152, .f1 = nlr_t155};
    NlrOwned1 nlr_l10 = nlr_t158;
    int64_t nlr_t176 = nlr_l10.f0.f0;
    (void)nlr_t176;
    bool nlr_t191 = (nlr_t176 == INT64_C(100));
    (void)nlr_t191;
    if (!(nlr_t191)) { fputs("Assertion failed\n", stderr); abort(); }
    int64_t nlr_t203 = nlr_l10.f1.f0;
    (void)nlr_t203;
    bool nlr_t218 = (nlr_t203 == INT64_C(8));
    (void)nlr_t218;
    if (!(nlr_t218)) { fputs("Assertion failed\n", stderr); abort(); }
    int64_t nlr_t220 = nlr_a1->f0;
    (void)nlr_t220;
    bool nlr_t234 = (nlr_t220 == INT64_C(4));
    (void)nlr_t234;
    if (!(nlr_t234)) { fputs("Assertion failed\n", stderr); abort(); }
    NlrOwned1 nlr_t236 = nlr_l10;
    NlrOwned1 nlr_l11 = nlr_t236;
    NlrOwned0 nlr_t242_0 = nlr_l11.f0;
    (void)nlr_t242_0;
    NlrOwned0 nlr_t242_1 = nlr_l11.f1;
    (void)nlr_t242_1;
    NlrOwned0 nlr_l13 = nlr_t242_1;
    NlrOwned0 nlr_l12 = nlr_t242_0;
    NlrOwned0 nlr_t251 = nlr_l13;
    NlrOwned0 nlr_l14 = nlr_t251;
    NlrOwned0 nlr_t257 = nlr_l12;
    NlrOwned0 nlr_l15 = nlr_t257;
    if (true) {
        nlr_l16 = INT64_C(1);
        nlr_l17 = false;
        int64_t nlr_t287 = nlr_l16;
        (void)nlr_t287;
        bool nlr_t290 = nlr_l17;
        (void)nlr_t290;
        NlrOwned0 nlr_t293 = (NlrOwned0){.f0 = nlr_t287, .f1 = nlr_t290};
        NlrOwned0 nlr_l18 = nlr_t293;
        int64_t nlr_t307 = nlr_l18.f0;
        (void)nlr_t307;
        bool nlr_t322 = (nlr_t307 == INT64_C(1));
        (void)nlr_t322;
        if (!(nlr_t322)) { fputs("Assertion failed\n", stderr); abort(); }
        NlrOwned0 nlr_t324 = nlr_l18;
        NlrOwned0 nlr_l19 = nlr_t324;
        int64_t nlr_t336 = nlr_l19.f0;
        (void)nlr_t336;
        nlr_l20 = nlr_t336;
        bool nlr_t351 = nlr_l19.f1;
        (void)nlr_t351;
        nlr_l21 = nlr_t351;
        int64_t nlr_t360 = nlr_l20;
        (void)nlr_t360;
        bool nlr_t372 = (nlr_t360 == INT64_C(1));
        (void)nlr_t372;
        if (!(nlr_t372)) { fputs("Assertion failed\n", stderr); abort(); }
        bool nlr_t374 = nlr_l21;
        (void)nlr_t374;
        bool nlr_t377 = (!nlr_t374);
        (void)nlr_t377;
        if (!(nlr_t377)) { fputs("Assertion failed\n", stderr); abort(); }
        int64_t nlr_t379_0 = nlr_l19.f0;
        (void)nlr_t379_0;
        bool nlr_t379_1 = nlr_l19.f1;
        (void)nlr_t379_1;
    } else {
        nlr_l22 = INT64_C(2);
        nlr_l23 = true;
        int64_t nlr_t406 = nlr_l22;
        (void)nlr_t406;
        bool nlr_t409 = nlr_l23;
        (void)nlr_t409;
        NlrOwned0 nlr_t412 = (NlrOwned0){.f0 = nlr_t406, .f1 = nlr_t409};
        NlrOwned0 nlr_l24 = nlr_t412;
        NlrOwned0 nlr_t420 = nlr_l24;
        NlrOwned0 nlr_l25 = nlr_t420;
        int64_t nlr_t432 = nlr_l25.f0;
        (void)nlr_t432;
        nlr_l26 = nlr_t432;
        bool nlr_t447 = nlr_l25.f1;
        (void)nlr_t447;
        nlr_l27 = nlr_t447;
        int64_t nlr_t456 = nlr_l26;
        (void)nlr_t456;
        bool nlr_t468 = (nlr_t456 == INT64_C(2));
        (void)nlr_t468;
        if (!(nlr_t468)) { fputs("Assertion failed\n", stderr); abort(); }
        bool nlr_t470 = nlr_l27;
        (void)nlr_t470;
        if (!(nlr_t470)) { fputs("Assertion failed\n", stderr); abort(); }
        int64_t nlr_t474_0 = nlr_l25.f0;
        (void)nlr_t474_0;
        bool nlr_t474_1 = nlr_l25.f1;
        (void)nlr_t474_1;
    }
    int64_t nlr_t479 = nlr_a1->f0;
    (void)nlr_t479;
    bool nlr_t493 = (nlr_t479 == INT64_C(4));
    (void)nlr_t493;
    if (!(nlr_t493)) { fputs("Assertion failed\n", stderr); abort(); }
    nlr_l28 = INT64_C(0);
    while ((nlr_l28 < INT64_C(2))) {
        int64_t nlr_t525 = nlr_l28;
        (void)nlr_t525;
        nlr_l29 = nlr_t525;
        nlr_l30 = true;
        int64_t nlr_t536 = nlr_l29;
        (void)nlr_t536;
        bool nlr_t539 = nlr_l30;
        (void)nlr_t539;
        NlrOwned0 nlr_t542 = (NlrOwned0){.f0 = nlr_t536, .f1 = nlr_t539};
        NlrOwned0 nlr_l31 = nlr_t542;
        int64_t nlr_t556 = nlr_l31.f0;
        (void)nlr_t556;
        int64_t nlr_t562 = nlr_l28;
        (void)nlr_t562;
        bool nlr_t565 = (nlr_t556 == nlr_t562);
        (void)nlr_t565;
        if (!(nlr_t565)) { fputs("Assertion failed\n", stderr); abort(); }
        NlrOwned0 nlr_t567 = nlr_l31;
        NlrOwned0 nlr_l32 = nlr_t567;
        int64_t nlr_t579 = nlr_l32.f0;
        (void)nlr_t579;
        nlr_l33 = nlr_t579;
        bool nlr_t594 = nlr_l32.f1;
        (void)nlr_t594;
        nlr_l34 = nlr_t594;
        int64_t nlr_t603 = nlr_l33;
        (void)nlr_t603;
        int64_t nlr_t606 = nlr_l28;
        (void)nlr_t606;
        bool nlr_t609 = (nlr_t603 == nlr_t606);
        (void)nlr_t609;
        if (!(nlr_t609)) { fputs("Assertion failed\n", stderr); abort(); }
        bool nlr_t611 = nlr_l34;
        (void)nlr_t611;
        if (!(nlr_t611)) { fputs("Assertion failed\n", stderr); abort(); }
        int64_t nlr_t615 = nlr_l28;
        (void)nlr_t615;
        int64_t nlr_t627 = nlr_i64_add(nlr_t615, INT64_C(1));
        (void)nlr_t627;
        nlr_l28 = nlr_t627;
        int64_t nlr_t631_0 = nlr_l32.f0;
        (void)nlr_t631_0;
        bool nlr_t631_1 = nlr_l32.f1;
        (void)nlr_t631_1;
    }
    NlrOwned0 nlr_t641 = nlr_l14;
    NlrOwned0 nlr_l35 = nlr_t641;
    int64_t nlr_t653 = nlr_l35.f0;
    (void)nlr_t653;
    nlr_l36 = nlr_t653;
    bool nlr_t668 = nlr_l35.f1;
    (void)nlr_t668;
    nlr_l37 = nlr_t668;
    int64_t nlr_t677 = nlr_l36;
    (void)nlr_t677;
    bool nlr_t689 = (nlr_t677 == INT64_C(8));
    (void)nlr_t689;
    if (!(nlr_t689)) { fputs("Assertion failed\n", stderr); abort(); }
    bool nlr_t691 = nlr_l37;
    (void)nlr_t691;
    bool nlr_t694 = (!nlr_t691);
    (void)nlr_t694;
    if (!(nlr_t694)) { fputs("Assertion failed\n", stderr); abort(); }
    NlrOwned0 nlr_t696 = nlr_l15;
    NlrOwned0 nlr_l38 = nlr_t696;
    int64_t nlr_t708 = nlr_l38.f0;
    (void)nlr_t708;
    nlr_l39 = nlr_t708;
    bool nlr_t723 = nlr_l38.f1;
    (void)nlr_t723;
    nlr_l40 = nlr_t723;
    int64_t nlr_t732 = nlr_l39;
    (void)nlr_t732;
    bool nlr_t744 = (nlr_t732 == INT64_C(100));
    (void)nlr_t744;
    if (!(nlr_t744)) { fputs("Assertion failed\n", stderr); abort(); }
    bool nlr_t746 = nlr_l40;
    (void)nlr_t746;
    if (!(nlr_t746)) { fputs("Assertion failed\n", stderr); abort(); }
    int64_t nlr_t750 = nlr_a0->f0;
    (void)nlr_t750;
    int64_t nlr_t755_0 = nlr_l35.f0;
    (void)nlr_t755_0;
    bool nlr_t755_1 = nlr_l35.f1;
    (void)nlr_t755_1;
    int64_t nlr_t760_0 = nlr_l38.f0;
    (void)nlr_t760_0;
    bool nlr_t760_1 = nlr_l38.f1;
    (void)nlr_t760_1;
    return nlr_t750;
}
int main(void) {
    return (int)nlr_f0_owned_0();
}
