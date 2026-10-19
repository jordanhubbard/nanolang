#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { const unsigned char *data; size_t length; } NlrString;
static const unsigned char nlr_string_11[] = {65};
static const unsigned char nlr_string_12[] = {66};
typedef struct { int64_t f0; } NlrOwned0;
typedef struct { NlrOwned0 f0; bool f1; } NlrOwned1;
typedef struct { NlrOwned1 f0; int64_t f1; } NlrOwned2;
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
int64_t nlr_f0_owned_0(void);
int64_t nlr_f1_owned_1(int64_t nlr_a0, NlrString nlr_a1);
int64_t nlr_f2_owned_2(NlrOwned2 nlr_a0);
static int64_t nlr_i64_bits(uint64_t bits) {
    if (bits <= (uint64_t)INT64_MAX) return (int64_t)bits;
    return -INT64_C(1) - (int64_t)(UINT64_MAX - bits);
}
static int64_t nlr_i64_add(int64_t a, int64_t b) { return nlr_i64_bits((uint64_t)a + (uint64_t)b); }
int64_t nlr_f0_owned_0(void) {
    int64_t nlr_l0 = 0;
    (void)nlr_l0;
    int64_t nlr_l1 = 0;
    (void)nlr_l1;
    int64_t nlr_l2 = 0;
    (void)nlr_l2;
    bool nlr_l5 = false;
    (void)nlr_l5;
    int64_t nlr_l8 = 0;
    (void)nlr_l8;
    int64_t nlr_l10 = 0;
    (void)nlr_l10;
    bool nlr_l13 = false;
    (void)nlr_l13;
    int64_t nlr_l16 = 0;
    (void)nlr_l16;
    nlr_l0 = INT64_C(1);
    nlr_l1 = INT64_C(0);
    while ((nlr_l1 < nlr_l0)) {
        int64_t nlr_t50 = nlr_f1_owned_1(INT64_C(7), (NlrString){nlr_string_11, 1});
        (void)nlr_t50;
        nlr_l2 = nlr_t50;
        int64_t nlr_t58 = nlr_l2;
        (void)nlr_t58;
        NlrOwned0 nlr_t61 = (NlrOwned0){.f0 = nlr_t58};
        NlrOwned0 nlr_l3 = nlr_t61;
        NlrOwned0 nlr_t69 = nlr_l3;
        NlrOwned0 nlr_l4 = nlr_t69;
        nlr_l5 = true;
        NlrOwned0 nlr_t80 = nlr_l4;
        bool nlr_t83 = nlr_l5;
        (void)nlr_t83;
        NlrOwned1 nlr_t86 = (NlrOwned1){.f0 = nlr_t80, .f1 = nlr_t83};
        NlrOwned1 nlr_l6 = nlr_t86;
        NlrOwned1 nlr_t94 = nlr_l6;
        NlrOwned1 nlr_l7 = nlr_t94;
        int64_t nlr_t114 = nlr_f1_owned_1(INT64_C(9), (NlrString){nlr_string_12, 1});
        (void)nlr_t114;
        nlr_l8 = nlr_t114;
        NlrOwned1 nlr_t122 = nlr_l7;
        int64_t nlr_t125 = nlr_l8;
        (void)nlr_t125;
        NlrOwned2 nlr_t128 = (NlrOwned2){.f0 = nlr_t122, .f1 = nlr_t125};
        NlrOwned2 nlr_l9 = nlr_t128;
        NlrOwned2 nlr_t136 = nlr_l9;
        int64_t nlr_t139 = nlr_f2_owned_2(nlr_t136);
        (void)nlr_t139;
        bool nlr_t153 = (nlr_t139 == INT64_C(16));
        (void)nlr_t153;
        if (!(nlr_t153)) { (void)fflush(stdout); fputs("Assertion failed\n", stderr); abort(); }
        int64_t nlr_t169 = nlr_f1_owned_1(INT64_C(7), (NlrString){nlr_string_11, 1});
        (void)nlr_t169;
        nlr_l10 = nlr_t169;
        int64_t nlr_t177 = nlr_l10;
        (void)nlr_t177;
        NlrOwned0 nlr_t180 = (NlrOwned0){.f0 = nlr_t177};
        NlrOwned0 nlr_l11 = nlr_t180;
        NlrOwned0 nlr_t188 = nlr_l11;
        NlrOwned0 nlr_l12 = nlr_t188;
        nlr_l13 = true;
        NlrOwned0 nlr_t199 = nlr_l12;
        bool nlr_t202 = nlr_l13;
        (void)nlr_t202;
        NlrOwned1 nlr_t205 = (NlrOwned1){.f0 = nlr_t199, .f1 = nlr_t202};
        NlrOwned1 nlr_l14 = nlr_t205;
        NlrOwned1 nlr_t213 = nlr_l14;
        NlrOwned1 nlr_l15 = nlr_t213;
        int64_t nlr_t233 = nlr_f1_owned_1(INT64_C(9), (NlrString){nlr_string_12, 1});
        (void)nlr_t233;
        nlr_l16 = nlr_t233;
        NlrOwned1 nlr_t241 = nlr_l15;
        int64_t nlr_t244 = nlr_l16;
        (void)nlr_t244;
        NlrOwned2 nlr_t247 = (NlrOwned2){.f0 = nlr_t241, .f1 = nlr_t244};
        int64_t nlr_t252 = nlr_f2_owned_2(nlr_t247);
        (void)nlr_t252;
        bool nlr_t266 = (nlr_t252 == INT64_C(16));
        (void)nlr_t266;
        if (!(nlr_t266)) { (void)fflush(stdout); fputs("Assertion failed\n", stderr); abort(); }
        int64_t nlr_t268 = nlr_l1;
        (void)nlr_t268;
        int64_t nlr_t280 = nlr_i64_add(nlr_t268, INT64_C(1));
        (void)nlr_t280;
        nlr_l1 = nlr_t280;
    }
    return INT64_C(0);
}
int64_t nlr_f1_owned_1(int64_t nlr_a0, NlrString nlr_a1) {
    int64_t nlr_l0 = nlr_a0;
    (void)nlr_l0;
    (void)nlr_a1;
    (void)fwrite((nlr_a1).data, 1, (nlr_a1).length, stdout);
    int64_t nlr_t4 = nlr_l0;
    (void)nlr_t4;
    return nlr_t4;
}
int64_t nlr_f2_owned_2(NlrOwned2 nlr_a0) {
    int64_t nlr_l3 = 0;
    (void)nlr_l3;
    int64_t nlr_l5 = 0;
    (void)nlr_l5;
    bool nlr_l8 = false;
    (void)nlr_l8;
    bool nlr_l10 = false;
    (void)nlr_l10;
    int64_t nlr_l12 = 0;
    (void)nlr_l12;
    NlrOwned2 nlr_l0 = nlr_a0;
    NlrOwned2 nlr_t0 = nlr_l0;
    NlrOwned2 nlr_l1 = nlr_t0;
    NlrOwned1 nlr_t6_0 = nlr_l1.f0;
    (void)nlr_t6_0;
    int64_t nlr_t6_1 = nlr_l1.f1;
    (void)nlr_t6_1;
    nlr_l3 = nlr_t6_1;
    NlrOwned1 nlr_l2 = nlr_t6_0;
    NlrOwned1 nlr_t15 = nlr_l2;
    NlrOwned1 nlr_l4 = nlr_t15;
    int64_t nlr_t21 = nlr_l3;
    (void)nlr_t21;
    nlr_l5 = nlr_t21;
    NlrOwned1 nlr_t27 = nlr_l4;
    NlrOwned1 nlr_l6 = nlr_t27;
    NlrOwned0 nlr_t33_0 = nlr_l6.f0;
    (void)nlr_t33_0;
    bool nlr_t33_1 = nlr_l6.f1;
    (void)nlr_t33_1;
    nlr_l8 = nlr_t33_1;
    NlrOwned0 nlr_l7 = nlr_t33_0;
    NlrOwned0 nlr_t42 = nlr_l7;
    NlrOwned0 nlr_l9 = nlr_t42;
    bool nlr_t48 = nlr_l8;
    (void)nlr_t48;
    nlr_l10 = nlr_t48;
    NlrOwned0 nlr_t54 = nlr_l9;
    NlrOwned0 nlr_l11 = nlr_t54;
    int64_t nlr_t66 = nlr_l11.f0;
    (void)nlr_t66;
    nlr_l12 = nlr_t66;
    bool nlr_t75 = nlr_l10;
    (void)nlr_t75;
    if (!(nlr_t75)) { (void)fflush(stdout); fputs("Assertion failed\n", stderr); abort(); }
    int64_t nlr_t79 = nlr_l12;
    (void)nlr_t79;
    int64_t nlr_t82 = nlr_l5;
    (void)nlr_t82;
    int64_t nlr_t85 = nlr_i64_add(nlr_t79, nlr_t82);
    (void)nlr_t85;
    int64_t nlr_t86_0 = nlr_l11.f0;
    (void)nlr_t86_0;
    return nlr_t85;
}
int main(void) {
    return (int)nlr_f0_owned_0();
}
