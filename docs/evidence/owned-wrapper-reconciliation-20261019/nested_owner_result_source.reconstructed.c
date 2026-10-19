#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { const unsigned char *data; size_t length; } NlrString;
static const unsigned char nlr_string_13[] = {65};
static const unsigned char nlr_string_14[] = {66};
typedef struct { int64_t f0; } NlrOwned0;
typedef struct { NlrOwned0 f0; bool f1; } NlrOwned1;
typedef struct { NlrOwned1 f0; int64_t f1; } NlrOwned2;
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
int64_t nlr_f0_owned_0(void);
int64_t nlr_f1_owned_1(int64_t nlr_a0, NlrString nlr_a1);
NlrOwned2 nlr_f2_owned_2(bool nlr_a0);
NlrOwned2 nlr_f3_owned_3(NlrOwned2 nlr_a0);
int64_t nlr_f4_owned_4(NlrOwned2 nlr_a0);
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
    nlr_l0 = INT64_C(2);
    nlr_l1 = INT64_C(0);
    while ((nlr_l1 < nlr_l0)) {
        int64_t nlr_t36 = nlr_l1;
        (void)nlr_t36;
        bool nlr_t48 = (nlr_t36 == INT64_C(0));
        (void)nlr_t48;
        NlrOwned2 nlr_t49 = nlr_f2_owned_2(nlr_t48);
        (void)nlr_t49;
        NlrOwned2 nlr_t54 = nlr_f3_owned_3(nlr_t49);
        (void)nlr_t54;
        NlrOwned2 nlr_l2 = nlr_t54;
        NlrOwned2 nlr_t62 = nlr_l2;
        int64_t nlr_t65 = nlr_f4_owned_4(nlr_t62);
        (void)nlr_t65;
        bool nlr_t79 = (nlr_t65 == INT64_C(16));
        (void)nlr_t79;
        if (!(nlr_t79)) { (void)fflush(stdout); fputs("Assertion failed\n", stderr); abort(); }
        int64_t nlr_t81 = nlr_l1;
        (void)nlr_t81;
        int64_t nlr_t93 = nlr_i64_add(nlr_t81, INT64_C(1));
        (void)nlr_t93;
        nlr_l1 = nlr_t93;
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
NlrOwned2 nlr_f2_owned_2(bool nlr_a0) {
    bool nlr_l0 = nlr_a0;
    (void)nlr_l0;
    int64_t nlr_l1 = 0;
    (void)nlr_l1;
    bool nlr_l4 = false;
    (void)nlr_l4;
    int64_t nlr_l7 = 0;
    (void)nlr_l7;
    int64_t nlr_t14 = nlr_f1_owned_1(INT64_C(7), (NlrString){nlr_string_13, 1});
    (void)nlr_t14;
    nlr_l1 = nlr_t14;
    int64_t nlr_t22 = nlr_l1;
    (void)nlr_t22;
    NlrOwned0 nlr_t25 = (NlrOwned0){.f0 = nlr_t22};
    NlrOwned0 nlr_l2 = nlr_t25;
    NlrOwned0 nlr_t33 = nlr_l2;
    NlrOwned0 nlr_l3 = nlr_t33;
    nlr_l4 = true;
    NlrOwned0 nlr_t44 = nlr_l3;
    bool nlr_t47 = nlr_l4;
    (void)nlr_t47;
    NlrOwned1 nlr_t50 = (NlrOwned1){.f0 = nlr_t44, .f1 = nlr_t47};
    NlrOwned1 nlr_l5 = nlr_t50;
    NlrOwned1 nlr_t58 = nlr_l5;
    NlrOwned1 nlr_l6 = nlr_t58;
    int64_t nlr_t78 = nlr_f1_owned_1(INT64_C(9), (NlrString){nlr_string_14, 1});
    (void)nlr_t78;
    nlr_l7 = nlr_t78;
    NlrOwned1 nlr_t86 = nlr_l6;
    int64_t nlr_t89 = nlr_l7;
    (void)nlr_t89;
    NlrOwned2 nlr_t92 = (NlrOwned2){.f0 = nlr_t86, .f1 = nlr_t89};
    NlrOwned2 nlr_l8 = nlr_t92;
    bool nlr_t100 = nlr_l0;
    (void)nlr_t100;
    if (nlr_t100) {
        NlrOwned2 nlr_t108 = nlr_l8;
        return nlr_t108;
    }
    NlrOwned2 nlr_t112 = nlr_l8;
    return nlr_t112;
}
NlrOwned2 nlr_f3_owned_3(NlrOwned2 nlr_a0) {
    NlrOwned2 nlr_l0 = nlr_a0;
    NlrOwned2 nlr_t0 = nlr_l0;
    return nlr_t0;
}
int64_t nlr_f4_owned_4(NlrOwned2 nlr_a0) {
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
