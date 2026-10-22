#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { const unsigned char *data; size_t length; } NlrString;
static const unsigned char nlr_string_7[] = {65};
static const unsigned char nlr_string_8[] = {66};
static const unsigned char nlr_string_9[] = {67};
typedef struct { int64_t f0; } NlrOwned0;
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
int64_t nlr_f0_owned_0(void);
int64_t nlr_f1_owned_1(int64_t nlr_a0, NlrString nlr_a1);
NlrOwned0 nlr_f2_owned_2(int64_t nlr_a0, NlrString nlr_a1);
NlrOwned0 nlr_f3_owned_3(NlrOwned0 nlr_a0);
int64_t nlr_f4_owned_4(NlrOwned0 nlr_a0, NlrOwned0 nlr_a1, int64_t nlr_a2);
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
    nlr_l0 = INT64_C(3);
    nlr_l1 = INT64_C(0);
    while ((nlr_l1 < nlr_l0)) {
        int64_t nlr_t50 = nlr_f1_owned_1(INT64_C(1), (NlrString){nlr_string_7, 1});
        (void)nlr_t50;
        nlr_l2 = nlr_t50;
        int64_t nlr_t58 = nlr_l2;
        (void)nlr_t58;
        NlrOwned0 nlr_t61 = (NlrOwned0){.f0 = nlr_t58};
        NlrOwned0 nlr_t80 = nlr_f2_owned_2(INT64_C(2), (NlrString){nlr_string_8, 1});
        (void)nlr_t80;
        NlrOwned0 nlr_t85 = nlr_f3_owned_3(nlr_t80);
        (void)nlr_t85;
        int64_t nlr_t104 = nlr_f1_owned_1(INT64_C(3), (NlrString){nlr_string_9, 1});
        (void)nlr_t104;
        int64_t nlr_t109 = nlr_f4_owned_4(nlr_t61, nlr_t85, nlr_t104);
        (void)nlr_t109;
        bool nlr_t123 = (nlr_t109 == INT64_C(3));
        (void)nlr_t123;
        if (!(nlr_t123)) { (void)fflush(stdout); fputs("Assertion failed\n", stderr); abort(); }
        int64_t nlr_t125 = nlr_l1;
        (void)nlr_t125;
        int64_t nlr_t137 = nlr_i64_add(nlr_t125, INT64_C(1));
        (void)nlr_t137;
        nlr_l1 = nlr_t137;
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
NlrOwned0 nlr_f2_owned_2(int64_t nlr_a0, NlrString nlr_a1) {
    int64_t nlr_l0 = nlr_a0;
    (void)nlr_l0;
    int64_t nlr_l2 = 0;
    (void)nlr_l2;
    (void)nlr_a1;
    (void)fwrite((nlr_a1).data, 1, (nlr_a1).length, stdout);
    int64_t nlr_t4 = nlr_l0;
    (void)nlr_t4;
    nlr_l2 = nlr_t4;
    int64_t nlr_t10 = nlr_l2;
    (void)nlr_t10;
    NlrOwned0 nlr_t13 = (NlrOwned0){.f0 = nlr_t10};
    return nlr_t13;
}
NlrOwned0 nlr_f3_owned_3(NlrOwned0 nlr_a0) {
    NlrOwned0 nlr_l0 = nlr_a0;
    NlrOwned0 nlr_t0 = nlr_l0;
    return nlr_t0;
}
int64_t nlr_f4_owned_4(NlrOwned0 nlr_a0, NlrOwned0 nlr_a1, int64_t nlr_a2) {
    int64_t nlr_l2 = nlr_a2;
    (void)nlr_l2;
    int64_t nlr_l4 = 0;
    (void)nlr_l4;
    int64_t nlr_l5 = 0;
    (void)nlr_l5;
    int64_t nlr_l7 = 0;
    (void)nlr_l7;
    NlrOwned0 nlr_l0 = nlr_a0;
    NlrOwned0 nlr_l1 = nlr_a1;
    NlrOwned0 nlr_t0 = nlr_l0;
    NlrOwned0 nlr_l3 = nlr_t0;
    int64_t nlr_t12 = nlr_l3.f0;
    (void)nlr_t12;
    nlr_l4 = nlr_t12;
    int64_t nlr_t21 = nlr_l4;
    (void)nlr_t21;
    nlr_l5 = nlr_t21;
    NlrOwned0 nlr_t27 = nlr_l1;
    NlrOwned0 nlr_l6 = nlr_t27;
    int64_t nlr_t39 = nlr_l6.f0;
    (void)nlr_t39;
    nlr_l7 = nlr_t39;
    int64_t nlr_t48 = nlr_l2;
    (void)nlr_t48;
    bool nlr_t60 = (nlr_t48 == INT64_C(3));
    (void)nlr_t60;
    if (!(nlr_t60)) { (void)fflush(stdout); fputs("Assertion failed\n", stderr); abort(); }
    int64_t nlr_t62 = nlr_l5;
    (void)nlr_t62;
    int64_t nlr_t65 = nlr_l7;
    (void)nlr_t65;
    int64_t nlr_t68 = nlr_i64_add(nlr_t62, nlr_t65);
    (void)nlr_t68;
    int64_t nlr_t69_0 = nlr_l3.f0;
    (void)nlr_t69_0;
    int64_t nlr_t73_0 = nlr_l6.f0;
    (void)nlr_t73_0;
    return nlr_t68;
}
int main(void) {
    return (int)nlr_f0_owned_0();
}
