/* I check runtime operands and exact result representations under default FP. */
#include "binary64_arithmetic.h"
#include "binary64_bits.h"
#include <assert.h>
#include <fenv.h>
#include <stdio.h>

static double from(uint64_t bits) { double x; memcpy(&x,&bits,8); return x; }
static uint64_t bits(double x) { uint64_t u; memcpy(&u,&x,8); return u; }
static unsigned checks;
static double operate(unsigned op, double a, double b) {
    switch(op) {
        case 0:return nano_rt_f64_add(a,b);
        case 1:return nano_rt_f64_sub(a,b);
        case 2:return nano_rt_f64_mul(a,b);
        default:return nano_rt_f64_div(a,b);
    }
}
static void check(unsigned op,uint64_t a,uint64_t b,uint64_t want) {
    volatile uint64_t input_a=a,input_b=b;
    double x=from(input_a),y=from(input_b);
    assert(bits(operate(op,x,y))==want);
    assert(bits(x)==a && bits(y)==b);
    assert((uint64_t)nl_float_to_bits(nl_float_from_bits(nl_float_to_bits(x)))==a);
    assert(bits(-x)==(a^UINT64_C(0x8000000000000000)));
    ++checks;
}
/* Ordinary source f64_* functions keep their legacy names beside my runtime. */
static int nl_f64_add(void){return 1;}
static int nl_f64_sub(void){return 2;}
static int nl_f64_mul(void){return 3;}
static int nl_f64_div(void){return 4;}
int main(void) {
    assert(fegetround()==FE_TONEAREST);
    const uint64_t q=UINT64_C(0x7ff8000000000000);
    const uint64_t one=UINT64_C(0x3ff0000000000000);
    const uint64_t inf=UINT64_C(0x7ff0000000000000);
    const uint64_t sign=UINT64_C(0x8000000000000000);
    const uint64_t nans[]={UINT64_C(0x7ff0000000000001),UINT64_C(0xfff0000000000042),
        UINT64_C(0x7ff8123456789abc),UINT64_C(0xfff8000000001234)};
    for(unsigned op=0;op<4;op++) for(unsigned i=0;i<4;i++) {
        check(op,nans[i],one,q);check(op,one,nans[i],q);
        for(unsigned j=0;j<4;j++) check(op,nans[i],nans[j],q);
    }
    const uint64_t numerators[]={0,sign,one,inf,inf|sign,nans[0],nans[1],nans[2],nans[3]};
    for(unsigned i=0;i<9;i++){check(3,numerators[i],0,0);check(3,numerators[i],sign,0);}
    check(0,inf,inf|sign,q);check(1,inf,inf,q);
    check(2,0,inf,q);check(2,inf,sign,q);check(3,inf,inf,q);
    check(0,sign,sign,sign);check(0,sign,0,0);check(1,sign,0,sign);
    check(2,sign,one,sign);check(3,sign,one,sign);
    check(0,one,UINT64_C(0x3ca0000000000000),one); /* half an ulp, even */
    check(0,one+1,UINT64_C(0x3ca0000000000000),one+2); /* odd rounds up */
    check(2,1,UINT64_C(0x3fe0000000000000),0);
    check(2,3,UINT64_C(0x3fe0000000000000),2);
    check(2,sign|1,UINT64_C(0x3fe0000000000000),sign);
    check(0,1,1,2); /* gradual underflow is required */
    check(0,UINT64_C(0x7fefffffffffffff),UINT64_C(0x7fefffffffffffff),inf);
    check(3,one,UINT64_C(0x4008000000000000),UINT64_C(0x3fd5555555555555));
    volatile uint64_t a=one+1,b=UINT64_C(0x3feffffffffffffe);
    double product=nano_rt_f64_mul(from(a),from(b));
    assert(bits(nano_rt_f64_sub(product,from(one)))==0);++checks;
    assert(nl_f64_add()+nl_f64_sub()+nl_f64_mul()+nl_f64_div()==10);
    printf("I passed %u arithmetic bit checks.\n",checks);
    return 0;
}
