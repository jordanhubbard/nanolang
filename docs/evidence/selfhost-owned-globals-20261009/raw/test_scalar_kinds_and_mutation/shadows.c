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
/* I transfer unique owners; field observations retain only their shell. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#ifndef NOWN_ALLOC
#define NOWN_ALLOC calloc
#endif
#ifndef NOWN_FREE
#define NOWN_FREE free
#endif
typedef struct nown_record nown_record;
typedef struct { size_t refs,length; unsigned char bytes[]; } nown_string;
typedef struct { int64_t scalar; nown_record *record; nown_string *string; double floating; uint8_t tag; } nown_value;
static inline double nown_float_bits(uint64_t bits) { double value; memcpy(&value,&bits,sizeof(value)); return value; }
struct nown_record { size_t refs, count; uint32_t layout, variant; nown_value fields[]; };
typedef struct { unsigned root,region,exclusive,parent,depth,origin; uint64_t generation; uint16_t fields[32]; } nown_reference;
static nown_record *nown_referent(nown_value *const origins[2],const uint64_t generations[2],const nown_reference *ref) {
 if(ref->origin>1 || !origins[ref->origin] || ref->generation!=generations[ref->origin])return NULL;
 nown_record *record=origins[ref->origin][ref->root].record;
 for(unsigned i=0;i<ref->depth;i++) record=record->fields[ref->fields[i]].record;
 return record;
}
static void nown_release(nown_value v) {
 if (v.string && --v.string->refs==0) NOWN_FREE(v.string);
 if (v.record && --v.record->refs==0) {
  for(size_t i=0;i<v.record->count;i++) nown_release(v.record->fields[i]);
  NOWN_FREE(v.record);
 }
}
static int nown_retain(nown_value v) {
 if((v.string && v.string->refs==SIZE_MAX) || (v.record && v.record->refs==SIZE_MAX)) return 0;
 if(v.string) ++v.string->refs;
 if(v.record) ++v.record->refs;
 return 1;
}
static nown_string *nown_string_new(const unsigned char *bytes,size_t length) {
 if(length>SIZE_MAX-sizeof(nown_string)-1) return NULL;
 nown_string *s=NOWN_ALLOC(1,sizeof(*s)+length+1); if(!s) return NULL;
 s->refs=1; s->length=length; if(length) memcpy(s->bytes,bytes,length); s->bytes[length]=0; return s;
}
static int nown_string_equal(const nown_string *a,const nown_string *b) {
 if(a==b) return 1;
 if(!a || !b) return 0;
 return a->length==b->length && !memcmp(a->bytes,b->bytes,a->length);
}
/* I return status separately so allocation failure still cleans every root. */
static int nown_function_1(nown_value *argument,uint64_t *next_generation,uint64_t generation,nown_value *result,nown_value *globals);
static int nown_function_1(nown_value *argument,uint64_t *next_generation,uint64_t generation,nown_value *result,nown_value *globals) {
 nown_value t[256]={{0}}, l[256]={{0}}, a={0}, c={0}, pending={0};
 nown_reference refs[256]={{0}}; unsigned region=0;
 int status=0; (void)result; (void)a; (void)c; (void)nown_retain; (void)nown_string_new; (void)nown_string_equal; (void)refs; (void)region; (void)nown_referent;
 (void)globals;
 nown_value *origins[2]={NULL,l}; uint64_t generations[2]={0,generation}; (void)argument; (void)next_generation;
 (void)nown_float_bits;
 (void)nano_rt_f64_add; (void)nano_rt_f64_sub; (void)nano_rt_f64_mul; (void)nano_rt_f64_div;
 (void)origins; (void)generations; goto L0;
L0:;
 t[0]=(nown_value){.scalar=(int64_t)UINT64_C(7),.tag=1};
 goto L9;
L9:;
 nown_release(l[0]); l[0]=t[0]; t[0]=(nown_value){0};
 goto L12;
L12:;
 if(!nown_retain(l[0])){status=1;goto cleanup;} t[0]=l[0];
 goto L15;
L15:;
 a.record=NOWN_ALLOC(1,sizeof(nown_record)+1*sizeof(nown_value));
 if(!a.record){status=1;goto cleanup;} a.record->refs=1; a.record->count=1; a.record->layout=0;
 a.record->fields[0]=t[0]; t[0]=(nown_value){0};
 a.tag=8; t[0]=a; a=(nown_value){0};
 goto L20;
L20:;
 nown_release(l[1]); l[1]=t[0]; t[0]=(nown_value){0};
 goto L23;
L23:;
 t[0]=l[1]; l[1]=(nown_value){0};
 goto L26;
L26:;
 nown_release(l[2]); l[2]=t[0]; t[0]=(nown_value){0};
 goto L29;
L29:;
 ++region;
 goto L30;
L30:;
 refs[0]=(nown_reference){.root=2,.region=region,.exclusive=0,.parent=65535,.depth=0};
 refs[0].origin=1; refs[0].generation=generations[1];
 goto L35;
L35:;
 { nown_record *record=nown_referent(origins,generations,&refs[0]); if(!record){status=3;goto cleanup;} t[0]=record->fields[0]; }
 goto L40;
L40:;
 for(unsigned r=0;r<256;r++) if(refs[r].region==region) refs[r].region=0;
 --region;
 goto L41;
L41:;
 nown_release(l[3]); l[3]=t[0]; t[0]=(nown_value){0};
 goto L44;
L44:;
 if(!nown_retain(l[3])){status=1;goto cleanup;} t[0]=l[3];
 goto L47;
L47:;
 t[1]=(nown_value){.scalar=(int64_t)UINT64_C(7),.tag=1};
 goto L56;
L56:;
 { int comparison;  if(t[0].tag==5) comparison=nown_string_equal(t[0].string,t[1].string); else  if(t[0].tag==3) {
 comparison=t[0].floating == t[1].floating;
 } else comparison=t[0].scalar == t[1].scalar; nown_release(t[0]); nown_release(t[1]); t[0]=(nown_value){.scalar=comparison,.tag=4}; t[1]=(nown_value){0}; }
 goto L57;
L57:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L58;
L58:;
 if(!nown_retain(globals[0])){status=1;goto cleanup;} t[0]=globals[0];
 goto L63;
L63:;
 t[1]=(nown_value){.scalar=(int64_t)UINT64_C(3),.tag=1};
 goto L72;
L72:;
 { int comparison;  if(t[0].tag==5) comparison=nown_string_equal(t[0].string,t[1].string); else  if(t[0].tag==3) {
 comparison=t[0].floating == t[1].floating;
 } else comparison=t[0].scalar == t[1].scalar; nown_release(t[0]); nown_release(t[1]); t[0]=(nown_value){.scalar=comparison,.tag=4}; t[1]=(nown_value){0}; }
 goto L73;
L73:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L74;
L74:;
 if(!nown_retain(globals[0])){status=1;goto cleanup;} t[0]=globals[0];
 goto L79;
L79:;
 t[1]=(nown_value){.scalar=(int64_t)UINT64_C(1),.tag=1};
 goto L88;
L88:;
 t[0].scalar=(int64_t)((uint64_t)t[0].scalar + (uint64_t)t[1].scalar); t[1]=(nown_value){0};
 goto L89;
L89:;
 nown_release(globals[0]); globals[0]=t[0]; t[0]=(nown_value){0};
 goto L94;
L94:;
 if(!nown_retain(globals[1])){status=1;goto cleanup;} t[0]=globals[1];
 goto L99;
L99:;
 t[0].scalar=!t[0].scalar;
 goto L100;
L100:;
 nown_release(globals[1]); globals[1]=t[0]; t[0]=(nown_value){0};
 goto L105;
L105:;
 if(!nown_retain(globals[2])){status=1;goto cleanup;} t[0]=globals[2];
 goto L110;
L110:;
 t[1]=(nown_value){.floating=nown_float_bits(UINT64_C(0x3fe8000000000000)),.tag=3};
 goto L119;
L119:;
 t[0]=(nown_value){.floating=nano_rt_f64_add(t[0].floating,t[1].floating),.tag=3}; t[1]=(nown_value){0};
 goto L120;
L120:;
 nown_release(globals[2]); globals[2]=t[0]; t[0]=(nown_value){0};
 goto L125;
L125:;
 a.string=nown_string_new((const unsigned char *)"second",6); if(!a.string){status=1;goto cleanup;} a.tag=5; t[0]=a; a=(nown_value){0};
 goto L130;
L130:;
 nown_release(globals[3]); globals[3]=t[0]; t[0]=(nown_value){0};
 goto L135;
L135:;
 if(!nown_retain(globals[0])){status=1;goto cleanup;} t[0]=globals[0];
 goto L140;
L140:;
 t[1]=(nown_value){.scalar=(int64_t)UINT64_C(4),.tag=1};
 goto L149;
L149:;
 { int comparison;  if(t[0].tag==5) comparison=nown_string_equal(t[0].string,t[1].string); else  if(t[0].tag==3) {
 comparison=t[0].floating == t[1].floating;
 } else comparison=t[0].scalar == t[1].scalar; nown_release(t[0]); nown_release(t[1]); t[0]=(nown_value){.scalar=comparison,.tag=4}; t[1]=(nown_value){0}; }
 goto L150;
L150:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L151;
L151:;
 if(!nown_retain(globals[1])){status=1;goto cleanup;} t[0]=globals[1];
 goto L156;
L156:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L157;
L157:;
 if(!nown_retain(globals[2])){status=1;goto cleanup;} t[0]=globals[2];
 goto L162;
L162:;
 t[1]=(nown_value){.floating=nown_float_bits(UINT64_C(0x4000000000000000)),.tag=3};
 goto L171;
L171:;
 t[0]=(nown_value){.scalar=(t[0].floating == t[1].floating),.tag=4}; t[1]=(nown_value){0};
 goto L172;
L172:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L173;
L173:;
 if(!nown_retain(globals[3])){status=1;goto cleanup;} t[0]=globals[3];
 goto L178;
L178:;
 a.string=nown_string_new((const unsigned char *)"second",6); if(!a.string){status=1;goto cleanup;} a.tag=5; t[1]=a; a=(nown_value){0};
 goto L183;
L183:;
 { int comparison;  if(t[0].tag==5) comparison=nown_string_equal(t[0].string,t[1].string); else  if(t[0].tag==3) {
 comparison=t[0].floating == t[1].floating;
 } else comparison=t[0].scalar == t[1].scalar; nown_release(t[0]); nown_release(t[1]); t[0]=(nown_value){.scalar=comparison,.tag=4}; t[1]=(nown_value){0}; }
 goto L184;
L184:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L185;
L185:;
 t[0]=(nown_value){.scalar=(int64_t)UINT64_C(0),.tag=1};
 goto L194;
L194:;
 a=l[2]; l[2]=(nown_value){0};
 t[1]=a.record->fields[0]; a.record->fields[0]=(nown_value){0};
 nown_release(a); a=(nown_value){0};
 goto L197;
L197:;
 nown_release(t[1]); t[1]=(nown_value){0};
 goto L198;
L198:;
 pending=t[0]; t[0]=(nown_value){0};
 goto cleanup;
cleanup:;
 for(size_t i=0;i<256;i++){nown_release(t[i]);nown_release(l[i]);}
 if(!status){*result=pending; pending=(nown_value){0};}
 nown_release(pending); return status;
}
int nvm_owned_entry(int64_t *result) {
 nown_value t[256]={{0}}, l[256]={{0}}, a={0}, c={0}, pending={0};
 nown_reference refs[256]={{0}}; unsigned region=0;
 int status=0; (void)result; (void)a; (void)c; (void)nown_retain; (void)nown_string_new; (void)nown_string_equal; (void)refs; (void)region; (void)nown_referent;
 nown_value globals[4]={{0}};
 nown_value *origins[2]={l,NULL}; uint64_t generations[2]={1,0},next_generation=1; (void)next_generation;
 (void)nown_function_1;
 (void)nown_float_bits;
 (void)nano_rt_f64_add; (void)nano_rt_f64_sub; (void)nano_rt_f64_mul; (void)nano_rt_f64_div;
 (void)origins; (void)generations; goto L0;
L0:;
 t[0]=(nown_value){.scalar=(int64_t)UINT64_C(3),.tag=1};
 goto L9;
L9:;
 nown_release(globals[0]); globals[0]=t[0]; t[0]=(nown_value){0};
 goto L14;
L14:;
 t[0]=(nown_value){.scalar=0,.tag=4};
 goto L16;
L16:;
 nown_release(globals[1]); globals[1]=t[0]; t[0]=(nown_value){0};
 goto L21;
L21:;
 t[0]=(nown_value){.floating=nown_float_bits(UINT64_C(0x3ff4000000000000)),.tag=3};
 goto L30;
L30:;
 nown_release(globals[2]); globals[2]=t[0]; t[0]=(nown_value){0};
 goto L35;
L35:;
 a.string=nown_string_new((const unsigned char *)"first",5); if(!a.string){status=1;goto cleanup;} a.tag=5; t[0]=a; a=(nown_value){0};
 goto L40;
L40:;
 nown_release(globals[3]); globals[3]=t[0]; t[0]=(nown_value){0};
 goto L45;
L45:;
 goto L46;
L46:;
 if(next_generation==UINT64_MAX){status=3;goto cleanup;}
 status=nown_function_1(&t[0],&next_generation,++next_generation,&t[0],globals); if(status)goto cleanup;
 goto L51;
L51:;
 t[1]=(nown_value){.scalar=(int64_t)UINT64_C(0),.tag=1};
 goto L60;
L60:;
 { int comparison;  if(t[0].tag==5) comparison=nown_string_equal(t[0].string,t[1].string); else  if(t[0].tag==3) {
 comparison=t[0].floating == t[1].floating;
 } else comparison=t[0].scalar == t[1].scalar; nown_release(t[0]); nown_release(t[1]); t[0]=(nown_value){.scalar=comparison,.tag=4}; t[1]=(nown_value){0}; }
 goto L61;
L61:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L62;
L62:;
 t[0]=(nown_value){.scalar=(int64_t)UINT64_C(0),.tag=1};
 goto L71;
L71:;
 pending=t[0]; t[0]=(nown_value){0};
 goto cleanup;
cleanup:;
 for(size_t i=0;i<256;i++){nown_release(t[i]);nown_release(l[i]);}
 for(size_t i=0;i<4;i++)nown_release(globals[i]);
 if(!status){*result=pending.scalar; pending=(nown_value){0};}
 nown_release(pending); return status;
}
#ifndef NVM2C_NO_MAIN
int main(void){int64_t result=0;return nvm_owned_entry(&result)?1:(int)result;}
#endif
