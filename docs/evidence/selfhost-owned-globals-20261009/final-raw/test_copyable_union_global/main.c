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
int nvm_owned_entry(int64_t *result) {
 nown_value t[256]={{0}}, l[256]={{0}}, a={0}, c={0}, pending={0};
 nown_reference refs[256]={{0}}; unsigned region=0;
 int status=0; (void)result; (void)a; (void)c; (void)nown_retain; (void)nown_string_new; (void)nown_string_equal; (void)refs; (void)region; (void)nown_referent;
 nown_value globals[1]={{0}};
 nown_value *origins[2]={l,NULL}; uint64_t generations[2]={1,0},next_generation=1; (void)next_generation;
 (void)nown_float_bits;
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
 if(!a.record){status=1;goto cleanup;} a.record->refs=1; a.record->count=1; a.record->layout=1; a.record->variant=0;
 a.record->fields[0]=t[0]; t[0]=(nown_value){0};
 a.tag=10; t[0]=a; a=(nown_value){0};
 goto L25;
L25:;
 nown_release(globals[0]); globals[0]=t[0]; t[0]=(nown_value){0};
 goto L30;
L30:;
 t[0]=(nown_value){.scalar=(int64_t)UINT64_C(7),.tag=1};
 goto L39;
L39:;
 nown_release(l[1]); l[1]=t[0]; t[0]=(nown_value){0};
 goto L42;
L42:;
 if(!nown_retain(l[1])){status=1;goto cleanup;} t[0]=l[1];
 goto L45;
L45:;
 a.record=NOWN_ALLOC(1,sizeof(nown_record)+1*sizeof(nown_value));
 if(!a.record){status=1;goto cleanup;} a.record->refs=1; a.record->count=1; a.record->layout=0;
 a.record->fields[0]=t[0]; t[0]=(nown_value){0};
 a.tag=8; t[0]=a; a=(nown_value){0};
 goto L50;
L50:;
 nown_release(l[2]); l[2]=t[0]; t[0]=(nown_value){0};
 goto L53;
L53:;
 t[0]=l[2]; l[2]=(nown_value){0};
 goto L56;
L56:;
 nown_release(l[3]); l[3]=t[0]; t[0]=(nown_value){0};
 goto L59;
L59:;
 ++region;
 goto L60;
L60:;
 refs[0]=(nown_reference){.root=3,.region=region,.exclusive=0,.parent=65535,.depth=0};
 refs[0].origin=0; refs[0].generation=generations[0];
 goto L65;
L65:;
 { nown_record *record=nown_referent(origins,generations,&refs[0]); if(!record){status=3;goto cleanup;} t[0]=record->fields[0]; }
 goto L70;
L70:;
 for(unsigned r=0;r<256;r++) if(refs[r].region==region) refs[r].region=0;
 --region;
 goto L71;
L71:;
 nown_release(l[4]); l[4]=t[0]; t[0]=(nown_value){0};
 goto L74;
L74:;
 if(!nown_retain(l[4])){status=1;goto cleanup;} t[0]=l[4];
 goto L77;
L77:;
 t[1]=(nown_value){.scalar=(int64_t)UINT64_C(7),.tag=1};
 goto L86;
L86:;
 { int comparison;  if(t[0].tag==5) comparison=nown_string_equal(t[0].string,t[1].string); else  if(t[0].tag==3) {
 comparison=t[0].floating == t[1].floating;
 } else comparison=t[0].scalar == t[1].scalar; nown_release(t[0]); nown_release(t[1]); t[0]=(nown_value){.scalar=comparison,.tag=4}; t[1]=(nown_value){0}; }
 goto L87;
L87:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L88;
L88:;
 if(!nown_retain(globals[0])){status=1;goto cleanup;} t[0]=globals[0];
 goto L93;
L93:;
 if(t[0].tag==10 && t[0].record && t[0].record->variant==0) {goto L105;}
 goto L100;
L100:;
 goto L132;
L105:;
 if(!nown_retain(t[0])){status=1;goto cleanup;} t[1]=t[0];
 goto L106;
L106:;
 nown_release(l[5]); l[5]=t[1]; t[1]=(nown_value){0};
 goto L109;
L109:;
 nown_release(t[0]); t[0]=(nown_value){0};
 goto L110;
L110:;
 if(!nown_retain(l[5])){status=1;goto cleanup;} t[0]=l[5];
 goto L113;
L113:;
 if(t[0].tag!=10 || !t[0].record || 0>=t[0].record->count){status=3;goto cleanup;}
 if(!nown_retain(t[0].record->fields[0])){status=1;goto cleanup;} a=t[0]; t[0]=a.record->fields[0]; nown_release(a); a=(nown_value){0};
 goto L116;
L116:;
 t[1]=(nown_value){.scalar=(int64_t)UINT64_C(7),.tag=1};
 goto L125;
L125:;
 { int comparison;  if(t[0].tag==5) comparison=nown_string_equal(t[0].string,t[1].string); else  if(t[0].tag==3) {
 comparison=t[0].floating == t[1].floating;
 } else comparison=t[0].scalar == t[1].scalar; nown_release(t[0]); nown_release(t[1]); t[0]=(nown_value){.scalar=comparison,.tag=4}; t[1]=(nown_value){0}; }
 goto L126;
L126:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L127;
L127:;
 goto L162;
L132:;
 if(t[0].tag==10 && t[0].record && t[0].record->variant==1) {goto L144;}
 status=3;goto cleanup;
L144:;
 if(!nown_retain(t[0])){status=1;goto cleanup;} t[1]=t[0];
 goto L145;
L145:;
 nown_release(l[6]); l[6]=t[1]; t[1]=(nown_value){0};
 goto L148;
L148:;
 nown_release(t[0]); t[0]=(nown_value){0};
 goto L149;
L149:;
 t[0]=(nown_value){.scalar=0,.tag=4};
 goto L151;
L151:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L152;
L152:;
 goto L162;
L162:;
 a.record=NOWN_ALLOC(1,sizeof(nown_record)+0*sizeof(nown_value));
 if(!a.record){status=1;goto cleanup;} a.record->refs=1; a.record->count=0; a.record->layout=1; a.record->variant=1;
 a.tag=10; t[0]=a; a=(nown_value){0};
 goto L172;
L172:;
 nown_release(globals[0]); globals[0]=t[0]; t[0]=(nown_value){0};
 goto L177;
L177:;
 if(!nown_retain(globals[0])){status=1;goto cleanup;} t[0]=globals[0];
 goto L182;
L182:;
 if(t[0].tag==10 && t[0].record && t[0].record->variant==0) {goto L194;}
 goto L189;
L189:;
 goto L207;
L194:;
 if(!nown_retain(t[0])){status=1;goto cleanup;} t[1]=t[0];
 goto L195;
L195:;
 nown_release(l[7]); l[7]=t[1]; t[1]=(nown_value){0};
 goto L198;
L198:;
 nown_release(t[0]); t[0]=(nown_value){0};
 goto L199;
L199:;
 t[0]=(nown_value){.scalar=0,.tag=4};
 goto L201;
L201:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L202;
L202:;
 goto L237;
L207:;
 if(t[0].tag==10 && t[0].record && t[0].record->variant==1) {goto L219;}
 status=3;goto cleanup;
L219:;
 if(!nown_retain(t[0])){status=1;goto cleanup;} t[1]=t[0];
 goto L220;
L220:;
 nown_release(l[8]); l[8]=t[1]; t[1]=(nown_value){0};
 goto L223;
L223:;
 nown_release(t[0]); t[0]=(nown_value){0};
 goto L224;
L224:;
 t[0]=(nown_value){.scalar=1,.tag=4};
 goto L226;
L226:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L227;
L227:;
 goto L237;
L237:;
 t[0]=(nown_value){.scalar=(int64_t)UINT64_C(0),.tag=1};
 goto L246;
L246:;
 a=l[3]; l[3]=(nown_value){0};
 t[1]=a.record->fields[0]; a.record->fields[0]=(nown_value){0};
 nown_release(a); a=(nown_value){0};
 goto L249;
L249:;
 nown_release(t[1]); t[1]=(nown_value){0};
 goto L250;
L250:;
 pending=t[0]; t[0]=(nown_value){0};
 goto cleanup;
cleanup:;
 for(size_t i=0;i<256;i++){nown_release(t[i]);nown_release(l[i]);}
 for(size_t i=0;i<1;i++)nown_release(globals[i]);
 if(!status){*result=pending.scalar; pending=(nown_value){0};}
 nown_release(pending); return status;
}
#ifndef NVM2C_NO_MAIN
int main(void){int64_t result=0;return nvm_owned_entry(&result)?1:(int)result;}
#endif
