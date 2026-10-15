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
 nown_value globals[2]={{0}};
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
 if(!nown_retain(globals[0])){status=1;goto cleanup;} t[0]=globals[0];
 goto L35;
L35:;
 nown_release(globals[1]); globals[1]=t[0]; t[0]=(nown_value){0};
 goto L40;
L40:;
 t[0]=(nown_value){.scalar=(int64_t)UINT64_C(7),.tag=1};
 goto L49;
L49:;
 nown_release(l[1]); l[1]=t[0]; t[0]=(nown_value){0};
 goto L52;
L52:;
 if(!nown_retain(l[1])){status=1;goto cleanup;} t[0]=l[1];
 goto L55;
L55:;
 a.record=NOWN_ALLOC(1,sizeof(nown_record)+1*sizeof(nown_value));
 if(!a.record){status=1;goto cleanup;} a.record->refs=1; a.record->count=1; a.record->layout=0;
 a.record->fields[0]=t[0]; t[0]=(nown_value){0};
 a.tag=8; t[0]=a; a=(nown_value){0};
 goto L60;
L60:;
 nown_release(l[2]); l[2]=t[0]; t[0]=(nown_value){0};
 goto L63;
L63:;
 t[0]=l[2]; l[2]=(nown_value){0};
 goto L66;
L66:;
 nown_release(l[3]); l[3]=t[0]; t[0]=(nown_value){0};
 goto L69;
L69:;
 ++region;
 goto L70;
L70:;
 refs[0]=(nown_reference){.root=3,.region=region,.exclusive=0,.parent=65535,.depth=0};
 refs[0].origin=0; refs[0].generation=generations[0];
 goto L75;
L75:;
 { nown_record *record=nown_referent(origins,generations,&refs[0]); if(!record){status=3;goto cleanup;} t[0]=record->fields[0]; }
 goto L80;
L80:;
 for(unsigned r=0;r<256;r++) if(refs[r].region==region) refs[r].region=0;
 --region;
 goto L81;
L81:;
 nown_release(l[4]); l[4]=t[0]; t[0]=(nown_value){0};
 goto L84;
L84:;
 if(!nown_retain(l[4])){status=1;goto cleanup;} t[0]=l[4];
 goto L87;
L87:;
 t[1]=(nown_value){.scalar=(int64_t)UINT64_C(7),.tag=1};
 goto L96;
L96:;
 { int comparison;  if(t[0].tag==5) comparison=nown_string_equal(t[0].string,t[1].string); else  if(t[0].tag==3) {
 comparison=t[0].floating == t[1].floating;
 } else comparison=t[0].scalar == t[1].scalar; nown_release(t[0]); nown_release(t[1]); t[0]=(nown_value){.scalar=comparison,.tag=4}; t[1]=(nown_value){0}; }
 goto L97;
L97:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L98;
L98:;
 if(!nown_retain(globals[1])){status=1;goto cleanup;} t[0]=globals[1];
 goto L103;
L103:;
 if(t[0].tag==10 && t[0].record && t[0].record->variant==0) {goto L115;}
 goto L110;
L110:;
 goto L142;
L115:;
 if(!nown_retain(t[0])){status=1;goto cleanup;} t[1]=t[0];
 goto L116;
L116:;
 nown_release(l[5]); l[5]=t[1]; t[1]=(nown_value){0};
 goto L119;
L119:;
 nown_release(t[0]); t[0]=(nown_value){0};
 goto L120;
L120:;
 if(!nown_retain(l[5])){status=1;goto cleanup;} t[0]=l[5];
 goto L123;
L123:;
 if(t[0].tag!=10 || !t[0].record || 0>=t[0].record->count){status=3;goto cleanup;}
 if(!nown_retain(t[0].record->fields[0])){status=1;goto cleanup;} a=t[0]; t[0]=a.record->fields[0]; nown_release(a); a=(nown_value){0};
 goto L126;
L126:;
 t[1]=(nown_value){.scalar=(int64_t)UINT64_C(7),.tag=1};
 goto L135;
L135:;
 { int comparison;  if(t[0].tag==5) comparison=nown_string_equal(t[0].string,t[1].string); else  if(t[0].tag==3) {
 comparison=t[0].floating == t[1].floating;
 } else comparison=t[0].scalar == t[1].scalar; nown_release(t[0]); nown_release(t[1]); t[0]=(nown_value){.scalar=comparison,.tag=4}; t[1]=(nown_value){0}; }
 goto L136;
L136:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L137;
L137:;
 goto L172;
L142:;
 if(t[0].tag==10 && t[0].record && t[0].record->variant==1) {goto L154;}
 status=3;goto cleanup;
L154:;
 if(!nown_retain(t[0])){status=1;goto cleanup;} t[1]=t[0];
 goto L155;
L155:;
 nown_release(l[6]); l[6]=t[1]; t[1]=(nown_value){0};
 goto L158;
L158:;
 nown_release(t[0]); t[0]=(nown_value){0};
 goto L159;
L159:;
 t[0]=(nown_value){.scalar=0,.tag=4};
 goto L161;
L161:;
 a=t[0]; t[0]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};
 goto L162;
L162:;
 goto L172;
L172:;
 t[0]=(nown_value){.scalar=(int64_t)UINT64_C(0),.tag=1};
 goto L181;
L181:;
 a=l[3]; l[3]=(nown_value){0};
 t[1]=a.record->fields[0]; a.record->fields[0]=(nown_value){0};
 nown_release(a); a=(nown_value){0};
 goto L184;
L184:;
 nown_release(t[1]); t[1]=(nown_value){0};
 goto L185;
L185:;
 pending=t[0]; t[0]=(nown_value){0};
 goto cleanup;
cleanup:;
 for(size_t i=0;i<256;i++){nown_release(t[i]);nown_release(l[i]);}
 for(size_t i=0;i<2;i++)nown_release(globals[i]);
 if(!status){*result=pending.scalar; pending=(nown_value){0};}
 nown_release(pending); return status;
}
#ifndef NVM2C_NO_MAIN
int main(void){int64_t result=0;return nvm_owned_entry(&result)?1:(int)result;}
#endif
