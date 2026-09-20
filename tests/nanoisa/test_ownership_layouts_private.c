/* I test declaration structure only, without ownership or execution admission. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../src/nanoisa/ownership_layouts_private.h"
#include "../../src/nanoisa/isa.h"
static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"I failed line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
#ifdef LAYOUT_INSTRUMENT
static unsigned allocations,live;static int allowance=-1;static unsigned transient;
static void *layout_calloc(size_t n,size_t width){
 allocations++;if(!allowance || (transient && allocations==transient))return NULL;
 if(allowance>0)allowance--;
 void *p=calloc(n,width);if(p)live++;return p;
}
static void layout_free(void *p){if(p){CHECK(live);live--;}free(p);}
#define calloc layout_calloc
#define free layout_free
#include "../../src/nanoisa/nvm_v2_layouts.c"
#undef calloc
#undef free
#endif
static void u32(unsigned char *p,uint32_t n){for(unsigned i=0;i<4;i++)p[i]=(unsigned char)(n>>(i*8));}
static void u16(unsigned char *p,uint16_t n){p[0]=(unsigned char)n;p[1]=(unsigned char)(n>>8);}
static void header(unsigned char *p,uint16_t fields){p[0]=NVM_V2_LAYOUT_STRUCT;p[1]=0;u16(p+2,fields);u32(p+4,NVM_V2_NO_INDEX);}
static void field(unsigned char *p,uint8_t tag,uint32_t nested){p[0]=tag;p[1]=p[2]=p[3]=0;u32(p+4,nested);u32(p+8,NVM_V2_NO_INDEX);}
static NvmV2Layouts query(const unsigned char *p,size_t n,NvmV2Result expected){
 NvmV2Layouts result={(void *)(uintptr_t)1,73},before=result;
 NvmV2Result actual=nvm_ownership_layouts_private_decode(p,n,&result);
 if(actual!=expected)fprintf(stderr,"I expected %u, received %u\n",expected,actual);
 CHECK(actual==expected);
 if(actual!=NVM_V2_OK){CHECK(result.items==before.items && result.count==before.count);return (NvmV2Layouts){0};}
 return result;
}
static void release(NvmV2Layouts *p){nvm_v2_layouts_free(p);
#ifdef LAYOUT_INSTRUMENT
 CHECK(!live);
#endif
}
static void two(unsigned char *p){memset(p,0,44);u32(p,2);header(p+4,1);field(p+12,TAG_STRUCT,1);header(p+24,1);field(p+32,TAG_ARRAY,NVM_V2_NO_INDEX);}
static void structure(void){
 unsigned char bytes[45];two(bytes);NvmV2Layouts old={0};
 CHECK(nvm_v2_layouts_decode(bytes,44,&old)==NVM_V2_ERR_INDEX_RANGE && !old.items);
 NvmV2Layouts p=query(bytes,44,NVM_V2_OK);CHECK(p.count==2 && p.items[0].fields[0].nested_idx==1 && p.items[1].fields[0].type_tag==TAG_ARRAY);
 memset(bytes,0,sizeof bytes);CHECK(p.items[0].kind==NVM_V2_LAYOUT_STRUCT && p.items[1].field_count==1 && p.items[1].fields[0].nested_idx==NVM_V2_NO_INDEX);release(&p);
 two(bytes);bytes[32]=TAG_STRING;p=query(bytes,44,NVM_V2_OK);release(&p);
 CHECK(nvm_v2_layouts_decode(bytes,44,&old)==NVM_V2_OK);release(&old);
 two(bytes);field(bytes+32,TAG_STRUCT,0);query(bytes,44,NVM_V2_ERR_INDEX_RANGE);
 two(bytes);u32(bytes+16,0);query(bytes,44,NVM_V2_ERR_INDEX_RANGE);
 two(bytes);u32(bytes+36,0);query(bytes,44,NVM_V2_ERR_INDEX_RANGE);
 two(bytes);bytes[33]=1;query(bytes,44,NVM_V2_ERR_RESERVED_FLAGS);
 two(bytes);bytes[24]=NVM_V2_LAYOUT_UNION;query(bytes,44,NVM_V2_ERR_INDEX_RANGE);
 two(bytes);bytes[32]=TAG_FUNCTION;query(bytes,44,NVM_V2_ERR_INDEX_RANGE);
 two(bytes);query(bytes,43,NVM_V2_ERR_TRUNCATED);bytes[44]=0;query(bytes,45,NVM_V2_ERR_SECTION_RANGE);
 two(bytes);CHECK(nvm_ownership_layouts_private_decode(bytes,44,NULL)==NVM_V2_ERR_SECTION_RANGE);
 query(NULL,44,NVM_V2_ERR_SECTION_RANGE);
 /* I preserve the original broader prior-only structural policy. A later
  * authority reader must still validate every tag/referent combination. */
 two(bytes);field(bytes+12,TAG_ARRAY,NVM_V2_NO_INDEX);field(bytes+32,TAG_STRUCT,0);
 p=query(bytes,44,NVM_V2_OK);release(&p);CHECK(nvm_v2_layouts_decode(bytes,44,&old)==NVM_V2_OK);release(&old);
 unsigned char empty[4]={0};p=query(empty,4,NVM_V2_OK);CHECK(!p.items && !p.count);release(&p);
}
static void boundaries(void){
 unsigned char *bytes=calloc(1,4+256*20);CHECK(bytes);u32(bytes,256);
 for(unsigned i=0;i<256;i++){header(bytes+4+i*20,1);field(bytes+12+i*20,i==255?TAG_ARRAY:TAG_STRUCT,i==255?NVM_V2_NO_INDEX:i+1);}
 NvmV2Layouts p=query(bytes,4+256*20,NVM_V2_OK);CHECK(p.count==256);release(&p);
 u32(bytes,257);
#ifdef LAYOUT_INSTRUMENT
 allocations=0;
#endif
 query(bytes,4+256*20,NVM_V2_ERR_INDEX_RANGE);
#ifdef LAYOUT_INSTRUMENT
 CHECK(!allocations);
#endif
 free(bytes);
 size_t size=4+16+65536u*12;bytes=calloc(1,size+12);CHECK(bytes);u32(bytes,2);header(bytes+4,65535);
 for(unsigned i=0;i<65535;i++)field(bytes+12+i*12,TAG_ARRAY,NVM_V2_NO_INDEX);
 size_t second=12+65535u*12;header(bytes+second,1);field(bytes+second+8,TAG_ARRAY,NVM_V2_NO_INDEX);
 p=query(bytes,size,NVM_V2_OK);CHECK(p.count==2 && p.items[0].field_count==65535 && p.items[1].field_count==1);release(&p);
 header(bytes+second,2);field(bytes+second+20,TAG_ARRAY,NVM_V2_NO_INDEX);
#ifdef LAYOUT_INSTRUMENT
 allocations=0;
#endif
 query(bytes,size+12,NVM_V2_ERR_INDEX_RANGE);
#ifdef LAYOUT_INSTRUMENT
 CHECK(!allocations);
#endif
 free(bytes);
 bytes=malloc(NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_BYTES+1u);CHECK(bytes);
 query(bytes,NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_BYTES+1u,NVM_V2_ERR_INDEX_RANGE);free(bytes);
}
static void faults(void){
#ifdef LAYOUT_INSTRUMENT
 unsigned char bytes[44];two(bytes);allocations=0;
 NvmV2Layouts p=query(bytes,sizeof bytes,NVM_V2_OK);unsigned count=allocations;release(&p);CHECK(count==5);
 for(unsigned i=0;i<count;i++){
  allocations=0;allowance=(int)i;query(bytes,sizeof bytes,NVM_V2_ERR_TRUNCATED);allowance=-1;CHECK(!live);
  p=query(bytes,sizeof bytes,NVM_V2_OK);release(&p);
  allocations=0;transient=i+1;query(bytes,sizeof bytes,NVM_V2_ERR_TRUNCATED);transient=0;CHECK(!live);
  p=query(bytes,sizeof bytes,NVM_V2_OK);release(&p);
 }
 printf("I checked %u allocating sites in both failure modes.\n",count);
#endif
}
int main(void){structure();boundaries();faults();printf("PASS %u private layout structure checks; no array authority\n",checks);return 0;}
