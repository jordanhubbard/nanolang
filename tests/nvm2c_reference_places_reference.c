/* I call the actual public place APIs on retained producer descriptors. */
#include "reference_places.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct { const uint8_t *bytes; size_t size, pos; bool ok; } Reader;
static uint32_t word(Reader *r) {
    if (!r->ok || r->pos>r->size || r->size-r->pos<4) { r->ok=false;return 0; }
    const uint8_t *p=r->bytes+r->pos;r->pos+=4;
    return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
}
typedef struct { NvmReferencePlace value; uint16_t *fields; bool present; } Place;
static Place place(Reader *r) {
    Place p={0};p.present=word(r)!=0;
    uint64_t low=word(r),high=word(r);p.value.invocation=low|(high<<32);
    uint32_t local=word(r);p.value.local=(uint16_t)local;
    p.value.root_layout=word(r);bool present=word(r)!=0;uint32_t count=word(r);
    p.value.referent_layout=word(r);p.value.mode=(NvmReferenceMode)word(r);
    if (local>UINT16_MAX || count>UINT16_MAX || count>(r->size-r->pos)/4) {r->ok=false;return p;}
    p.value.field_count=(uint16_t)count;p.fields=calloc(count?count:1,sizeof *p.fields);
    if (!p.fields)exit(2);
    for (uint32_t i=0;i<count;i++) {uint32_t f=word(r);if(f>UINT16_MAX)r->ok=false;p.fields[i]=(uint16_t)f;}
    p.value.fields=present?p.fields:NULL;return p;
}
static void describe(const uint8_t *bytes,size_t size) {
    Reader r={bytes,size,0,true};bool table_present=word(&r)!=0,items_present=word(&r)!=0;
    uint32_t count=word(&r),authority_a=word(&r),authority_b=word(&r);
    if (!r.ok || count>65536 || count>(size-r.pos)/12) {puts("D");return;}
    NvmV2Layout *rows=calloc(count?count:1,sizeof *rows);
    NvmV2LayoutField **fields=calloc(count?count:1,sizeof *fields);
    if (!rows || !fields)exit(2);
    for (uint32_t i=0;r.ok && i<count;i++) {
        uint32_t kind=word(&r);bool present=word(&r)!=0;uint32_t n=word(&r);
        if (!r.ok || kind>UINT8_MAX || n>UINT16_MAX || n>(size-r.pos)/8) {r.ok=false;break;}
        rows[i].kind=(uint8_t)kind;rows[i].field_count=(uint16_t)n;
        fields[i]=calloc(n?n:1,sizeof **fields);if(!fields[i])exit(2);
        for (uint32_t f=0;f<n;f++) {
            uint32_t tag=word(&r);if(tag>UINT8_MAX)r.ok=false;
            fields[i][f].type_tag=(uint8_t)tag;fields[i][f].nested_idx=word(&r);
        }
        rows[i].fields=present?fields[i]:NULL;
    }
    Place a=place(&r),b=place(&r);
    if (!r.ok || r.pos!=size)puts("D");
    else {
        NvmV2Layouts layouts={items_present?rows:NULL,count};
        NvmV2Layouts *table=table_present?&layouts:NULL;
        NvmReferencePlace *left=a.present?&a.value:NULL,*right=b.present?&b.value:NULL;
        printf("%d%d%d%d%d%d%d%d%d%d\n",
            nvm_reference_place_valid(table,authority_a,left),nvm_reference_place_valid(table,authority_b,right),
            nvm_reference_places_overlap(left,right),nvm_reference_places_overlap(right,left),
            nvm_reference_holds_conflict(left,right),nvm_reference_holds_conflict(right,left),
            nvm_reference_owner_access_conflicts(left,right,false),nvm_reference_owner_access_conflicts(left,right,true),
            nvm_reference_owner_access_conflicts(right,left,false),nvm_reference_owner_access_conflicts(right,left,true));
    }
    free(a.fields);free(b.fields);
    for (uint32_t i=0;i<count;i++)free(fields[i]);
    free(fields);free(rows);
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *f=fopen(argv[i],"rb");if(!f || fseek(f,0,SEEK_END))return 2;
        long size=ftell(f);if(size<0 || fseek(f,0,SEEK_SET))return 2;
        uint8_t *bytes=malloc(size?(size_t)size:1);if(!bytes)return 2;
        if(fread(bytes,1,(size_t)size,f)!=(size_t)size || fclose(f))return 2;
        describe(bytes,(size_t)size);free(bytes);
    }
    return 0;
}
