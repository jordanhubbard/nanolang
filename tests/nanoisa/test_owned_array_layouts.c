/* I describe pending modules; I never execute them. */
#include "mixed_layout_view.h"
#include "owned_array_layouts.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"
#include "isa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
static long budget = -1;
#define CHECK(x) do { checks++; assert(x); } while (0)
void *owned_array_test_malloc(size_t n) {
    if (budget == 0) return NULL;
    if (budget > 0) budget--;
    return malloc(n);
}
void *owned_array_test_calloc(size_t n, size_t s) {
    if (budget == 0) return NULL;
    if (budget > 0) budget--;
    return calloc(n, s);
}
static void word(uint8_t *p, uint32_t x) {
    for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(x >> (8 * i));
}
static void retain(NvmModule *m, NvmV2Layout *rows, uint32_t n) {
    NvmV2Layouts l = {rows, n};
    CHECK(nvm_retain_layouts(m, &l) == NVM_V2_OK);
}
static void ownership(NvmModule *m, const uint8_t *flags, uint32_t n, unsigned version) {
    uint32_t at = (8 + n + 3) & ~3u;
    free(m->ownership_data);
    m->ownership_size = at + 4 + (m->function_count ? 20 : 0) + (version == 2 ? 12 : 0);
    m->ownership_data = calloc(m->ownership_size, 1); CHECK(m->ownership_data);
    uint8_t *p = m->ownership_data;
    word(p, version); word(p + 4, n); memcpy(p + 8, flags, n);
    word(p + at, m->function_count); at += 4;
    if (m->function_count) {
        p[at] = 1; p[at + 2] = 1; /* One local, one parameter. */
        p[at + 4] = TAG_VOID; word(p + at + 8, NVM_V2_NO_INDEX);
        p[at + 12] = TAG_STRUCT; word(p + at + 16, 0); at += 20;
    }
    if (version == 2) {
        word(p + at, 1); p[at + 4] = 1; /* One path, one numeric field. */
        p[at + 8] = 7;
    }
}
static NvmOwnedArrayLayouts *describe(NvmModule *m) {
    NvmOwnedArrayLayouts *v=NULL;
    NvmRecordPlanResult r=nvm_describe_owned_array_layouts(m,&v);
    CHECK(r.status==NVM_RECORD_DESCRIBED && v);return v;
}
static void refuse(NvmModule *m) {
    unsigned char sentinel;
    NvmOwnedArrayLayouts *v=(NvmOwnedArrayLayouts *)&sentinel;
    CHECK(nvm_describe_owned_array_layouts(m,&v).status!=NVM_RECORD_DESCRIBED);
    CHECK(v==(NvmOwnedArrayLayouts *)&sentinel);
}
static void clean(NvmModule *m) {free(m->layout_data);free(m->ownership_data);}
static void facts_and_faults(void) {
    NvmModule m={0};m.struct_count=6;m.enum_count=1;
    NvmV2LayoutField integer={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField array={TAG_ARRAY,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField bundle[]={{TAG_STRUCT,2,NVM_V2_NO_INDEX},
        {TAG_ARRAY,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX},{TAG_STRING,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX}};
    NvmV2LayoutField outer[]={{TAG_STRUCT,3,NVM_V2_NO_INDEX},{TAG_STRUCT,3,NVM_V2_NO_INDEX}};
    NvmV2Layout rows[]={
        {NVM_V2_LAYOUT_ENUM,0,NVM_V2_NO_INDEX,NULL},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&integer},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&integer},
        {NVM_V2_LAYOUT_STRUCT,3,NVM_V2_NO_INDEX,bundle},
        {NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,outer},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&array},
        {NVM_V2_LAYOUT_STRUCT,0,NVM_V2_NO_INDEX,NULL}};
    uint8_t flags[]={0,1,3,3,3,1,3};retain(&m,rows,7);ownership(&m,flags,7,2);
    NvmOwnedArrayLayouts *v=describe(&m);NvmOwnedArrayLayoutCounts counts;
    CHECK(nvm_owned_array_layout_counts(v,&counts));
    CHECK(counts.layouts==7 && counts.source_records==6 && counts.managed_records==1 && counts.leaf_paths==10);
    for(uint32_t i=0;i<7;i++) {
        NvmOwnedArrayLayoutFact f;CHECK(nvm_owned_array_layout_fact(v,i,&f));
        CHECK(f.global_layout==i && f.flags==flags[i]);
        CHECK(f.source_record==(i?i-1:NVM_V2_NO_INDEX));
        CHECK(f.managed_record==(i==1?0:NVM_V2_NO_INDEX));
    }
    uint32_t global=99;CHECK(nvm_owned_array_source_layout(v,3,&global) && global==4);
    CHECK(nvm_owned_array_managed_layout(v,0,&global) && global==1);
    NvmOwnedArrayLayoutFact f;CHECK(nvm_owned_array_layout_fact(v,4,&f));
    CHECK(f.path_start==4 && f.path_count==6 && f.owner_depth==2);
    CHECK(f.has_array && f.has_string && f.whole_root_borrow_unsuitable);
    CHECK(nvm_owned_array_layout_fact(v,6,&f) && !f.path_count && !f.has_array);
    NvmOwnedArrayLeafPath path;CHECK(nvm_owned_array_leaf_path(v,4,&path));
    CHECK(path.root_layout==4 && path.terminal_layout==2 && path.length==3 && path.tag==TAG_INT);
    CHECK(path.fields[0]==0 && path.fields[1]==0 && path.fields[2]==0);
    CHECK(nvm_owned_array_leaf_path(v,8,&path));
    CHECK(path.root_layout==4 && path.terminal_layout==3 && path.length==2 && path.tag==TAG_ARRAY);
    CHECK(path.fields[0]==1 && path.fields[1]==1);
    NvmV2LayoutField field;CHECK(nvm_owned_array_layout_field(v,3,1,&field));
    CHECK(field.type_tag==TAG_ARRAY && field.nested_idx==NVM_V2_NO_INDEX);
    NvmOwnedArrayTransport transport;CHECK(nvm_owned_array_transport(v,&transport));
    CHECK(transport.layout_size==m.layout_size && transport.ownership_size==m.ownership_size);
    CHECK(!memcmp(transport.layouts,m.layout_data,m.layout_size));
    CHECK(!memcmp(transport.ownership,m.ownership_data,m.ownership_size));
    NvmOwnedArrayLayoutFact old=f;CHECK(!nvm_owned_array_layout_fact(v,7,&f) && !memcmp(&f,&old,sizeof f));
    NvmOwnedArrayLeafPath old_path=path;CHECK(!nvm_owned_array_leaf_path(v,10,&path) && !memcmp(&path,&old_path,sizeof path));
    global=99;CHECK(!nvm_owned_array_source_layout(v,6,&global) && global==99);
    CHECK(!nvm_owned_array_managed_layout(v,1,&global) && global==99);
    NvmV2LayoutField old_field=field;CHECK(!nvm_owned_array_layout_field(v,3,3,&field) && !memcmp(&field,&old_field,sizeof field));
    NvmOwnedArrayTransport old_transport=transport;CHECK(!nvm_owned_array_transport(NULL,&transport) && !memcmp(&transport,&old_transport,sizeof transport));
    NvmOwnedArrayLayoutCounts old_counts=counts;CHECK(!nvm_owned_array_layout_counts(NULL,&counts) && !memcmp(&counts,&old_counts,sizeof counts));
    bool needs=false;CHECK(nvm_ownership_contracts_validate(&m,&needs)!=NVM_V2_OK);
    NvmMixedLayoutView *mixed=NULL;CHECK(nvm_describe_mixed_layouts(&m,&mixed).status!=NVM_RECORD_DESCRIBED && !mixed);
    unsigned failures=0,success=0;
    for(long n=0;n<40;n++) {
        unsigned char sentinel;NvmOwnedArrayLayouts *next=(NvmOwnedArrayLayouts *)&sentinel;
        budget=n;NvmRecordPlanResult r=nvm_describe_owned_array_layouts(&m,&next);budget=-1;
        if(r.status==NVM_RECORD_MEMORY){CHECK(next==(NvmOwnedArrayLayouts *)&sentinel);failures++;}
        else {CHECK(r.status==NVM_RECORD_DESCRIBED);nvm_owned_array_layouts_free(next);success++;}
        CHECK(!memcmp(transport.layouts,m.layout_data,m.layout_size));
        CHECK(!memcmp(transport.ownership,m.ownership_data,m.ownership_size));
    }
    CHECK(failures>=8 && success);
    /* I exercise refusal only; these descriptor containers are never executed. */
    m.ownership_data[10]=0;refuse(&m);m.ownership_data[10]=3;
    m.ownership_data[10]=4;refuse(&m);m.ownership_data[10]=3;
    uint8_t service=0;m.service_data=&service;m.service_size=0;refuse(&m);m.service_data=NULL;
    m.service_size=1;refuse(&m);m.service_size=0;
    NvmImportEntry import={0};import.kind=NVM_IMPORT_SERVICE;m.imports=&import;m.import_count=1;
    refuse(&m);m.imports=NULL;m.import_count=0;
    clean(&m);CHECK(nvm_owned_array_leaf_path(v,8,&path) && path.tag==TAG_ARRAY);
    CHECK(transport.layouts && transport.ownership);nvm_owned_array_layouts_free(v);
}
static void bounds(void) {
    NvmModule m={0};NvmV2Layout rows[34];NvmV2LayoutField fields[34][2];uint8_t flags[34];
    for(unsigned i=0;i<34;i++) {
        fields[i][0]=(NvmV2LayoutField){i?TAG_STRUCT:TAG_ARRAY,i?i-1:NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
        fields[i][1]=fields[i][0];rows[i]=(NvmV2Layout){NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,fields[i]};flags[i]=3;
    }
    m.struct_count=32;retain(&m,rows,32);ownership(&m,flags,32,1);
    NvmOwnedArrayLayouts *v=describe(&m);NvmOwnedArrayLayoutCounts c;CHECK(nvm_owned_array_layout_counts(v,&c) && c.leaf_paths==32);
    NvmOwnedArrayLeafPath p;CHECK(nvm_owned_array_leaf_path(v,31,&p) && p.length==32);nvm_owned_array_layouts_free(v);
    m.struct_count=33;retain(&m,rows,33);ownership(&m,flags,33,1);refuse(&m);
    rows[0].field_count=0;v=0;retain(&m,rows,33);v=describe(&m);
    CHECK(nvm_owned_array_layout_counts(v,&c) && c.leaf_paths==0);nvm_owned_array_layouts_free(v);
    m.struct_count=34;retain(&m,rows,34);ownership(&m,flags,34,1);refuse(&m);
    rows[0].field_count=1;
    for(unsigned i=1;i<17;i++)rows[i].field_count=2;
    m.struct_count=16;retain(&m,rows,16);ownership(&m,flags,16,1);v=describe(&m);
    CHECK(nvm_owned_array_layout_counts(v,&c) && c.leaf_paths==65535);nvm_owned_array_layouts_free(v);
    m.struct_count=17;retain(&m,rows,17);ownership(&m,flags,17,1);refuse(&m);clean(&m);
}
static void borrowed_and_categories(void) {
    NvmModule m={0};m.struct_count=1;m.function_count=1;
    NvmFunctionEntry function={0};function.arity=function.local_count=1;m.functions=&function;
    uint8_t tag=TAG_STRUCT,*parameters[]={&tag};m.function_param_types=parameters;
    NvmV2LayoutField fields[]={{TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX},{TAG_ARRAY,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX}};
    NvmV2Layout row={NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,fields};uint8_t flags[]={3};
    retain(&m,&row,1);ownership(&m,flags,1,1);m.ownership_data[29]=1;refuse(&m);
    m.ownership_data[29]=2;refuse(&m);m.ownership_data[29]=0;
    NvmOwnedArrayLayouts *v=describe(&m);nvm_owned_array_layouts_free(v);
    fields[1].type_tag=TAG_STRING;retain(&m,&row,1);m.ownership_data[29]=1;refuse(&m);
    fields[1].type_tag=TAG_INT;retain(&m,&row,1);v=describe(&m);nvm_owned_array_layouts_free(v);
    m.ownership_data[29]=0;fields[1].type_tag=TAG_FLOAT;retain(&m,&row,1);refuse(&m);
    fields[1].type_tag=TAG_HASHMAP;retain(&m,&row,1);refuse(&m);clean(&m);
    memset(&m,0,sizeof m);refuse(&m);refuse(NULL);nvm_owned_array_layouts_free(NULL);
}
int main(void) {
    facts_and_faults();bounds();borrowed_and_categories();
    printf("%u owned ARRAY descriptor checks passed; no module execution\n",checks);return 0;
}
