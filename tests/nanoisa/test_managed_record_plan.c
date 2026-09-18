/* I test descriptions and ownership refusal without granting runtime admission. */
#include "managed_record_plan.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
#include "isa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(x) do { checks++; assert(x); } while(0)
static long budget=-1;
void *nrp_test_calloc(size_t count,size_t size) {
    if(budget==0)return NULL;
    if(budget>0)budget--;
    return calloc(count,size);
}
static void word(uint8_t *p,uint32_t x) {for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(x>>(8*i));}
static void expect(NvmModule *m,NvmRecordPlanStatus status) {
    NvmRecordPlan sentinel={0},*plan=&sentinel;
    NvmRecordPlanResult r=nvm_describe_managed_records(m,&plan);
    CHECK(r.status==status);
    if(status==NVM_RECORD_DESCRIBED) {
        CHECK(plan!=&sentinel && plan->authority==NVM_RECORD_AUTHORITY_UNKNOWN);
        nvm_record_plan_free(plan);
    } else CHECK(plan==&sentinel);
}
static NvmModule *assemble(const char *text) {
    AsmResult error;NvmModule *m=asm_assemble(text,&error);CHECK(m);return m;
}
static void retain(NvmModule *m,NvmV2Layout *items,uint32_t count) {
    NvmV2Layouts layouts={items,count};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
}
static void empty_and_authority(void) {
    NvmModule *m=assemble(".types 1 0 0\n.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n");
    expect(m,NVM_RECORD_UNRESOLVED); /* Count alone supplies no empty layout. */
    NvmV2Layout empty={NVM_V2_LAYOUT_STRUCT,0,NVM_V2_NO_INDEX,NULL};retain(m,&empty,1);
    expect(m,NVM_RECORD_DESCRIBED);
    char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);
    NvmModule *copy=assemble(text);CHECK(copy->layout_size==m->layout_size);
    CHECK(!memcmp(copy->layout_data,m->layout_data,m->layout_size));expect(copy,NVM_RECORD_DESCRIBED);
    free(text);nvm_module_free(copy);
    m->ownership_data=calloc(28,1);CHECK(m->ownership_data);m->ownership_size=28;
    word(m->ownership_data,1);word(m->ownership_data+4,1);word(m->ownership_data+12,1);
    m->ownership_data[20]=TAG_INT;word(m->ownership_data+24,NVM_V2_NO_INDEX);
    for(unsigned flag=0;flag<=3;flag++) {
        if(flag==2)continue;
        m->ownership_data[8]=(uint8_t)flag;
        bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK);
        CHECK(needs==(flag==3));expect(m,NVM_RECORD_UNRESOLVED);
    }
    free(m->ownership_data);m->ownership_data=NULL;m->ownership_size=0;
    expect(m,NVM_RECORD_DESCRIBED);nvm_module_free(m);
}
static void mapping_and_allocation(const char *output) {
    NvmModule *m=assemble(".types 3 1 1\n.entry main\n.function main 0 0 0 int 1\nPUSH_I64 42\nAGG_PACK 0 1 0 1\nAGG_GET 0\nPUSH_I64 42\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n");
    uint32_t same=nvm_add_string(m,"same",4),other=nvm_add_string(m,"other",5);
    NvmV2LayoutField scalar={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField fields[]={{TAG_STRUCT,0,NVM_V2_NO_INDEX},{TAG_STRUCT,3,NVM_V2_NO_INDEX},
        {TAG_STRING,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX},{TAG_ARRAY,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX},
        {TAG_ENUM,1,NVM_V2_NO_INDEX}};
    NvmV2Layout layouts[]={{NVM_V2_LAYOUT_STRUCT,1,same,&scalar},{NVM_V2_LAYOUT_ENUM,0,NVM_V2_NO_INDEX,NULL},
        {NVM_V2_LAYOUT_TUPLE,0,NVM_V2_NO_INDEX,NULL},{NVM_V2_LAYOUT_STRUCT,1,same,&scalar},
        {NVM_V2_LAYOUT_UNION,0,NVM_V2_NO_INDEX,NULL},{NVM_V2_LAYOUT_STRUCT,5,other,fields}};
    retain(m,layouts,6);CHECK(nvm_verify(m).ok);
    char *before=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(before);
    unsigned failed=0,passed=0;
    for(long n=0;n<12;n++) {
        NvmRecordPlan sentinel={0},*plan=&sentinel;budget=n;
        NvmRecordPlanResult r=nvm_describe_managed_records(m,&plan);budget=-1;
        if(r.status==NVM_RECORD_MEMORY){CHECK(plan==&sentinel);failed++;}
        else {CHECK(r.status==NVM_RECORD_DESCRIBED);CHECK(plan!=&sentinel);nvm_record_plan_free(plan);passed++;}
        char *after=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(after && !strcmp(before,after));free(after);
    }
    CHECK(failed>=7 && passed);
    NvmRecordPlan *plan=NULL;CHECK(nvm_describe_managed_records(m,&plan).status==NVM_RECORD_DESCRIBED);
    CHECK(plan->record_count==3 && plan->layouts.count==6 && plan->authority==NVM_RECORD_AUTHORITY_UNKNOWN);
    CHECK(plan->record_to_layout[0]==0 && plan->record_to_layout[1]==3 && plan->record_to_layout[2]==5);
    CHECK(plan->layout_to_record[1]==NVM_V2_NO_INDEX && plan->layout_to_record[2]==NVM_V2_NO_INDEX && plan->layout_to_record[4]==NVM_V2_NO_INDEX);
    CHECK(plan->layout_to_record[0]==0 && plan->layout_to_record[3]==1 && plan->layout_to_record[5]==2);
    CHECK(plan->layouts.items[0].name_idx==plan->layouts.items[3].name_idx);
    for (unsigned i=0;i<5;i++) {
        NvmV2LayoutField *actual=&plan->layouts.items[5].fields[i];
        CHECK(actual->type_tag==fields[i].type_tag);
        CHECK(actual->nested_idx==fields[i].nested_idx);
        CHECK(actual->name_idx==fields[i].name_idx);
    }
    NvmModule *copy=assemble(before);free(before);
    CHECK(copy->layout_size==m->layout_size && !memcmp(copy->layout_data,m->layout_data,m->layout_size));
    NvmRecordPlan *copy_plan=NULL;CHECK(nvm_describe_managed_records(copy,&copy_plan).status==NVM_RECORD_DESCRIBED);
    CHECK(!memcmp(copy_plan->record_to_layout,plan->record_to_layout,3*sizeof(uint32_t)));
    nvm_record_plan_free(copy_plan);nvm_module_free(copy);
    if(output) {
        NvmV2Module v2;CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);size_t size;
        CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);uint8_t *bytes=malloc(size);CHECK(bytes);
        CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
        FILE *f=fopen(output,"wb");CHECK(f && fwrite(bytes,1,size,f)==size && fclose(f)==0);
        free(bytes);nvm_v2_module_free(&v2);
    }
    nvm_module_free(m); /* No plan field/map aliases the source. */
    CHECK(plan->layouts.items[5].fields[1].nested_idx==3 && plan->record_to_layout[1]==3);
    nvm_record_plan_free(plan);
}
static void boundaries(void) {
    expect(NULL,NVM_RECORD_INVALID);CHECK(nvm_describe_managed_records(NULL,NULL).status==NVM_RECORD_INVALID);
    NvmModule m={0};NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout layouts[]={{NVM_V2_LAYOUT_ENUM,0,NVM_V2_NO_INDEX,NULL},{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field}};
    m.struct_count=m.enum_count=1;retain(&m,layouts,2);expect(&m,NVM_RECORD_DESCRIBED);
    field.type_tag=TAG_STRUCT;field.nested_idx=0;retain(&m,layouts,2);expect(&m,NVM_RECORD_INVALID);
    field.type_tag=TAG_STRING;retain(&m,layouts,2);expect(&m,NVM_RECORD_INVALID);
    field.type_tag=TAG_HASHMAP;field.nested_idx=NVM_V2_NO_INDEX;retain(&m,layouts,2);expect(&m,NVM_RECORD_UNRESOLVED);
    m.struct_count=2;expect(&m,NVM_RECORD_INVALID);m.struct_count=1;
    m.layout_size--;expect(&m,NVM_RECORD_INVALID);m.layout_size++;free(m.layout_data);
    NvmV2Layout *many=calloc(257,sizeof *many);CHECK(many);
    for(unsigned i=0;i<257;i++){many[i].kind=NVM_V2_LAYOUT_STRUCT;many[i].name_idx=NVM_V2_NO_INDEX;}
    memset(&m,0,sizeof m);m.struct_count=257;retain(&m,many,257);expect(&m,NVM_RECORD_LIMIT);free(m.layout_data);free(many);
    NvmV2LayoutField *fields=calloc(65535,sizeof *fields);CHECK(fields);
    for(unsigned i=0;i<65535;i++){fields[i].type_tag=TAG_INT;fields[i].nested_idx=fields[i].name_idx=NVM_V2_NO_INDEX;}
    NvmV2Layout large[]={{NVM_V2_LAYOUT_STRUCT,65535,NVM_V2_NO_INDEX,fields},{NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,fields}};
    memset(&m,0,sizeof m);m.struct_count=2;retain(&m,large,2);expect(&m,NVM_RECORD_LIMIT);
    large[1].field_count=1;retain(&m,large,2);expect(&m,NVM_RECORD_DESCRIBED);free(m.layout_data);free(fields);
}
int main(int argc,char **argv) {
    empty_and_authority();mapping_and_allocation(argc==2?argv[1]:NULL);boundaries();
    printf("%u record plan checks passed\n",checks);return 0;
}
