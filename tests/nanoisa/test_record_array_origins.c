/* I construct query inputs only. No VM or generated service executes. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../../src/nanoisa/managed_record_array_origins.h"
#include "../../src/nanoisa/record_array_structure_private.h"
#include "../../src/nanoisa/isa.h"
#include "../../src/nanoisa/ownership_contracts.h"
#include "record_array_alloc.h"
#ifdef RA_WHITEBOX
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#include "../../src/nanoisa/managed_array_shapes.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
#endif
static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"I failed line %u: %s\n",__LINE__,#x);exit(1);}} while(0)
#define NO UINT32_MAX
#define MASK(t) ((uint16_t)(1u<<(t)))
typedef struct {
    NvmModule m;NvmFunctionEntry fn[3];char *strings[4];uint32_t lengths[4];
    uint8_t code[32768],layouts[2048],ownership[4096];size_t n,l,o;
    unsigned element;bool interleaved;uint8_t params[2],*param_rows[3];
} Input;
static void b8(uint8_t *p,size_t *n,uint8_t v){p[(*n)++]=v;}
static void b16(uint8_t *p,size_t *n,uint16_t v){b8(p,n,(uint8_t)v);b8(p,n,(uint8_t)(v>>8));}
static void b32(uint8_t *p,size_t *n,uint32_t v){for(unsigned i=0;i<4;i++)b8(p,n,(uint8_t)(v>>(8*i)));}
static void op(Input *c,uint8_t v){CHECK(c->n+16<sizeof c->code);b8(c->code,&c->n,v);}
static void arg16(Input *c,uint8_t v,uint16_t a){op(c,v);b16(c->code,&c->n,a);}
static void arg32(Input *c,uint8_t v,uint32_t a){op(c,v);b32(c->code,&c->n,a);}
static void scalar(Input *c,uint8_t tag){
    if(tag==TAG_INT||tag==TAG_FLOAT){op(c,tag==TAG_INT?OP_PUSH_I64:OP_PUSH_F64);for(unsigned i=0;i<8;i++)b8(c->code,&c->n,0);}
    else if(tag==TAG_U8||tag==TAG_BOOL){op(c,tag==TAG_U8?OP_PUSH_U8:OP_PUSH_BOOL);b8(c->code,&c->n,1);}
    else if(tag==TAG_STRING)arg32(c,OP_PUSH_STR,1);
    else op(c,OP_PUSH_VOID);
}
static void array(Input *c,uint8_t tag){op(c,OP_ARR_NEW);b8(c->code,&c->n,tag);}
static void record(Input *c,unsigned ordinal){op(c,OP_STRUCT_LITERAL);b32(c->code,&c->n,ordinal);b16(c->code,&c->n,1);}
static void desc(Input *c,uint8_t tag){b8(c->ownership,&c->o,tag);b8(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,NO);}
static void init(Input *c,uint8_t element,bool interleaved){
    memset(c,0,sizeof *c);c->element=element;c->interleaved=interleaved;
    c->strings[0]="main";c->strings[1]="leaf";c->strings[2]="helper";c->strings[3]="__init__";
    for(unsigned i=0;i<4;i++)c->lengths[i]=(uint32_t)strlen(c->strings[i]);
    memcpy(c->m.header.magic,"NVM\1",4);c->m.header.format_version=NVM_FORMAT_VERSION;c->m.header.flags=NVM_FLAG_HAS_MAIN;
    c->m.functions=c->fn;c->m.function_count=1;c->fn[0].result_tag=TAG_INT;c->fn[0].result_count=1;c->fn[0].local_count=3;
    c->m.strings=c->strings;c->m.string_lengths=c->lengths;c->m.string_count=4;c->m.source_file_idx=NO;
    c->m.code=c->code;c->m.layout_data=c->layouts;c->m.ownership_data=c->ownership;c->m.struct_count=2;c->m.union_count=interleaved?1:0;
}
static void authority(Input *c){
    c->l=c->o=0;unsigned count=2+c->interleaved;b32(c->layouts,&c->l,count);
    for(unsigned i=0;i<count;i++){
        bool un=c->interleaved&&i==0;b8(c->layouts,&c->l,un?NVM_V2_LAYOUT_UNION:NVM_V2_LAYOUT_STRUCT);b8(c->layouts,&c->l,0);b16(c->layouts,&c->l,1);b32(c->layouts,&c->l,NO);
        b8(c->layouts,&c->l,un?TAG_INT:TAG_ARRAY);b8(c->layouts,&c->l,0);b16(c->layouts,&c->l,0);b32(c->layouts,&c->l,NO);b32(c->layouts,&c->l,NO);
    }
    b32(c->ownership,&c->o,3);b32(c->ownership,&c->o,count);
    for(unsigned i=0;i<count;i++)b8(c->ownership,&c->o,c->interleaved&&i==0?0:NVM_LAYOUT_COMPLETE);
    while(c->o%4)b8(c->ownership,&c->o,0);
    b32(c->ownership,&c->o,c->m.function_count);
    for(unsigned i=0;i<c->m.function_count;i++){
        b16(c->ownership,&c->o,c->fn[i].local_count);b16(c->ownership,&c->o,c->fn[i].arity);desc(c,c->fn[i].result_tag);
        for(unsigned j=0;j<c->fn[i].local_count;j++)desc(c,j<c->fn[i].arity?c->param_rows[i][j]:TAG_VOID);
    }
    b32(c->ownership,&c->o,4);b32(c->ownership,&c->o,0);b32(c->ownership,&c->o,1+c->interleaved);
    if(c->interleaved){
        b16(c->ownership,&c->o,1);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,20);b32(c->ownership,&c->o,1);
        b32(c->ownership,&c->o,0);b16(c->ownership,&c->o,1);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,1);b16(c->ownership,&c->o,0);b16(c->ownership,&c->o,1);
    }
    b16(c->ownership,&c->o,2);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,40);b32(c->ownership,&c->o,1);
    b8(c->ownership,&c->o,(uint8_t)c->element);b8(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,NO);b32(c->ownership,&c->o,2);
    for(unsigned i=0;i<2;i++){b32(c->ownership,&c->o,i+c->interleaved);b16(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);}
    c->m.layout_size=(uint32_t)c->l;c->m.ownership_size=(uint32_t)c->o;c->m.code_size=(uint32_t)c->n;
}
static void finish(Input *c){scalar(c,TAG_INT);op(c,OP_RET);c->fn[0].code_length=(uint32_t)c->n;authority(c);}
static NvmRecordArrayOrigins *query(Input *c,NvmArrayEligibilityStatus expected){
    NvmRecordArrayOrigins *p=(void *)(uintptr_t)1;NvmArrayEligibilityResult r=nvm_analyze_record_array_origins(&c->m,&p);
    if(r.status!=expected)fprintf(stderr,"I expected %u, got %u: %s\n",expected,r.status,r.message);
    CHECK(r.status==expected);CHECK(expected==NVM_ARRAY_ELIGIBLE?p!=(void *)(uintptr_t)1:p==(void *)(uintptr_t)1);return p;
}
static void positive(Input *c){NvmRecordArrayOrigins *p=query(c,NVM_ARRAY_ELIGIBLE);nvm_record_array_origins_free(p);}
static void basic(Input *c,uint8_t tag,bool un){init(c,tag,un);array(c,tag);scalar(c,tag);op(c,OP_ARR_PUSH);record(c,0);op(c,OP_POP);finish(c);}
#define UNCHANGED(Type,call) do{Type out,old;memset(&out,0xa5,sizeof out);memcpy(&old,&out,sizeof out);CHECK(!(call));CHECK(!memcmp(&out,&old,sizeof out));}while(0)
static void copied_facts(void){
    for(uint8_t t=TAG_INT;t<=TAG_STRING;t++){
        Input *c=malloc(sizeof *c);CHECK(c);basic(c,t,true);NvmRecordArrayOrigins *p=query(c,NVM_ARRAY_ELIGIBLE);
        memset(c,0,sizeof *c);free(c);NvmRecordArrayOriginCounts n;CHECK(nvm_record_array_origins_counts(p,&n));CHECK(n.origins==2&&n.fields==1&&n.checked_array_writes==1&&n.checked_field_writes==1);
        CHECK(n.peak_bytes_reserved<=NVM_RA_BYTES&&n.work_reserved<=NVM_RA_STEPS);
        NvmRecordHeapOrigin a,r;CHECK(nvm_record_array_origin(p,0,&a)&&a.kind==NVM_HEAP_ORIGIN_ARRAY&&a.declared_tag==t);
        CHECK(nvm_record_array_origin(p,1,&r)&&r.kind==NVM_HEAP_ORIGIN_RECORD&&r.record_ordinal==0&&r.layout_index==1&&r.field_count==1);
        NvmRecordValueOrigins v;CHECK(nvm_record_array_field_value(p,0,&v)&&v.tags==MASK(TAG_ARRAY)&&v.origins==1&&!v.unknown);
        uint16_t mask=0;CHECK(nvm_record_array_required_elements(p,0,&mask)&&mask==MASK(t));
        NvmDeclarationCounts dc;CHECK(nvm_record_array_declaration_counts(p,&dc)&&dc.layouts==3&&dc.bindings==2&&dc.unions==1);
        NvmDeclarationLayout dl;CHECK(nvm_record_array_declaration_layout(p,1,&dl)&&dl.kind==NVM_V2_LAYOUT_STRUCT&&dl.fields==1);
        NvmV2LayoutField df;CHECK(nvm_record_array_declaration_field(p,1,0,&df)&&df.type_tag==TAG_ARRAY);
        NvmOrdinaryArrayType dt;CHECK(nvm_record_array_declaration_type(p,0,&dt)&&dt.tag==t);
        NvmOrdinaryArrayBinding db;CHECK(nvm_record_array_declaration_binding(p,1,&db)&&db.layout==2&&db.element_type==0);
        NvmUnionVariantFact dv;CHECK(nvm_record_array_declaration_variant(p,0,0,&dv)&&dv.layout==0&&dv.field_count==1);
        UNCHANGED(NvmRecordArrayOriginCounts,nvm_record_array_origins_counts(NULL,&out));
        UNCHANGED(NvmRecordHeapOrigin,nvm_record_array_origin(p,n.origins,&out));
        UNCHANGED(NvmRecordValueOrigins,nvm_record_array_field_value(p,n.fields,&out));
        UNCHANGED(uint16_t,nvm_record_array_required_elements(p,n.origins,&out));
        UNCHANGED(NvmDeclarationCounts,nvm_record_array_declaration_counts(NULL,&out));
        UNCHANGED(NvmDeclarationLayout,nvm_record_array_declaration_layout(p,3,&out));
        UNCHANGED(NvmV2LayoutField,nvm_record_array_declaration_field(p,1,1,&out));
        UNCHANGED(NvmOrdinaryArrayType,nvm_record_array_declaration_type(p,1,&out));
        UNCHANGED(NvmOrdinaryArrayBinding,nvm_record_array_declaration_binding(p,2,&out));
        UNCHANGED(NvmUnionVariantFact,nvm_record_array_declaration_variant(p,0,1,&out));
        CHECK(!nvm_record_array_origins_counts(p,NULL));
        CHECK(!nvm_record_array_origin(p,0,NULL));CHECK(!nvm_record_array_field_value(p,0,NULL));CHECK(!nvm_record_array_required_elements(p,0,NULL));
        CHECK(!nvm_record_array_declaration_counts(p,NULL));CHECK(!nvm_record_array_declaration_layout(p,0,NULL));CHECK(!nvm_record_array_declaration_field(p,1,0,NULL));
        CHECK(!nvm_record_array_declaration_type(p,0,NULL));CHECK(!nvm_record_array_declaration_binding(p,0,NULL));CHECK(!nvm_record_array_declaration_variant(p,0,0,NULL));
        nvm_record_array_origins_free(p);
    }
}
static void aliases_and_copies(void){
    for(unsigned after=0;after<2;after++)for(unsigned bad=0;bad<2;bad++){
        Input c;init(&c,TAG_FLOAT,false);array(&c,TAG_FLOAT);arg16(&c,OP_STORE_LOCAL,0);
        if(after){arg16(&c,OP_LOAD_LOCAL,0);record(&c,0);arg16(&c,OP_STORE_LOCAL,1);}
        arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,bad?TAG_INT:TAG_FLOAT);op(&c,OP_ARR_PUSH);op(&c,OP_POP);
        if(!after){arg16(&c,OP_LOAD_LOCAL,0);record(&c,0);op(&c,OP_POP);}finish(&c);
        NvmRecordArrayOrigins *p=query(&c,bad?NVM_ARRAY_UNRESOLVED:NVM_ARRAY_ELIGIBLE);if(!bad)nvm_record_array_origins_free(p);
    }
    Input c;init(&c,TAG_INT,false);array(&c,TAG_INT);arg16(&c,OP_STORE_LOCAL,0);
    arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_INT);scalar(&c,TAG_INT);op(&c,OP_ARR_SLICE);record(&c,0);op(&c,OP_POP);finish(&c);
    NvmRecordArrayOrigins *p=query(&c,NVM_ARRAY_ELIGIBLE);NvmRecordArrayOriginCounts n;CHECK(nvm_record_array_origins_counts(p,&n)&&n.origins==3);
    NvmRecordValueOrigins v;CHECK(nvm_record_array_field_value(p,0,&v)&&v.origins==4);uint16_t mask;
    CHECK(nvm_record_array_required_elements(p,0,&mask)&&mask==0);CHECK(nvm_record_array_required_elements(p,2,&mask)&&mask==MASK(TAG_INT));nvm_record_array_origins_free(p);
}
static void refusals(void){
    Input c;
    const uint8_t tags[]={TAG_ARRAY,TAG_STRUCT,TAG_VOID,TAG_ENUM};
    for(unsigned i=0;i<sizeof tags;i++){init(&c,TAG_INT,false);array(&c,tags[i]);op(&c,OP_POP);finish(&c);query(&c,NVM_ARRAY_UNRESOLVED);}
    init(&c,TAG_INT,false);array(&c,TAG_STRING);array(&c,TAG_INT);op(&c,OP_ARR_PUSH);op(&c,OP_POP);finish(&c);query(&c,NVM_ARRAY_UNRESOLVED);
    init(&c,TAG_INT,false);array(&c,TAG_STRING);op(&c,OP_PUSH_VOID);op(&c,OP_ARR_PUSH);op(&c,OP_POP);finish(&c);query(&c,NVM_ARRAY_UNRESOLVED);
    init(&c,TAG_INT,false);op(&c,OP_NOP);c.fn[0].code_length=(uint32_t)c.n;authority(&c);query(&c,NVM_ARRAY_INVALID);
    init(&c,TAG_INT,false);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;authority(&c);query(&c,NVM_ARRAY_INVALID);
    init(&c,TAG_INT,false);authority(&c);query(&c,NVM_ARRAY_INVALID);
    basic(&c,TAG_INT,false);c.m.service_size=1;query(&c,NVM_ARRAY_UNRESOLVED);
    basic(&c,TAG_INT,false);c.fn[0].local_count=257;query(&c,NVM_ARRAY_LIMIT);
    basic(&c,TAG_INT,false);c.m.layout_size--;query(&c,NVM_ARRAY_INVALID);
    NvmRecordArrayOrigins *p=(void *)(uintptr_t)1;CHECK(nvm_analyze_record_array_origins(NULL,&p).status==NVM_ARRAY_INVALID&&p==(void *)(uintptr_t)1);
    CHECK(nvm_analyze_record_array_origins(&c.m,NULL).status==NVM_ARRAY_INVALID);nvm_record_array_origins_free(NULL);
}
static void field_call_global_join(void){
    Input c;init(&c,TAG_INT,false);array(&c,TAG_INT);record(&c,0);arg16(&c,OP_STORE_LOCAL,0);
    arg16(&c,OP_LOAD_LOCAL,0);array(&c,TAG_INT);arg16(&c,OP_AGG_SET,0);arg32(&c,OP_STORE_GLOBAL,0);
    arg32(&c,OP_LOAD_GLOBAL,0);arg16(&c,OP_AGG_GET,0);op(&c,OP_POP);finish(&c);
    NvmRecordArrayOrigins *p=query(&c,NVM_ARRAY_ELIGIBLE);NvmRecordValueOrigins v;CHECK(nvm_record_array_field_value(p,0,&v)&&v.origins==5);nvm_record_array_origins_free(p);
    init(&c,TAG_INT,false);array(&c,TAG_INT);arg32(&c,OP_CALL,1);record(&c,0);op(&c,OP_POP);finish(&c);
    c.m.function_count=2;c.fn[1].name_idx=2;c.fn[1].code_offset=(uint32_t)c.n;c.fn[1].local_count=c.fn[1].arity=1;c.fn[1].result_count=1;c.fn[1].result_tag=TAG_ARRAY;
    c.params[0]=TAG_ARRAY;c.param_rows[1]=c.params;c.m.function_param_types=c.param_rows;
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;authority(&c);positive(&c);
    /* I join both syntactic branches; the concrete condition is irrelevant. */
    init(&c,TAG_INT,false);scalar(&c,TAG_BOOL);size_t branch=c.n;arg32(&c,OP_JMP_FALSE,0);array(&c,TAG_INT);size_t jump=c.n;arg32(&c,OP_JMP,0);
    size_t other=c.n;array(&c,TAG_INT);size_t join=c.n;record(&c,0);op(&c,OP_POP);finish(&c);
    size_t at=branch+1;b32(c.code,&at,(uint32_t)(other-branch));at=jump+1;b32(c.code,&at,(uint32_t)(join-jump));
    p=query(&c,NVM_ARRAY_ELIGIBLE);CHECK(nvm_record_array_field_value(p,0,&v)&&v.origins==3);nvm_record_array_origins_free(p);
}
static void old_mode(void){
    Input c;basic(&c,TAG_INT,false);NvmRecordEligibilityReport *old=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult before=nvm_analyze_managed_records(&c.m,&old);CHECK(before.status!=NVM_ARRAY_ELIGIBLE&&old==(void *)(uintptr_t)1);
    NvmVerifyResult vb=nvm_verify(&c.m);positive(&c);old=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult after=nvm_analyze_managed_records(&c.m,&old);NvmVerifyResult va=nvm_verify(&c.m);
    CHECK(before.status==after.status&&!strcmp(before.message,after.message)&&old==(void *)(uintptr_t)1);CHECK(vb.ok==va.ok&&!strcmp(vb.error_msg,va.error_msg));
    init(&c,TAG_INT,false);array(&c,TAG_ARRAY);array(&c,TAG_INT);op(&c,OP_ARR_PUSH);op(&c,OP_POP);finish(&c);
    c.m.layout_data=NULL;c.m.layout_size=0;c.m.ownership_data=NULL;c.m.ownership_size=0;c.m.struct_count=0;
    NvmArrayGraphEligibilityReport *graph=NULL;CHECK(nvm_analyze_managed_array_graphs(&c.m,&graph).status==NVM_ARRAY_ELIGIBLE);
    CHECK(graph->arrays.origin_count==2&&graph->child_origins[0]==2);nvm_array_graph_eligibility_free(graph);
}
static void boundaries(void){
    NvmRecordArrayBudget b={0};CHECK(nvm_ra_bytes(&b,NVM_RA_BYTES));CHECK(b.peak==NVM_RA_BYTES);CHECK(!nvm_ra_bytes(&b,1)&&b.bytes==NVM_RA_BYTES&&b.limited);
    memset(&b,0,sizeof b);CHECK(nvm_ra_steps(&b,NVM_RA_STEPS));CHECK(!nvm_ra_steps(&b,1)&&b.work==NVM_RA_STEPS&&b.limited);
    memset(&b,0,sizeof b);CHECK(!nvm_ra_alloc(&b,SIZE_MAX,2)&&b.limited);
    for(unsigned count=64;count<=65;count++){
        Input c;init(&c,TAG_INT,false);for(unsigned i=0;i<count;i++){array(&c,TAG_INT);op(&c,OP_POP);}finish(&c);
        NvmRecordArrayOrigins *p=query(&c,count==64?NVM_ARRAY_ELIGIBLE:NVM_ARRAY_LIMIT);
        if(count==64){NvmRecordArrayOriginCounts n;CHECK(nvm_record_array_origins_counts(p,&n)&&n.origins==64);nvm_record_array_origins_free(p);}
    }
#ifdef RA_WHITEBOX
    /* I drive actual allocation and transfer callers at exact remaining caps. */
    for(unsigned over=0;over<2;over++){
        Analysis a={0};NvmRecordArrayStructure structure={0};a.structure=&structure;
        a.budget.bytes=NVM_RA_BYTES-sizeof(Value)+over;size_t before=ra_calls;
        void *p=analysis_alloc(&a,1,sizeof(Value));
        if(over)CHECK(!p&&a.budget.limited&&ra_calls==before);
        else {CHECK(p&&a.budget.peak==NVM_RA_BYTES);ra_test_free(p);}
        CHECK(!ra_live&&!ra_bytes);
        Input c;init(&c,TAG_INT,false);c.fn[0].result_count=0;c.fn[0].result_tag=TAG_VOID;
        memset(&a,0,sizeof a);a.module=&c.m;a.structure=&structure;a.budget.work=NVM_RA_STEPS-2049+over;
        Value states[1]={{0}};uint16_t depths[1]={0};uint32_t queue[1]={0};uint8_t seen[1]={1},queued[1]={0};
        Function *f=&a.functions[0];f->states=states;f->depths=depths;f->queue=queue;f->seen=seen;f->queued=queued;f->stride=1;
        int ok=walk(&a,0);CHECK(over?(!ok&&a.budget.limited):(ok&&a.budget.work==NVM_RA_STEPS));
    }
    for(unsigned explicit_return=0;explicit_return<2;explicit_return++){
        Input c;init(&c,TAG_INT,false);Analysis a={0};NvmRecordArrayStructure structure={0};a.module=&c.m;a.structure=&structure;
        Function *f=&a.functions[0];Value states[1]={{0}};uint16_t depths[1]={0};uint32_t queue[1]={0};uint8_t seen[1]={1},queued[1]={0};VmDecodedInstruction ins={0};ins.instruction.opcode=OP_RET;
        f->states=states;f->depths=depths;f->queue=queue;f->seen=seen;f->queued=queued;f->stride=1;f->decoded.instructions=&ins;f->decoded.instruction_count=explicit_return;
        /* One queued state suffices: both paths must refuse before result read. */
        if(explicit_return){uint8_t seen2[2]={1,0},queued2[2]={0};uint32_t q2[2]={0};f->seen=seen2;f->queued=queued2;f->queue=q2;CHECK(!walk(&a,0)&&a.result.status==NVM_ARRAY_INVALID);}
        else CHECK(!walk(&a,0)&&a.result.status==NVM_ARRAY_INVALID);
    }
    Input c;basic(&c,TAG_INT,true);CHECK(!ra_live&&!ra_bytes);ra_calls=ra_peak=0;
    NvmRecordArrayOrigins *p=query(&c,NVM_ARRAY_ELIGIBLE);size_t measured=ra_calls,actual_peak=ra_peak;NvmRecordArrayOriginCounts n;CHECK(nvm_record_array_origins_counts(p,&n));CHECK(actual_peak<=n.peak_bytes_reserved);nvm_record_array_origins_free(p);CHECK(!ra_live&&!ra_bytes&&measured>20);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<measured;i++){
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;p=(void *)(uintptr_t)1;NvmArrayEligibilityResult r=nvm_analyze_record_array_origins(&c.m,&p);
        CHECK(r.status!=NVM_ARRAY_ELIGIBLE&&p==(void *)(uintptr_t)1);CHECK(!ra_live&&!ra_bytes);
        ra_fail=SIZE_MAX;ra_persistent=0;positive(&c);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I measured %zu allocation positions and %zu peak payload bytes; every refusal recovered\n",measured,actual_peak);
#endif
}
int main(void){setvbuf(stdout,NULL,_IONBF,0);puts("I begin copied facts");copied_facts();puts("I begin alias/copy facts");aliases_and_copies();puts("I begin refusal boundaries");refusals();puts("I begin call/global/join facts");field_call_global_join();puts("I begin old-mode compatibility");old_mode();puts("I begin bounded accounting and recovery");boundaries();printf("PASS %u record-array origin checks; no runtime admission\n",checks);return 0;}
