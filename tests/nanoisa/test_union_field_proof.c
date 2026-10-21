/* I qualify descriptive union proof without enrolling public consumers. */
#define main union_declaration_controls_main
#include "test_union_field_declarations.c"
#undef main
#include "../../src/nanoisa/managed_union_field_proof_private.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanoisa/nvm2c.h"
#ifdef UFP_ALLOC
#include "record_array_alloc.h"
#include "../../src/nanoisa/record_union_structure_private.h"
static uint64_t proof_extra_bytes,proof_extra_work;
/* Test-only positive reservation after the real structure constructor. This
 * exercises remaining-analysis cap handling, not early-constructor limits. */
static NvmArrayEligibilityResult proof_reserved_structure(const NvmModule *m,
    NvmRecordArrayBudget *budget,NvmRecordUnionStructure **out) {
    NvmRecordUnionStructure *staged=NULL;
    NvmArrayEligibilityResult r=nvm_record_union_structure_prepare(m,budget,&staged);
    if(r.status!=NVM_ARRAY_ELIGIBLE)return r;
    if(!nvm_ra_bytes(budget,proof_extra_bytes)||!nvm_ra_steps(budget,proof_extra_work)) {
        nvm_record_union_structure_free(staged);r.status=NVM_ARRAY_LIMIT;
        snprintf(r.message,sizeof r.message,"I reached test-only positive remaining-analysis reservation.");return r;
    }
    *out=staged;return r;
}
#define nvm_record_union_structure_prepare proof_reserved_structure
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#include "../../src/nanoisa/managed_array_shapes.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
#undef nvm_record_union_structure_prepare
#endif
static void proof_header(UnionCase *v,uint8_t *code,size_t length) {
    static char *strings[]={"Box","Outer","value","boxed","Some","None"};
    static uint32_t lengths[]={3,5,5,5,4,4};
    NvmModule *m=&v->c.module;
    m->strings=strings;m->string_lengths=lengths;m->code=code;m->code_size=(uint32_t)length;
    m->header.magic[0]=NVM_MAGIC_0;m->header.magic[1]=NVM_MAGIC_1;
    m->header.magic[2]=NVM_MAGIC_2;m->header.magic[3]=NVM_MAGIC_3;
    m->header.format_version=NVM_FORMAT_VERSION;m->header.flags=NVM_FLAG_HAS_MAIN;
    m->header.entry_point=0;v->function.code_length=(uint32_t)length;v->function.name_idx=0;
}
static void proof_i64(uint8_t *code,size_t *n,int64_t value) {
    u8(code,n,OP_PUSH_I64);
    for(unsigned i=0;i<8;i++)u8(code,n,(uint8_t)((uint64_t)value>>(i*8)));
}
static void proof_pack(uint8_t *code,size_t *n,uint8_t kind,uint16_t variant,uint16_t count) {
    u8(code,n,OP_AGG_PACK);u8(code,n,kind);u32(code,n,0);u16(code,n,variant);u16(code,n,count);
}
static NvmUnionFieldProof *proof_query(UnionCase *v,NvmArrayEligibilityStatus expected) {
    NvmUnionFieldProof *p=(void *)(uintptr_t)1;bool needs=false;
    CHECK(nvm_ownership_contracts_validate(&v->c.module,&needs)==NVM_V2_OK&&needs);
    NvmArrayEligibilityResult r=nvm_union_field_proof_prepare(&v->c.module,&p);
    if(r.status!=expected)fprintf(stderr,"union proof expected=%u actual=%u function=%u pc=%u: %s\n",
        expected,r.status,r.function,r.pc,r.message);
    CHECK(r.status==expected);
    CHECK(expected==NVM_ARRAY_ELIGIBLE?p!=(void *)(uintptr_t)1:p==(void *)(uintptr_t)1);
    return p;
}
static void proof_public_refusal(const NvmModule *m) {
    CHECK(!nvm_verify(m).ok);
    char error[512]={0};char *text=nvm2c_emit(m,error,sizeof error);
    CHECK(!text&&error[0]);
    NvmV2Module wire;memset(&wire,0,sizeof wire);
    /* Generic v2 conversion is descriptive and already accepts this metadata.
     * It is not verification or execution admission. */
    CHECK(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&wire,&copy)==NVM_V2_OK);
    CHECK(copy->layout_size==m->layout_size&&!memcmp(copy->layout_data,m->layout_data,m->layout_size));
    CHECK(copy->ownership_size==m->ownership_size&&!memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
    CHECK(!nvm_verify(copy).ok);memset(error,0,sizeof error);
    text=nvm2c_emit(copy,error,sizeof error);CHECK(!text&&error[0]);
    nvm_module_free(copy);nvm_v2_module_free(&wire);
}
static void proof_positive(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
    proof_pack(code,&n,AGG_RECORD,0,1);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    proof_header(&v,code,n);
    NvmUnionFieldProof *p=proof_query(&v,NVM_ARRAY_ELIGIBLE);
    proof_public_refusal(&v.c.module);
    NvmUnionFieldProofCounts counts;CHECK(nvm_union_field_proof_counts(p,&counts));
    CHECK(counts.origins==2&&counts.functions==1&&counts.globals==0);
    NvmUnionFieldOrigin origin;
    CHECK(nvm_union_field_proof_origin(p,0,&origin)&&origin.kind==NVM_UNION_FIELD_ORIGIN&&origin.variant==0&&origin.layout_index==0);
    CHECK(origin.member.tags==(1u<<TAG_INT)&&!origin.member.origins&&!origin.member.unknown);
    CHECK(nvm_union_field_proof_origin(p,1,&origin)&&origin.kind==NVM_HEAP_ORIGIN_RECORD&&origin.variant==UINT8_MAX&&origin.layout_index==1);
    CHECK(origin.member.tags==(1u<<TAG_UNION)&&origin.member.origins==1);
    NvmRecordValueOrigins result;CHECK(nvm_union_field_proof_result(p,0,&result)&&result.tags==(1u<<TAG_INT));
    NvmUnionFieldOrigin saved=origin;CHECK(!nvm_union_field_proof_origin(p,2,&origin)&&!memcmp(&saved,&origin,sizeof saved));
    memset(&v,0,sizeof v);memset(code,0,sizeof code);
    CHECK(nvm_union_field_proof_origin(p,1,&origin)&&origin.layout_index==1&&origin.member.origins==1);
    nvm_union_field_proof_free(p);
}
static void proof_empty(void) {
    UnionCase v;union_case(&v);uint8_t code[64];size_t n=0;
    proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_AGG_TAG);u8(code,&n,OP_RET);
    proof_header(&v,code,n);NvmUnionFieldProof *p=proof_query(&v,NVM_ARRAY_ELIGIBLE);
    NvmUnionFieldOrigin row;CHECK(nvm_union_field_proof_origin(p,0,&row)&&row.variant==1&&!row.member_count&&!row.member.tags);
    nvm_union_field_proof_free(p);
    n=0;proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    proof_header(&v,code,n);(void)proof_query(&v,NVM_ARRAY_UNRESOLVED);
}
static void proof_desc(uint8_t *bytes,size_t *n,uint8_t tag) {
    u8(bytes,n,tag);u8(bytes,n,0);u16(bytes,n,0);
    u32(bytes,n,tag==TAG_STRUCT?1:tag==TAG_UNION?0:UINT32_MAX);
}
static void proof_functions(UnionCase *v,NvmFunctionEntry *functions,uint32_t count) {
    Case *c=&v->c;size_t suffix=v->local+8;
    uint8_t extension[64];size_t length=c->o-suffix;CHECK(length<=sizeof extension);
    memcpy(extension,c->ownership+suffix,length);c->o=12;
    u32(c->ownership,&c->o,count);
    for(uint32_t i=0;i<count;i++) {
        u16(c->ownership,&c->o,functions[i].local_count);u16(c->ownership,&c->o,functions[i].arity);
        proof_desc(c->ownership,&c->o,functions[i].result_count?functions[i].result_tag:TAG_VOID);
        for(uint32_t j=0;j<functions[i].local_count;j++)proof_desc(c->ownership,&c->o,
            j<functions[i].arity&&c->module.function_param_types&&c->module.function_param_types[i]?
                c->module.function_param_types[i][j]:TAG_VOID);
    }
    CHECK(c->o+length<=sizeof c->ownership);memcpy(c->ownership+c->o,extension,length);c->o+=length;
    c->module.functions=functions;c->module.function_count=count;c->module.ownership_size=(uint32_t)c->o;
}
static void proof_calls(bool tail,bool blocked) {
    UnionCase v;union_case(&v);uint8_t code[256];size_t n=0;
    NvmFunctionEntry functions[3]={{0}};functions[0].result_count=1;functions[0].result_tag=TAG_INT;
    u8(code,&n,OP_CALL);u32(code,&n,1);
    if(blocked){proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);}
    proof_pack(code,&n,AGG_RECORD,0,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);functions[0].code_length=(uint32_t)n;
    functions[1].code_offset=(uint32_t)n;functions[1].name_idx=1;
    if(blocked) {
        /* This callee never returns. The caller's union use/RET must not be
         * interpreted as unreachable evidence for a successful proof. */
        u8(code,&n,OP_JMP);u32(code,&n,0);
    } else {
        functions[1].result_count=1;functions[1].result_tag=TAG_UNION;
        if(tail){u8(code,&n,OP_TAIL_CALL);u32(code,&n,2);}
        else {proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);u8(code,&n,OP_RET);}
    }
    functions[1].code_length=(uint32_t)n-functions[1].code_offset;
    if(tail&&!blocked) {
        functions[2].code_offset=(uint32_t)n;functions[2].name_idx=2;
        functions[2].result_count=1;functions[2].result_tag=TAG_UNION;
        proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);u8(code,&n,OP_RET);
        functions[2].code_length=(uint32_t)n-functions[2].code_offset;
    }
    proof_header(&v,code,n);proof_functions(&v,functions,tail&&!blocked?3:2);
    NvmUnionFieldProof *p=proof_query(&v,blocked?NVM_ARRAY_UNRESOLVED:NVM_ARRAY_ELIGIBLE);
    if(!blocked) {
        NvmRecordValueOrigins result;CHECK(nvm_union_field_proof_result(p,1,&result)&&result.tags==(1u<<TAG_UNION)&&result.origins);
        nvm_union_field_proof_free(p);
    }
}
static void proof_halt(bool value_callee) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    NvmFunctionEntry functions[2]={{0}};
    functions[0].result_count=1;functions[0].result_tag=TAG_INT;
    u8(code,&n,OP_CALL);u32(code,&n,1);
    if(!value_callee){proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);}
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    functions[0].code_length=(uint32_t)n;functions[1].name_idx=1;functions[1].code_offset=(uint32_t)n;
    functions[1].result_count=value_callee?1:0;functions[1].result_tag=value_callee?TAG_UNION:TAG_VOID;
    u8(code,&n,OP_HALT);functions[1].code_length=1;
    proof_header(&v,code,n);proof_functions(&v,functions,2);
    (void)proof_query(&v,NVM_ARRAY_UNRESOLVED);
}
static void proof_return_or_halt(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);size_t branch=n;
    u8(code,&n,OP_JMP_FALSE);u32(code,&n,0);
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    patch(code,branch+1,(uint32_t)(n-branch));u8(code,&n,OP_HALT);
    proof_header(&v,code,n);NvmUnionFieldProof *p=proof_query(&v,NVM_ARRAY_ELIGIBLE);
    nvm_union_field_proof_free(p);
}
static void proof_detached(bool direct) {
    UnionCase v;union_case(&v);uint8_t code[256];size_t n=0;
    NvmFunctionEntry function={0};function.local_count=2;function.result_count=1;function.result_tag=TAG_INT;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
    if(direct){u8(code,&n,OP_DUP);u8(code,&n,OP_STORE_LOCAL);u16(code,&n,1);}
    proof_pack(code,&n,AGG_RECORD,0,1);u8(code,&n,OP_STORE_LOCAL);u16(code,&n,0);
    if(!direct) {
        u8(code,&n,OP_LOAD_LOCAL);u16(code,&n,0);u8(code,&n,OP_AGG_GET);u16(code,&n,0);
        u8(code,&n,OP_STORE_LOCAL);u16(code,&n,1);
    }
    u8(code,&n,OP_LOAD_LOCAL);u16(code,&n,0);proof_pack(code,&n,AGG_VARIANT,1,0);
    u8(code,&n,OP_AGG_SET);u16(code,&n,0);u8(code,&n,OP_POP);
    u8(code,&n,OP_LOAD_LOCAL);u16(code,&n,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    proof_header(&v,code,n);function.code_length=(uint32_t)n;proof_functions(&v,&function,1);
    NvmUnionFieldProof *p=proof_query(&v,direct?NVM_ARRAY_ELIGIBLE:NVM_ARRAY_UNRESOLVED);
    if(!direct)return; /* Historical read aliases are deliberately conservative. */
    NvmUnionFieldOrigin row;
    CHECK(nvm_union_field_proof_origin(p,1,&row)&&row.member.origins==5&&row.member.tags==(1u<<TAG_UNION));
    CHECK(nvm_union_field_proof_origin(p,0,&row)&&row.member.tags==(1u<<TAG_INT));
    nvm_union_field_proof_free(p);
    /* A receiver from the containing field retains the complete Some/None
     * history; a known current spelling cannot justify its payload read. */
    code[n-6]=0; /* Read local0 (record), then its field, then payload. */
    n--;u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    function.code_length=(uint32_t)n;v.c.module.code_size=(uint32_t)n;
    (void)proof_query(&v,NVM_ARRAY_UNRESOLVED);
}
static void proof_origins(unsigned count) {
    UnionCase v;union_case(&v);uint8_t code[2048];size_t n=0;
    for(unsigned i=0;i<count;i++) {
        proof_i64(code,&n,(int64_t)i);proof_pack(code,&n,AGG_VARIANT,0,1);
        if(i+1<count)u8(code,&n,OP_POP);
    }
    proof_pack(code,&n,AGG_RECORD,0,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);proof_header(&v,code,n);
    NvmUnionFieldProof *p=proof_query(&v,count==63?NVM_ARRAY_ELIGIBLE:NVM_ARRAY_LIMIT);
    if(count==63) {
        NvmUnionFieldProofCounts counts;CHECK(nvm_union_field_proof_counts(p,&counts)&&counts.origins==64);
        NvmUnionFieldOrigin row;CHECK(nvm_union_field_proof_origin(p,63,&row)&&row.member.origins==(UINT64_C(1)<<62));
        nvm_union_field_proof_free(p);
    }
}
static void proof_reversed(void) {
    UnionCase v;union_case(&v);uint8_t code[64];size_t n=0;
    v.c.ownership[v.variant+6]=0;v.c.ownership[v.variant+12]=0;v.c.ownership[v.variant+14]=1;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,1,1);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);proof_header(&v,code,n);
    NvmUnionFieldProof *p=proof_query(&v,NVM_ARRAY_ELIGIBLE);NvmUnionFieldOrigin row;
    CHECK(nvm_union_field_proof_origin(p,0,&row)&&row.variant==1&&row.member_count==1);
    nvm_union_field_proof_free(p);
    n=0;proof_pack(code,&n,AGG_VARIANT,0,0);u8(code,&n,OP_AGG_TAG);u8(code,&n,OP_RET);proof_header(&v,code,n);
    p=proof_query(&v,NVM_ARRAY_ELIGIBLE);CHECK(nvm_union_field_proof_origin(p,0,&row)&&row.variant==0&&!row.member_count);
    nvm_union_field_proof_free(p);
}
static void proof_globals(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);proof_pack(code,&n,AGG_RECORD,0,1);
    u8(code,&n,OP_STORE_GLOBAL);u32(code,&n,0);
    u8(code,&n,OP_LOAD_GLOBAL);u32(code,&n,0);u8(code,&n,OP_AGG_GET);u16(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);proof_header(&v,code,n);
    NvmUnionFieldProof *p=proof_query(&v,NVM_ARRAY_ELIGIBLE);NvmRecordValueOrigins global;
    CHECK(nvm_union_field_proof_global(p,0,&global)&&global.origins==2&&
        global.tags==((1u<<TAG_VOID)|(1u<<TAG_STRUCT))&&!global.unknown);
    nvm_union_field_proof_free(p);
}
#ifdef UFP_ALLOC
static void proof_allocations(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);proof_pack(code,&n,AGG_RECORD,0,1);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    proof_header(&v,code,n);UnionCase saved=v;uint8_t saved_code[128];memcpy(saved_code,code,n);
    NvmUnionFieldDeclarations *declarations=NULL;
    CHECK(nvm_union_field_declarations_prepare(&v.c.module,&declarations).status==NVM_UFD_PREPARED);
    size_t declaration_bytes=ra_bytes;nvm_union_field_declarations_free(declarations);CHECK(!ra_live&&!ra_bytes);
    ra_calls=ra_peak=0;NvmUnionFieldProof *p=NULL;
    CHECK(nvm_union_field_proof_prepare(&v.c.module,&p).status==NVM_ARRAY_ELIGIBLE);
    size_t measured=ra_calls,retained=ra_bytes,peak=ra_peak;NvmUnionFieldProofCounts counts;
    CHECK(nvm_union_field_proof_counts(p,&counts));
    size_t exact=sizeof *p+declaration_bytes+counts.origins*sizeof(NvmUnionFieldOrigin)+
        (counts.globals+counts.functions)*sizeof(NvmRecordValueOrigins)+counts.functions*sizeof(uint16_t);
    CHECK(retained==exact&&peak<=counts.bytes_reserved);
    nvm_union_field_proof_free(p);CHECK(!ra_live&&!ra_bytes);
    NvmUnionFieldProof *prior=NULL;
    CHECK(nvm_union_field_proof_prepare(&v.c.module,&prior).status==NVM_ARRAY_ELIGIBLE);
    size_t prior_live=ra_live,prior_bytes=ra_bytes;
    for(unsigned mode=0;mode<2;mode++)for(size_t at=0;at<measured;at++) {
        p=prior;ra_calls=0;ra_fail=at;ra_persistent=(int)mode;
        NvmArrayEligibilityResult result=nvm_union_field_proof_prepare(&v.c.module,&p);
        if(result.status!=NVM_ARRAY_MEMORY)fprintf(stderr,"prefix=%zu mode=%u status=%u: %s\n",at,mode,result.status,result.message);
        CHECK(result.status==NVM_ARRAY_MEMORY&&p==prior&&ra_live==prior_live&&ra_bytes==prior_bytes);
        CHECK(!memcmp(&v,&saved,sizeof v)&&!memcmp(code,saved_code,n));
        ra_fail=SIZE_MAX;ra_persistent=0;NvmUnionFieldProofCounts before;
        CHECK(nvm_union_field_proof_counts(prior,&before)&&before.origins==counts.origins&&before.globals==counts.globals&&
            before.functions==counts.functions&&before.bytes_reserved==counts.bytes_reserved&&before.work_reserved==counts.work_reserved);
        p=NULL;CHECK(nvm_union_field_proof_prepare(&v.c.module,&p).status==NVM_ARRAY_ELIGIBLE);
        nvm_union_field_proof_free(p);CHECK(ra_live==prior_live&&ra_bytes==prior_bytes);
    }
    for(unsigned resource=0;resource<2;resource++) {
        proof_extra_bytes=resource?0:NVM_RA_BYTES-counts.bytes_reserved;
        proof_extra_work=resource?NVM_RA_STEPS-counts.work_reserved:0;
        p=NULL;CHECK(nvm_union_field_proof_prepare(&v.c.module,&p).status==NVM_ARRAY_ELIGIBLE);
        NvmUnionFieldProofCounts exact_counts;CHECK(nvm_union_field_proof_counts(p,&exact_counts));
        CHECK(resource?exact_counts.work_reserved==NVM_RA_STEPS:exact_counts.bytes_reserved==NVM_RA_BYTES);
        nvm_union_field_proof_free(p);
        if(resource)proof_extra_work++;else proof_extra_bytes++;
        p=prior;CHECK(nvm_union_field_proof_prepare(&v.c.module,&p).status==NVM_ARRAY_LIMIT&&p==prior);
        CHECK(ra_live==prior_live&&ra_bytes==prior_bytes&&!memcmp(&v,&saved,sizeof v)&&!memcmp(code,saved_code,n));
        proof_extra_bytes=proof_extra_work=0;p=NULL;
        CHECK(nvm_union_field_proof_prepare(&v.c.module,&p).status==NVM_ARRAY_ELIGIBLE);
        nvm_union_field_proof_free(p);CHECK(ra_live==prior_live&&ra_bytes==prior_bytes);
    }
    nvm_union_field_proof_free(prior);CHECK(!ra_live&&!ra_bytes);
    printf("sizes analysis=%zu value=%zu wrapper=%zu context=%zu owner=%zu row=%zu retained=%zu peak=%zu reserved=%llu work=%llu prefixes=%zu\n",
        sizeof(Analysis),sizeof(Value),sizeof(NvmRecordUnionStructure),sizeof(NvmUnionAnalysisContext),sizeof(NvmUnionFieldProof),sizeof(NvmUnionFieldOrigin),
        retained,peak,(unsigned long long)counts.bytes_reserved,(unsigned long long)counts.work_reserved,measured);
}
#endif
static void proof_unknown_history(bool global) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    NvmFunctionEntry functions[2]={{0}};uint8_t parameter=TAG_UNION;uint8_t *parameters[2]={NULL,&parameter};
    functions[0].result_count=1;functions[0].result_tag=TAG_INT;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    functions[0].code_length=(uint32_t)n;
    functions[1].name_idx=1;functions[1].code_offset=(uint32_t)n;
    functions[1].arity=1;functions[1].local_count=1;functions[1].result_count=1;functions[1].result_tag=global?TAG_INT:TAG_UNION;
    u8(code,&n,OP_LOAD_LOCAL);u16(code,&n,0);
    if(global) {
        u8(code,&n,OP_STORE_GLOBAL);u32(code,&n,0);
        u8(code,&n,OP_PUSH_VOID);u8(code,&n,OP_STORE_GLOBAL);u32(code,&n,0);
        proof_i64(code,&n,0);
    }
    u8(code,&n,OP_RET);functions[1].code_length=(uint32_t)n-functions[1].code_offset;
    proof_header(&v,code,n);v.c.module.function_param_types=parameters;proof_functions(&v,functions,2);
    (void)proof_query(&v,NVM_ARRAY_UNRESOLVED);
}
static void proof_blocked_void(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;NvmFunctionEntry functions[2]={{0}};
    functions[0].result_tag=TAG_VOID;functions[1].result_tag=TAG_VOID;functions[1].name_idx=1;
    u8(code,&n,OP_CALL);u32(code,&n,1);proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
    u8(code,&n,OP_POP);u8(code,&n,OP_RET);functions[0].code_length=(uint32_t)n;
    functions[1].code_offset=(uint32_t)n;u8(code,&n,OP_HALT);functions[1].code_length=1;
    proof_header(&v,code,n);proof_functions(&v,functions,2);
    (void)proof_query(&v,NVM_ARRAY_UNRESOLVED);
    NvmUnionFieldProof *p=(void *)(uintptr_t)1;NvmArrayEligibilityResult r=nvm_union_field_proof_prepare(&v.c.module,&p);
    CHECK(r.status==NVM_ARRAY_UNRESOLVED&&p==(void *)(uintptr_t)1&&strstr(r.message,"reachable direct-call return"));
}
static void proof_wrong_operands(void) {
    for(unsigned mode=0;mode<5;mode++) {
        UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
        if(mode!=1) {
            if(mode==2){u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);}
            else proof_i64(code,&n,42);
        }
        size_t pack=n;proof_pack(code,&n,AGG_VARIANT,mode==0?2:0,mode==1?0:1);
        if(mode==3)patch(code,pack+2,1);
        if(mode==4){proof_i64(code,&n,43);u8(code,&n,OP_AGG_SET);u16(code,&n,0);}
        u8(code,&n,OP_AGG_TAG);u8(code,&n,OP_RET);proof_header(&v,code,n);
        (void)proof_query(&v,mode==3?NVM_ARRAY_INVALID:NVM_ARRAY_UNRESOLVED);
    }
}
static void proof_original(const char *path) {
    FILE *file=fopen(path,"rb");CHECK(file);CHECK(!fseek(file,0,SEEK_END));
    long size=ftell(file);CHECK(size>0&&(unsigned long)size<=UINT32_MAX);rewind(file);
    uint8_t *bytes=malloc((size_t)size);CHECK(bytes);
    CHECK(fread(bytes,1,(size_t)size,file)==(size_t)size);CHECK(!fclose(file));
    NvmV2Module original_wire;memset(&original_wire,0,sizeof original_wire);
    CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&original_wire)==NVM_V2_OK);
    NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&original_wire,&m)==NVM_V2_OK);
    CHECK(m&&nvm_verify(m).ok);
    CHECK(!m->layout_size&&!m->ownership_size&&m->struct_count==1&&m->union_count==1&&m->string_count>=6);
    UnionCase v;union_case(&v);Case *c=&v.c;size_t suffix=v.local+8;
    uint8_t extension[64];size_t extension_size=c->o-suffix;CHECK(extension_size<=sizeof extension);
    memcpy(extension,c->ownership+suffix,extension_size);c->o=12;
    u32(c->ownership,&c->o,m->function_count);
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *f=&m->functions[i];CHECK(f->local_count<64);
        u16(c->ownership,&c->o,f->local_count);u16(c->ownership,&c->o,f->arity);
        proof_desc(c->ownership,&c->o,f->result_count?f->result_tag:TAG_VOID);
        for(uint32_t j=0;j<f->local_count;j++)proof_desc(c->ownership,&c->o,
            j<f->arity&&m->function_param_types&&m->function_param_types[i]?m->function_param_types[i][j]:TAG_VOID);
    }
    CHECK(c->o+extension_size<=sizeof c->ownership);memcpy(c->ownership+c->o,extension,extension_size);c->o+=extension_size;
    m->layout_data=c->layouts;m->layout_size=(uint32_t)c->l;
    m->ownership_data=c->ownership;m->ownership_size=(uint32_t)c->o;
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);
    NvmUnionFieldProof *proof=NULL;NvmArrayEligibilityResult result=nvm_union_field_proof_prepare(m,&proof);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"original %s status=%u fn=%u pc=%u: %s\n",path,result.status,result.function,result.pc,result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);proof_public_refusal(m);
    NvmUnionFieldProofCounts counts;CHECK(nvm_union_field_proof_counts(proof,&counts));
    printf("original %s origins=%u functions=%u reserved=%llu work=%llu\n",path,counts.origins,counts.functions,
        (unsigned long long)counts.bytes_reserved,(unsigned long long)counts.work_reserved);
    m->layout_data=NULL;m->layout_size=0;m->ownership_data=NULL;m->ownership_size=0;
    nvm_module_free(m);nvm_v2_module_free(&original_wire);free(bytes);memset(&v,0,sizeof v);
    NvmUnionFieldOrigin row;CHECK(nvm_union_field_proof_origin(proof,0,&row));
    nvm_union_field_proof_free(proof);
}
int main(int argc,char **argv) {
    CHECK(argc==1||argc==7);
#ifdef UFP_ALLOC
    if(argc==1){proof_allocations();printf("PASS %u union proof allocation checks\n",checks);return 0;}
#endif
    CHECK(union_declaration_controls_main()==0);
    proof_positive();proof_empty();proof_calls(false,false);proof_calls(true,false);proof_calls(false,true);proof_halt(false);proof_halt(true);proof_return_or_halt();proof_detached(false);proof_detached(true);proof_origins(63);proof_origins(64);proof_reversed();proof_globals();proof_unknown_history(false);proof_unknown_history(true);proof_blocked_void();proof_wrong_operands();nvm_union_field_proof_free(NULL);
    for(int i=1;i<argc;i++)proof_original(argv[i]);
    printf("PASS %u union proof checks; public execution remains refused\n",checks);
    return 0;
}
