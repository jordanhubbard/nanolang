/* I reuse unchanged ordinary module builders, not execution paths. */
#define main origin_fixture_main
#include "test_owned_array_origins.c"
#undef main
#include "owned_array_authority.h"
#include "owned_array_admission.h"
#include "verifier.h"
static bool structural_boundary, failed_structural, failed_direct;
void *authority_test_malloc(size_t n) {
    if(budget==0){if(structural_boundary)failed_structural=true;else failed_direct=true;}
    return owner_origin_test_malloc(n);
}
void *authority_test_calloc(size_t n,size_t width) {
    if(budget==0){if(structural_boundary)failed_structural=true;else failed_direct=true;}
    return owner_origin_test_calloc(n,width);
}
bool authority_structural_layout_valid(const NvmModule *m) {
    CHECK(!structural_boundary);structural_boundary=true;
    bool valid=nvm_retained_layouts_valid(m);structural_boundary=false;return valid;
}
static NvmOwnedArrayPlan *authority(NvmModule *m,NvmOwnerAuthorityStatus wanted) {
    unsigned char sentinel;NvmOwnedArrayPlan *p=(void *)&sentinel;
    NvmOwnerAuthorityResult r=nvm_prepare_owned_array_authority(m,&p);
    if(r.status!=wanted)fprintf(stderr,"authority wanted%d got%d f%u pc%u: %s\n",wanted,r.status,r.function,r.pc,r.message);
    CHECK(r.status==wanted);
    if(wanted==NVM_OWNER_AUTH_PREPARED){CHECK(p!=(void *)&sentinel);return p;}
    CHECK(p==(void *)&sentinel);return NULL;
}
static void origin_only_refusal(const char *body,const Type *locals,unsigned count) {
    Function f={body,0,(uint16_t)count,T(TAG_INT),{{0}}};
    for(unsigned n=0;n<count;n++)f.locals[n]=locals[n];
    NvmModule *m=build(&f,1);NvmOwnedArrayOrigins *o=expect(m,NVM_OWNER_ORIGIN_PROVED);nvm_owned_array_origins_free(o);
    authority(m,NVM_OWNER_AUTH_UNRESOLVED);nvm_module_free(m);
}
static NvmModule *authority_fixture(void) {
    Function f[]={
        {"CALL 1\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 1\nARR_LEN\nPOP\n"
         "OWN_MOVE_LOCAL 0\nCALL 3\nPOP\nPUSH_I64 0\nRET\n",0,1,T(TAG_INT),{OWNER(2)}},
        {"PUSH_I64 7\nOWN_PACK 0\nARR_NEW 3\nPUSH_STR value\nOWN_PACK 1\n"
         "PUSH_I64 8\nOWN_PACK 0\nARR_NEW 3\nPUSH_STR value\nOWN_PACK 1\nOWN_PACK 2\nCALL 2\nRET\n",0,0,OWNER(2),{{0}}},
        {"OWN_MOVE_LOCAL 0\nRET\n",1,1,OWNER(2),{OWNER(2)}},
        {"OWN_UNPACK_LOCAL 0\nOWN_STORE_LOCAL 1\nOWN_STORE_LOCAL 2\n"
         "OWN_UNPACK_LOCAL 1\nPOP\nPOP\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nPOP\n"
         "OWN_UNPACK_LOCAL 2\nPOP\nPOP\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nRET\n",1,4,T(TAG_INT),{OWNER(2),OWNER(1),OWNER(1),OWNER(0)}},
        {"OWN_MOVE_LOCAL 0\nRET\n",1,1,OWNER(3),{OWNER(3)}}
    };
    return build(f,5);
}
static void prepared_and_faults(void) {
    NvmModule *m=authority_fixture();NvmOwnedArrayPlan *p=authority(m,NVM_OWNER_AUTH_PREPARED);
    NvmOwnerAuthorityCounts counts;CHECK(nvm_owned_array_plan_counts(p,&counts));
    CHECK(counts.functions==5 && counts.instructions>30 && counts.persisted_cells>0 && counts.visits>0 && counts.work>0);
    NvmOwnerSignature s;CHECK(nvm_owned_array_plan_signature(p,0,&s));
    CHECK(s.max_stack==1 && s.local_count==1 && !s.parameter_count && s.result.tag==TAG_INT && !s.result.owner);
    CHECK(nvm_owned_array_plan_signature(p,1,&s));CHECK(s.max_stack==4 && s.result.owner && s.result.global_layout==2 && s.result_fields==2);
    CHECK(nvm_owned_array_plan_signature(p,2,&s));CHECK(s.parameter_count==1 && s.parameters[0].owner && s.parameters[0].global_layout==2 && s.result.owner);
    NvmOwnerDeclaration local;CHECK(nvm_owned_array_plan_local(p,3,2,&local));CHECK(local.owner && local.tag==TAG_STRUCT && local.global_layout==1 && !local.mode);
    NvmOwnedArrayLayoutFact row;CHECK(nvm_owned_array_plan_layout(p,2,&row));CHECK(row.flags==3 && row.source_record==2 && row.managed_record==NVM_V2_NO_INDEX && row.has_array && row.has_string);
    uint32_t global=99;CHECK(nvm_owned_array_plan_source(p,3,&global) && global==3);
    NvmOwnedArrayTransport transport;CHECK(nvm_owned_array_plan_transport(p,&transport));
    CHECK(transport.layouts!=m->layout_data && transport.ownership!=m->ownership_data);
    CHECK(transport.layout_size==m->layout_size && transport.ownership_size==m->ownership_size);
    CHECK(!memcmp(transport.layouts,m->layout_data,m->layout_size) && !memcmp(transport.ownership,m->ownership_data,m->ownership_size));
    uint32_t index=0;
    for(uint32_t f=0;f<m->function_count;f++)for(uint32_t pc=0;pc<m->functions[f].code_length;) {
        DecodedInstruction in;uint32_t width=isa_decode(m->code+m->functions[f].code_offset+pc,m->functions[f].code_length-pc,&in);CHECK(width);
        NvmOwnerOriginObligation ob;CHECK(nvm_owned_array_plan_obligation(p,index++,&ob) && ob.function==f && ob.pc==pc);pc+=width;
    }
    CHECK(index==counts.instructions);
    NvmOwnerSignature saved_s=s;CHECK(!nvm_owned_array_plan_signature(p,5,&s) && !memcmp(&s,&saved_s,sizeof s));
    NvmOwnerDeclaration saved_local=local;CHECK(!nvm_owned_array_plan_local(p,3,4,&local) && !memcmp(&local,&saved_local,sizeof local));
    NvmOwnedArrayLayoutFact saved_row=row;CHECK(!nvm_owned_array_plan_layout(p,5,&row) && !memcmp(&row,&saved_row,sizeof row));
    global=99;CHECK(!nvm_owned_array_plan_source(p,5,&global) && global==99);
    NvmOwnedArrayTransport saved_transport=transport;CHECK(!nvm_owned_array_plan_transport(NULL,&transport) && !memcmp(&transport,&saved_transport,sizeof transport));
    NvmOwnerAuthorityCounts saved_counts=counts;CHECK(!nvm_owned_array_plan_counts(NULL,&counts) && !memcmp(&counts,&saved_counts,sizeof counts));
    NvmOwnerOriginObligation ob={.function=8,.pc=9,.actual_tags=10,.required_tags=11,.read_tags=12},saved_ob=ob;CHECK(!nvm_owned_array_plan_obligation(p,counts.instructions,&ob) && !memcmp(&ob,&saved_ob,sizeof ob));
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);CHECK(nvm_verify(m).ok);
    uint8_t *code=malloc(m->code_size),*owned=malloc(m->ownership_size),*layout=malloc(m->layout_size);CHECK(code && owned && layout);
    memcpy(code,m->code,m->code_size);memcpy(owned,m->ownership_data,m->ownership_size);memcpy(layout,m->layout_data,m->layout_size);
    for(unsigned public_mode=0;public_mode<2;public_mode++) {
    uint8_t **original_params=m->function_param_types;if(public_mode)m->function_param_types=NULL;
    unsigned failures=0,structural_failures=0;bool success=false;
    for(long limit=0;limit<1024;limit++) {
        unsigned char sentinel;NvmOwnedArrayPlan *next=(void *)&sentinel;
        failed_structural=failed_direct=false;budget=limit;NvmOwnerAuthorityResult r=public_mode?nvm_owned_array_admit(m,&next):nvm_prepare_owned_array_authority(m,&next);budget=-1;
        if(r.status==NVM_OWNER_AUTH_MEMORY){CHECK(next==(void *)&sentinel && failed_direct && !failed_structural);failures++;}
        else if(r.status==NVM_OWNER_AUTH_INVALID && failed_structural && !failed_direct) {
            CHECK(next==(void *)&sentinel);structural_failures++;
            printf("I retain structural INVALID at allocation prefix %ld, exact retained-layout boundary.\n",limit);
        }
        else {if(r.status!=NVM_OWNER_AUTH_PREPARED)fprintf(stderr,"fault%ld status%d %s\n",limit,r.status,r.message);CHECK(r.status==NVM_OWNER_AUTH_PREPARED);nvm_owned_array_plan_free(next);success=true;}
        CHECK(!memcmp(code,m->code,m->code_size) && !memcmp(owned,m->ownership_data,m->ownership_size) && !memcmp(layout,m->layout_data,m->layout_size));
        if(success)break;
    }
    CHECK(failures>100 && structural_failures>0 && success);printf("I preserve %s plan outputs through %u MEMORY and %u attributed structural INVALID failures.\n",public_mode?"public absent-sidecar":"private",failures,structural_failures);
    CHECK(m->function_param_types==(public_mode?NULL:original_params));m->function_param_types=original_params;
    }
    nvm_module_free(m);CHECK(nvm_owned_array_plan_signature(p,4,&s) && s.result.owner && s.result.global_layout==3);
    CHECK(!memcmp(transport.layouts,layout,transport.layout_size) && !memcmp(transport.ownership,owned,transport.ownership_size));
    nvm_owned_array_plan_free(p);free(code);free(owned);free(layout);
}
static void independent_exits(void) {
    Type pair[]={OWNER(3)};
    origin_only_refusal("ARR_NEW 3\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 0\nPUSH_I64 0\nRET\n",pair,1);
    origin_only_refusal("OWN_PACK 4\nOWN_STORE_LOCAL 0\nPUSH_I64 0\nRET\n",(Type[]){OWNER(4)},1);
    Function f[]={
        {"PUSH_I64 0\nRET\n",0,0,T(TAG_INT),{{0}}},
        {"PUSH_I64 0\nRET\n",1,1,T(TAG_INT),{OWNER(3)}}
    };
    NvmModule *m=build(f,2);NvmOwnedArrayOrigins *o=expect(m,NVM_OWNER_ORIGIN_PROVED);nvm_owned_array_origins_free(o);
    authority(m,NVM_OWNER_AUTH_UNRESOLVED);nvm_module_free(m);
    f[1].body="OWN_UNPACK_LOCAL 0\nPOP\nPOP\nPUSH_I64 0\nRET\n";
    m=build(f,2);NvmOwnedArrayPlan *p=authority(m,NVM_OWNER_AUTH_PREPARED);nvm_owned_array_plan_free(p);nvm_module_free(m);
}
static void joins_and_refusals(void) {
    struct {const char *body;bool accepted;} cases[]={
        {"PUSH_BOOL 1\nJMP_FALSE other\nARR_NEW 3\nSTORE_LOCAL 0\nPUSH_STR value\nSTORE_LOCAL 1\nJMP done\n"
         "other:\nARR_NEW 3\nSTORE_LOCAL 0\nPUSH_STR value\nSTORE_LOCAL 1\ndone:\nLOAD_LOCAL 0\nARR_LEN\nPOP\nLOAD_LOCAL 1\nPOP\nPUSH_I64 0\nRET\n",true},
        {"PUSH_BOOL 1\nJMP_FALSE done\nARR_NEW 3\nSTORE_LOCAL 0\ndone:\nLOAD_LOCAL 0\nPOP\nPUSH_I64 0\nRET\n",false},
        {"PUSH_BOOL 1\nJMP_FALSE done\nPUSH_STR value\nSTORE_LOCAL 1\ndone:\nLOAD_LOCAL 1\nPOP\nPUSH_I64 0\nRET\n",false},
        {"ARR_NEW 3\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 2\nARR_NEW 3\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 2\nPUSH_I64 0\nRET\n",false},
        {"ARR_NEW 3\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 2\nLOAD_LOCAL 2\nOWN_MOVE_LOCAL 2\nPUSH_I64 0\nRET\n",false},
        {"PUSH_BOOL 1\nJMP_FALSE done\nARR_NEW 3\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 2\ndone:\nPUSH_I64 0\nRET\n",false},
        {"ARR_NEW 3\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 2\nLOAD_LOCAL 2\nAGG_GET 0\nSTORE_LOCAL 0\n"
         "OWN_UNPACK_LOCAL 2\nPOP\nPOP\nLOAD_LOCAL 0\nPUSH_F64 3.0\nARR_PUSH\nPOP\nPUSH_I64 0\nRET\n",true},
        {"ARR_NEW 3\nPUSH_I64 0\nARR_GET\nPUSH_F64 1.5\nF64_EQ\nPOP\nPUSH_I64 0\nRET\n",true},
        {"PUSH_BOOL 1\nJMP_FALSE other\nOWN_PACK 4\nJMP done\nother:\nOWN_PACK 4\ndone:\nOWN_STORE_LOCAL 3\n"
         "OWN_UNPACK_LOCAL 3\nPUSH_I64 0\nRET\n",true}
    };
    for(unsigned n=0;n<sizeof cases/sizeof cases[0];n++) {
        Function f={cases[n].body,0,4,T(TAG_INT),{T(TAG_ARRAY),T(TAG_STRING),OWNER(3),OWNER(4)}};
        NvmModule *m=build(&f,1);NvmOwnedArrayPlan *p=authority(m,cases[n].accepted?NVM_OWNER_AUTH_PREPARED:NVM_OWNER_AUTH_UNRESOLVED);
        nvm_owned_array_plan_free(p);nvm_module_free(m);
    }
    NvmModule *m=authority_fixture();m->service_size=1;authority(m,NVM_OWNER_AUTH_UNRESOLVED);m->service_size=0;nvm_module_free(m);
    unsigned char sentinel;NvmOwnedArrayPlan *p=(void *)&sentinel;
    CHECK(nvm_prepare_owned_array_authority(NULL,&p).status==NVM_OWNER_AUTH_INVALID && p==(void *)&sentinel);
    CHECK(nvm_prepare_owned_array_authority(NULL,NULL).status==NVM_OWNER_AUTH_INVALID);nvm_owned_array_plan_free(NULL);
}
static void comparison_rows(const char *body,uint8_t opcode,uint8_t mask,bool revisit) {
    Function f={body,0,0,T(TAG_INT),{{0}}};NvmModule *m=build(&f,1);
    NvmOwnedArrayOrigins *origins=expect(m,NVM_OWNER_ORIGIN_PROVED);
    NvmOwnedArrayPlan *p=authority(m,NVM_OWNER_AUTH_PREPARED);
    NvmOwnerAuthorityCounts counts;CHECK(nvm_owned_array_plan_counts(p,&counts));
    if(revisit)CHECK(counts.visits>counts.instructions);
    unsigned comparisons=0;uint32_t index=0;
    for(uint32_t pc=0;pc<m->functions[0].code_length;) {
        DecodedInstruction in;uint32_t width=isa_decode(m->code+pc,m->functions[0].code_length-pc,&in);CHECK(width);
        NvmOwnerOriginObligation o,a;CHECK(nvm_owned_array_origin_obligation(origins,index,&o));
        CHECK(nvm_owned_array_plan_obligation(p,index++,&a));
        CHECK(a.function==0 && a.pc==pc && a.opcode==in.opcode);
        CHECK(a.operand_count==o.operand_count && a.runtime_checks==o.runtime_checks);
        for(unsigned n=0;n<2;n++)CHECK(a.operand_actual[n]==o.operand_actual[n] && a.operand_required[n]==o.operand_required[n]);
        if(in.opcode==opcode) {
            comparisons++;CHECK(a.operand_count==2 && a.runtime_checks==mask);
            for(unsigned n=0;n<2;n++) {
                CHECK(a.operand_required[n]==(1u<<TAG_FLOAT));
                CHECK(a.operand_actual[n]==((1u<<TAG_FLOAT)|((mask&(1u<<n))?(1u<<TAG_VOID):0)));
            }
        } else CHECK(!a.runtime_checks);
        if(in.opcode==OP_RET)CHECK(!a.operand_count && !a.operand_actual[0] && !a.operand_actual[1]);
        pc+=width;
    }
    CHECK(comparisons==1 && index==counts.instructions);CHECK(nvm_verify(m).ok);
    nvm_owned_array_plan_free(p);nvm_owned_array_origins_free(origins);nvm_module_free(m);
}
static void optional_operands(void) {
    const char *names[]={"F64_EQ","F64_NE","F64_LT","F64_LE","F64_GT","F64_GE"};
    uint8_t opcodes[]={OP_F64_EQ,OP_F64_NE,OP_F64_LT,OP_F64_LE,OP_F64_GT,OP_F64_GE};
    const char *exact="PUSH_F64 1.5\n",*optional="ARR_NEW 3\nPUSH_I64 0\nARR_GET\n";
    char body[2048];
    for(unsigned op=0;op<6;op++)for(unsigned mask=0;mask<4;mask++) {
        snprintf(body,sizeof body,"%s%s%s\nPOP\nPUSH_I64 0\nRET\n",mask&1?optional:exact,mask&2?optional:exact,names[op]);
        comparison_rows(body,opcodes[op],(uint8_t)mask,false);
    }
    comparison_rows("PUSH_BOOL 1\nJMP_FALSE other\nPUSH_F64 1.5\nJMP join\n"
        "other:\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nARR_NEW 3\nPUSH_I64 0\nARR_GET\njoin:\nPUSH_F64 1.5\nF64_EQ\nPOP\nPUSH_I64 0\nRET\n",OP_F64_EQ,1,true);
    comparison_rows("PUSH_F64 1.5\nagain:\nDUP\nPUSH_F64 1.5\nF64_NE\nPOP\nPUSH_BOOL 0\nJMP_FALSE done\n"
        "POP\nARR_NEW 3\nPUSH_I64 0\nARR_GET\nJMP again\ndone:\nPOP\nPUSH_I64 0\nRET\n",OP_F64_NE,1,true);
    const char *refused[]={
        "PUSH_VOID\nPUSH_F64 1.5\nF64_EQ\nPOP\nPUSH_I64 0\nRET\n",
        "PUSH_I64 1\nPUSH_F64 1.5\nF64_EQ\nPOP\nPUSH_I64 0\nRET\n",
        "PUSH_STR value\nPUSH_F64 1.5\nF64_EQ\nPOP\nPUSH_I64 0\nRET\n",
        "LOAD_LOCAL 0\nPUSH_F64 1.5\nF64_EQ\nPOP\nPUSH_I64 0\nRET\n",
        "ARR_NEW 3\nPUSH_I64 0\nARR_GET\nPUSH_F64 1.5\nF64_ADD\nPOP\nPUSH_I64 0\nRET\n",
        "ARR_NEW 3\nPUSH_I64 0\nARR_GET\nF64_NEG\nPOP\nPUSH_I64 0\nRET\n",
        "ARR_NEW 3\nPUSH_I64 0\nARR_GET\nSTORE_LOCAL 0\nPUSH_I64 0\nRET\n",
        "ARR_NEW 3\nPUSH_I64 0\nARR_GET\nF64_TO_BITS\nPOP\nPUSH_I64 0\nRET\n"
    };
    for(unsigned n=0;n<sizeof refused/sizeof refused[0];n++) {
        Function f={refused[n],0,1,T(TAG_INT),{T(TAG_FLOAT)}};NvmModule *m=build(&f,1);
        authority(m,NVM_OWNER_AUTH_UNRESOLVED);nvm_module_free(m);
    }
}
static void integer_output(void) {
    const char *values[]={"PUSH_I64 7\n","PUSH_BOOL 1\n","PUSH_F64 1.5\n","PUSH_STR value\n","PUSH_VOID\n","ARR_NEW 3\n","PUSH_U8 1\n"};
    for(unsigned newline=0;newline<2;newline++)for(unsigned n=0;n<sizeof values/sizeof values[0];n++) {
        char body[256];snprintf(body,sizeof body,"%s%s\nPUSH_I64 0\nRET\n",values[n],newline?"PRINTLN":"PRINT");
        Function f={body,0,0,T(TAG_INT),{{0}}};NvmModule *m=build(&f,1);
        NvmOwnedArrayPlan *p=authority(m,n?NVM_OWNER_AUTH_UNRESOLVED:NVM_OWNER_AUTH_PREPARED);
        if(p) {
            NvmOwnerAuthorityCounts c;CHECK(nvm_owned_array_plan_counts(p,&c));unsigned output=0;
            for(uint32_t i=0;i<c.instructions;i++) {NvmOwnerOriginObligation o;CHECK(nvm_owned_array_plan_obligation(p,i,&o));
                if(o.opcode==(newline?OP_PRINTLN:OP_PRINT)) {output++;CHECK(o.operand_count==1 && !o.runtime_checks && o.operand_actual[0]==(1u<<TAG_INT) && o.operand_required[0]==(1u<<TAG_INT));}}
            CHECK(output==1);nvm_owned_array_plan_free(p);
        }
        nvm_module_free(m);
    }
}

static void public_parameter_bounds(void) {
    NvmModule *m=authority_fixture();unsigned char sentinel;
    uint32_t functions=m->function_count;NvmFunctionEntry *entries=m->functions;
    uint32_t size=m->ownership_size;uint8_t *bytes=malloc(size);CHECK(bytes);memcpy(bytes,m->ownership_data,size);
    /* Five flags occupy bytes8..12; the aligned function-count word is16. */
    CHECK(size>20 && bytes[16]==functions && bytes[17]==0 && bytes[18]==0 && bytes[19]==0);
    for(unsigned which=0;which<7;which++) {
        NvmOwnedArrayPlan *plan=(void *)&sentinel;NvmOwnerAuthorityStatus wanted=NVM_OWNER_AUTH_INVALID;
        uint16_t arity=entries[0].arity,locals=entries[0].local_count;
        if(which==0)m->functions=NULL;
        if(which==1){m->function_count=9;m->ownership_data[16]=9;wanted=NVM_OWNER_AUTH_LIMIT;}
        if(which==2){entries[0].arity=9;wanted=NVM_OWNER_AUTH_LIMIT;}
        if(which==3){entries[0].local_count=257;wanted=NVM_OWNER_AUTH_LIMIT;}
        if(which==4)m->ownership_size=20; /* No function descriptor header. */
        if(which==5)m->ownership_size=25; /* Header plus incomplete result descriptor. */
        if(which==6)m->ownership_data[25]=1; /* Nonparameter mode: full query must reject. */
        NvmModule before=*m;NvmFunctionEntry saved_entry=entries[0];
        NvmOwnerAuthorityResult result=nvm_owned_array_admit(m,&plan);
        fprintf(stderr,"public parameter boundary=%u status=%u\n",which,(unsigned)result.status);
        CHECK(result.status==wanted && plan==(void *)&sentinel);
        CHECK(!memcmp(&before,m,sizeof before) && !memcmp(&saved_entry,&entries[0],sizeof saved_entry));
        m->functions=entries;m->function_count=functions;m->ownership_size=size;
        entries[0].arity=arity;entries[0].local_count=locals;memcpy(m->ownership_data,bytes,size);
        plan=NULL;CHECK(nvm_owned_array_admit(m,&plan).status==NVM_OWNER_AUTH_PREPARED);nvm_owned_array_plan_free(plan);
    }
    /* Retain an exact parameter descriptor but reject an unsupported tag before
     * the wrapper can substitute it into the temporary view. */
    NvmV2Cursor cursor;uint32_t word;const uint8_t *flags;
    nvm_v2_cursor_init(&cursor,m->ownership_data,size);
    CHECK(nvm_v2_u32(&cursor,&word)==NVM_V2_OK && nvm_v2_u32(&cursor,&word)==NVM_V2_OK);
    CHECK(nvm_v2_take(&cursor,word,&flags)==NVM_V2_OK && nvm_v2_align4(&cursor)==NVM_V2_OK && nvm_v2_u32(&cursor,&word)==NVM_V2_OK);
    uint32_t parameter=0;
    for(uint32_t f=0;f<functions;f++) {
        uint16_t locals,params;const uint8_t *row;
        CHECK(nvm_v2_u16(&cursor,&locals)==NVM_V2_OK && nvm_v2_u16(&cursor,&params)==NVM_V2_OK);
        for(uint32_t n=0;n<=(uint32_t)locals;n++) {
            CHECK(nvm_v2_take(&cursor,8,&row)==NVM_V2_OK);
            if(!parameter && params && n==1)parameter=(uint32_t)(row-m->ownership_data);
        }
    }
    CHECK(parameter);m->ownership_data[parameter]=TAG_FLOAT;
    NvmOwnedArrayPlan *plan=(void *)&sentinel;
    CHECK(nvm_owned_array_admit(m,&plan).status==NVM_OWNER_AUTH_UNRESOLVED && plan==(void *)&sentinel);
    CHECK(m->ownership_data[parameter]==TAG_FLOAT);memcpy(m->ownership_data,bytes,size);
    CHECK(nvm_owned_array_admit(m,&plan).status==NVM_OWNER_AUTH_PREPARED);nvm_owned_array_plan_free(plan);
    CHECK(!memcmp(bytes,m->ownership_data,size));free(bytes);nvm_module_free(m);
}
int main(void) {
    public_parameter_bounds();prepared_and_faults();independent_exits();joins_and_refusals();optional_operands();integer_output();
    printf("%u private owner ARRAY authority checks passed; no pending module execution\n",checks);return 0;
}
