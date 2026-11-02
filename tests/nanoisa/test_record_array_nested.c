/* I qualify exact depth-two primitive fields through actual public consumers. */
#ifndef NESTED_TEST_LEAF
#define NESTED_TEST_LEAF TAG_INT
#endif
#ifndef NESTED_TEST_DEPTH
#define NESTED_TEST_DEPTH 2
#endif
#define main retained_origin_fixture_main
#include "test_record_array_origins.c"
#undef main
#include "../../src/nanoisa/record_array_admission.h"
#include "../../src/nanoisa/nvm2c.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/disassembler.h"
#include "../../src/nanovm/vm.h"
#include "../../src/nanovm/record_array_runtime_private.h"
int g_argc=0;char **g_argv=NULL;
#ifdef RA_NESTED_WHITEBOX
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#include "../../src/nanovm/vm.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
#endif
static void pack_nested(Input *c) {
    op(c,OP_AGG_PACK);b8(c->code,&c->n,AGG_RECORD);b32(c->code,&c->n,0);
    b16(c->code,&c->n,0);b16(c->code,&c->n,1);
}
static void integer(Input *c,unsigned value) {
    scalar(c,TAG_INT);c->code[c->n-8]=(uint8_t)value;
}
static bool exact_float_bits;
static uint64_t float_payload;
static uint64_t leaf_bits(bool final) {
    if(exact_float_bits)return float_payload;
    if(NESTED_TEST_LEAF==TAG_FLOAT) {
        double value=final?11.5:7.25;uint64_t bits;memcpy(&bits,&value,sizeof bits);return bits;
    }
    if(NESTED_TEST_LEAF==TAG_BOOL)return final?0:1;
    return final?11:7;
}
static void leaf_value(Input *c,bool final) {
    if(NESTED_TEST_LEAF==TAG_STRING){arg32(c,OP_PUSH_STR,final?2:1);return;}
    uint64_t bits=leaf_bits(final);
    scalar(c,NESTED_TEST_LEAF);
    if(NESTED_TEST_LEAF==TAG_INT || NESTED_TEST_LEAF==TAG_FLOAT) {
        for(unsigned i=0;i<8;i++)c->code[c->n-8+i]=(uint8_t)(bits>>(8*i));
    } else c->code[c->n-1]=(uint8_t)bits;
}
static void leaf_equal(Input *c) {
    bool nan=exact_float_bits && (float_payload&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000) && (float_payload&UINT64_C(0x000fffffffffffff));
    op(c,NESTED_TEST_LEAF==TAG_FLOAT?(nan?OP_F64_NE:OP_F64_EQ):OP_EQ);
}
static void fixture(Input *c,unsigned mode) {
    init(c,NESTED_TEST_LEAF,false);c->fn[0].local_count=NESTED_TEST_DEPTH==3?5:4;
    array(c,NESTED_TEST_LEAF);leaf_value(c,false);op(c,OP_ARR_PUSH);arg16(c,OP_STORE_LOCAL,1);
    array(c,TAG_ARRAY);if(mode>=5)arg32(c,OP_CALL,1);arg16(c,OP_STORE_LOCAL,0);
    if(NESTED_TEST_DEPTH==3) {
        array(c,TAG_ARRAY);
        for(unsigned i=0;i<2;i++){arg16(c,OP_LOAD_LOCAL,0);op(c,OP_ARR_PUSH);}
        arg16(c,OP_STORE_LOCAL,4);arg16(c,OP_LOAD_LOCAL,4);
    } else arg16(c,OP_LOAD_LOCAL,0);
    pack_nested(c);arg32(c,OP_STORE_GLOBAL,0);
    /* I mutate the outer through its original alias after field construction. */
    for(unsigned i=0;i<2;i++) {
        arg16(c,OP_LOAD_LOCAL,0);arg16(c,OP_LOAD_LOCAL,1);op(c,OP_ARR_PUSH);op(c,OP_POP);
    }
    if(mode==1) {
        arg16(c,OP_LOAD_LOCAL,0);integer(c,0);integer(c,2);op(c,OP_PUSH_BOOL);b8(c->code,&c->n,1);op(c,OP_ASSERT);op(c,OP_ARR_SLICE);arg16(c,OP_STORE_LOCAL,0);
    } else if(mode==2) {
        arg16(c,OP_LOAD_LOCAL,0);op(c,OP_ARR_POP);arg16(c,OP_STORE_LOCAL,1);
    } else if(mode==3) {
        arg16(c,OP_LOAD_LOCAL,0);integer(c,0);op(c,OP_ARR_REMOVE);op(c,OP_POP);
    } else if(mode==4) {
        /* I replace the canonical record field while retaining its old alias. */
        arg32(c,OP_LOAD_GLOBAL,0);array(c,TAG_ARRAY);op(c,OP_AGG_SET);b16(c->code,&c->n,0);op(c,OP_POP);
    }
    arg16(c,OP_LOAD_LOCAL,1);integer(c,0);leaf_value(c,true);op(c,OP_ARR_SET);op(c,OP_POP);
    arg16(c,OP_LOAD_LOCAL,0);integer(c,0);op(c,OP_ARR_GET);integer(c,0);op(c,OP_ARR_GET);
    leaf_value(c,true);leaf_equal(c);op(c,OP_ASSERT);
    if(mode!=4) {
        arg32(c,OP_LOAD_GLOBAL,0);arg16(c,OP_AGG_GET,0);integer(c,0);op(c,OP_ARR_GET);
        if(NESTED_TEST_DEPTH==3){integer(c,0);op(c,OP_ARR_GET);}
        integer(c,0);op(c,OP_ARR_GET);leaf_value(c,true);leaf_equal(c);op(c,OP_ASSERT);
    }
    integer(c,0);op(c,OP_RET);c->fn[0].code_length=(uint32_t)c->n;
    if(mode>=5) {
        c->strings[3]="forward";c->lengths[3]=7;
        c->m.function_count=mode==5?2:3;c->m.function_param_types=c->param_rows;c->params[0]=TAG_ARRAY;
        for(unsigned f=1;f<c->m.function_count;f++) {
            c->fn[f].name_idx=f+1;c->fn[f].arity=c->fn[f].local_count=1;
            c->fn[f].result_count=1;c->fn[f].result_tag=TAG_ARRAY;c->param_rows[f]=c->params;
            c->fn[f].code_offset=(uint32_t)c->n;arg16(c,OP_LOAD_LOCAL,0);
            if(f==1 && mode==6)arg32(c,OP_TAIL_CALL,2);else op(c,OP_RET);
            c->fn[f].code_length=(uint32_t)c->n-c->fn[f].code_offset;
        }
    }
    if(NESTED_TEST_DEPTH==3)depth_three_authority(c,NESTED_TEST_LEAF);
    else {nested_authority(c);c->ownership[c->o-36]=NESTED_TEST_LEAF;}
}
static void execute(const NvmModule *m) {
    VmState vm;vm_init(&vm,m);VmResult result=vm_execute(&vm);
    if(result!=VM_OK)fprintf(stderr,"I failed nested VM: %s\n",vm.error_msg);
    CHECK(result==VM_OK && vm.frame_count==0 && vm.stack_size==1 && vm.stack[0].tag==TAG_INT && vm.stack[0].as.i64==0);
    if(exact_float_bits) {
        CHECK(vm.globals[0].tag==TAG_STRUCT);
        VmArray *outer=vm.globals[0].as.sval->fields[0].as.array;
        CHECK(outer->elem_type==TAG_ARRAY && outer->length==2);
        if(NESTED_TEST_DEPTH==3)outer=outer->elements[0].as.array;
        VmArray *inner=outer->elements[0].as.array;uint64_t bits;
        CHECK(inner->elem_type==TAG_FLOAT && inner->unboxed && inner->length==1);
        memcpy(&bits,inner->packed,sizeof bits);CHECK(bits==float_payload);
    }
    vm_destroy(&vm);CHECK(vm.heap.stats.num_objects==0);
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    VmRecordArrayPrivate *private=NULL;
    CHECK(vm_record_array_private_create(m,&private).status==NVM_ARRAY_ELIGIBLE);
    for(unsigned attempt=0;attempt<2;attempt++) {
        CHECK(vm_record_array_private_run(private)==VM_OK);
        VmRecordArrayPrivateStats stats;CHECK(vm_record_array_private_stats(private,&stats));
        CHECK(!stats.active_frames && !stats.active_stack && stats.has_result);
        uint32_t path[]={0};VmRecordArrayObservation outer;
        CHECK(vm_record_array_private_observe(private,VM_RA_GLOBAL,0,path,1,&outer));
        CHECK(outer.tag==TAG_ARRAY && outer.element_tag==TAG_ARRAY);
        if(outer.length) {
            uint32_t inner_path[]={0,0,0,0};VmRecordArrayObservation value;
            CHECK(vm_record_array_private_observe(private,VM_RA_GLOBAL,0,inner_path,NESTED_TEST_DEPTH+1,&value));
            CHECK(value.tag==NESTED_TEST_LEAF);
            if(NESTED_TEST_LEAF==TAG_STRING)CHECK(value.length==6);
            else CHECK(value.scalar_bits==leaf_bits(true));
        }
    }
    vm_record_array_private_destroy(private);
#endif
}
static void admitted(Input *c) {
    NvmRecordArrayExecutionPlan *plan=NULL;
    NvmArrayEligibilityResult result=nvm_record_array_admit(&c->m,&plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"I failed nested proof: %u %s\n",result.status,result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);nvm_record_array_execution_free(plan);CHECK(nvm_verify(&c->m).ok);
}
static void public_routes(Input *c,const char *directory,unsigned id) {
    admitted(c);execute(&c->m);
    char *assembly=disasm_module_styled(&c->m,DISASM_STYLE_CANONICAL);CHECK(assembly);
    AsmResult result;NvmModule *copy=asm_assemble(assembly,&result);
    if(!copy)fprintf(stderr,"I failed nested assembly: %s\n",result.message);
    CHECK(copy);CHECK(copy->ownership_size==c->m.ownership_size && !memcmp(copy->ownership_data,c->m.ownership_data,c->m.ownership_size));
    execute(copy);nvm_module_free(copy);free(assembly);
    NvmV2Module wire;CHECK(nvm_v2_from_nvm_module(&c->m,&wire)==NVM_V2_OK);
    size_t size=0;CHECK(nvm_v2_module_serialize(&wire,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);CHECK(nvm_v2_module_serialize(&wire,bytes,size,NULL)==NVM_V2_OK);
    NvmV2Module decoded;CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);execute(copy);
    nvm_module_free(copy);
    /* I require the decoded transport consumer to rederive nested authority;
     * an already serialized container is not a reusable proof token. */
    uint8_t *retained=(uint8_t *)decoded.ownership_data;
    size_t leaf=decoded.ownership_size-36;CHECK(retained[leaf]==NESTED_TEST_LEAF);
    retained[leaf]=NESTED_TEST_LEAF==TAG_STRING?TAG_BOOL:TAG_STRING;copy=NULL;
    CHECK(nvm_v2_to_nvm_module(&decoded,&copy)!=NVM_V2_OK && !copy);
    retained[leaf]=NESTED_TEST_LEAF;
    CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);execute(copy);
    nvm_module_free(copy);nvm_v2_module_free(&decoded);nvm_v2_module_free(&wire);free(bytes);
    char error[256];char *native=nvm2c_emit(&c->m,error,sizeof error);
    if(!native)fprintf(stderr,"I failed nested native emission: %s\n",error);
    CHECK(native);char path[1024];snprintf(path,sizeof path,"%s/nested-%u.c",directory,id);
    FILE *file=fopen(path,"w");CHECK(file);CHECK(fputs("#define main fixture_main\n",file)>=0);
    if(exact_float_bits) {
        const char *cleanup=strstr(native,"    nrec_release_snapshots();");
        CHECK(cleanup && !strstr(cleanup+1,"    nrec_release_snapshots();"));
        CHECK(fputs("static void verify_bits(void);\n",file)>=0);
        CHECK(fwrite(native,1,(size_t)(cleanup-native),file)==(size_t)(cleanup-native));
        CHECK(fputs("    verify_bits();\n",file)>=0);CHECK(fputs(cleanup,file)>=0);
        CHECK(fputs("\nstatic void verify_bits(void) { nrec_t record=nvalue_require_record(nglobal[0]); nmap_value outer={7,record.k[0],(char *)record.a[0]}; ",file)>=0);
        if(NESTED_TEST_DEPTH==3)CHECK(fputs("outer=nvalue_array_get(outer,0); ",file)>=0);
        CHECK(fputs("nmap_value value=nvalue_array_get(nvalue_array_get(outer,0),0); ",file)>=0);
        CHECK(fprintf(file,"if(value.kind!=3 || (uint64_t)value.integer!=UINT64_C(0x%016llx)) abort(); }\n",(unsigned long long)float_payload)>0);
    } else CHECK(fputs(native,file)>=0);
    CHECK(fputs("\n#undef main\nint main(void) { nmap_collection_budget=1; int result=fixture_main(); nglobal[0]=(nmap_value){0}; nmap_collect(); if(nrec_owned_head || narr_owners || nagg_live_bytes) abort(); return result; }\n",file)>=0);
    CHECK(!fclose(file));free(native);
}
static void refusal_controls(void) {
    for(unsigned variant=0;variant<5;variant++) {
        Input c;fixture(&c,0);size_t row=c.o-36;
        if(variant==0)c.ownership[row]=NESTED_TEST_LEAF==TAG_STRING?TAG_BOOL:TAG_STRING;
        if(variant==1){size_t at=c.o-40;b32(c.ownership,&at,0);} /* cycle */
        if(variant==2){size_t at=c.o-40;b32(c.ownership,&at,9);} /* invalid referent */
        if(variant==3)c.ownership[row]=TAG_VOID;
        if(variant==4)c.code[1]=NESTED_TEST_LEAF==TAG_STRING?TAG_BOOL:TAG_STRING; /* known wrong child origin */
        NvmRecordArrayExecutionPlan *plan=(void *)(uintptr_t)1;
        NvmArrayEligibilityResult result=nvm_record_array_admit(&c.m,&plan);
        CHECK(result.status==(variant==0 || variant==4?NVM_ARRAY_UNRESOLVED:NVM_ARRAY_INVALID));
        CHECK(plan==(void *)(uintptr_t)1 && !nvm_verify(&c.m).ok);
        char *text=disasm_module_styled(&c.m,DISASM_STYLE_CANONICAL);CHECK(text);
        AsmResult error;NvmModule *bad=asm_assemble(text,&error);CHECK(!bad);free(text);
        NvmV2Module wire={0};
        CHECK(nvm_v2_from_nvm_module(&c.m,&wire)!=NVM_V2_OK);
        nvm_v2_module_free(&wire);
        fixture(&c,0);admitted(&c);execute(&c.m);
    }
}
static void bounds_cleanup(void) {
    for(unsigned negative=0;negative<2;negative++) {
        Input c;fixture(&c,0);uint32_t at=0;bool found=false;
        while(at<c.n) {
            DecodedInstruction in;uint32_t width=isa_decode(c.code+at,(uint32_t)c.n-at,&in);CHECK(width);
            if(in.opcode==OP_ARR_SET) {
                size_t value_width=(NESTED_TEST_LEAF==TAG_INT || NESTED_TEST_LEAF==TAG_FLOAT)?9:(NESTED_TEST_LEAF==TAG_STRING?5:2);
                CHECK(at>=value_width+9 && c.code[at-value_width-9]==OP_PUSH_I64);
                memset(c.code+at-value_width-8,negative?0xff:0,8);
                if(!negative)c.code[at-value_width-8]=1;
                found=true;break;
            }
            at+=width;
        }
        CHECK(found);admitted(&c);
        VmState vm;vm_init(&vm,&c.m);CHECK(vm_execute(&vm)==VM_ERR_OUT_OF_BOUNDS);
        CHECK(!vm.frame_count && !vm.stack_size);vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
        VmRecordArrayPrivate *private=NULL;CHECK(vm_record_array_private_create(&c.m,&private).status==NVM_ARRAY_ELIGIBLE);
        CHECK(vm_record_array_private_run(private)==VM_ERR_OUT_OF_BOUNDS);
        VmRecordArrayPrivateStats stats;CHECK(vm_record_array_private_stats(private,&stats));
        CHECK(!stats.active_frames && !stats.active_stack && !stats.has_result);
        vm_record_array_private_destroy(private);CHECK(!ra_live && !ra_bytes);
#endif
        fixture(&c,0);execute(&c.m);CHECK(!ra_live && !ra_bytes);
    }
}
static void allocation_prefixes(void) {
    Input c;fixture(&c,1);
    CHECK(!ra_live && !ra_bytes);
    NvmRecordArrayExecutionPlan *plan=NULL;ra_calls=0;
    CHECK(nvm_record_array_admit(&c.m,&plan).status==NVM_ARRAY_ELIGIBLE);
    size_t proof_total=ra_calls;nvm_record_array_execution_free(plan);CHECK(!ra_live && !ra_bytes);
    for(int persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<proof_total;fail++) {
        Input before=c;plan=(void *)(uintptr_t)1;ra_calls=0;ra_fail=fail;ra_persistent=persistent;
        NvmArrayEligibilityResult result=nvm_record_array_admit(&c.m,&plan);
        CHECK(ra_calls>fail && result.status==NVM_ARRAY_MEMORY && plan==(void *)(uintptr_t)1);
        CHECK(!memcmp(&before,&c,sizeof c) && !ra_live && !ra_bytes);
        ra_fail=SIZE_MAX;ra_persistent=0;plan=NULL;
        CHECK(nvm_record_array_admit(&c.m,&plan).status==NVM_ARRAY_ELIGIBLE);
        nvm_record_array_execution_free(plan);CHECK(!ra_live && !ra_bytes);
    }
    printf("I checked %zu typed nested proof allocation positions in both failure modes.\n",proof_total);
    VmState vm;vm_init(&vm,&c.m);
    ra_calls=0;CHECK(vm_execute(&vm)==VM_OK);size_t total=ra_calls;
    vm_destroy(&vm);CHECK(total && !ra_live && !ra_bytes);
    for(int persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<total;fail++) {
        ra_fail=SIZE_MAX;ra_persistent=0;vm_init(&vm,&c.m);
        ra_calls=0;ra_fail=fail;ra_persistent=persistent;VmResult result=vm_execute(&vm);
        CHECK(ra_calls>fail);
        bool refused=result==VM_ERR_TYPE_ERROR &&
            !strcmp(vm.error_msg,"I require reference lifetime and ownership instruction verification before execution");
        if(result!=VM_OK && result!=VM_ERR_MEMORY && !refused)
            fprintf(stderr,"I failed nested allocation %zu/%zu mode %d: %u %s\n",fail,total,persistent,result,vm.error_msg);
        CHECK(result==VM_OK || result==VM_ERR_MEMORY || refused);
        CHECK(vm.frame_count==0);if(result!=VM_OK)CHECK(vm.stack_size==0);
        ra_fail=SIZE_MAX;ra_persistent=0;vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
        vm_init(&vm,&c.m);CHECK(vm_execute(&vm)==VM_OK);vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
    }
    printf("I checked %zu nested allocation positions in transient and persistent modes with independent recovery.\n",total);
}
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
static void private_prefixes(void) {
    Input c;fixture(&c,1);VmRecordArrayPrivate *p=NULL;
    CHECK(!ra_live && !ra_bytes);ra_calls=0;
    CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
    size_t preparation=ra_calls;CHECK(vm_record_array_private_run(p)==VM_OK);
    ra_calls=0;CHECK(vm_record_array_private_run(p)==VM_OK);size_t execution=ra_calls;
    vm_record_array_private_destroy(p);CHECK(!ra_live && !ra_bytes && preparation && execution);
    for(unsigned phase=0;phase<2;phase++)for(int persistent=0;persistent<2;persistent++)
    for(size_t fail=0;fail<(phase?execution:preparation);fail++) {
        ra_fail=SIZE_MAX;ra_persistent=0;p=NULL;
        if(phase) {
            CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
            CHECK(vm_record_array_private_run(p)==VM_OK);
        }
        ra_calls=0;ra_fail=fail;ra_persistent=persistent;
        if(phase) {
            CHECK(vm_record_array_private_run(p)==VM_ERR_MEMORY && ra_calls>fail);
            VmRecordArrayPrivateStats stats;VmRecordArrayObservation prior;
            CHECK(vm_record_array_private_stats(p,&stats) && !stats.active_frames && !stats.active_stack);
            CHECK(vm_record_array_private_observe(p,VM_RA_RESULT,0,NULL,0,&prior) && prior.tag==TAG_INT && prior.scalar_bits==0);
            vm_record_array_private_destroy(p);
        } else {
            p=(void *)(uintptr_t)1;
            CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_MEMORY && p==(void *)(uintptr_t)1 && ra_calls>fail);
        }
        CHECK(!ra_live && !ra_bytes);ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
        CHECK(vm_record_array_private_run(p)==VM_OK);vm_record_array_private_destroy(p);
        CHECK(!ra_live && !ra_bytes);
    }
    printf("I checked %zu private preparation and %zu execution failure positions in both modes.\n",preparation,execution);
}
#endif
#ifdef RA_NESTED_WHITEBOX
static void slice_headroom(void) {
    Input c;fixture(&c,1);VmState vm;vm_init(&vm,&c.m);
    uint32_t active=0;VmOwnedInvocationProof proof={0};
    proof.module=&c.m;proof.ordinary_arrays=true;proof.ordinary_active_root=&active;
    VmCallFrame *frame=&vm.frames[0];memset(frame,0,sizeof *frame);
    frame->module=&c.m;frame->local_count=c.fn[0].local_count;frame->owned_callable=val_void();
    for(uint16_t i=0;i<frame->local_count;i++)vm.stack[vm.stack_size++]=val_void();
    vm.frame_count=1;vm.current_fn=0;vm.ip=0;
    VmTrap trap=vm_core_execute_scoped(&vm,&proof,NULL,NULL);CHECK(trap.type==TRAP_ASSERT);
    vm_release(&vm.heap,trap.data.assert_check.condition);
    NanoValue outer=stack_peek(&vm,2);CHECK(outer.tag==TAG_ARRAY && outer.as.array->length==2);
    VmArray *child=outer.as.array->elements[0].as.array;
    CHECK(child==outer.as.array->elements[1].as.array);
    uint32_t refs=child->header.ref_count,ip=vm.ip,stack=vm.stack_size;
    child->header.ref_count=UINT32_MAX-1;
    trap=vm_core_execute_scoped(&vm,&proof,NULL,NULL);
    CHECK(trap.type==TRAP_ERROR && trap.data.error.code==VM_ERR_MEMORY);
    CHECK(vm.stack_size==stack && child->header.ref_count==UINT32_MAX-1 && outer.as.array->length==2);
    child->header.ref_count=refs;vm.ip=ip;
    do {
        trap=vm_core_execute_scoped(&vm,&proof,NULL,NULL);
        if(trap.type==TRAP_ASSERT) {CHECK(val_truthy(trap.data.assert_check.condition));vm_release(&vm.heap,trap.data.assert_check.condition);}
    } while(trap.type==TRAP_ASSERT);
    CHECK(trap.type==TRAP_NONE && !vm.frame_count && vm.stack_size==1 && vm.stack[0].as.i64==0);
    vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
    execute(&c.m);CHECK(!ra_live && !ra_bytes);
}
#endif
#if defined(NANO_RECORD_ARRAY_PRIVATE_RUNTIME) && NESTED_TEST_DEPTH==3
static void maximum_private_release(void) {
    Input c;fixture(&c,0);uint8_t *layouts=calloc(1,4+256*20);CHECK(layouts);
    size_t n=0;b32(layouts,&n,256);
    for(unsigned i=0;i<256;i++) {
        b8(layouts,&n,NVM_V2_LAYOUT_STRUCT);b8(layouts,&n,0);b16(layouts,&n,1);b32(layouts,&n,NO);
        b8(layouts,&n,i?TAG_STRUCT:TAG_ARRAY);b8(layouts,&n,0);b16(layouts,&n,0);
        b32(layouts,&n,i?i-1:NO);b32(layouts,&n,NO);
    }
    c.m.layout_data=layouts;c.m.layout_size=(uint32_t)n;c.m.struct_count=256;
    c.o=0;b32(c.ownership,&c.o,3);b32(c.ownership,&c.o,256);
    for(unsigned i=0;i<256;i++)b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);
    b32(c.ownership,&c.o,1);b16(c.ownership,&c.o,5);b16(c.ownership,&c.o,0);desc(&c,TAG_INT);
    for(unsigned i=0;i<5;i++)desc(&c,TAG_VOID);
    b32(c.ownership,&c.o,4);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b16(c.ownership,&c.o,2);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,44);b32(c.ownership,&c.o,3);
    b8(c.ownership,&c.o,TAG_ARRAY);b8(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b8(c.ownership,&c.o,TAG_ARRAY);b8(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,2);
    b8(c.ownership,&c.o,TAG_STRING);b8(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,NO);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);
    c.m.ownership_size=(uint32_t)c.o;
    VmRecordArrayPrivate *p=NULL;CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);free(layouts);
    CHECK(vm_record_array_private_run(p)==VM_OK && p->layouts[255].rank==256);
    VmHeap *heap=&p->vm.heap;uint64_t objects=heap->stats.num_objects,bytes=heap->stats.allocated-heap->stats.freed;
    VmString *text=vm_string_new(heap,"deep dynamic string",19);CHECK(text);
    VmArray *inner=vm_array_new(heap,TAG_STRING,0),*middle=vm_array_new(heap,TAG_ARRAY,0),*outer=vm_array_new(heap,TAG_ARRAY,0);CHECK(inner&&middle&&outer);
    CHECK(vm_array_push(heap,inner,val_string(text)));vm_release(heap,val_string(text));
    CHECK(vm_array_push(heap,middle,val_array(inner)));vm_release(heap,val_array(inner));
    CHECK(vm_array_push(heap,outer,val_array(middle)));vm_release(heap,val_array(middle));
    NanoValue child=val_array(outer);
    for(unsigned i=0;i<256;i++) {
        CHECK(vm_ra_field_accepts(p,i,0,child));VmStruct *owner=vm_struct_new(heap,i,1);CHECK(owner);
        owner->fields[0]=child;child=val_struct(owner);
    }
    struct PreviousField { NvmV2LayoutField declaration;uint8_t element,nested_leaf; };
    CHECK(sizeof(VmRecordArrayField)==sizeof(struct PreviousField));
    printf("I preserve copied field bytes: %zu; release entries: %zu.\n",sizeof(VmRecordArrayField),sizeof heap->private_release/sizeof heap->private_release[0]);
    CHECK(sizeof heap->private_release/sizeof heap->private_release[0]==259);
    CHECK(VM_RECORD_ARRAY_PATH_MAX==258);
    uint32_t path[259]={0};VmRecordArrayObservation observation={0};
    NanoValue previous=p->vm.globals[0];p->vm.globals[0]=child;
    CHECK(vm_record_array_private_observe(p,VM_RA_GLOBAL,0,path,258,&observation));
    CHECK(observation.tag==TAG_ARRAY && observation.element_tag==TAG_STRING);
    VmRecordArrayObservation retained=observation;
    CHECK(!vm_record_array_private_observe(p,VM_RA_GLOBAL,0,path,259,&observation));
    CHECK(!memcmp(&observation,&retained,sizeof observation));
    p->vm.globals[0]=previous;
    size_t calls=ra_calls;ra_fail=calls;ra_persistent=1;vm_release(heap,child);
    CHECK(ra_calls==calls && heap->stats.num_objects==objects && heap->stats.allocated-heap->stats.freed==bytes);
    ra_fail=SIZE_MAX;ra_persistent=0;
    /* Empty physical arrays do not justify an unbounded fourth layer. */
    VmArray *deep=vm_array_new(heap,TAG_ARRAY,0);CHECK(deep);
    for(unsigned i=0;i<3;i++) {
        VmArray *parent=vm_array_new(heap,TAG_ARRAY,0);CHECK(parent);
        CHECK(vm_array_push(heap,parent,val_array(deep)));vm_release(heap,val_array(deep));deep=parent;
    }
    CHECK(!vm_ra_array_valid(deep,TAG_ARRAY));
    CHECK(!vm_ra_array_exact(deep,3,TAG_STRING));
    vm_release(heap,val_array(deep));
    CHECK(heap->stats.num_objects==objects && heap->stats.allocated-heap->stats.freed==bytes);
    vm_record_array_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
#endif
int main(int argc,char **argv) {
#ifdef RA_NESTED_WHITEBOX
    slice_headroom();
#endif
    if(argc==2 && !strcmp(argv[1],"allocations")){
        allocation_prefixes();
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
        private_prefixes();
#endif
        return 0;
    }
    refusal_controls();bounds_cleanup();
#if defined(NANO_RECORD_ARRAY_PRIVATE_RUNTIME) && NESTED_TEST_DEPTH==3
    if(NESTED_TEST_LEAF==TAG_STRING)maximum_private_release();
#endif
    CHECK(argc==2);for(unsigned mode=0;mode<7;mode++) {
        Input c;fixture(&c,mode);
        if(mode==3) {
            NvmRecordArrayExecutionPlan *plan=(void *)(uintptr_t)1;
            CHECK(nvm_record_array_admit(&c.m,&plan).status==NVM_ARRAY_UNRESOLVED && plan==(void *)(uintptr_t)1);
            CHECK(!nvm_verify(&c.m).ok);
        } else public_routes(&c,argv[1],mode);
    }
    if(NESTED_TEST_LEAF==TAG_FLOAT) {
        const uint64_t bits[]={0,UINT64_C(0x8000000000000000),1,
            UINT64_C(0x7fefffffffffffff),UINT64_C(0x7ff0000000000000),
            UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000001234)};
        exact_float_bits=true;
        for(unsigned i=0;i<sizeof bits/sizeof bits[0];i++) {
            float_payload=bits[i];Input c;fixture(&c,1);public_routes(&c,argv[1],100+i);
        }
        exact_float_bits=false;
    }
    printf("PASS %u nested record-array public checks\n",checks);return 0;
}
