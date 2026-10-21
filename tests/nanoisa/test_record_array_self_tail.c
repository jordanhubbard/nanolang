/* I exercise the selected self-tail path, not a recursive native C call. */
#define RA_TAIL_GLOBAL_COUNT 3
#define RA_TAIL_CLEAR_EXTRA_GLOBALS "nglobal[1]=(nmap_value){0}; nglobal[2]=(nmap_value){0}; "
#define RA_TAIL_MAIN retained_tail_fixture_main
#include "test_record_array_tail.c"
#undef RA_TAIL_MAIN
#include "../../src/nanovm/record_array_runtime_private.h"

static void number(Input *c,unsigned value) {
    scalar(c,TAG_INT);size_t at=c->n-8;for(unsigned i=0;i<8;i++)c->code[at+i]=(uint8_t)((uint64_t)value>>(8*i));
}
static bool pause_self_tail;
static void self_fixture(Input *c,uint8_t leaf,unsigned width,unsigned iterations,bool entry) {
    fixture(c,leaf,width,entry?TAG_INT:TAG_STRUCT,entry);
    uint8_t first[4096],last[4096];uint32_t first_size=c->fn[0].code_length,last_size=c->fn[2].code_length;
    CHECK(first_size<=sizeof first && last_size<=sizeof last);
    memcpy(first,c->code,first_size);memcpy(last,c->code+c->fn[2].code_offset,last_size);
    c->n=0;number(c,0);arg32(c,OP_STORE_GLOBAL,1);
    memcpy(c->code+c->n,first,first_size);c->n+=first_size;c->fn[0].code_length=(uint32_t)c->n;
    c->fn[1].code_offset=(uint32_t)c->n;
    arg16(c,OP_LOAD_LOCAL,0);arg32(c,OP_STORE_GLOBAL,2);
    arg32(c,OP_LOAD_GLOBAL,1);number(c,iterations);op(c,OP_LT);
    size_t branch=c->n;arg32(c,OP_JMP_FALSE,0);
    arg32(c,OP_LOAD_GLOBAL,1);number(c,1);op(c,OP_I64_ADD);arg32(c,OP_STORE_GLOBAL,1);
    /* Duplicate record arguments are retained cumulatively on every restart. */
    for(unsigned i=0;i<width;i++)arg16(c,OP_LOAD_LOCAL,(uint16_t)(width-1-i));
    if(pause_self_tail){op(c,OP_PUSH_BOOL);b8(c->code,&c->n,1);op(c,OP_ASSERT);}
    arg32(c,OP_TAIL_CALL,1);
    size_t at=branch+1;b32(c->code,&at,(uint32_t)(c->n-branch));
    for(unsigned i=0;i<width;i++)arg16(c,OP_LOAD_LOCAL,(uint16_t)i);
    arg32(c,OP_TAIL_CALL,2);c->fn[1].code_length=(uint32_t)c->n-c->fn[1].code_offset;
    c->fn[2].code_offset=(uint32_t)c->n;memcpy(c->code+c->n,last,last_size);c->n+=last_size;c->fn[2].code_length=last_size;
    authority(c);
}
static void self_execute(Input *c,unsigned iterations,bool entry) {
    VmState vm;vm_init(&vm,&c->m);vm.profile.enabled=true;
    VmResult status=vm_execute(&vm);if(status!=VM_OK)fprintf(stderr,"I failed self-tail VM: %u %s\n",status,vm.error_msg);
    CHECK(status==VM_OK);CHECK(vm.globals[1].tag==TAG_INT && vm.globals[1].as.i64==iterations);
    CHECK(vm.globals[0].tag==TAG_STRUCT && vm.globals[2].tag==TAG_STRUCT && vm.globals[0].as.sval==vm.globals[2].as.sval);
    CHECK(vm.profile.max_frame_depth<=(entry?1u:2u));vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    VmRecordArrayPrivate *p=NULL;CHECK(vm_record_array_private_create(&c->m,&p).status==NVM_ARRAY_ELIGIBLE);
    CHECK(vm_record_array_private_run(p)==VM_OK);VmRecordArrayPrivateStats stats;CHECK(vm_record_array_private_stats(p,&stats));
    CHECK(stats.maximum_frames<=(entry?1u:2u));
    VmRecordArrayObservation counter,a,b;
    CHECK(vm_record_array_private_observe(p,VM_RA_GLOBAL,1,NULL,0,&counter) && counter.scalar_bits==iterations);
    CHECK(vm_record_array_private_observe(p,VM_RA_GLOBAL,0,NULL,0,&a));
    CHECK(vm_record_array_private_observe(p,VM_RA_GLOBAL,2,NULL,0,&b) && a.identity==b.identity);
    vm_record_array_private_destroy(p);CHECK(!ra_live && !ra_bytes);
#endif
}
static void self_allocation_prefixes(void) {
    Input c;self_fixture(&c,TAG_STRING,17,3,false);
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
    printf("I checked %zu self-tail proof allocation positions in both failure modes.\n",proof_total);
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
            fprintf(stderr,"I failed self-tail allocation %zu/%zu mode %d: %u %s\n",fail,total,persistent,result,vm.error_msg);
        CHECK(result==VM_OK || result==VM_ERR_MEMORY || refused);
        CHECK(vm.frame_count==0);if(result!=VM_OK)CHECK(vm.stack_size==0);
        ra_fail=SIZE_MAX;ra_persistent=0;vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
        vm_init(&vm,&c.m);CHECK(vm_execute(&vm)==VM_OK);vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
    }
    printf("I checked %zu self-tail allocation positions in transient and persistent modes with independent recovery.\n",total);
}

static void private_staging_prefixes(void) {
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    Input c;self_fixture(&c,TAG_STRING,17,3,true);VmRecordArrayPrivate *p=NULL;
    CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
    CHECK(vm_record_array_private_run(p)==VM_OK);ra_calls=0;
    CHECK(vm_record_array_private_run(p)==VM_OK);size_t total=ra_calls;
    vm_record_array_private_destroy(p);CHECK(total && !ra_live && !ra_bytes);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<total;fail++) {
        CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
        CHECK(vm_record_array_private_run(p)==VM_OK);
        VmRecordArrayPrivateStats before,after;VmRecordArrayObservation old,retained;
        CHECK(vm_record_array_private_stats(p,&before));
        CHECK(vm_record_array_private_observe(p,VM_RA_RESULT,0,NULL,0,&old));
        ra_calls=0;ra_fail=fail;ra_persistent=(int)persistent;
        CHECK(vm_record_array_private_run(p)==VM_ERR_MEMORY && ra_calls>fail);
        ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(vm_record_array_private_stats(p,&after));
        CHECK(!after.active_frames && !after.active_stack && after.maximum_frames==1);
        CHECK(after.has_result);
        CHECK(vm_record_array_private_observe(p,VM_RA_RESULT,0,NULL,0,&retained));
        CHECK(retained.tag==old.tag && retained.scalar_bits==old.scalar_bits);
        CHECK(vm_record_array_private_run(p)==VM_OK);
        CHECK(vm_record_array_private_stats(p,&after));
        CHECK(after.heap_objects==before.heap_objects && after.heap_live_bytes==before.heap_live_bytes);
        vm_record_array_private_destroy(p);CHECK(!ra_live && !ra_bytes);
    }
    printf("I checked %zu private self-tail staging positions with previous result preservation.\n",total);
#endif
}
static void failure_after_restart(void) {
    for(unsigned bounds=0;bounds<2;bounds++) {
        Input c;self_fixture(&c,TAG_INT,1,3,true);
        uint32_t begin=c.fn[2].code_offset,end=begin+c.fn[2].code_length;
        for(uint32_t pc=begin;pc<end;) {
            DecodedInstruction ins;uint32_t width=isa_decode(c.code+pc,end-pc,&ins);CHECK(width);
            if(ins.opcode==OP_PUSH_BOOL && c.code[pc+width]==OP_ASSERT)c.code[pc+1]=0;
            pc+=width;
        }
        if(bounds) {
            uint8_t saved[4096];size_t length=c.fn[2].code_length;CHECK(length<sizeof saved);memcpy(saved,c.code+begin,length);
            c.n=begin;arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_AGG_GET,0);number(&c,999);number(&c,0);op(&c,OP_ARR_SET);op(&c,OP_POP);
            memcpy(c.code+c.n,saved,length);c.n+=length;c.fn[2].code_length=(uint32_t)c.n-begin;authority(&c);
        }
        VmResult expected=bounds?VM_ERR_OUT_OF_BOUNDS:VM_ERR_ASSERT_FAILED;
        CHECK(nvm_verify(&c.m).ok);VmState vm;vm_init(&vm,&c.m);
        VmResult actual=vm_execute(&vm);if(actual!=expected)fprintf(stderr,"I expected restart failure %u, got %u (bounds=%u): %s\n",expected,actual,bounds,vm.error_msg);
        CHECK(actual==expected);CHECK(vm.globals[1].tag==TAG_INT && vm.globals[1].as.i64==3);
        CHECK(!vm.frame_count && !vm.stack_size);vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
        VmRecordArrayPrivate *p=NULL;CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
        CHECK(vm_record_array_private_run(p)==expected);VmRecordArrayPrivateStats stats;
        CHECK(vm_record_array_private_stats(p,&stats));
        CHECK(!stats.active_frames && !stats.active_stack && !stats.has_result && stats.maximum_frames==1);
        vm_record_array_private_destroy(p);CHECK(!ra_live && !ra_bytes);
#endif
    }
}
#ifndef RA_SELF_MAIN
#define RA_SELF_MAIN main
#endif
int RA_SELF_MAIN(int argc,char **argv) {
    if(argc==2 && !strcmp(argv[1],"allocations")){self_allocation_prefixes();private_staging_prefixes();failure_after_restart();return 0;}
    CHECK(argc==2);unsigned id=0;
    for(uint8_t leaf=TAG_INT;leaf<=TAG_STRING;leaf++)for(unsigned width=16;width<=17;width++) {
        Input c;self_fixture(&c,leaf,width,3,false);self_execute(&c,3,false);public_routes(&c,argv[1],id++);
    }
    for(unsigned entry=0;entry<2;entry++)for(unsigned n=0;n<2;n++) {
        unsigned iterations=n?4096:0;Input c;self_fixture(&c,TAG_STRING,1,iterations,entry!=0);
        self_execute(&c,iterations,entry!=0);public_routes(&c,argv[1],id++);
    }
    printf("PASS %u self-tail runtime checks; %u native fixtures\n",checks,id);return 0;
}
