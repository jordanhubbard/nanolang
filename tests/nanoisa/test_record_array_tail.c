/* I qualify acyclic direct tail transfer through public record-array routes. */
#define main retained_origin_fixture_main
#include "test_record_array_origins.c"
#undef main
#include "../../src/nanoisa/record_array_admission.h"
#include "../../src/nanoisa/nvm2c.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/disassembler.h"
#include "../../src/nanovm/vm.h"
int g_argc=0;char **g_argv=NULL;
static void pack(Input *c) {
    op(c,OP_AGG_PACK);b8(c->code,&c->n,AGG_RECORD);b32(c->code,&c->n,0);
    b16(c->code,&c->n,0);b16(c->code,&c->n,1);
}
/* Three functions suffice for a CALL containing a tail, a chain ending at a
 * distinct target, and explicit self/mixed-cycle refusal controls. */
static void fixture(Input *c,uint8_t element,unsigned arity,uint8_t result,bool entry_tail) {
    static uint8_t parameters[256];memset(parameters,TAG_STRUCT,sizeof parameters);
    init(c,element,false);c->strings[3]="last";c->lengths[3]=4;
    c->m.function_count=3;c->m.function_param_types=c->param_rows;
    arg32(c,OP_LOAD_GLOBAL,0);op(c,OP_TYPE_CHECK);b8(c->code,&c->n,TAG_VOID);arg16(c,OP_STORE_LOCAL,2);
    array(c,element);scalar(c,element);op(c,OP_ARR_PUSH);pack(c);
    arg16(c,OP_STORE_LOCAL,0);
    for(unsigned i=0;i<arity;i++)arg16(c,OP_LOAD_LOCAL,0);
    arg32(c,entry_tail?OP_TAIL_CALL:OP_CALL,1);
    if(entry_tail) { c->fn[0].result_tag=result;c->fn[0].result_count=result!=TAG_VOID; }
    else {
        if(result==TAG_STRUCT)arg16(c,OP_AGG_GET,0);
        if(result==TAG_STRUCT || result==TAG_ARRAY) {
            op(c,OP_ARR_LEN);scalar(c,TAG_INT); /* compare against two after mutation */
            c->code[c->n-8]=2;op(c,OP_EQ);op(c,OP_ASSERT);
        } else if(result!=TAG_VOID)op(c,OP_POP);
        /* The pre-call predicate is no longer authority for the changed
         * global. I still consume its concrete true value after the tail. */
        arg16(c,OP_LOAD_LOCAL,2);size_t branch=c->n;arg32(c,OP_JMP_FALSE,0);
        arg32(c,OP_LOAD_GLOBAL,0);arg16(c,OP_AGG_GET,0);op(c,OP_ARR_LEN);
        scalar(c,TAG_INT);c->code[c->n-8]=2;op(c,OP_EQ);op(c,OP_ASSERT);
        size_t operand=branch+1;b32(c->code,&operand,(uint32_t)(c->n-branch));
        scalar(c,TAG_INT);op(c,OP_RET);
    }
    c->fn[0].code_length=(uint32_t)c->n;
    for(unsigned fn=1;fn<=2;fn++) {
        c->fn[fn].name_idx=fn+1;c->fn[fn].code_offset=(uint32_t)c->n;
        c->fn[fn].arity=(uint16_t)arity;c->fn[fn].local_count=(uint16_t)(arity+fn);
        c->fn[fn].result_count=result!=TAG_VOID;c->fn[fn].result_tag=result;
        c->param_rows[fn]=parameters;
        if(fn==1) {
            for(unsigned i=0;i<arity;i++)arg16(c,OP_LOAD_LOCAL,(uint16_t)i);
            arg32(c,OP_TAIL_CALL,2);
        } else {
            arg16(c,OP_LOAD_LOCAL,0);arg16(c,OP_AGG_GET,0);scalar(c,element);op(c,OP_ARR_PUSH);op(c,OP_POP);
            arg16(c,OP_LOAD_LOCAL,0);arg32(c,OP_STORE_GLOBAL,0);
            op(c,OP_PUSH_BOOL);b8(c->code,&c->n,1);op(c,OP_ASSERT);
            if(result==TAG_STRUCT || result==TAG_ARRAY) {
                arg16(c,OP_LOAD_LOCAL,(uint16_t)(arity-1));
                if(result==TAG_ARRAY)arg16(c,OP_AGG_GET,0);
            } else if(result!=TAG_VOID)scalar(c,result);
            op(c,OP_RET);
        }
        c->fn[fn].code_length=(uint32_t)c->n-c->fn[fn].code_offset;
    }
    authority(c);
}
static void forward_tail(Input *c) {
    fixture(c,TAG_STRING,17,TAG_STRUCT,false);
    uint8_t original[32768];memcpy(original,c->code,c->n);c->n=0;
    NvmFunctionEntry functions[3];memcpy(functions,c->fn,sizeof functions);
    uint32_t mapping[32769]={0};
    for(unsigned fi=0;fi<3;fi++) {
        c->fn[fi].code_offset=(uint32_t)c->n;
        for(uint32_t at=functions[fi].code_offset;at<functions[fi].code_offset+functions[fi].code_length;) {
            mapping[at]=(uint32_t)c->n;
            DecodedInstruction in;uint32_t width=isa_decode(original+at,functions[fi].code_offset+functions[fi].code_length-at,&in);CHECK(width);
            if(in.opcode==OP_AGG_PACK) {
                op(c,OP_AGG_PACK);b8(c->code,&c->n,AGG_RECORD);b32(c->code,&c->n,1);b16(c->code,&c->n,0);b16(c->code,&c->n,1);
            }
            CHECK(c->n+width<sizeof c->code);memcpy(c->code+c->n,original+at,width);c->n+=width;
            if(in.opcode==OP_AGG_GET)arg16(c,OP_AGG_GET,0);
            at+=width;
        }
        mapping[functions[fi].code_offset+functions[fi].code_length]=(uint32_t)c->n;
        c->fn[fi].code_length=(uint32_t)c->n-c->fn[fi].code_offset;
    }
    for(unsigned fi=0;fi<3;fi++)for(uint32_t at=functions[fi].code_offset;at<functions[fi].code_offset+functions[fi].code_length;) {
        DecodedInstruction in;uint32_t width=isa_decode(original+at,functions[fi].code_offset+functions[fi].code_length-at,&in);CHECK(width);
        if(in.opcode==OP_JMP_FALSE) {size_t operand=mapping[at]+1;b32(c->code,&operand,mapping[at+in.operands[0].i32]-mapping[at]);}
        at+=width;
    }
    authority(c);c->layouts[12]=TAG_STRUCT;size_t at=16;b32(c->layouts,&at,1);
    at=c->o-44;b32(c->ownership,&at,28);at=c->o-28;b32(c->ownership,&at,1);
    memmove(c->ownership+c->o-24,c->ownership+c->o-12,12);c->o-=12;c->m.ownership_size=(uint32_t)c->o;
}
static void new_record_tail(Input *c) {
    fixture(c,TAG_INT,1,TAG_STRUCT,false);
    /* My callee returns a new shell sharing the original array. */
    CHECK(c->code[c->n-1]==OP_RET);c->n--;
    arg16(c,OP_AGG_GET,0);pack(c);op(c,OP_RET);
    c->fn[2].code_length=(uint32_t)c->n-c->fn[2].code_offset;authority(c);
}
static void initializer_tail(Input *c,uint8_t element) {
    init(c,element,false);c->m.function_count=3;
    arg32(c,OP_TAIL_CALL,1);c->fn[0].code_length=(uint32_t)c->n;
    c->fn[1].name_idx=2;c->fn[1].code_offset=(uint32_t)c->n;c->fn[1].local_count=4;
    c->fn[1].result_count=1;c->fn[1].result_tag=TAG_INT;
    arg32(c,OP_LOAD_GLOBAL,0);arg16(c,OP_AGG_GET,0);scalar(c,element);op(c,OP_ARR_PUSH);op(c,OP_POP);
    op(c,OP_PUSH_BOOL);b8(c->code,&c->n,1);op(c,OP_ASSERT);
    scalar(c,TAG_INT);op(c,OP_RET);c->fn[1].code_length=(uint32_t)c->n-c->fn[1].code_offset;
    c->fn[2].name_idx=3;c->fn[2].code_offset=(uint32_t)c->n;
    c->fn[2].result_count=1;c->fn[2].result_tag=TAG_INT;
    array(c,element);pack(c);arg32(c,OP_STORE_GLOBAL,0);arg32(c,OP_TAIL_CALL,1);
    c->fn[2].code_length=(uint32_t)c->n-c->fn[2].code_offset;authority(c);
}
#ifndef RA_TAIL_GLOBAL_COUNT
#define RA_TAIL_GLOBAL_COUNT 1
#endif
#ifndef RA_TAIL_CLEAR_EXTRA_GLOBALS
#define RA_TAIL_CLEAR_EXTRA_GLOBALS ""
#endif
static void execute(const NvmModule *m) {
    VmState vm;vm_init(&vm,m);VmResult status=vm_execute(&vm);
    if(status!=VM_OK)fprintf(stderr,"I failed tail VM: %s\n",vm.error_msg);
    CHECK(status==VM_OK && vm.frame_count==0);
    CHECK(vm.stack_size==m->functions[0].result_count);
    if(vm.stack_size)CHECK(vm.stack[0].tag==m->functions[0].result_tag);
    CHECK(vm.global_count==RA_TAIL_GLOBAL_COUNT && vm.globals[0].tag==TAG_STRUCT);
    NanoValue leaf=vm.globals[0];
    for(unsigned depth=0;depth<2 && leaf.tag==TAG_STRUCT;depth++)leaf=leaf.as.sval->fields[0];
    CHECK(leaf.tag==TAG_ARRAY && leaf.as.array->length==2);
    vm_destroy(&vm);CHECK(vm.heap.stats.num_objects==0);
}
static void admitted(Input *c) {
    NvmRecordArrayExecutionPlan *p=NULL;NvmArrayEligibilityResult result=nvm_record_array_admit(&c->m,&p);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"I failed tail admission: %u %s\n",result.status,result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);CHECK(nvm_verify(&c->m).ok);
    NvmRecordArrayExecutionCounts counts;CHECK(nvm_record_array_execution_counts(p,&counts));unsigned tails=0;
    for(uint32_t i=0;i<counts.instructions;i++) {
        NvmRecordArrayExecutionInstruction in;CHECK(nvm_record_array_execution_instruction(p,i,&in));
        if(in.opcode==OP_TAIL_CALL) {
            tails++;CHECK(in.recipe==NVM_RA_ROOT_TAIL && !in.successor_count && !in.pushes);
            CHECK(in.callee<c->m.function_count && in.pops==c->fn[in.callee].arity);
            CHECK((in.obligations&(NVM_RA_STAGE_OPERANDS|NVM_RA_RETAIN_BEFORE_RELEASE|NVM_RA_FIRST_ERROR_CLEANUP))==
                  (NVM_RA_STAGE_OPERANDS|NVM_RA_RETAIN_BEFORE_RELEASE|NVM_RA_FIRST_ERROR_CLEANUP));
        }
    }
    CHECK(tails);nvm_record_array_execution_free(p);
}
static void public_routes(Input *c,const char *directory,unsigned id) {
    admitted(c);execute(&c->m);
    char *assembly=disasm_module_styled(&c->m,DISASM_STYLE_CANONICAL);CHECK(assembly);
    AsmResult assembled;NvmModule *copy=asm_assemble(assembly,&assembled);
    if(!copy)fprintf(stderr,"I failed tail assembly: %s\n",assembled.message);
    CHECK(copy);execute(copy);nvm_module_free(copy);free(assembly);
    NvmV2Module wire;CHECK(nvm_v2_from_nvm_module(&c->m,&wire)==NVM_V2_OK);
    size_t size=0;CHECK(nvm_v2_module_serialize(&wire,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&wire,bytes,size,NULL)==NVM_V2_OK);
    NvmV2Module decoded;CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);execute(copy);
    nvm_module_free(copy);nvm_v2_module_free(&decoded);nvm_v2_module_free(&wire);free(bytes);
    char error[256];char *native=nvm2c_emit(&c->m,error,sizeof error);
    if(!native)fprintf(stderr,"I failed tail native emission: %s\n",error);
    CHECK(native);char path[1024];snprintf(path,sizeof path,"%s/tail-%u.c",directory,id);
    FILE *file=fopen(path,"w");CHECK(file);CHECK(fputs("#define main fixture_main\n",file)>=0);CHECK(fputs(native,file)>=0);
    CHECK(fprintf(file,"\n#undef main\nint main(void) { nmap_collection_budget=1; int result=fixture_main(); nglobal[0]=(nmap_value){0}; " RA_TAIL_CLEAR_EXTRA_GLOBALS "nmap_collect(); if(nrec_owned_head || %s || nagg_live_bytes) abort(); return result; }\n",c->element==TAG_STRING?"nsarr_owners":"narr_owners")>=0);
    CHECK(!fclose(file));free(native);
}
static void refusal(Input *c,NvmArrayEligibilityStatus expected) {
    NvmRecordArrayExecutionPlan *p=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult r=nvm_record_array_admit(&c->m,&p);
    if(r.status!=expected)fprintf(stderr,"I expected tail refusal %u, got %u: %s\n",expected,r.status,r.message);
    CHECK(r.status==expected && p==(void *)(uintptr_t)1);CHECK(!nvm_verify(&c->m).ok);
}
#ifdef RA_TAIL_ALLOCATIONS
static void allocation_prefixes(void) {
    Input c;fixture(&c,TAG_STRING,17,TAG_STRUCT,false);
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
    printf("I checked %zu typed tail proof allocation positions in both failure modes.\n",proof_total);
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
            fprintf(stderr,"I failed tail allocation %zu/%zu mode %d: %u %s\n",fail,total,persistent,result,vm.error_msg);
        CHECK(result==VM_OK || result==VM_ERR_MEMORY || refused);
        CHECK(vm.frame_count==0);if(result!=VM_OK)CHECK(vm.stack_size==0);
        ra_fail=SIZE_MAX;ra_persistent=0;vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
        vm_init(&vm,&c.m);CHECK(vm_execute(&vm)==VM_OK);vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
    }
    printf("I checked %zu tail allocation positions in transient and persistent modes with independent recovery.\n",total);
}
#endif
#ifndef RA_TAIL_MAIN
#define RA_TAIL_MAIN main
#endif
int RA_TAIL_MAIN(int argc,char **argv) {
#ifdef RA_TAIL_ALLOCATIONS
    if(argc==2 && !strcmp(argv[1],"allocations")){allocation_prefixes();return 0;}
#endif
    CHECK(argc==2);unsigned id=0;
    for(uint8_t tag=TAG_INT;tag<=TAG_STRING;tag++)for(unsigned width=16;width<=17;width++) {
        Input c;fixture(&c,tag,width,TAG_STRUCT,false);public_routes(&c,argv[1],id++);
    }
    const uint8_t results[]={TAG_VOID,TAG_INT,TAG_ARRAY,TAG_STRUCT};
    for(unsigned i=0;i<sizeof results;i++)for(unsigned entry=0;entry<2;entry++) {
        Input c;fixture(&c,TAG_INT,1,results[i],entry!=0);
        if(entry && results[i]!=TAG_INT)refusal(&c,NVM_ARRAY_UNRESOLVED);
        else public_routes(&c,argv[1],id++);
    }
    Input c;new_record_tail(&c);public_routes(&c,argv[1],id++);
    forward_tail(&c);public_routes(&c,argv[1],id++);
    initializer_tail(&c,TAG_STRING);public_routes(&c,argv[1],id++);
    fixture(&c,TAG_INT,1,TAG_STRUCT,false);
    size_t target=c.fn[1].code_offset+4;b32(c.code,&target,1);refusal(&c,NVM_ARRAY_UNRESOLVED);
    fixture(&c,TAG_INT,1,TAG_STRUCT,false);target=c.fn[1].code_offset+4;b32(c.code,&target,99);refusal(&c,NVM_ARRAY_INVALID);
    fixture(&c,TAG_INT,1,TAG_STRUCT,false);c.fn[2].result_tag=TAG_ARRAY;authority(&c);refusal(&c,NVM_ARRAY_INVALID);
    /* I reject both pure and mixed cycles, including unreachable tail edges. */
    for(unsigned mixed=0;mixed<2;mixed++) {
        fixture(&c,TAG_INT,1,TAG_STRUCT,false);c.n=c.fn[2].code_offset;
        arg16(&c,OP_LOAD_LOCAL,0);arg32(&c,mixed?OP_CALL:OP_TAIL_CALL,1);
        if(mixed)op(&c,OP_RET);
        c.fn[2].code_length=(uint32_t)c.n-c.fn[2].code_offset;authority(&c);
        refusal(&c,NVM_ARRAY_UNRESOLVED);
    }
    fixture(&c,TAG_INT,1,TAG_STRUCT,false);
    size_t extra=c.fn[1].code_offset+3;memmove(c.code+extra+1,c.code+extra,c.n-extra);
    c.code[extra]=OP_DUP;c.n++;c.fn[1].code_length++;c.fn[2].code_offset++;authority(&c);
    refusal(&c,NVM_ARRAY_UNRESOLVED);
    fixture(&c,TAG_INT,1,TAG_STRUCT,false);c.param_rows[2][0]=TAG_INT;authority(&c);
    refusal(&c,NVM_ARRAY_UNRESOLVED);
    printf("PASS %u direct tail public checks; %u native fixtures\n",checks,id);return 0;
}
