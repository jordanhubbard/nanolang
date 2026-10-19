/* I exercise actual public consumers of complete array-field authority. */
#define main retained_origin_fixture_main
#include "test_record_array_origins.c"
#undef main
#include "../../src/nanoisa/record_array_admission.h"
#include "../../src/nanoisa/nvm2c.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/disassembler.h"
#include "../../src/nanovm/vm.h"
int g_argc=0;char **g_argv=NULL;
static void execute(const NvmModule *module) {
    VmState vm;vm_init(&vm,module);
    CHECK(vm_call_function(&vm,module->header.entry_point,NULL,0)==VM_ERR_TYPE_ERROR);
    CHECK(vm.stack_size==0 && vm.frame_count==0);
    NanoValue output=val_int(771);
    CHECK(vm_invoke(&vm,module->header.entry_point,NULL,0,&output)==VM_ERR_TYPE_ERROR);
    CHECK(output.tag==TAG_INT && output.as.i64==771);
    CHECK(vm_invoke_callable(&vm,val_function(module->header.entry_point),NULL,0,&output)==VM_ERR_TYPE_ERROR);
    CHECK(output.tag==TAG_INT && output.as.i64==771);
    CHECK(vm_core_execute(&vm).type==TRAP_ERROR);
    VmResult status=vm_execute(&vm);
    if(status!=VM_OK)fprintf(stderr,"I failed public VM: %s\n",vm.error_msg);
    CHECK(status==VM_OK && vm.stack_size==1 && vm.stack[0].tag==TAG_INT);
    CHECK(vm.stack[0].as.i64==0 && vm.frame_count==0);
    vm_destroy(&vm);
    CHECK(vm.heap.stats.num_objects==0);
}
static void assembly_roundtrip(const NvmModule *module) {
    char *text=disasm_module_styled(module,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult result;NvmModule *copy=asm_assemble(text,&result);
    if(!copy)fprintf(stderr,"I failed canonical assembly: %s\n",result.message);
    CHECK(copy);
    CHECK(copy->layout_size==module->layout_size && !memcmp(copy->layout_data,module->layout_data,module->layout_size));
    CHECK(copy->ownership_size==module->ownership_size && !memcmp(copy->ownership_data,module->ownership_data,module->ownership_size));
    execute(copy);nvm_module_free(copy);free(text);
}
static void integer(Input *c,int64_t value) {
    op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)b8(c->code,&c->n,(uint8_t)((uint64_t)value>>(8*i)));
}
static void public_basic(Input *c,uint8_t tag,bool interleaved) {
    init(c,tag,interleaved);array(c,tag);scalar(c,tag);op(c,OP_ARR_PUSH);
    op(c,OP_DUP);arg16(c,OP_STORE_LOCAL,0);
    op(c,OP_AGG_PACK);b8(c->code,&c->n,AGG_RECORD);b32(c->code,&c->n,0);
    b16(c->code,&c->n,0);b16(c->code,&c->n,1);arg16(c,OP_STORE_LOCAL,1);
    /* I mutate through the retained field and observe the original alias. */
    arg16(c,OP_LOAD_LOCAL,1);arg16(c,OP_AGG_GET,0);scalar(c,tag);op(c,OP_ARR_PUSH);op(c,OP_POP);
    arg16(c,OP_LOAD_LOCAL,0);op(c,OP_ARR_LEN);integer(c,2);op(c,OP_EQ);op(c,OP_ASSERT);
    finish(c);
}
static void aggregate(Input *c, uint32_t ordinal) {
    op(c,OP_AGG_PACK);b8(c->code,&c->n,AGG_RECORD);b32(c->code,&c->n,ordinal);
    b16(c->code,&c->n,0);b16(c->code,&c->n,1);
}
static void fail_assertion(Input *c) {
    op(c,OP_PUSH_BOOL);b8(c->code,&c->n,0);op(c,OP_ASSERT);
}
static void public_flow(Input *c,uint8_t tag,unsigned fail_at) {
    init(c,tag,false);
    /* My initializer owns the original array; a helper mutates and returns
     * the shared alias, then nested records retain that same allocation. */
    arg32(c,OP_LOAD_GLOBAL,0);arg32(c,OP_CALL,1);
    aggregate(c,1);aggregate(c,0);arg32(c,OP_STORE_GLOBAL,1);
    arg32(c,OP_LOAD_GLOBAL,1);arg16(c,OP_AGG_GET,0);arg16(c,OP_AGG_GET,0);
    op(c,OP_ARR_LEN);integer(c,2);op(c,OP_EQ);op(c,OP_ASSERT);
    arg32(c,OP_LOAD_GLOBAL,0);scalar(c,tag);op(c,OP_ARR_PUSH);op(c,OP_POP);
    arg32(c,OP_LOAD_GLOBAL,1);arg16(c,OP_AGG_GET,0);arg16(c,OP_AGG_GET,0);
    op(c,OP_ARR_LEN);integer(c,3);op(c,OP_EQ);op(c,OP_ASSERT);
    finish(c);
    c->m.function_count=3;
    c->fn[1].name_idx=2;c->fn[1].code_offset=(uint32_t)c->n;
    c->fn[1].local_count=c->fn[1].arity=1;
    c->fn[1].result_count=1;c->fn[1].result_tag=TAG_ARRAY;
    c->params[0]=TAG_ARRAY;c->param_rows[1]=c->params;c->m.function_param_types=c->param_rows;
    arg16(c,OP_LOAD_LOCAL,0);scalar(c,tag);op(c,OP_ARR_PUSH);
    if(fail_at==1)fail_assertion(c);
    op(c,OP_RET);c->fn[1].code_length=(uint32_t)c->n-c->fn[1].code_offset;
    c->fn[2].name_idx=3;c->fn[2].code_offset=(uint32_t)c->n;
    c->fn[2].result_count=1;c->fn[2].result_tag=TAG_INT;
    array(c,tag);scalar(c,tag);op(c,OP_ARR_PUSH);arg32(c,OP_STORE_GLOBAL,0);
    if(fail_at==2)fail_assertion(c);
    integer(c,0);op(c,OP_RET);c->fn[2].code_length=(uint32_t)c->n-c->fn[2].code_offset;
    authority(c);
    /* My first record is a forward reference to the second record. Only
     * that inner record declares an array field, so I retain its binding. */
    c->layouts[12]=TAG_STRUCT;size_t at=16;b32(c->layouts,&at,1);
    at=c->o-44;b32(c->ownership,&at,28);
    at=c->o-28;b32(c->ownership,&at,1);
    memmove(c->ownership+c->o-24,c->ownership+c->o-12,12);
    c->o-=12;c->m.ownership_size=(uint32_t)c->o;
}
static void exercise_flow(uint8_t tag,const char *directory) {
    Input c;public_flow(&c,tag,0);
    NvmRecordArrayExecutionPlan *plan=NULL;
    NvmArrayEligibilityResult admitted=nvm_record_array_admit(&c.m,&plan);
    if(admitted.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"I failed public flow admission: %s\n",admitted.message);
    CHECK(admitted.status==NVM_ARRAY_ELIGIBLE);
    NvmRecordArrayExecutionCounts counts;CHECK(nvm_record_array_execution_counts(plan,&counts));
    CHECK(counts.functions==3 && counts.initializer==2 && counts.globals==2 && counts.records==2);
    nvm_record_array_execution_free(plan);
    execute(&c.m);assembly_roundtrip(&c.m);
    NvmV2Module wire;NvmV2Result converted=nvm_v2_from_nvm_module(&c.m,&wire);
    if(converted!=NVM_V2_OK)fprintf(stderr,"I failed public flow forward conversion with status %u\n",converted);
    CHECK(converted==NVM_V2_OK);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&wire,&copy)==NVM_V2_OK);
    execute(copy);nvm_module_free(copy);
    size_t wire_size=0;CHECK(nvm_v2_module_serialize(&wire,NULL,0,&wire_size)==NVM_V2_OK);
    uint8_t *wire_bytes=malloc(wire_size);CHECK(wire_bytes);
    CHECK(nvm_v2_module_serialize(&wire,wire_bytes,wire_size,NULL)==NVM_V2_OK);
    NvmV2Module decoded;CHECK(nvm_v2_module_deserialize(wire_bytes,wire_size,&decoded)==NVM_V2_OK);
    CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);
    execute(copy);nvm_module_free(copy);nvm_v2_module_free(&decoded);
    free(wire_bytes);nvm_v2_module_free(&wire);
    char error[256];char *source=nvm2c_emit(&c.m,error,sizeof error);
    if(!source)fprintf(stderr,"I failed public flow native translation: %s\n",error);
    CHECK(source);char path[1024];snprintf(path,sizeof path,"%s/flow-%u.c",directory,tag);
    FILE *file=fopen(path,"w");CHECK(file);CHECK(fputs(source,file)>=0);CHECK(fclose(file)==0);free(source);
    for(unsigned failure=1;failure<=2;failure++) {
        public_flow(&c,tag,failure);
        CHECK(nvm_verify(&c.m).ok);
        VmState vm;vm_init(&vm,&c.m);
        CHECK(vm_execute(&vm)==VM_ERR_ASSERT_FAILED);
        CHECK(vm.stack_size==0 && vm.frame_count==0);
        /* My failed prefix never publishes the nested record. The initialized
         * global array remains VM-owned until ordinary instance destruction. */
        CHECK(vm.global_count==1 && vm.globals[0].tag==TAG_ARRAY);
        vm_destroy(&vm);CHECK(vm.heap.stats.num_objects==0);
    }
}

int main(int argc,char **argv) {
    CHECK(argc==2);
    for(uint8_t tag=TAG_INT;tag<=TAG_STRING;tag++) {
        Input c;public_basic(&c,tag,false);
        NvmRecordArrayExecutionPlan *plan=NULL;
        NvmArrayEligibilityResult admission=nvm_record_array_admit(&c.m,&plan);
        if(admission.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"I failed admission: %s\n",admission.message);
        CHECK(admission.status==NVM_ARRAY_ELIGIBLE);nvm_record_array_execution_free(plan);
        CHECK(nvm_verify(&c.m).ok);execute(&c.m);assembly_roundtrip(&c.m);
        NvmV2Module v2;CHECK(nvm_v2_from_nvm_module(&c.m,&v2)==NVM_V2_OK);
        NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&v2,&copy)==NVM_V2_OK);execute(copy);
        nvm_module_free(copy);
        size_t encoded_size=0;CHECK(nvm_v2_module_serialize(&v2,NULL,0,&encoded_size)==NVM_V2_OK);
        uint8_t *encoded=malloc(encoded_size);CHECK(encoded);
        CHECK(nvm_v2_module_serialize(&v2,encoded,encoded_size,&encoded_size)==NVM_V2_OK);
        NvmV2Layout *saved_layouts=v2.layouts.items;
        CHECK(v2.layouts.count>0);v2.layouts.items=NULL;
        NvmModule *malformed_copy=(void *)(uintptr_t)1;
        CHECK(nvm_v2_to_nvm_module(&v2,&malformed_copy)==NVM_V2_ERR_INDEX_RANGE);
        CHECK(malformed_copy==NULL);v2.layouts.items=saved_layouts;
        uint16_t declared=v2.functions.items[0].max_stack;CHECK(declared>1);
        uint8_t *previous=malloc(encoded_size);CHECK(previous);memcpy(previous,encoded,encoded_size);
        v2.functions.items[0].max_stack=1;
        NvmModule *refused_copy=(void *)(uintptr_t)1;
        CHECK(nvm_v2_to_nvm_module(&v2,&refused_copy)!=NVM_V2_OK && refused_copy==NULL);
        CHECK(nvm_v2_module_serialize(&v2,encoded,encoded_size,NULL)!=NVM_V2_OK);
        CHECK(!memcmp(previous,encoded,encoded_size));
        v2.functions.items[0].max_stack=declared;free(previous);
        NvmV2Module decoded;CHECK(nvm_v2_module_deserialize(encoded,encoded_size,&decoded)==NVM_V2_OK);
        CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);execute(copy);
        nvm_module_free(copy);nvm_v2_module_free(&decoded);nvm_v2_module_free(&v2);free(encoded);
        uint32_t legacy_size=77;CHECK(!nvm_serialize(&c.m,&legacy_size) && legacy_size==0);
        char error[256];char *source=nvm2c_emit(&c.m,error,sizeof error);
        if(!source)fprintf(stderr,"I failed native translation: %s\n",error);
        CHECK(source);char path[1024];snprintf(path,sizeof path,"%s/array-%u.c",argv[1],tag);
        FILE *file=fopen(path,"w");CHECK(file);CHECK(fputs(source,file)>=0);CHECK(fclose(file)==0);free(source);
        /* I retain first failure and drain frame/stack owners at a failed assertion. */
        NvmRecordArrayExecutionPlan *failure_plan=NULL;
        CHECK(nvm_record_array_admit(&c.m,&failure_plan).status==NVM_ARRAY_ELIGIBLE);
        NvmRecordArrayExecutionCounts counts;CHECK(nvm_record_array_execution_counts(failure_plan,&counts));
        uint32_t comparison=NO;
        for(uint32_t i=0;i<counts.instructions;i++) {
            NvmRecordArrayExecutionInstruction instruction;
            CHECK(nvm_record_array_execution_instruction(failure_plan,i,&instruction));
            if(instruction.opcode==OP_EQ){comparison=instruction.pc;break;}
        }
        nvm_record_array_execution_free(failure_plan);CHECK(comparison!=NO);
        c.code[comparison]=OP_NE;
        VmState failing;vm_init(&failing,&c.m);size_t baseline=failing.heap.stats.num_objects;
        CHECK(vm_execute(&failing)==VM_ERR_ASSERT_FAILED);
        vm_gc_collect_cycles(&failing.heap);
        CHECK(failing.stack_size==0 && failing.frame_count==0 && failing.heap.stats.num_objects==baseline);
        vm_destroy(&failing);CHECK(failing.heap.stats.num_objects==0);
        c.code[comparison]=OP_EQ;
        /* I honor the supported CODE mutation boundary before re-admission. */
        VmState rebuilt;vm_init(&rebuilt,&c.m);vm_invalidate_module(&rebuilt,&c.m);
        CHECK(c.code[c.n-10]==OP_PUSH_I64 && c.code[c.n-1]==OP_RET);
        c.code[c.n-9]=17;
        CHECK(vm_rebuild_module(&rebuilt,&c.m));CHECK(vm_execute(&rebuilt)==VM_OK);
        CHECK(rebuilt.stack_size==1 && rebuilt.stack[0].tag==TAG_INT && rebuilt.stack[0].as.i64==17);
        vm_destroy(&rebuilt);CHECK(rebuilt.heap.stats.num_objects==0);c.code[c.n-9]=0;
        /* I reclassify changed original bytes; a prior plan is not cached trust. */
        VmState existing;vm_init(&existing,&c.m);
        CHECK(existing.decoded_module_valid);
        size_t retained_constants=existing.heap.stats.num_objects;
        c.code[1]=TAG_STRUCT;
        CHECK(vm_execute(&existing)!=VM_OK);
        CHECK(existing.stack_size==0 && existing.frame_count==0 && existing.global_count==0);
        CHECK(existing.heap.stats.num_objects==retained_constants);
        vm_destroy(&existing);CHECK(existing.heap.stats.num_objects==0);
        CHECK(!nvm_verify(&c.m).ok);CHECK(!nvm2c_emit(&c.m,error,sizeof error));
        VmState refused;vm_init(&refused,&c.m);CHECK(vm_execute(&refused)!=VM_OK);vm_destroy(&refused);
        public_basic(&c,tag,true);CHECK(!nvm_verify(&c.m).ok);
        basic(&c,tag,false);CHECK(!nvm_verify(&c.m).ok);
    }
    for(uint8_t tag=TAG_INT;tag<=TAG_STRING;tag++)exercise_flow(tag,argv[1]);
    printf("PASS %u public record-array checks\n",checks);return 0;
}
