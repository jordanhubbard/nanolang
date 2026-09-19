/* I execute only modules accepted by the complete public conjunction. */
#define main previous_shape_fixture_main
#include "test_mixed_float_proof.c"
#undef main
#include "nvm2c.h"
#include "disassembler.h"
#include "../../src/nanovm/vm.h"
int g_argc=0;char **g_argv=NULL;
static const char *consume=".function close 1 1 0 int 1\nOWN_UNPACK_LOCAL 0\nRET\n.end\n.parameters 1 struct\n";
static uint8_t *runtime_row(NvmModule *m,unsigned function) {
    uint8_t *row=m->ownership_data+16;
    for(unsigned f=0;f<function;f++)row+=4+8*(m->functions[f].local_count+1);
    return row;
}
static NvmModule *runtime_fixture(unsigned index) {
    const char *operations[]={
        "LOAD_LOCAL 1\nPUSH_F64 2.5\nARR_PUSH\nSTORE_LOCAL 2\n"
        "LOAD_LOCAL 2\nPUSH_I64 0\nPUSH_F64 9.5\nARR_SET\nPOP\n"
        "LOAD_LOCAL 1\nARR_LEN\nPUSH_I64 2\nEQ\nASSERT\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 0\nARR_GET\nPUSH_F64 9.5\nF64_EQ\nASSERT\n"
        "ARR_NEW 3\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 1\nPUSH_I64 1\nARR_GET\nPUSH_F64 2.5\nF64_EQ\nASSERT\n"
        "LOAD_LOCAL 1\nPUSH_I64 -1\nARR_GET\nPUSH_VOID\nEQ\nASSERT\n",
        "LOAD_LOCAL 1\nPUSH_I64 99\nARR_GET\nPUSH_F64 1.0\nF64_ADD\nPOP\n",
        "PUSH_F64 1.0\nLOAD_LOCAL 1\nPUSH_I64 99\nARR_GET\nF64_SUB\nPOP\n",
        "LOAD_LOCAL 1\nPUSH_I64 99\nARR_GET\nLOAD_LOCAL 1\nPUSH_I64 -1\nARR_GET\nF64_EQ\nPOP\n",
        "PUSH_BOOL 0\nASSERT\n",
        "LOAD_LOCAL 1\nPUSH_I64 99\nARR_GET\nPUSH_F64 0.0\nNE\nASSERT\n"
        "PUSH_F64 -0.0\nPUSH_F64 0.0\nEQ\nASSERT\n",
        "PUSH_F64 nan\nPUSH_F64 nan\nNE\nASSERT\n"
        "PUSH_F64 nan\nPUSH_F64 1.0\nLE\nASSERT\n"
        "PUSH_F64 nan\nPUSH_F64 1.0\nF64_LE\nBOOL_NOT\nASSERT\n"
        "PUSH_F64 nan\nPUSH_F64 -0.0\nF64_DIV\nPUSH_F64 0.0\nF64_EQ\nASSERT\n"
        "PUSH_F64 1.5\nPUSH_F64 2.5\nF64_ADD\nPUSH_F64 4.0\nF64_EQ\nASSERT\n"
    };
    CHECK(index<12);
    char body[4096];snprintf(body,sizeof body,
        "PUSH_F64 1.5\nARR_LITERAL 3 1\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nSTORE_LOCAL 2\n"
        "PUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 3\n%s"
        "OWN_MOVE_LOCAL 3\nCALL 1\nPUSH_I64 7\nEQ\nASSERT\nPUSH_I64 0\nRET\n",operations[index==9?6:index<6?index:0]);
    Type locals[]={{TAG_STRUCT,1},SCALAR(TAG_ARRAY),SCALAR(TAG_ARRAY),{TAG_STRUCT,0}};
    if(index==6 || index==10) {
        char helpers[6000];snprintf(helpers,sizeof helpers,"%s.function original_main 0 4 0 int 1\n%s.end\n",consume,body);
        if(index==10)strcat(helpers,".function forward_main 0 0 0 int 1\nCALL 2\nRET\n.end\n");
        const char *entry=index==10?"PUSH_I64 7\nOWN_PACK 0\nCALL 1\nPUSH_I64 7\nEQ\nASSERT\nCALL 3\nPUSH_I64 0\nEQ\nASSERT\nPUSH_I64 0\nRET\n":"PUSH_I64 7\nOWN_PACK 0\nCALL 1\nPUSH_I64 7\nEQ\nASSERT\nCALL 2\nPUSH_I64 0\nEQ\nASSERT\nPUSH_I64 0\nRET\n";
        NvmModule *m=build(entry,NULL,0,helpers,false);
        uint8_t *row=runtime_row(m,2)+12;
        for(unsigned n=0;n<4;n++){row[n*8]=locals[n].tag;word(row+n*8+4,locals[n].layout);}
        return m;
    }
    if(index==7) {
        const char *needle="OWN_MOVE_LOCAL 3\nCALL 1";char *at=strstr(body,needle);CHECK(at);
        char tail[512];snprintf(tail,sizeof tail,"%s",at);
        snprintf(at,sizeof body-(size_t)(at-body),"CALL 2\nCALL 3\nCALL 1\nPUSH_I64 11\nEQ\nASSERT\n%s",tail);
        char helpers[1200];snprintf(helpers,sizeof helpers,"%s.function factory 0 0 0 struct 1\nPUSH_I64 11\nOWN_PACK 0\nRET\n.end\n.function relay 1 1 0 struct 1\nOWN_MOVE_LOCAL 0\nRET\n.end\n.parameters 3 struct\n",consume);
        NvmModule *m=build(body,locals,4,helpers,false);
        word(runtime_row(m,2)+8,0);word(runtime_row(m,3)+8,0);
        return m;
    }
    if(index==11) {
        const char *needle="OWN_MOVE_LOCAL 3\nCALL 1";char *at=strstr(body,needle);CHECK(at);
        char tail[512];snprintf(tail,sizeof tail,"%s",at);
        snprintf(at,sizeof body-(size_t)(at-body),
            "PUSH_I64 0\nSTORE_LOCAL 4\nrepeat:\nLOAD_LOCAL 4\nPUSH_I64 2\nLT\nJMP_FALSE repeated\n"
            "ARR_NEW 3\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\nLOAD_LOCAL 4\nPUSH_I64 1\nADD\nSTORE_LOCAL 4\nJMP repeat\n"
            "repeated:\nPUSH_BOOL 0\nJMP_FALSE skipped\nARR_NEW 3\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\nskipped:\n%s",tail);
        Type repeated[]={{TAG_STRUCT,1},SCALAR(TAG_ARRAY),SCALAR(TAG_ARRAY),{TAG_STRUCT,0},SCALAR(TAG_INT)};
        return build(body,repeated,5,consume,false);
    }
    const char *helper=index==8?".function close 1 1 0 int 1\nPUSH_BOOL 0\nASSERT\nOWN_UNPACK_LOCAL 0\nRET\n.end\n.parameters 1 struct\n":consume;
    return build(body,locals,4,helper,false);
}
static void runtime_artifacts(NvmModule *m,const char *dir,unsigned index) {
    NvmV2Module v2;size_t size;CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    char path[1024];snprintf(path,sizeof path,"%s/case%u.nvm",dir,index);
    FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(bytes,1,size,f)==size);CHECK(!fclose(f));free(bytes);nvm_v2_module_free(&v2);
    char error[256];char *source=nvm2c_emit(m,error,sizeof error);if(!source)fprintf(stderr,"native: %s\n",error);CHECK(source);
    snprintf(path,sizeof path,"%s/case%u.c",dir,index);f=fopen(path,"w");CHECK(f);CHECK(fputs(source,f)>=0);CHECK(!fclose(f));free(source);
}
static VmResult runtime_api(VmState *vm,unsigned api,NanoValue *out) {
    return api==0?vm_invoke(vm,0,NULL,0,out):api==1?vm_execute(vm):api==2?vm_call_function(vm,0,NULL,0):vm_invoke_callable(vm,val_function(0),NULL,0,out);
}
/* I count only the finite fixture graph; a buffer entry is not an owning edge. */
static bool runtime_root_graph(VmHeap *heap,size_t baseline,bool report) {
    enum { CAP=256 };
    VmHeapHeader *nodes[CAP];uint32_t incoming[CAP]={0};unsigned count=0;
    for(uint32_t n=0;n<heap->cycle_count;n++) {
        VmHeapHeader *h=heap->cycle_buf[n];if(!h)return false;
        unsigned at=0;while(at<count && nodes[at]!=h)at++;
        if(at==count){if(count==CAP)return false;nodes[count++]=h;}
    }
    for(unsigned n=0;n<count;n++) {
        VmHeapHeader *h=nodes[n];
        if(h->obj_type==TAG_ARRAY) {
            VmArray *array=(VmArray *)h;
            if(array->elem_type!=TAG_FLOAT || !array->unboxed || array->elements)
                return false;
        } else if(h->obj_type==TAG_STRUCT) {
            VmStruct *record=(VmStruct *)h;
            if(record->field_names || (record->field_count && !record->fields))return false;
            for(uint32_t f=0;f<record->field_count;f++) {
                NanoValue value=record->fields[f];
                if(value.tag==TAG_ARRAY || value.tag==TAG_STRUCT) {
                    VmHeapHeader *child=value.as.obj;if(!child || child->obj_type!=value.tag)return false;
                    unsigned at=0;while(at<count && nodes[at]!=child)at++;
                    if(at==count){if(count==CAP)return false;nodes[count++]=child;}
                    if(incoming[at]==UINT32_MAX)return false;
                    incoming[at]++;
                } else if(value.tag!=TAG_VOID && value.tag!=TAG_INT &&
                          value.tag!=TAG_BOOL && value.tag!=TAG_U8 && value.tag!=TAG_FLOAT)
                    return false;
            }
        } else return false;
    }
    if(report)fprintf(stderr,"mixed graph allocated=%zu baseline=%zu nodes=%u buffered=%u\n",
        (size_t)heap->stats.num_objects,baseline,count,heap->cycle_count);
    if(baseline>SIZE_MAX-count || heap->stats.num_objects!=baseline+count)return false;
    bool no_external=true;
    for(unsigned n=0;n<count;n++) {
        if(report)fprintf(stderr,"mixed graph node=%u kind=%u refs=%u incoming=%u\n",n,
            (unsigned)nodes[n]->obj_type,(unsigned)nodes[n]->ref_count,incoming[n]);
        if(nodes[n]->ref_count!=incoming[n])no_external=false;
    }
    return no_external;
}
static void runtime_clean(VmState *vm,size_t baseline) {
    CHECK(!vm->stack_size && !vm->frame_count);
    CHECK(!vm->references.active && !vm->callee_references.active);
    for(unsigned f=0;f<NVM_OWNED_MAX_FUNCTIONS-2;f++)CHECK(!vm->value_references[f].active);
    CHECK(runtime_root_graph(&vm->heap,baseline,true));
    if(vm->heap.cycle_count) {
        VmHeapHeader *h=vm->heap.cycle_buf[0];
        NanoValue retained={0};retained.tag=h->obj_type;retained.as.obj=h;
        CHECK(h->ref_count<UINT32_MAX);
        vm_retain(&vm->heap,retained);
        CHECK(!runtime_root_graph(&vm->heap,baseline,false));
        vm_release(&vm->heap,retained);
        CHECK(runtime_root_graph(&vm->heap,baseline,false));
    }
    vm_gc_collect_cycles(&vm->heap);
    CHECK(vm->heap.stats.num_objects==baseline);
}
static void runtime_core(NvmModule *m,unsigned index,VmResult wanted) {
    VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);
    size_t baseline=vm.heap.stats.num_objects;
    vm.frame_count=1;vm.current_fn=0;vm.ip=m->functions[0].code_offset;
    vm.frames[0]=(VmCallFrame){.fn_idx=0,.local_count=m->functions[0].local_count,.module=m};
    vm.stack_size=m->functions[0].local_count;
    for(uint32_t n=0;n<vm.stack_size;n++)vm.stack[n]=val_void();
    VmTrap trap;unsigned traps=0;bool failed_assert=false;
    do {
        CHECK(traps++<64);trap=vm_core_execute(&vm);
        if(trap.type==TRAP_ASSERT) {
            CHECK(trap.data.assert_check.condition.tag==TAG_BOOL);
            failed_assert=!val_truthy(trap.data.assert_check.condition);
            vm_release(&vm.heap,trap.data.assert_check.condition);
            if(failed_assert)break;
        }
    } while(trap.type==TRAP_ASSERT);
    fprintf(stderr,"mixed direct core case=%u traps=%u terminal=%u\n",index,traps,(unsigned)trap.type);
    if(wanted==VM_OK) {
        CHECK(trap.type==TRAP_NONE && vm.stack_size==1);
        NanoValue value=vm.stack[--vm.stack_size];CHECK(value.tag==TAG_INT && !value.as.i64);vm_release(&vm.heap,value);
    } else if(wanted==VM_ERR_ASSERT_FAILED)CHECK(failed_assert && trap.type==TRAP_ASSERT);
    else CHECK(trap.type==TRAP_ERROR && trap.data.error.code==wanted);
    runtime_clean(&vm,baseline);vm_destroy(&vm);
}
#ifndef MIXED_RUNTIME_ALLOC_TEST

int main(int argc,char **argv) {
    CHECK(argc==2);
    for(unsigned index=0;index<12;index++) {
        NvmModule *m=runtime_fixture(index);NvmVerifyResult verified=nvm_verify(m);
        if(!verified.ok)fprintf(stderr,"case%u: %s\n",index,verified.error_msg);
        CHECK(verified.ok);
        CHECK(!nvm_verify_owned_module(m).ok);CHECK(nvm_verify_linked(m,NULL,0).ok);
        runtime_artifacts(m,argv[1],index);
        char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);AsmResult error;
        NvmModule *copy=asm_assemble(text,&error);CHECK(copy);CHECK(nvm_verify(copy).ok);
        CHECK(copy->ownership_size==m->ownership_size && !memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
        CHECK(copy->layout_size==m->layout_size && !memcmp(copy->layout_data,m->layout_data,m->layout_size));
        free(text);nvm_module_free(copy);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        VmResult wanted=(index==4||index==8)?VM_ERR_ASSERT_FAILED:index>=1&&index<=3?VM_ERR_TYPE_ERROR:VM_OK;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<2;repeat++) {
            fprintf(stderr,"mixed case=%u api=%u repeat=%u\n",index,api,repeat);
            NanoValue out=val_int(-91);VmResult result=runtime_api(&vm,api,&out);
            if(result!=wanted)fprintf(stderr,"case%u api%u repeat%u wanted%d got%d: %s\n",index,api,repeat,wanted,result,vm.error_msg);
            CHECK(result==wanted);
            if(result==VM_OK){if(api==1||api==2){CHECK(vm.stack_size==1);out=vm.stack[--vm.stack_size];}CHECK(out.tag==TAG_INT && !out.as.i64);vm_release(&vm.heap,out);}
            runtime_clean(&vm,baseline);
        }
        vm_destroy(&vm);runtime_core(m,index,wanted);nvm_module_free(m);printf("case %u %u 0\n",index,(index==4||index==8)?2:index>=1&&index<=3?3:0);
    }
    printf("%u mixed runtime lifecycle checks passed\n",checks);return 0;
}

#endif
