/* I execute only the explicitly reviewed private adapter. Normal selectors
 * continue to refuse these fresh modules. */
#define main retained_origin_fixture_main
#include "test_owned_array_origins.c"
#undef main
#include "owned_array_authority.h"
#include "owned_array_runtime_private.h"
#include "../../src/nanovm/owned_array_runtime_private.h"
#include "nvm2c.h"
#include "verifier.h"
int g_argc=0;char **g_argv=NULL;
static unsigned heap_attempts,heap_fail,heap_hits;
void *private_array_malloc(size_t n){if(n && ++heap_attempts==heap_fail){heap_hits++;return NULL;}return malloc(n);}
void *private_array_calloc(size_t n,size_t s){if(n && s && ++heap_attempts==heap_fail){heap_hits++;return NULL;}return calloc(n,s);}
void *private_array_realloc(void *p,size_t n){if(n && ++heap_attempts==heap_fail){heap_hits++;return NULL;}return realloc(p,n);}
static const char *factory=
    "PUSH_F64 1.5\nPUSH_F64 2.5\nARR_LITERAL 3 2\nSTORE_LOCAL 0\n"
    "PUSH_I64 7\nOWN_PACK 0\nLOAD_LOCAL 0\nPUSH_STR value\nOWN_PACK 1\n"
    "PUSH_I64 8\nOWN_PACK 0\nLOAD_LOCAL 0\nPUSH_STR value\nOWN_PACK 1\nOWN_PACK 2\nRET\n";
static NvmModule *runtime_module(unsigned which) {
    const char *body=
        "PUSH_I64 7\nPRINTLN\nCALL 1\nCALL 2\nOWN_STORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 1\nSTORE_LOCAL 1\n"
        "LOAD_LOCAL 1\nPUSH_I64 0\nPUSH_F64 4.5\nARR_SET\nPOP\n"
        "LOAD_LOCAL 0\nAGG_GET 1\nAGG_GET 1\nPUSH_I64 0\nARR_GET\nPUSH_F64 4.5\nF64_EQ\nASSERT\n"
        "OWN_UNPACK_LOCAL 0\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 4\n"
        "OWN_UNPACK_LOCAL 3\nSTORE_LOCAL 2\nPOP\nOWN_STORE_LOCAL 5\nOWN_UNPACK_LOCAL 5\nPOP\n"
        "OWN_UNPACK_LOCAL 4\nPOP\nPOP\nOWN_STORE_LOCAL 5\nOWN_UNPACK_LOCAL 5\nPOP\n"
        "LOAD_LOCAL 2\nCALL 3\nPUSH_I64 42\nI64_EQ\nASSERT\n"
        "PUSH_I64 0\nSTORE_LOCAL 6\nagain:\nLOAD_LOCAL 6\nPUSH_I64 12\nI64_LT_S\nJMP_FALSE done\n"
        "LOAD_LOCAL 1\nPUSH_F64 3.5\nARR_PUSH\nPOP\nLOAD_LOCAL 6\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 6\nJMP again\n"
        "done:\nLOAD_LOCAL 1\nARR_LEN\nPUSH_I64 14\nI64_EQ\nASSERT\n"
        "LOAD_LOCAL 1\nPUSH_I64 0\nPUSH_F64 9.5\nARR_SET\nPOP\n"
        "LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_F64 9.5\nF64_EQ\nASSERT\nPUSH_I64 0\nRET\n";
    char alternate[2048];
    if(which) {
        const char *operation=which==1?"LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_F64 1.5\nF64_EQ\nASSERT\n":
            which==2?"PUSH_F64 1.5\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nF64_EQ\nASSERT\n":
            which==3?"LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nF64_EQ\nASSERT\n":
            which==4?"PUSH_BOOL 0\nASSERT\n":
            which==5?"LOAD_LOCAL 1\nPUSH_I64 0\nPUSH_F64 1.5\nARR_SET\nPOP\n":
            "LOAD_LOCAL 1\nARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n";
        snprintf(alternate,sizeof alternate,"PUSH_I64 7\nPRINTLN\nARR_NEW 3\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 7\n"
            "LOAD_LOCAL 7\nAGG_GET 0\nSTORE_LOCAL 1\n%sOWN_UNPACK_LOCAL 7\nPOP\nPOP\nOWN_PACK 4\nOWN_STORE_LOCAL 5\nOWN_UNPACK_LOCAL 5\nPUSH_I64 0\nRET\n",operation);
        body=alternate;
    }
    Function f[]={
        {body,0,8,T(TAG_INT),{OWNER(2),T(TAG_ARRAY),T(TAG_STRING),OWNER(1),OWNER(1),OWNER(0),T(TAG_INT),OWNER(3)}},
        {factory,0,1,OWNER(2),{T(TAG_ARRAY)}},
        {"OWN_MOVE_LOCAL 0\nRET\n",1,1,OWNER(2),{OWNER(2)}},
        {"LOAD_LOCAL 0\nPOP\nPUSH_I64 42\nRET\n",1,1,T(TAG_INT),{T(TAG_STRING)}}
    };
    /* The empty owner control has its own exact local declaration. */
    if(which)f[0].locals[5]=(Type)OWNER(4);
    return build(f,4);
}
/* I account for this fixture's finite graph before asking GC to reclaim it.
 * Module constants are the only permitted external heap roots. */
static bool no_extra_roots(VmState *vm,size_t baseline,bool report) {
    enum { CAP=256 };VmHeapHeader *nodes[CAP];
    uint32_t edges[CAP]={0},module_roots[CAP]={0};unsigned count=0,constant_nodes=0;
    for(uint32_t n=0;n<vm->module_constants.count;n++) {
        VmHeapHeader *h=(VmHeapHeader *)vm->module_constants.strings[n];
        if(!h || h->obj_type!=TAG_STRING)return false;
        unsigned at=0;while(at<count && nodes[at]!=h)at++;
        if(at==count){if(count==CAP)return false;nodes[count++]=h;constant_nodes++;}
        if(module_roots[at]==UINT32_MAX)return false;
        module_roots[at]++;
    }
    if(constant_nodes!=baseline)return false;
    for(uint32_t n=0;n<vm->heap.cycle_count;n++) {
        VmHeapHeader *h=vm->heap.cycle_buf[n];if(!h || !h->buffered)return false;
        unsigned at=0;while(at<count && nodes[at]!=h)at++;
        if(at==count){if(count==CAP)return false;nodes[count++]=h;}
    }
    for(unsigned n=0;n<count;n++) {
        VmHeapHeader *h=nodes[n];
        if(h->obj_type==TAG_STRING) {if(!module_roots[n])return false;}
        else if(h->obj_type==TAG_ARRAY) {
            VmArray *array=(VmArray *)h;
            if(array->elem_type!=TAG_FLOAT || !array->unboxed || array->elements)return false;
        } else if(h->obj_type==TAG_STRUCT) {
            VmStruct *record=(VmStruct *)h;
            if(record->field_names || (record->field_count && !record->fields))return false;
            for(uint32_t f=0;f<record->field_count;f++) {
                NanoValue value=record->fields[f];
                if(value.tag==TAG_ARRAY || value.tag==TAG_STRUCT || value.tag==TAG_STRING) {
                    VmHeapHeader *child=value.as.obj;if(!child || child->obj_type!=value.tag)return false;
                    unsigned at=0;while(at<count && nodes[at]!=child)at++;
                    if(value.tag==TAG_STRING && (at==count || !module_roots[at]))return false;
                    if(at==count){if(count==CAP)return false;nodes[count++]=child;}
                    if(edges[at]==UINT32_MAX)return false;
                    edges[at]++;
                } else if(value.tag!=TAG_VOID && value.tag!=TAG_INT && value.tag!=TAG_BOOL && value.tag!=TAG_U8 && value.tag!=TAG_FLOAT)return false;
            }
        } else return false;
    }
    if(report)fprintf(stderr,"private graph objects=%zu constants=%u total=%u buffered=%u\n",vm->heap.stats.num_objects,constant_nodes,count,vm->heap.cycle_count);
    if(vm->heap.stats.num_objects!=count)return false;
    bool exact=true;
    for(unsigned n=0;n<count;n++) {
        if(report)fprintf(stderr,"private graph node=%u tag=%u refs=%u internal=%u module=%u\n",n,nodes[n]->obj_type,nodes[n]->ref_count,edges[n],module_roots[n]);
        if(edges[n]>UINT32_MAX-module_roots[n] || nodes[n]->ref_count!=edges[n]+module_roots[n])exact=false;
    }
    return exact;
}
static void clean(VmState *vm,size_t objects,size_t bytes) {
    CHECK(!vm->stack_size && !vm->frame_count && !vm->references.active && !vm->callee_references.active);
    for(unsigned f=0;f<NVM_OWNED_MAX_FUNCTIONS-2;f++)CHECK(!vm->value_references[f].active);
    CHECK(no_extra_roots(vm,objects,true));
    if(vm->module_constants.count) {
        NanoValue extra=val_string(vm->module_constants.strings[0]);
        CHECK(extra.as.string->header.ref_count<UINT32_MAX);vm_retain(&vm->heap,extra);
        CHECK(!no_extra_roots(vm,objects,false));vm_release(&vm->heap,extra);
        CHECK(no_extra_roots(vm,objects,false));
    }
    if(vm->heap.cycle_count) {
        VmHeapHeader *h=vm->heap.cycle_buf[0];NanoValue extra={0};extra.tag=h->obj_type;extra.as.obj=h;
        CHECK(h->ref_count<UINT32_MAX);vm_retain(&vm->heap,extra);
        CHECK(!no_extra_roots(vm,objects,false));vm_release(&vm->heap,extra);
        CHECK(no_extra_roots(vm,objects,false));
    }
    vm_gc_collect_cycles(&vm->heap);
    CHECK(no_extra_roots(vm,objects,true));CHECK(vm->heap.stats.num_objects==objects);
    CHECK(vm->heap.stats.allocated-vm->heap.stats.freed==bytes);
}
static void check_output(FILE *output) {
    CHECK(!fflush(output));CHECK(ftell(output)==2);rewind(output);CHECK(fgetc(output)=='7' && fgetc(output)=='\n');
}
static void invoke(VmState *vm,VmResult wanted,size_t objects,size_t bytes,bool inject) {
    FILE *output=tmpfile();CHECK(output);vm->output=output;
    fprintf(stderr,"private invoke phase=%s fault=%u baseline=%zu bytes=%zu\n",inject?"fault":"ordinary/recovery",heap_fail,objects,bytes);
    NanoValue result=val_int(-91);VmResult status=vm_execute_owned_array_private(vm,&result);heap_fail=0;
    fprintf(stderr,"private invoke status=%d hits=%u objects=%zu bytes=%zu\n",status,heap_hits,vm->heap.stats.num_objects,vm->heap.stats.allocated-vm->heap.stats.freed);
    if(inject && heap_hits){CHECK(heap_hits==1 && status==VM_ERR_MEMORY);CHECK(result.tag==TAG_INT && result.as.i64==-91);}
    else {CHECK(status==wanted);CHECK(result.tag==TAG_INT && result.as.i64==(wanted==VM_OK?0:-91));}
    check_output(output);CHECK(!fclose(output));vm->output=NULL;clean(vm,objects,bytes);
}
static void retain_boundary(void) {
    Function f={"PUSH_STR value\nPOP\nPUSH_I64 0\nRET\n",0,0,T(TAG_INT),{{0}}};
    NvmModule *m=build(&f,1);VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);
    unsigned index=0;while(index<m->string_count && strcmp(m->strings[index],"retained"))index++;
    CHECK(index<m->string_count);VmString *string=vm.module_constants.strings[index];
    uint32_t saved=string->header.ref_count;string->header.ref_count=UINT32_MAX;
    NanoValue result=val_int(-91);CHECK(vm_execute_owned_array_private(&vm,&result)==VM_ERR_MEMORY);
    CHECK(result.tag==TAG_INT && result.as.i64==-91 && string->header.ref_count==UINT32_MAX);
    CHECK(!vm.stack_size && !vm.frame_count);string->header.ref_count=saved;
    CHECK(vm_execute_owned_array_private(&vm,&result)==VM_OK && result.as.i64==0);
    vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);nvm_module_free(m);
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    for(unsigned which=0;which<7;which++) {
        fprintf(stderr,"private owner ARRAY case=%u\n",which);
        NvmModule *m=runtime_module(which);NvmOwnedArrayPlan *plan=NULL;
        NvmOwnerAuthorityResult q=nvm_prepare_owned_array_authority(m,&plan);
        if(q.status!=NVM_OWNER_AUTH_PREPARED)fprintf(stderr,"prepare: %s f%u pc%u\n",q.message,q.function,q.pc);
        CHECK(q.status==NVM_OWNER_AUTH_PREPARED);nvm_owned_array_plan_free(plan);
        CHECK(!nvm_verify(m).ok);char error[256];char *normal=nvm2c_emit(m,error,sizeof error);CHECK(!normal);
        VmResult wanted=which>=1&&which<=3?VM_ERR_TYPE_ERROR:which==4?VM_ERR_ASSERT_FAILED:which==5?VM_ERR_OUT_OF_BOUNDS:VM_OK;
        for(unsigned fused=0;fused<2;fused++) {
            VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);
            VmDispatchProfile profile={.fuse_load_local_field=fused};vm_set_dispatch_profile(&vm,profile);CHECK(vm.dispatch_module_valid);
            size_t objects=vm.heap.stats.num_objects,bytes=vm.heap.stats.allocated-vm.heap.stats.freed;
            fprintf(stderr,"private public-refusal case=%u fused=%u baseline=%zu bytes=%zu\n",which,fused,objects,bytes);
            NanoValue untouched=val_int(-91);CHECK(vm_call_function(&vm,0,NULL,0)!=VM_OK);CHECK(untouched.as.i64==-91);clean(&vm,objects,bytes);
            invoke(&vm,wanted,objects,bytes,false);
            bool done=false;
            for(unsigned fault=1;fault<128;fault++) {
                fprintf(stderr,"private VM case=%u fused=%u fault=%u\n",which,fused,fault);
                heap_attempts=heap_hits=0;heap_fail=fault;invoke(&vm,wanted,objects,bytes,true);
                if(!heap_hits){done=true;break;}
                invoke(&vm,wanted,objects,bytes,false);
            }
            CHECK(done);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
        }
        char *source=NULL;CHECK(nvm2c_emit_owned_array_private(m,&source,error,sizeof error));CHECK(source);
        char path[1024];snprintf(path,sizeof path,"%s/case%u.c",argv[1],which);FILE *file=fopen(path,"w");CHECK(file);CHECK(fputs(source,file)>=0);CHECK(!fclose(file));free(source);
        nvm_module_free(m);printf("case %u %u\n",which,which>=1&&which<=3?3:which==4?2:which==5?4:0);
    }
    retain_boundary();
    printf("%u private owner ARRAY runtime checks passed\n",checks);return 0;
}
