/* Actual private union execution; public record-union admission stays closed. */
#include "../../src/nanovm/union_field_runtime_private.h"
#define UFE_MAIN retained_union_recipe_controls
#include "test_union_field_execution.c"
int g_argc=0;char **g_argv=NULL;
#ifdef UNION_EXECUTION_WHITEBOX
#ifdef UNION_EXECUTION_ALLOC
#include "record_array_alloc.h"
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#endif
#include "../../src/nanovm/vm.c"
#ifdef UNION_EXECUTION_ALLOC
#undef malloc
#undef calloc
#undef realloc
#undef free
#endif
#endif
static VmUnionFieldPrivate *runtime_prepare(const NvmModule *m) {
    VmUnionFieldPrivate *p=NULL;NvmArrayEligibilityResult r=vm_union_field_private_prepare(m,&p);
    if(r.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"prepare status=%u %s\n",r.status,r.message);
    CHECK(r.status==NVM_ARRAY_ELIGIBLE && p);return p;
}
static void runtime_check(VmUnionFieldPrivate *p,int64_t expected) {
    VmUnionFieldRunResult result;memset(&result,0xa5,sizeof result);
    VmResult status=vm_union_field_private_execute(p,&result);
    if(status!=VM_OK) {
#ifdef UNION_EXECUTION_WHITEBOX
        fprintf(stderr,"execute status=%u error=%s ip=%u fn=%u\n",status,p->vm.error_msg,p->vm.ip,p->vm.current_fn);
#else
        fprintf(stderr,"execute status=%u\n",status);
#endif
    }
    CHECK(status==VM_OK && result.outcome==VM_UNION_FIELD_RETURNED && result.result.tag==TAG_INT && result.result.as.i64==expected);
#ifdef UNION_EXECUTION_WHITEBOX
    CHECK(!p->vm.stack_size && !p->vm.frame_count && !p->vm.activation_floor && !p->vm.heap.gc_running);
    CHECK(!vm_rm_private_dag(&p->vm.heap));
#endif
}
static void runtime_original(const char *path) {
    FILE *file=fopen(path,"rb");CHECK(file);CHECK(!fseek(file,0,SEEK_END));
    long size=ftell(file);CHECK(size>0&&(unsigned long)size<=UINT32_MAX);rewind(file);
    uint8_t *bytes=malloc((size_t)size);CHECK(bytes);
    CHECK(fread(bytes,1,(size_t)size,file)==(size_t)size);CHECK(!fclose(file));
    NvmV2Module original_wire;memset(&original_wire,0,sizeof original_wire);
    CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&original_wire)==NVM_V2_OK);
    NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&original_wire,&m)==NVM_V2_OK);
    CHECK(m&&nvm_verify(m).ok);
    /* Actual old NULL route uses this newly compiled vm.c, not installed CLI. */
    VmState *ordinary=calloc(1,sizeof *ordinary);CHECK(ordinary);vm_init(ordinary,m);
    CHECK(vm_execute(ordinary)==VM_OK);NanoValue ordinary_result=vm_get_result(ordinary);
    CHECK(ordinary_result.tag==TAG_INT && ordinary_result.as.i64==0);
    vm_destroy(ordinary);CHECK(!ordinary->heap.stats.num_objects);free(ordinary);
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
    VmUnionFieldPrivate *p=runtime_prepare(m);runtime_check(p,0);
    VmUnionFieldPreparation stats;CHECK(vm_union_field_private_preparation(p,&stats));
    printf("prepared original %s rows=%u functions=%u image=%llu/%llu\n",path,stats.instructions,stats.functions,
        (unsigned long long)stats.image_overlap_bytes,(unsigned long long)stats.image_overlap_work);
    m->layout_data=NULL;m->layout_size=0;m->ownership_data=NULL;m->ownership_size=0;
    nvm_module_free(m);nvm_v2_module_free(&original_wire);free(bytes);memset(&v,0,sizeof v);
    runtime_check(p,0);vm_union_field_private_destroy(p);
}
static void runtime_basic(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);
    VmUnionFieldPrivate *p=runtime_prepare(&v.c.module);
    memset(&v,0xa5,sizeof v);memset(code,0xa5,sizeof code);runtime_check(p,42);runtime_check(p,42);
    vm_union_field_private_destroy(p);
#ifdef UNION_EXECUTION_WHITEBOX
    printf("sizes owner=%zu scope=%zu VmState=%zu VmHeap=%zu outcome=%zu\n",sizeof(VmUnionFieldPrivate),sizeof(VmUnionFieldInvocation),sizeof(VmState),sizeof(VmHeap),sizeof(VmUnionFieldRunResult));
#endif
}
static void runtime_payload(uint8_t *code,size_t *n) {
    proof_i64(code,n,42);proof_pack(code,n,AGG_VARIANT,0,1);proof_pack(code,n,AGG_RECORD,0,1);
    u8(code,n,OP_AGG_GET);u16(code,n,0);u8(code,n,OP_AGG_GET);u16(code,n,0);
}
static void runtime_lifecycle(unsigned mode) {
    UnionCase v;union_case(&v);uint8_t code[512];size_t n=0;
    NvmFunctionEntry fn[2]={{0}};fn[0].result_tag=TAG_INT;fn[0].result_count=1;
    bool init=mode==3 || mode==4 || mode==9,helper=mode>=5 && mode<=8;
    if(init) {
        u8(code,&n,OP_LOAD_GLOBAL);u32(code,&n,0);proof_i64(code,&n,16);u8(code,&n,OP_EQ);u8(code,&n,OP_ASSERT);
        proof_i64(code,&n,99);u8(code,&n,OP_STORE_GLOBAL);u32(code,&n,0);
    }
    if(helper){u8(code,&n,(mode==6 || mode==8)?OP_TAIL_CALL:OP_CALL);u32(code,&n,1);if(mode==5 || mode==7)u8(code,&n,OP_RET);}
    else {
        if(mode<=1){u8(code,&n,OP_PUSH_BOOL);u8(code,&n,mode==0);u8(code,&n,OP_ASSERT);}
        size_t branch=0;
        if(mode==2){u8(code,&n,OP_PUSH_BOOL);u8(code,&n,0);branch=n;u8(code,&n,OP_JMP_FALSE);u32(code,&n,0);}
        runtime_payload(code,&n);u8(code,&n,OP_RET);
        if(mode==2){patch(code,branch+1,(uint32_t)(n-branch));proof_i64(code,&n,123);u8(code,&n,OP_HALT);}
    }
    fn[0].code_length=(uint32_t)n;
    if(init || helper) {
        fn[1].name_idx=init?6:1;fn[1].code_offset=(uint32_t)n;fn[1].result_tag=mode==9?TAG_VOID:TAG_INT;fn[1].result_count=mode==9?0:1;
        if(init){proof_i64(code,&n,16);u8(code,&n,OP_STORE_GLOBAL);u32(code,&n,0);}
        u8(code,&n,OP_PUSH_BOOL);u8(code,&n,mode==4 || mode>=7);size_t branch=n;u8(code,&n,OP_JMP_FALSE);u32(code,&n,0);
        if(init){if(mode!=9)proof_i64(code,&n,16);}else {
            if(mode>=7){u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);u8(code,&n,OP_ASSERT);}
            runtime_payload(code,&n);
        }
        u8(code,&n,OP_RET);patch(code,branch+1,(uint32_t)(n-branch));proof_i64(code,&n,321);u8(code,&n,OP_HALT);
        fn[1].code_length=(uint32_t)n-fn[1].code_offset;
    }
    proof_header(&v,code,n);char *strings[7];uint32_t lengths[7];
    for(unsigned i=0;i<6;i++){strings[i]=v.c.module.strings[i];lengths[i]=v.c.module.string_lengths[i];}
    strings[6]="__init__";lengths[6]=8;v.c.module.strings=strings;v.c.module.string_lengths=lengths;v.c.module.string_count=7;
    proof_functions(&v,fn,init||helper?2:1);VmUnionFieldPrivate *p=runtime_prepare(&v.c.module);
    for(unsigned repeat=0;repeat<2;repeat++) {
        VmUnionFieldRunResult out;memset(&out,0xa5,sizeof out);VmUnionFieldRunResult saved=out;
        VmResult status=vm_union_field_private_execute(p,&out);
        printf("lifecycle mode=%u repeat=%u status=%u outcome=%u\n",mode,repeat,status,status==VM_OK?(unsigned)out.outcome:99u);
        if(mode==1){CHECK(status==VM_ERR_ASSERT_FAILED&&!memcmp(&out,&saved,sizeof out));}
        else if(mode==2 || mode==5 || mode==6)CHECK(status==VM_OK&&out.outcome==VM_UNION_FIELD_HALTED&&out.result.tag==TAG_VOID);
        else CHECK(status==VM_OK&&out.outcome==VM_UNION_FIELD_RETURNED&&out.result.tag==TAG_INT&&out.result.as.i64==42);
#ifdef UNION_EXECUTION_WHITEBOX
        CHECK(!p->vm.stack_size&&!p->vm.frame_count&&!p->vm.activation_floor);
        if(mode==1)CHECK(p->vm.last_error==VM_ERR_ASSERT_FAILED&&!strcmp(p->vm.error_msg,"Assertion failed"));
        if(init)CHECK(p->vm.globals[0].tag==TAG_INT&&p->vm.globals[0].as.i64==99);
#endif
    }
    vm_union_field_private_destroy(p);
}
static void runtime_replacement(void) {
    UnionCase v;union_case(&v);uint8_t code[256];size_t n=0;NvmFunctionEntry fn={0};
    fn.local_count=2;fn.result_count=1;fn.result_tag=TAG_INT;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
    u8(code,&n,OP_DUP);u8(code,&n,OP_STORE_LOCAL);u16(code,&n,1);
    proof_pack(code,&n,AGG_RECORD,0,1);u8(code,&n,OP_STORE_LOCAL);u16(code,&n,0);
    u8(code,&n,OP_LOAD_LOCAL);u16(code,&n,0);proof_pack(code,&n,AGG_VARIANT,1,0);
    u8(code,&n,OP_AGG_SET);u16(code,&n,0);u8(code,&n,OP_DUP);u8(code,&n,OP_STORE_GLOBAL);u32(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_AGG_TAG);proof_i64(code,&n,1);u8(code,&n,OP_EQ);u8(code,&n,OP_ASSERT);
    u8(code,&n,OP_LOAD_LOCAL);u16(code,&n,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    proof_header(&v,code,n);fn.code_length=(uint32_t)n;proof_functions(&v,&fn,1);
    VmUnionFieldPrivate *p=runtime_prepare(&v.c.module);runtime_check(p,42);
#ifdef UNION_EXECUTION_WHITEBOX
    NanoValue record=p->vm.globals[0];CHECK(record.tag==TAG_STRUCT&&record.as.sval->fields[0].tag==TAG_UNION);
    CHECK(record.as.sval->def_idx==0&&record.as.sval->fields[0].as.uval->def_idx==0&&record.as.sval->fields[0].as.uval->variant==1);
    CHECK(((VmHeapHeader *)record.as.obj)->buffered&&((VmHeapHeader *)record.as.obj)->colour==VM_GC_PURPLE);
    CHECK(vm_uf_shell(p,record));vm_gc_collect_cycles(&p->vm.heap);
    CHECK(p->vm.globals[0].as.obj==record.as.obj&&!p->vm.heap.gc_running);
#endif
    runtime_check(p,42);vm_union_field_private_destroy(p);
}
static void runtime_signature_refusal(void) {
    for(unsigned mode=0;mode<3;mode++) {
        UnionCase v;union_case(&v);uint8_t code[256];size_t n=0;NvmFunctionEntry fn[2]={{0}};
        if(mode==0){u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);fn[0].result_tag=TAG_BOOL;}
        else {runtime_payload(code,&n);fn[0].result_tag=TAG_INT;}
        fn[0].result_count=1;u8(code,&n,OP_RET);fn[0].code_length=(uint32_t)n;
        if(mode) {
            fn[1].name_idx=6;fn[1].code_offset=(uint32_t)n;fn[1].result_count=1;
            if(mode==1){proof_pack(code,&n,AGG_VARIANT,1,0);fn[1].result_tag=TAG_UNION;}
            else {u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);fn[1].result_tag=TAG_BOOL;}
            u8(code,&n,OP_RET);fn[1].code_length=(uint32_t)n-fn[1].code_offset;
        }
        proof_header(&v,code,n);char *strings[7];uint32_t lengths[7];
        for(unsigned i=0;i<6;i++){strings[i]=v.c.module.strings[i];lengths[i]=v.c.module.string_lengths[i];}
        strings[6]="__init__";lengths[6]=8;v.c.module.strings=strings;v.c.module.string_lengths=lengths;v.c.module.string_count=7;
        proof_functions(&v,fn,mode?2:1);VmUnionFieldPrivate *p=runtime_prepare(&v.c.module);
        VmUnionFieldRunResult out;memset(&out,0xa5,sizeof out);VmUnionFieldRunResult saved=out;
#ifdef UNION_EXECUTION_WHITEBOX
        VmState before=p->vm;
#endif
        CHECK(vm_union_field_private_execute(p,&out)==VM_ERR_TYPE_ERROR&&!memcmp(&out,&saved,sizeof out));
#ifdef UNION_EXECUTION_WHITEBOX
        CHECK(!memcmp(&before,&p->vm,sizeof before));
#endif
        vm_union_field_private_destroy(p);
    }
}
#ifdef UNION_EXECUTION_WHITEBOX
#ifdef UNION_EXECUTION_ALLOC
static void runtime_prefixes(const NvmModule *m,const char *label) {
    CHECK(!ra_live&&!ra_bytes);VmUnionFieldPrivate *p=runtime_prepare(m);ra_calls=0;
    runtime_check(p,42);size_t reached=ra_calls;vm_union_field_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    size_t failed=0,optional=0;
    for(int persistent=0;persistent<2;persistent++)for(size_t at=0;at<reached;at++) {
        p=runtime_prepare(m);VmUnionFieldRunResult out;memset(&out,0xa5,sizeof out);VmUnionFieldRunResult prior=out;
        ra_calls=0;ra_fail=at;ra_persistent=persistent;
        VmResult status=vm_union_field_private_execute(p,&out);
        ra_fail=SIZE_MAX;ra_persistent=0;
        if(status==VM_OK){optional++;CHECK(out.outcome==VM_UNION_FIELD_RETURNED&&out.result.tag==TAG_INT&&out.result.as.i64==42);}
        else {failed++;CHECK(status==VM_ERR_MEMORY&&!memcmp(&out,&prior,sizeof out));}
        CHECK(vm_uf_idle(p));runtime_check(p,42);vm_union_field_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    }
    printf("runtime prefixes %s reached=%zu failures=%zu optional=%zu\n",label,reached,failed,optional);
}
static void runtime_allocations(void) {
    for(unsigned mode=0;mode<3;mode++) {
        UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
        if(mode==1){u8(code,&n,OP_PUSH_STR);u32(code,&n,0);u8(code,&n,OP_PUSH_STR);u32(code,&n,1);u8(code,&n,OP_STR_CONCAT);u8(code,&n,OP_POP);}
        proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_POP);
        if(mode==2) {
            proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);proof_pack(code,&n,AGG_RECORD,0,1);
            u8(code,&n,OP_DUP);u8(code,&n,OP_STORE_GLOBAL);u32(code,&n,0);
            u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_AGG_GET);u16(code,&n,0);
        } else runtime_payload(code,&n);
        u8(code,&n,OP_RET);proof_header(&v,code,n);
        runtime_prefixes(&v.c.module,mode==2?"retained-global":mode?"dynamic-string":"record-and-empty-union");
    }
}
#endif
static void runtime_dormant(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    runtime_payload(code,&n);u8(code,&n,OP_RET);uint32_t dead=(uint32_t)n;
    proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_AGG_TAG);u8(code,&n,OP_RET);proof_header(&v,code,n);
    VmUnionFieldPrivate *p=runtime_prepare(&v.c.module);runtime_check(p,42);VmState *vm=&p->vm;
    uint32_t index=vm->decoded_module.functions[0].instruction_indices[dead]-1;
    NvmUnionFieldExecutionInstruction row;CHECK(nvm_union_field_image_instruction(p->image,index,&row));
    CHECK(row.site.state==UFS_NOT_ESTABLISHED&&vm_uf_site_same(&row.site,&p->instructions[index].site));
    VmUnionFieldInvocation view={p,vm,vm->module,0,0,true};VmCallFrame *f=&vm->frames[0];
    memset(f,0,sizeof *f);f->module=vm->module;f->owned_callable=val_void();f->local_count=p->functions[0].signature.local_count;
    for(unsigned i=0;i<f->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;vm->ip=dead;
    VmTrap t=vm_core_execute_union(vm,NULL,NULL,NULL,NULL,NULL,TAG_VOID,&view);
    CHECK(t.type==TRAP_ERROR&&vm->ip==dead&&vm->stack_size==f->local_count);
    vm_uf_unwind(&view);view.active=false;runtime_check(p,42);vm_union_field_private_destroy(p);
}
static void runtime_tail_owners(bool tail) {
    UnionCase v;union_case(&v);uint8_t code[512];size_t n=0;NvmFunctionEntry fn[2]={{0}};
    uint8_t tags[18];memset(tags,TAG_UNION,sizeof tags);uint8_t *params[2]={NULL,tags};
    fn[0].result_tag=TAG_INT;fn[0].result_count=1;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
    for(unsigned i=1;i<18;i++)u8(code,&n,OP_DUP);
    u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);u8(code,&n,OP_ASSERT);uint32_t call=(uint32_t)n;
    u8(code,&n,tail?OP_TAIL_CALL:OP_CALL);u32(code,&n,1);if(!tail)u8(code,&n,OP_RET);
    fn[0].code_length=(uint32_t)n;fn[1].code_offset=(uint32_t)n;fn[1].name_idx=1;
    fn[1].arity=18;fn[1].local_count=18;fn[1].result_tag=TAG_INT;fn[1].result_count=1;
    proof_i64(code,&n,42);u8(code,&n,OP_RET);fn[1].code_length=(uint32_t)n-fn[1].code_offset;
    proof_header(&v,code,n);v.c.module.function_param_types=params;proof_functions(&v,fn,2);
    VmUnionFieldPrivate *p=runtime_prepare(&v.c.module);runtime_check(p,42);VmState *vm=&p->vm;
    VmUnionFieldInvocation view={p,vm,vm->module,0,0,true};
    memset(&vm->frames[0],0,sizeof vm->frames[0]);vm->frames[0].module=vm->module;vm->frames[0].owned_callable=val_void();vm->frame_count=1;
    VmTrap t=vm_core_execute_union(vm,NULL,NULL,NULL,NULL,NULL,TAG_VOID,&view);
    CHECK(t.type==TRAP_ASSERT&&vm->ip==call&&vm->stack_size==18);vm_release(&vm->heap,t.data.assert_check.condition);
    VmHeapHeader *h=vm->stack[0].as.obj;CHECK(h->ref_count==18);
    uint32_t index=vm->dispatch_module.functions[0].offset_to_index[call]-1;
    const VmDispatchInstruction *in=&vm->dispatch_module.functions[0].instructions[index];
    h->ref_count=UINT32_MAX-17;
    CHECK(vm_uf_preflight(&view,in)==(tail?VM_ERR_MEMORY:VM_OK));
    CHECK(h->ref_count==UINT32_MAX-17&&vm->ip==call&&vm->stack_size==18);
    if(tail) {
        t=vm_core_execute_union(vm,NULL,NULL,NULL,NULL,NULL,TAG_VOID,&view);
        CHECK(t.type==TRAP_ERROR&&t.data.error.code==VM_ERR_MEMORY&&vm->ip==call&&vm->stack_size==18);
        h->ref_count=UINT32_MAX-18;CHECK(vm_uf_preflight(&view,in)==VM_OK);
    }
    h->ref_count=18;vm_uf_unwind(&view);view.active=false;runtime_check(p,42);vm_union_field_private_destroy(p);
#ifdef UNION_EXECUTION_ALLOC
    runtime_prefixes(&v.c.module,tail?"tail18":"call18");
#endif
}
static void runtime_paused_get(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);proof_pack(code,&n,AGG_RECORD,0,1);
    u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);u8(code,&n,OP_ASSERT);
    uint32_t get=(uint32_t)n;u8(code,&n,OP_AGG_GET);u16(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);proof_header(&v,code,n);
    VmUnionFieldPrivate *p=runtime_prepare(&v.c.module);VmState *vm=&p->vm;
    for(unsigned mode=0;mode<3;mode++) {
        CHECK(vm_uf_idle(p));VmUnionFieldInvocation view={p,vm,vm->module,0,0,true};
        VmCallFrame *f=&vm->frames[0];memset(f,0,sizeof *f);f->module=vm->module;f->owned_callable=val_void();
        f->local_count=p->functions[0].signature.local_count;
        for(unsigned i=0;i<f->local_count;i++)vm->stack[vm->stack_size++]=val_void();
        vm->frame_count=1;vm->current_fn=0;vm->ip=0;
        VmTrap t=vm_core_execute_union(vm,NULL,NULL,NULL,NULL,NULL,TAG_VOID,&view);
        CHECK(t.type==TRAP_ASSERT&&t.data.assert_check.condition.tag==TAG_BOOL&&t.data.assert_check.condition.as.boolean);
        vm_release(&vm->heap,t.data.assert_check.condition);
        CHECK(vm->ip==get&&vm->stack_size==f->local_count+1u&&vm->stack[f->local_count].tag==TAG_STRUCT);
        NanoValue record=vm->stack[f->local_count];NanoValue child=record.as.sval->fields[0];CHECK(child.tag==TAG_UNION);
        uint32_t refs=((VmHeapHeader *)child.as.obj)->ref_count;
        uint32_t index=vm->decoded_module.functions[0].instruction_indices[get]-1;
        NvmUnionFieldExecutionInstruction saved=p->instructions[index];
        if(mode==0)((VmHeapHeader *)child.as.obj)->ref_count=UINT32_MAX;
        if(mode==1)p->instructions[index].site.state=UFS_NOT_ESTABLISHED;
        if(mode==2)view.active=false;
        t=vm_core_execute_union(vm,NULL,NULL,NULL,NULL,NULL,TAG_VOID,&view);
        CHECK(t.type==TRAP_ERROR&&vm->ip==get&&vm->stack_size==f->local_count+1u&&vm->stack[f->local_count].as.obj==record.as.obj);
        if(mode==0)CHECK(((VmHeapHeader *)child.as.obj)->ref_count==UINT32_MAX);
        ((VmHeapHeader *)child.as.obj)->ref_count=refs;p->instructions[index]=saved;view.active=true;
        vm_uf_unwind(&view);view.active=false;CHECK(vm_uf_idle(p));runtime_check(p,42);
    }
    vm_union_field_private_destroy(p);
}
#endif
int main(int argc,char **argv) {
    CHECK(argc==7 || argc==1);runtime_basic();
    for(unsigned mode=0;mode<10;mode++)runtime_lifecycle(mode);
    runtime_signature_refusal();runtime_replacement();
#ifdef UNION_EXECUTION_WHITEBOX
    runtime_dormant();runtime_paused_get();runtime_tail_owners(false);runtime_tail_owners(true);
#endif
    #ifdef UNION_EXECUTION_ALLOC
    runtime_allocations();CHECK(!ra_live&&!ra_bytes);(void)runtime_original;(void)argv;
#else
    for(int i=1;i<argc;i++)runtime_original(argv[i]);
#endif
    printf("PASS %u private union execution checks; originals=%d; no public enrollment\n",checks,argc-1);return 0;
}
