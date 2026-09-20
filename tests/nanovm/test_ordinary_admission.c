/* I observe the real admission path; modeled host callbacks are labeled. */
#include <assert.h>
#include "../../src/nanovm/vm.h"
#include "../../src/nanovm/vm_ffi.h"
#include "../../src/nanoisa/service_bindings_module.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/retained_layouts.h"
static unsigned pending_calls, prints, external_calls, pumps, nested_calls;
static bool observe_pending(const NvmModule *m);
static void observe_print(NanoValue v, FILE *out);
static bool observe_ffi(VmState *,const NvmModule *,uint32_t,NanoValue *,int,NanoValue *,char *,size_t);
static int observe_pump(VmState *,bool);
#define nvm_service_execution_pending observe_pending
#define val_print observe_print
#define vm_ffi_call_vm observe_ffi
#define vm_callback_pump observe_pump
#include "../../src/nanovm/vm.c"
#undef nvm_service_execution_pending
#undef val_print
#undef vm_ffi_call_vm
#undef vm_callback_pump
int g_argc; char **g_argv;
static unsigned checks;
#define CHECK(c) do { ++checks; if (!(c)) { fprintf(stderr,"check %u: %s:%d: %s\n",checks,__FILE__,__LINE__,#c); abort(); } } while (0)
#include "../nanoisa/owned_fixture.h"
static VmState *host_vm;
static bool mutate_at_host, reenter_at_host, modeled_pump;
static bool observe_pending(const NvmModule *m) {
    ++pending_calls;
    return nvm_service_execution_pending(m);
}
static void host_boundary(void) {
    CHECK(host_vm);
    if (reenter_at_host) {
        NanoValue result=val_void();
        CHECK(vm_invoke_callable(host_vm,val_function(1),NULL,0,&result)==VM_OK);
        CHECK(result.tag==TAG_INT && result.as.i64==73);
        ++nested_calls;
    }
    if (mutate_at_host) {
        NvmModule *m=(NvmModule *)host_vm->module;
        CHECK(!m->ownership_data && !m->ownership_size);
        m->ownership_data=calloc(1,1); CHECK(m->ownership_data);
        m->ownership_size=1; /* Benign malformed declaration, no code writes. */
    }
}
static void observe_print(NanoValue v,FILE *out) {
    ++prints;
    if (host_vm) host_boundary();
    val_print(v,out); /* Real formatting/output after modeled embedding hook. */
}
static bool observe_ffi(VmState *vm,const NvmModule *m,uint32_t index,
                        NanoValue *args,int count,NanoValue *out,char *error,size_t size) {
    (void)args; (void)error; (void)size;
    CHECK(vm==host_vm && m==vm->module && index==0 && count==0);
    ++external_calls; host_boundary(); *out=val_int(7); return true;
}
static int observe_pump(VmState *vm,bool wait) {
    if (!modeled_pump) return vm_callback_pump(vm,wait);
    CHECK(vm->callbacks && !wait); ++pumps; return 0;
}
static NvmModule *ordinary(const char *body,bool import) {
    char text[8192];
    int n=snprintf(text,sizeof(text),".entry 0\n.function main 0 1 0 int 1\n%s.end\n"
        ".function helper 0 0 0 int 1\nPUSH_BOOL 1\nASSERT\nPUSH_I64 73\nRET\n.end\n",body);
    CHECK(n>0 && (size_t)n<sizeof(text));
    AsmResult error;NvmModule *m=asm_assemble_unverified(text,&error);CHECK(m);
    if(import) {
        uint32_t ns=nvm_add_string(m,"diagnostic",10),name=nvm_add_string(m,"host",4);
        CHECK(nvm_add_import(m,ns,name,0,TAG_INT,NULL)==0);
    }
    NvmVerifyResult v=nvm_verify(m);
    if(!v.ok)fprintf(stderr,"%s\n",v.error_msg);
    CHECK(v.ok);return m;
}
static void enter_core(VmState *vm) {
    CHECK(!vm->stack_size && !vm->frame_count);
    const NvmFunctionEntry *fn=&vm->module->functions[0];
    vm->frame_count=1;vm->current_fn=0;vm->ip=fn->code_offset;
    vm->frames[0]=(VmCallFrame){.fn_idx=0,.local_count=fn->local_count,
        .return_ip=UINT32_MAX,.module=vm->module,.owned_callable={.tag=TAG_VOID}};
    CHECK(vm->stack_capacity>=fn->local_count);
    for(unsigned i=0;i<fn->local_count;i++)vm->stack[vm->stack_size++]=val_void();
}
static void assertion(VmTrap t) {
    CHECK(t.type==TRAP_ASSERT && val_truthy(t.data.assert_check.condition));
    CHECK(!val_is_heap_obj(t.data.assert_check.condition));
}
static void private_transitions(VmDispatchProfile profile,unsigned width) {
    const char *stores[]={"MEM_STORE8","MEM_STORE16","MEM_STORE32","MEM_STORE64"};
    char body[1024];
    CHECK(snprintf(body,sizeof(body),"PUSH_BOOL 1\nASSERT\nCALL 1\nPOP\nPUSH_BOOL 1\nASSERT\n"
        "PUSH_I64 0\nPUSH_I64 9\n%s\nPUSH_BOOL 1\nASSERT\n"
        "PUSH_BOOL 1\nASSERT\nPUSH_I64 42\nRET\n",stores[width])>0);
    NvmModule *m=ordinary(body,false);
    VmState vm;vm_init(&vm,m);vm_set_dispatch_profile(&vm,profile);enter_core(&vm);
    vm.memory=calloc(8,1);CHECK(vm.memory);vm.memory_size=8;
    VmOrdinaryAdmission cert={0};VmOwnedInvocationProof proof={0};pending_calls=0;
    assertion(vm_core_execute_scoped(&vm,&proof,&cert));CHECK(cert.valid && pending_calls==5);
    assertion(vm_core_execute_scoped(&vm,&proof,&cert));CHECK(cert.valid && pending_calls==5 && vm.frame_count==2);
    assertion(vm_core_execute_scoped(&vm,&proof,&cert));CHECK(cert.valid && pending_calls==5 && vm.frame_count==1);
    assertion(vm_core_execute_scoped(&vm,&proof,&cert));CHECK(!cert.valid && pending_calls==5 && vm.memory[0]==9);
    assertion(vm_core_execute_scoped(&vm,&proof,&cert));CHECK(cert.valid && pending_calls==10);
    CHECK(vm_core_execute_scoped(&vm,&proof,&cert).type==TRAP_NONE && pending_calls==10);
    CHECK(vm.stack_size==1 && vm.stack[0].as.i64==42 && !vm.frame_count);
    vm_destroy(&vm);nvm_module_free(m);
}
static void public_and_direct(VmDispatchProfile profile) {
    NvmModule *m=ordinary("PUSH_BOOL 1\nASSERT\nCALL 1\nPOP\nPUSH_BOOL 1\nASSERT\nPUSH_I64 42\nRET\n",false);
    VmState vm;vm_init(&vm,m);vm_set_dispatch_profile(&vm,profile);vm_profile_enable(&vm,true);
    for(unsigned i=0;i<2;i++) {
        NanoValue result=val_int(-9);pending_calls=0;
        CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK);
        CHECK(result.tag==TAG_INT && result.as.i64==42 && pending_calls==15);
        CHECK(!vm.stack_size && !vm.frame_count);
    }
    CHECK(vm.profile.opcode_counts[OP_ASSERT]==6);
    enter_core(&vm);pending_calls=0;
    assertion(vm_core_execute(&vm));CHECK(pending_calls==5);
    assertion(vm_core_execute(&vm));CHECK(pending_calls==10);
    assertion(vm_core_execute(&vm));CHECK(pending_calls==15);
    CHECK(vm_core_execute(&vm).type==TRAP_NONE && pending_calls==20);
    vm_destroy(&vm);nvm_module_free(m);
}
static void exclusions(VmDispatchProfile profile) {
    for(unsigned mode=0;mode<3;mode++) {
        NvmModule *m=ordinary("PUSH_BOOL 1\nASSERT\nPUSH_BOOL 1\nASSERT\nPUSH_I64 42\nRET\n",false);
        NvmModule *linked=NULL;VmState vm;vm_init(&vm,m);vm_set_dispatch_profile(&vm,profile);
        if(mode==0)vm.opcode_trace=true;
        if(mode==1){modeled_pump=true;vm.callbacks=(NanoCallbackRuntime *)&modeled_pump;}
        if(mode==2){linked=ordinary("PUSH_I64 8\nRET\n",false);CHECK(vm_link_module(&vm,linked)!=UINT32_MAX);}
        enter_core(&vm);VmOrdinaryAdmission cert={0};VmOwnedInvocationProof proof={0};pending_calls=0;
        assertion(vm_core_execute_scoped(&vm,&proof,&cert));CHECK(!cert.valid);
        unsigned first=pending_calls;CHECK(first>0);
        assertion(vm_core_execute_scoped(&vm,&proof,&cert));CHECK(!cert.valid && pending_calls==2*first);
        CHECK(vm_core_execute_scoped(&vm,&proof,&cert).type==TRAP_NONE && pending_calls==3*first);
        if(mode==1){vm.callbacks=NULL;modeled_pump=false;}
        vm_destroy(&vm);nvm_module_free(linked);nvm_module_free(m);
    }
    NvmModule *m=ordinary("PUSH_BOOL 1\nASSERT\nPUSH_BOOL 1\nASSERT\nPUSH_I64 42\nRET\n",false);
    VmState vm;vm_init(&vm,m);modeled_pump=true;vm.callbacks=(NanoCallbackRuntime *)&modeled_pump;
    pending_calls=pumps=0;NanoValue out=val_void();CHECK(vm_invoke(&vm,0,NULL,0,&out)==VM_OK);
    CHECK(pending_calls==25 && pumps==2 && out.as.i64==42);
    vm.callbacks=NULL;modeled_pump=false;vm_destroy(&vm);nvm_module_free(m);
}
static void host_paths(VmDispatchProfile profile) {
    for(unsigned kind=0;kind<2;kind++)for(unsigned reenter=0;reenter<2;reenter++)for(unsigned change=0;change<2;change++) {
        NvmModule *m=ordinary(kind?"PUSH_BOOL 1\nASSERT\nCALL_EXTERN 0\nPOP\nPUSH_BOOL 1\nASSERT\nPUSH_I64 42\nRET\n":
            "PUSH_BOOL 1\nASSERT\nPUSH_I64 7\nPRINT\nPUSH_BOOL 1\nASSERT\nPUSH_I64 42\nRET\n",kind!=0);
        VmState vm;vm_init(&vm,m);vm_set_dispatch_profile(&vm,profile);
        FILE *output=tmpfile();CHECK(output);vm.output=output;host_vm=&vm;
        mutate_at_host=change;reenter_at_host=reenter;pending_calls=prints=external_calls=nested_calls=0;
        NanoValue out=val_int(-9);VmResult status=vm_invoke(&vm,0,NULL,0,&out);
        CHECK(status==(change?VM_ERR_TYPE_ERROR:VM_OK));
        CHECK(change?out.tag==TAG_VOID:(out.tag==TAG_INT && out.as.i64==42));
        CHECK(nested_calls==reenter && (kind?external_calls:prints)==1);
        if(!change)CHECK(pending_calls==20+10*reenter);
        CHECK(!vm.stack_size && !vm.frame_count);
        host_vm=NULL;mutate_at_host=reenter_at_host=false;
        CHECK(!fclose(output));vm.output=NULL;vm_destroy(&vm);nvm_module_free(m);
    }
}
static void fused_managed_effects(VmDispatchProfile profile) {
    const char *bodies[]={
        "PUSH_I64 42\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 0\nPUSH_BOOL 1\nASSERT\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 42\nEQ\nASSERT\nPUSH_I64 42\nRET\n",
        "PUSH_STR 0\nASSERT\nPUSH_STR 0\nASSERT\nPUSH_I64 42\nRET\n",
        "HANDLER_PUSH 0 handled 0 0\nPUSH_BOOL 1\nASSERT\nPERFORM 0 0\nPOP\n"
        "HANDLER_POP 1\nPUSH_BOOL 1\nASSERT\nPUSH_I64 42\nRET\n"
        "handled:\nPUSH_BOOL 1\nASSERT\nPUSH_I64 7\nEFFECT_RESUME\n"};
    for(unsigned n=0;n<3;n++) {
        NvmModule *m=ordinary(bodies[n],false);VmState vm;vm_init(&vm,m);
        vm_set_dispatch_profile(&vm,profile);CHECK(vm_rebuild_module(&vm,m));
        unsigned fused=0;
        for(unsigned i=0;i<vm.dispatch_module.functions[0].instruction_count;i++)
            fused+=vm.dispatch_module.functions[0].instructions[i].super_op==VM_SUPER_LOAD_LOCAL_FIELD;
        CHECK(fused==(unsigned)(n==0 && profile.fuse_load_local_field));
        size_t objects=vm.heap.stats.num_objects;uint32_t refs=vm.module_constants.strings[0]->header.ref_count;
        pending_calls=0;NanoValue out=val_void();CHECK(vm_invoke(&vm,0,NULL,0,&out)==VM_OK);
        CHECK(out.tag==TAG_INT && out.as.i64==42 && pending_calls==15);
        CHECK(!vm.frame_count && !vm.stack_size && !vm.handler_count);
        CHECK(vm.module_constants.strings[0]->header.ref_count==refs);
        vm_gc_collect_cycles(&vm.heap);CHECK(vm.heap.stats.num_objects==objects);
        vm_destroy(&vm);nvm_module_free(m);
    }
}
static void refusal_and_owned(void) {
    NvmModule *m=ordinary("PUSH_BOOL 1\nASSERT\nPUSH_I64 42\nRET\n",false);
    VmState vm;vm_init(&vm,m);NanoValue out=val_int(-9);
    CHECK(vm_invoke(&vm,0,NULL,0,&out)==VM_OK);
    m->ownership_data=calloc(1,1);CHECK(m->ownership_data);m->ownership_size=1;
    out=val_int(-9);CHECK(vm_invoke(&vm,0,NULL,0,&out)==VM_ERR_TYPE_ERROR && out.as.i64==-9);
    CHECK(!vm.stack_size && !vm.frame_count);vm_destroy(&vm);nvm_module_free(m);
    m=ordinary("PUSH_BOOL 0\nASSERT\nPUSH_I64 42\nRET\n",false);vm_init(&vm,m);
    CHECK(vm_invoke(&vm,0,NULL,0,&out)==VM_ERR_ASSERT_FAILED && !vm.stack_size && !vm.frame_count);
    vm_destroy(&vm);nvm_module_free(m);
    m=ordinary("PUSH_BOOL 1\nASSERT\nPUSH_I64 42\nRET\n",true);
    m->imports[0].kind=NVM_IMPORT_SERVICE;vm_init(&vm,m);out=val_int(-9);
    external_calls=prints=0;
    CHECK(vm_invoke(&vm,0,NULL,0,&out)==VM_ERR_TYPE_ERROR && out.as.i64==-9);
    CHECK(!external_calls && !prints && !vm.stack_size && !vm.frame_count);
    enter_core(&vm);VmOrdinaryAdmission rejected={0};VmOwnedInvocationProof absent={0};
    CHECK(vm_core_execute_scoped(&vm,&absent,&rejected).type==TRAP_ERROR && !rejected.valid);
    vm_destroy(&vm);nvm_module_free(m);
    m=fixture("PUSH_I64 42\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_BOOL 1\nASSERT\nOWN_UNPACK_LOCAL 0\nRET\n",false,false);
    CHECK(nvm_verify(m).ok);vm_init(&vm,m);VmOwnedInvocationProof proof={0};CHECK(vm_ownership_admit(&vm,&proof));
    enter_core(&vm);VmOrdinaryAdmission cert={0};assertion(vm_core_execute_scoped(&vm,&proof,&cert));
    CHECK(!cert.valid && vm.references.active);
    CHECK(vm_core_execute_scoped(&vm,&proof,&cert).type==TRAP_NONE && !cert.valid);
    CHECK(vm.stack_size==1 && vm.stack[0].as.i64==42 && !vm.references.active);
    vm_destroy(&vm);nvm_module_free(m);
}
int main(void) {
    for(unsigned p=0;p<2;p++) {
        VmDispatchProfile profile=p?vm_dispatch_profile_all():vm_dispatch_profile_none();
        for(unsigned width=0;width<4;width++) {
            private_transitions(profile,width);
        }
        public_and_direct(profile);
        exclusions(profile);
        host_paths(profile);
        fused_managed_effects(profile);
    }
    refusal_and_owned();
    printf("%u ordinary admission checks passed; real VM, modeled host hooks; no timing claim\n",checks);
    return 0;
}
