/* I pause the actual self-tail instruction with two references to one owner. */
#define RA_SELF_MAIN retained_self_tail_runtime_main
#include "test_record_array_self_tail.c"
#undef RA_SELF_MAIN
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#include "../../src/nanovm/vm.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
static void begin(VmState *vm,const NvmModule *module) {
    VmCallFrame *frame=&vm->frames[0];memset(frame,0,sizeof *frame);
    frame->module=module;frame->local_count=module->functions[0].local_count;frame->owned_callable=val_void();
    for(uint16_t i=0;i<frame->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;vm->current_fn=0;vm->ip=0;
}
int main(void) {
    Input c;pause_self_tail=true;self_fixture(&c,TAG_INT,2,3,true);CHECK(nvm_verify(&c.m).ok);
    for(unsigned private_route=0;private_route<2;private_route++) {
        VmState ordinary,*vm;VmRecordArrayPrivate *p=NULL;
        if(private_route){CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);vm=&p->vm;p->busy=true;}
        else {vm=&ordinary;vm_init(vm,&c.m);}
        begin(vm,vm->module);
        uint32_t active=0;VmOwnedInvocationProof proof={0};proof.module=vm->module;proof.ordinary_arrays=true;proof.ordinary_active_root=&active;
        VmTrap trap=vm_core_execute_scoped(vm,private_route?NULL:&proof,NULL,p);CHECK(trap.type==TRAP_ASSERT);
        vm_release(&vm->heap,trap.data.assert_check.condition);
        CHECK(vm->current_fn==1 && vm->frame_count==1);
        NanoValue value=stack_peek(vm,0);CHECK(value.tag==TAG_STRUCT && value.as.obj==stack_peek(vm,1).as.obj);
        VmHeapHeader *owner=(VmHeapHeader *)value.as.obj;uint32_t refs=owner->ref_count,ip=vm->ip,stack=vm->stack_size;
        owner->ref_count=UINT32_MAX-1;
        if(private_route) {
            const VmDispatchInstruction *tail=NULL;
            for(uint32_t i=0;i<p->functions[1].instruction_count;i++) {
                const VmDispatchInstruction *in=&vm->dispatch_module.functions[1].instructions[i];
                if(in->instruction.opcode==OP_TAIL_CALL && in->instruction.operands[0].u32==1){tail=in;break;}
            }
            CHECK(tail && vm_ra_preflight(p,tail)==VM_ERR_MEMORY);
            CHECK(owner->ref_count==UINT32_MAX-1 && vm->stack_size==stack && vm->current_fn==1);
            owner->ref_count=UINT32_MAX-2;CHECK(vm_ra_preflight(p,tail)==VM_OK);
        } else {
            trap=vm_core_execute_scoped(vm,&proof,NULL,NULL);
            CHECK(trap.type==TRAP_ERROR && trap.data.error.code==VM_ERR_MEMORY);
            CHECK(owner->ref_count==UINT32_MAX-1 && vm->stack_size==stack && vm->current_fn==1 && vm->frame_count==1);
        }
        owner->ref_count=refs;vm->ip=ip;
        trap=vm_core_execute_scoped(vm,private_route?NULL:&proof,NULL,p);CHECK(trap.type==TRAP_ASSERT);
        vm_release(&vm->heap,trap.data.assert_check.condition);
        CHECK(vm->current_fn==1 && vm->frame_count==1);
        if(private_route){vm_ra_unwind(p);p->busy=false;CHECK(vm_record_array_private_run(p)==VM_OK);vm_record_array_private_destroy(p);}
        else vm_destroy(vm);
        CHECK(!ra_live && !ra_bytes);
    }
    printf("PASS %u self-tail duplicate-owner headroom checks\n",checks);return 0;
}
