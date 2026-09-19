/* I exercise checked public boundaries; refused modules never enter handlers. */
#include "nvm_format_v2.h"
#include "nvm_v2_sections.h"
static void public_reject(NvmModule *m) {
    fprintf(stderr,"public checked refusal route=%u\n",(unsigned)nvm_owned_array_route(m));
    unsigned char sentinel;NvmOwnedArrayPlan *plan=(void *)&sentinel;
    CHECK(nvm_owned_array_admit(m,&plan).status!=NVM_OWNER_AUTH_PREPARED && plan==(void *)&sentinel);
    CHECK(!nvm_verify(m).ok && !nvm_verify_linked(m,NULL,0).ok);
    uint16_t depth=321;CHECK(!nvm_verify_function_max_stack(m,0,&depth).ok && depth==321);
    char error[256];CHECK(!nvm2c_emit(m,error,sizeof error));
    NvmV2Module v2;CHECK(nvm_v2_from_nvm_module(m,&v2)!=NVM_V2_OK);nvm_v2_module_free(&v2);
}
static void public_direct_core(NvmModule *m) {
    fprintf(stderr,"public direct-core state snapshot\n");
    VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);
    size_t objects=vm.heap.stats.num_objects,bytes=vm.heap.stats.allocated-vm.heap.stats.freed;
    VmArray *array=vm_array_new(&vm.heap,TAG_FLOAT,8);CHECK(array);CHECK(vm_array_push(&vm.heap,array,val_float(3.5)));
    VmClosure *closure=vm_closure_new(&vm.heap,0,0);CHECK(closure);
    vm.stack[0]=val_array(array);vm.stack[1]=val_string(vm.module_constants.strings[0]);vm_retain(&vm.heap,vm.stack[1]);vm.stack_size=2;
    vm.frame_count=1;vm.frames[0].module=m;vm.frames[0].owned_callable=val_closure(closure);vm.frames[0].closure=closure;
    vm.references.active=true;vm.references.generation=5;vm.callee_references.active=true;vm.callee_references.generation=6;
    vm.value_references[0].active=true;vm.value_references[0].generation=7;vm.reference_generation=9;
    vm.handler_count=1;vm.handlers[0].module=m;vm.ip=17;vm.current_fn=1;vm.activation_floor=1;
    NanoValue values[2];memcpy(values,vm.stack,sizeof values);
    VmState *before=malloc(sizeof vm);CHECK(before);memcpy(before,&vm,sizeof vm);
    unsigned ar=array->header.ref_count,sr=vm.stack[1].as.string->header.ref_count,cr=closure->header.ref_count;
    VmTrap trap=vm_core_execute(&vm);CHECK(trap.type==TRAP_ERROR && trap.data.error.code==VM_ERR_TYPE_ERROR);
    before->last_error=vm.last_error;memcpy(before->error_msg,vm.error_msg,sizeof vm.error_msg);
    CHECK(!memcmp(before,&vm,sizeof vm) && !memcmp(values,vm.stack,sizeof values));free(before);
    CHECK(array->header.ref_count==ar && values[1].as.string->header.ref_count==sr && closure->header.ref_count==cr);
    vm_release(&vm.heap,vm.stack[0]);vm_release(&vm.heap,vm.stack[1]);vm.stack_size=0;
    vm_release(&vm.heap,vm.frames[0].owned_callable);memset(&vm.frames[0],0,sizeof vm.frames[0]);vm.frame_count=0;
    memset(&vm.references,0,sizeof vm.references);memset(&vm.callee_references,0,sizeof vm.callee_references);memset(vm.value_references,0,sizeof vm.value_references);
    vm.handler_count=vm.activation_floor=0;vm.ip=vm.current_fn=0;clean(&vm,objects,bytes);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
static NvmOwnedArrayPlan *public_input_integrity(NvmModule *m,bool accepted) {
    NvmModule before=*m;uint8_t *rows[8]={0},tags[8][8]={{0}};
    CHECK(m->function_count<=8);
    for(uint32_t f=0;f<m->function_count;f++) {
        CHECK(m->functions[f].arity<=8);
        rows[f]=m->function_param_types?m->function_param_types[f]:NULL;
        if(rows[f])memcpy(tags[f],rows[f],m->functions[f].arity);
    }
    uint8_t *owned=malloc(m->ownership_size),*code=malloc(m->code_size),*layouts=malloc(m->layout_size);
    CHECK(owned && code && layouts);memcpy(owned,m->ownership_data,m->ownership_size);
    memcpy(code,m->code,m->code_size);memcpy(layouts,m->layout_data,m->layout_size);
    unsigned char sentinel;NvmOwnedArrayPlan *plan=(void *)&sentinel;
    NvmOwnerAuthorityResult r=nvm_owned_array_admit(m,&plan);
    if(r.status!=NVM_OWNER_AUTH_PREPARED)fprintf(stderr,"public preparation status=%u %s\n",(unsigned)r.status,r.message);
    CHECK((r.status==NVM_OWNER_AUTH_PREPARED)==accepted);
    CHECK(!memcmp(&before,m,sizeof before));
    CHECK(!memcmp(owned,m->ownership_data,m->ownership_size) && !memcmp(code,m->code,m->code_size) && !memcmp(layouts,m->layout_data,m->layout_size));
    for(uint32_t f=0;f<m->function_count;f++) {
        CHECK((m->function_param_types?m->function_param_types[f]:NULL)==rows[f]);
        if(rows[f])CHECK(!memcmp(tags[f],rows[f],m->functions[f].arity));
    }
    free(owned);free(code);free(layouts);
    if(!accepted){CHECK(plan==(void *)&sentinel);return NULL;}return plan;
}
static void public_wire(NvmModule *m,const char *dir) {
    fprintf(stderr,"public transport sidecars=%u\n",m->function_param_types!=NULL);
    NvmOwnedArrayPlan *plan=public_input_integrity(m,true);
    NvmV2Module v2;CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    for(uint32_t f=0;f<m->function_count;f++) {
        NvmOwnerSignature sig;CHECK(nvm_owned_array_plan_signature(plan,f,&sig));
        uint16_t depth=0;CHECK(nvm_verify_function_max_stack(m,f,&depth).ok && depth==sig.max_stack);
        CHECK(v2.functions.items[f].max_stack==depth);
        NvmV2Signature *wire=&v2.signatures.items[v2.functions.items[f].signature_idx];
        CHECK(wire->param_count==sig.parameter_count);
        for(uint16_t n=0;n<sig.parameter_count;n++)CHECK(wire->param_tags[n]==sig.parameters[n].tag);
    }
    NvmModule *back=NULL;CHECK(nvm_v2_to_nvm_module(&v2,&back)==NVM_V2_OK);
    CHECK(nvm_verify(back).ok && back->layout_size==m->layout_size && back->ownership_size==m->ownership_size);
    CHECK(!memcmp(back->layout_data,m->layout_data,m->layout_size) && !memcmp(back->ownership_data,m->ownership_data,m->ownership_size));nvm_module_free(back);
    size_t size;CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);uint8_t *data=malloc(size);CHECK(data);
    CHECK(nvm_v2_module_serialize(&v2,data,size,NULL)==NVM_V2_OK);
    char path[1024];snprintf(path,sizeof path,"%s/public.nvm",dir);FILE *file=fopen(path,"wb");CHECK(file);CHECK(fwrite(data,1,size,file)==size);CHECK(!fclose(file));
    uint16_t depth=v2.functions.items[0].max_stack;CHECK(depth>1);v2.functions.items[0].max_stack=1;
    unsigned char sentinel;back=(void *)&sentinel;CHECK(nvm_v2_to_nvm_module(&v2,&back)!=NVM_V2_OK && back==NULL);
    uint8_t *snapshot=malloc(size);CHECK(snapshot);memcpy(snapshot,data,size);
    CHECK(nvm_v2_module_serialize(&v2,data,size,NULL)!=NVM_V2_OK);CHECK(!memcmp(snapshot,data,size));free(snapshot);
    v2.functions.items[0].max_stack=0;CHECK(nvm_v2_to_nvm_module(&v2,&back)==NVM_V2_OK);nvm_module_free(back);
    v2.functions.items[0].max_stack=depth+1;CHECK(nvm_v2_to_nvm_module(&v2,&back)==NVM_V2_OK);nvm_module_free(back);
    free(data);nvm_v2_module_free(&v2);nvm_owned_array_plan_free(plan);
}
static void public_pending_file(void) {
    for(unsigned helper=0;helper<2;helper++) {
        /* Case6 never calls helper3; pending declarations still cover it. */
        NvmModule *m=runtime_module(6);uint32_t function=helper?3:0;
        uint32_t offset=m->functions[function].code_offset;uint8_t saved=m->code[offset];
        m->code[offset]=OP_FILE_DROP_STACK;
        fprintf(stderr,"public pending File root/helper=%u\n",helper);
        CHECK(!nvm_service_bindings_present(m) && nvm_service_execution_pending(m));
        CHECK(nvm_owned_array_route(m)==NVM_OWNER_ARRAY_NOT_SELECTED);public_reject(m);
        unsigned char sentinel;NvmOwnedArrayPlan *plan=(void *)&sentinel;
        CHECK(nvm_owned_array_admit(m,&plan).status==NVM_OWNER_AUTH_UNRESOLVED && plan==(void *)&sentinel);
        VmState vm;vm_init(&vm,m);NanoValue result=val_int(-91);
        CHECK(vm_invoke(&vm,0,NULL,0,&result)!=VM_OK && result.tag==TAG_INT && result.as.i64==-91);
        CHECK(vm_invoke_callable(&vm,val_function(0),NULL,0,&result)!=VM_OK && result.tag==TAG_INT && result.as.i64==-91);
        CHECK(vm_call_function(&vm,0,NULL,0)!=VM_OK && vm_execute(&vm)!=VM_OK);
        CHECK(vm_core_execute(&vm).type==TRAP_ERROR && !vm.stack_size && !vm.frame_count);
        vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
        m->code[offset]=saved;CHECK(nvm_verify(m).ok);nvm_module_free(m);
    }
}
static void public_boundaries(const char *dir) {
    fprintf(stderr,"public boundary controls begin\n");public_pending_file();
    NvmModule *m=runtime_module(0);CHECK(nvm_owned_array_route(m)==NVM_OWNER_ARRAY_SELECTED && nvm_verify(m).ok);
    CHECK(nvm_verify_linked(m,NULL,0).ok);
    const NvmModule *links[]={m};CHECK(!nvm_verify_linked(m,links,1).ok);
    for(int p=NVM_PROFILE_CLOSED_SCALAR;p<=NVM_PROFILE_CLOSED_MANAGED_STRINGS;p++)CHECK(!nvm_verify_profile(m,(NvmVerifyProfile)p).ok);
    public_wire(m,dir);public_direct_core(m);
    uint8_t **params=m->function_param_types;m->function_param_types=NULL;public_wire(m,dir);m->function_param_types=params;
    uint8_t *partial=params[3];params[3]=NULL;public_wire(m,dir);params[3]=partial;
    uint8_t old=params[3][0];params[3][0]=TAG_VOID;public_wire(m,dir);params[3][0]=old;
    params[3][0]=TAG_INT;public_input_integrity(m,false);public_reject(m);params[3][0]=old;
    m->function_param_types=NULL;NvmOwnedArrayPlan *detached=public_input_integrity(m,true);m->function_param_types=params;
    NvmModule *fresh=runtime_module(0);nvm_module_free(m);m=fresh;params=m->function_param_types;
    NvmOwnerSignature detached_signature;CHECK(nvm_owned_array_plan_signature(detached,3,&detached_signature));
    CHECK(detached_signature.parameter_count==1 && detached_signature.parameters[0].tag==TAG_STRING);
    NvmOwnedArrayTransport detached_transport;CHECK(nvm_owned_array_plan_transport(detached,&detached_transport));
    CHECK(detached_transport.ownership_size==m->ownership_size && !memcmp(detached_transport.ownership,m->ownership_data,m->ownership_size));
    nvm_owned_array_plan_free(detached);
    unsigned offsets[]={4,5,12,13,14,15};
    for(unsigned n=0;n<sizeof offsets/sizeof offsets[0];n++) {
        fprintf(stderr,"public layout envelope offset=%u\n",offsets[n]);
        unsigned at=offsets[n];uint8_t saved=m->layout_data[at];m->layout_data[at]=255;
        CHECK(nvm_owned_array_route(m)==NVM_OWNER_ARRAY_INVALID);public_reject(m);m->layout_data[at]=saved;
    }
    uint32_t length=m->layout_size;m->layout_size--;CHECK(nvm_owned_array_route(m)==NVM_OWNER_ARRAY_INVALID);public_reject(m);m->layout_size=length;
    unsigned owned_offsets[]={8,13,16};
    for(unsigned n=0;n<sizeof owned_offsets/sizeof owned_offsets[0];n++) {
        fprintf(stderr,"public ownership envelope offset=%u\n",owned_offsets[n]);
        unsigned at=owned_offsets[n];uint8_t saved=m->ownership_data[at];m->ownership_data[at]=255;
        CHECK(nvm_owned_array_route(m)==NVM_OWNER_ARRAY_INVALID);public_reject(m);m->ownership_data[at]=saved;
    }
    uint8_t flag_copy[5];memcpy(flag_copy,m->ownership_data+8,5);memset(m->ownership_data+8,0,5);
    CHECK(nvm_owned_array_route(m)==NVM_OWNER_ARRAY_NOT_SELECTED);memcpy(m->ownership_data+8,flag_copy,5);
    m->service_size=1;CHECK(nvm_owned_array_route(m)==NVM_OWNER_ARRAY_NOT_SELECTED);public_reject(m);m->service_size=0;
    for(public_api=0;public_api<4;public_api++) {
        fprintf(stderr,"public changed-module api=%u\n",public_api);
        VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);size_t objects=vm.heap.stats.num_objects,bytes=vm.heap.stats.allocated-vm.heap.stats.freed;
        uint8_t result_tag=m->functions[0].result_tag;m->functions[0].result_tag=TAG_FLOAT;
        NanoValue result=val_int(-91);CHECK(runtime_entry(&vm,&result)!=VM_OK && result.tag==TAG_INT && result.as.i64==-91);
        m->functions[0].result_tag=result_tag;clean(&vm,objects,bytes);vm_destroy(&vm);
    }
    VmState vm;vm_init(&vm,m);size_t objects=vm.heap.stats.num_objects,bytes=vm.heap.stats.allocated-vm.heap.stats.freed;
    NanoValue result=val_int(-91);CHECK(vm_invoke(&vm,1,NULL,0,&result)!=VM_OK && result.as.i64==-91);
    CHECK(vm_invoke_callable(&vm,val_function(1),NULL,0,&result)!=VM_OK && result.as.i64==-91);
    VmClosure *closure=vm_closure_new(&vm.heap,0,0);CHECK(closure);uint32_t refs=closure->header.ref_count;
    CHECK(vm_invoke_callable(&vm,val_closure(closure),NULL,0,&result)!=VM_OK && result.as.i64==-91 && closure->header.ref_count==refs);vm_release(&vm.heap,val_closure(closure));
    clean(&vm,objects,bytes);vm_destroy(&vm);nvm_module_free(m);
    Function unconsumed={"PUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_I64 0\nRET\n",0,1,T(TAG_INT),{OWNER(0)}};
    m=build(&unconsumed,1);NvmOwnedArrayOrigins *origins=NULL;
    CHECK(nvm_analyze_owned_array_origins(m,&origins).status==NVM_OWNER_ORIGIN_PROVED);
    nvm_owned_array_origins_free(origins);public_reject(m);nvm_module_free(m);
    public_api=0;
}
