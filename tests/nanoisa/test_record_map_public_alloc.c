/* Actual public transport/runtime failure prefixes, independent allocator ledger. */
#define MAP_PUBLIC_MAIN retained_public_controls
#include "test_record_map_public.c"
#undef MAP_PUBLIC_MAIN
static void public_allocation_operation(Input *c,unsigned kind,const NvmV2Module *wire,const uint8_t *bytes,size_t size) {
    size_t live=ra_live,allocated=ra_bytes;
    if(kind==0) {
        NvmV2Module before;memset(&before,0xa5,sizeof before);NvmV2Module output=before;
        NvmV2Result r=nvm_v2_from_nvm_module(&c->m,&output);
        if(r==NVM_V2_OK)nvm_v2_module_free(&output);else CHECK(!memcmp(&output,&before,sizeof output));
    } else if(kind==1) {
        /* Output storage allocated outside the injected operation. */
        uint8_t output[65536];CHECK(size<sizeof output);memset(output,0xa5,sizeof output);size_t written=991;
        NvmV2Result r=nvm_v2_module_serialize(wire,output,sizeof output,&written);
        if(r==NVM_V2_OK)CHECK(written==size&&!memcmp(output,bytes,size));
        else {CHECK(written==991);for(size_t i=0;i<sizeof output;i++)CHECK(output[i]==0xa5);}
    } else if(kind==2) {
        NvmV2Module output=*wire;NvmV2Result r=nvm_v2_module_deserialize(bytes,size,&output);
        if(r==NVM_V2_OK)nvm_v2_module_free(&output);else CHECK(!memcmp(&output,wire,sizeof output));
    } else {
        char error[512];char *output=nvm2c_emit(&c->m,error,sizeof error);
        if(output)free(output);else CHECK(error[0]);
    }
    CHECK(ra_live==live&&ra_bytes==allocated);
}
static __attribute__((noinline)) void public_vm_prefixes(Input *c) {
    VmState vm;vm_init(&vm,&c->m);ra_calls=0;CHECK(vm_execute(&vm)==VM_OK);size_t calls=ra_calls;
    vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<calls;i++) {
        vm_init(&vm,&c->m);ra_calls=0;ra_fail=i;ra_persistent=(int)mode;
        VmResult r=vm_execute(&vm);ra_fail=SIZE_MAX;ra_persistent=0;
        if(r==VM_OK){CHECK(vm.stack_size==1&&vm_get_result(&vm).as.i64==0);vm.stack_size--;}
        else CHECK(!vm.stack_size&&!vm.frame_count);
        CHECK(vm_execute(&vm)==VM_OK);CHECK(vm.stack_size==1&&vm_get_result(&vm).as.i64==0);
        vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I checked %zu public invocation allocation positions, transient/persistent and retry.\n",calls);
}
int main(void) {
    setvbuf(stdout,NULL,_IONBF,0);Input c;map_basic(&c);
    NvmV2Module wire;CHECK(nvm_v2_from_nvm_module(&c.m,&wire)==NVM_V2_OK);
    size_t size=0;CHECK(nvm_v2_module_serialize(&wire,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);CHECK(nvm_v2_module_serialize(&wire,bytes,size,NULL)==NVM_V2_OK);
    for(unsigned kind=0;kind<3;kind++) {
        ra_calls=0;public_allocation_operation(&c,kind,&wire,bytes,size);size_t calls=ra_calls;
        for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<calls;i++) {
            ra_calls=0;ra_fail=i;ra_persistent=(int)mode;public_allocation_operation(&c,kind,&wire,bytes,size);
            ra_fail=SIZE_MAX;ra_persistent=0;public_allocation_operation(&c,kind,&wire,bytes,size);
        }
        printf("I checked public transport operation%u at %zu positions with prior-output and retry.\n",kind,calls);
    }
    Input native;init(&native,TAG_INT,false);map_new(&native,TAG_STRING,TAG_INT);scalar(&native,TAG_STRING);scalar(&native,TAG_INT);op(&native,OP_HM_SET);
    op(&native,OP_AGG_PACK);b8(native.code,&native.n,AGG_RECORD);b32(native.code,&native.n,0);b16(native.code,&native.n,0);b16(native.code,&native.n,1);
    op(&native,OP_POP);map_finish(&native);char error[512];char *native_text=nvm2c_emit(&native.m,error,sizeof error);CHECK(native_text);free(native_text);
    ra_calls=0;public_allocation_operation(&native,3,&wire,bytes,size);size_t native_calls=ra_calls;
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<native_calls;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)mode;public_allocation_operation(&native,3,&wire,bytes,size);
        ra_fail=SIZE_MAX;ra_persistent=0;public_allocation_operation(&native,3,&wire,bytes,size);
    }
    printf("I checked %zu public native compiler positions with independent retry.\n",native_calls);
    nvm_v2_module_free(&wire);free(bytes);CHECK(!ra_live&&!ra_bytes);public_vm_prefixes(&c);
    printf("PASS %u actual public allocation checks; peak %zu payload bytes.\n",checks,ra_peak);return 0;
}
