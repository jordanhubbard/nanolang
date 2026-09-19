/* I retain the existing exact mixed module and test service claims before its
 * private preparation, public admission and runtime selection. */
#define MIXED_RUNTIME_ALLOC_TEST
#include "test_mixed_samples_runtime.c"
#include "mixed_samples_internal.h"
#include "service_bindings_module.h"

int main(void) {
    (void)runtime_artifacts;(void)runtime_core;
    NvmModule *m=runtime_fixture(0);
    CHECK(nvm_mixed_samples_candidate(m));CHECK(nvm_verify(m).ok);
    CHECK(!m->service_data && !m->service_size && !m->import_count);
    NvmImportEntry *original_imports=m->imports;
    for(unsigned variant=0;variant<3;variant++) {
        uint8_t byte=0;NvmImportEntry claim={0};
        if(variant==0)m->service_data=&byte;
        if(variant==1)m->service_size=1;
        if(variant==2){claim.kind=NVM_IMPORT_SERVICE;m->imports=&claim;m->import_count=1;}
        CHECK(nvm_service_bindings_present(m));CHECK(!nvm_mixed_samples_candidate(m));
        NvmMixedSamplesPlan *sentinel=(NvmMixedSamplesPlan *)(uintptr_t)1;
        CHECK(nvm_mixed_samples_prepare(m,&sentinel).status==NVM_MIXED_SHAPE_UNRESOLVED);
        CHECK(sentinel==(NvmMixedSamplesPlan *)(uintptr_t)1);
        CHECK(nvm_mixed_samples_admit(m,&sentinel).status==NVM_MIXED_SHAPE_UNRESOLVED);
        CHECK(sentinel==(NvmMixedSamplesPlan *)(uintptr_t)1);
        CHECK(!nvm_verify(m).ok);CHECK(!nvm_verify_linked(m,NULL,0).ok);
        uint16_t maximum=123;CHECK(!nvm_verify_function_max_stack(m,0,&maximum).ok);CHECK(maximum==123);
        char error[256];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
        NvmV2Module wire={0};CHECK(nvm_v2_from_nvm_module(m,&wire)!=NVM_V2_OK);nvm_v2_module_free(&wire);
        for(unsigned api=0;api<4;api++) {
            VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
            NanoValue result=val_int(-91);CHECK(runtime_api(&vm,api,&result)!=VM_OK);
            CHECK(result.tag==TAG_INT && result.as.i64==-91);runtime_clean(&vm,baseline);vm_destroy(&vm);
        }
        m->service_data=NULL;m->service_size=0;m->imports=original_imports;m->import_count=0;
        CHECK(nvm_mixed_samples_candidate(m));CHECK(nvm_verify(m).ok);
    }
    NvmV2Module wire={0};CHECK(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
    NvmModule *roundtrip=NULL;CHECK(nvm_v2_to_nvm_module(&wire,&roundtrip)==NVM_V2_OK);
    CHECK(nvm_verify(roundtrip).ok);nvm_v2_module_free(&wire);nvm_module_free(roundtrip);
    VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;NanoValue result=val_int(-91);
    CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK);CHECK(result.tag==TAG_INT && result.as.i64==0);
    vm_release(&vm.heap,result);runtime_clean(&vm,baseline);vm_destroy(&vm);nvm_module_free(m);
    printf("%u mixed service selection and recovery checks passed\n",checks);return 0;
}
