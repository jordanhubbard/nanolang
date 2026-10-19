/* I exercise every metadata bridge allocation boundary and release partial state. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned fail_at, calls;
static void *metadata_malloc(size_t n) { if (fail_at && ++calls == fail_at) return NULL; return malloc(n); }
static void *metadata_calloc(size_t n,size_t z) { if (fail_at && ++calls == fail_at) return NULL; return calloc(n,z); }
#define malloc metadata_malloc
#define calloc metadata_calloc
#include "../../src/nanoisa/nvm_format.c"
#include "../../src/nanoisa/nvm_v2_convert.c"
#undef malloc
#undef calloc
int main(void) {
    NvmModule *m=nvm_module_new(); assert(m);
    uint32_t k=nvm_add_string(m,"example.fact",12),v=nvm_add_string(m,"a\0b",3);
    for(unsigned i=0;i<12;i++) assert(nvm_add_metadata(m,k,v));
    unsigned checks=0; bool succeeded=false;
    for(unsigned attempt=1;attempt<128;attempt++) {
        calls=0;fail_at=attempt;NvmV2Module out;
        NvmV2Result r=nvm_v2_from_nvm_module(m,&out);fail_at=0;++checks;
        assert(m->metadata_count==12 && m->metadata[11].value_idx==v);
        if(r==NVM_V2_OK) { assert(out.metadata.count==12);nvm_v2_module_free(&out);succeeded=true;break; }
        assert(r==NVM_V2_ERR_TRUNCATED);
    }
    assert(succeeded);NvmV2Module wire;assert(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
    succeeded=false;
    for(unsigned attempt=1;attempt<128;attempt++) {
        calls=0;fail_at=attempt;NvmModule *out=(void *)1;
        NvmV2Result r=nvm_v2_to_nvm_module(&wire,&out);fail_at=0;++checks;
        if(r==NVM_V2_OK) { assert(out && out->metadata_count==12);nvm_module_free(out);succeeded=true;break; }
        assert(out==NULL);
    }
    assert(succeeded);nvm_v2_module_free(&wire);nvm_module_free(m);
    succeeded=false;
    for(unsigned attempt=1;attempt<128;attempt++) {
        m=nvm_module_new(); assert(m);
        NvmFunctionEntry function={0}; assert(nvm_add_function(m,&function)==0);
        calls=0;fail_at=attempt;
        bool added=nvm_add_function_purity(m,0,true);fail_at=0;++checks;
        assert(nvm_metadata_valid(m));
        if(added) { assert(nvm_function_purity(m,0)==1);succeeded=true; }
        else {
            assert(m->metadata_count==0 && nvm_function_purity(m,0)==-1);
            assert(nvm_add_function_purity(m,0,false));
            assert(nvm_function_purity(m,0)==0);
        }
        nvm_module_free(m);
        if(succeeded)break;
    }
    assert(succeeded);
    succeeded=false;
    for(unsigned attempt=1;attempt<128;attempt++) {
        m=nvm_module_new(); assert(m);
        NvmFunctionEntry function={0}; assert(nvm_add_function(m,&function)==0);
        calls=0;fail_at=attempt;
        bool added=nvm_add_function_effects(m,0,63);fail_at=0;++checks;
        assert(nvm_metadata_valid(m));
        if(added) { assert(nvm_function_effects(m,0)==63);succeeded=true; }
        else {
            assert(m->metadata_count==0 && nvm_function_effects(m,0)==-1);
            assert(nvm_add_function_effects(m,0,0));
            assert(nvm_function_effects(m,0)==0);
        }
        nvm_module_free(m);
        if(succeeded)break;
    }
    assert(succeeded);
    for (unsigned prior = 0; prior < 2; ++prior) {
        succeeded = false;
        for (unsigned attempt = 1; attempt < 128; ++attempt) {
            m = nvm_module_new(); assert(m);
            NvmFunctionEntry function = {0}; assert(nvm_add_function(m, &function) == 0);
            if (prior) assert(nvm_add_match_coverage(m, 0, 1, 1, 0, 1, "before.nano"));
            calls = 0; fail_at = attempt;
            bool added = nvm_add_match_coverage(m, 0, 2, 3, 2, 1, "after.nano");
            fail_at = 0; ++checks;
            assert(nvm_metadata_valid(m));
            NvmMatchCoverage view;
            if (prior) assert(nvm_match_coverage_at(m, 0, &view) && view.coverage == 0);
            if (added) succeeded = true;
            else {
                assert(m->metadata_count == prior);
                assert(nvm_add_match_coverage(m, 0, 2, 3, 2, 1, "after.nano"));
            }
            assert(nvm_match_coverage_at(m, prior, &view) && view.coverage == 2);
            nvm_module_free(m);
            if (succeeded) break;
        }
        assert(succeeded);
    }
    for(unsigned prior=0;prior<2;prior++) {
        succeeded=false;
        for(unsigned attempt=1;attempt<128;attempt++) {
            m=nvm_module_new();assert(m);
            NvmFunctionEntry function={0};assert(nvm_add_function(m,&function)==0);
            if(prior)assert(nvm_add_generic_binding(m,0,"same","source.nano","U","bool"));
            calls=0;fail_at=attempt;
            bool added=nvm_add_generic_binding(m,0,"same","source.nano","T","array<int>");
            fail_at=0;++checks;
            assert(nvm_metadata_valid(m));
            NvmGenericBinding view;
            if(prior)assert(nvm_generic_binding_at(m,0,&view) && view.variable[0]=='U');
            if(added) {
                assert(nvm_generic_binding_at(m,prior,&view) && view.variable[0]=='T');
                succeeded=true;
            } else {
                assert(m->metadata_count==prior);
                assert(nvm_add_generic_binding(m,0,"same","source.nano","T","array<int>"));
                assert(nvm_generic_binding_at(m,prior,&view) && view.variable[0]=='T');
            }
            nvm_module_free(m);
            if(succeeded)break;
        }
        assert(succeeded);
    }
    printf("I passed %u metadata conversion allocation boundaries.\n",checks);return 0;
}
