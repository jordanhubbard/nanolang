#define main prior_affine_state_main
#include "test_affine_state.c"
#undef main
#include "affine_bytecode.h"
#include "nvm2c.h"

static NvmModule *call_fixture_profile(const char *call,const char *helper,bool nested,
                                       bool strings) {
    NvmModule *m=fixture();
    char source[16384];
    snprintf(source,sizeof(source),
        "%s.types 3 0 0\n.entry 0\n.function main 0 8 0 int 1\n"
        "PUSH_I64 10\nOWN_PACK 0\nOWN_STORE_LOCAL 2\n"
        "PUSH_I64 32\nOWN_PACK 0\nOWN_STORE_LOCAL 3\n"
        "OWN_MOVE_LOCAL 2\nOWN_MOVE_LOCAL 3\nOWN_PACK 1\nOWN_STORE_LOCAL 5\n"
        "%s%s\nPOP\nREGION_END\n%s"
        "OWN_UNPACK_LOCAL 2\nOWN_UNPACK_LOCAL 3\nADD\nRET\n.end\n"
        ".function inspect 1 2 0 int 1\n%s\n.end\n.parameters 1 struct\n",
        strings?".string borrowed \"borrowed\"\n":"",
        nested?"REGION_BEGIN\nBORROW_PATH_EXCLUSIVE 0 5 0\n":
            "OWN_UNPACK_LOCAL 5\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 2\nREGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 2\n",call,
        nested?"OWN_UNPACK_LOCAL 5\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 2\n":"",helper);
    AsmResult result;NvmModule *code=asm_assemble_unverified(source,&result);if(!code)fprintf(stderr,"assemble: %s\n",result.message);CHECK(code);
    free(m->code);m->code=code->code;code->code=NULL;m->code_size=code->code_size;
    memcpy(m->functions,code->functions,2*sizeof(*m->functions));
    if(strings) {
        m->strings=code->strings;code->strings=NULL;
        m->string_lengths=code->string_lengths;code->string_lengths=NULL;
        m->string_count=code->string_count;code->string_count=0;
    }
    nvm_module_free(code);
    slot(m->ownership_data+20,TAG_INT,0,NVM_V2_NO_INDEX);
    slot(m->ownership_data+96,TAG_INT,0,NVM_V2_NO_INDEX);
    m->ownership_data=realloc(m->ownership_data,132);CHECK(m->ownership_data);
    memset(m->ownership_data+112,0,20);m->ownership_size=132;
    m->ownership_data[92]=2;slot(m->ownership_data+112,TAG_INT,0,NVM_V2_NO_INDEX);
    word(m->ownership_data,2);word(m->ownership_data+120,1);m->ownership_data[124]=1;
    if(!nested){word(m->ownership_data,1);m->ownership_size=120;}
    bool needs;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK);
    return m;
}
static NvmModule *call_fixture_impl(const char *call,const char *helper,bool nested) {
    return call_fixture_profile(call,helper,nested,false);
}
static NvmModule *call_fixture(const char *call,const char *helper) {
    return call_fixture_impl(call,helper,true);
}
#ifndef CALLER_RUNTIME_TEST
int main(void) {
    const char *helpers[]={"REF_GET 0 0\nRET", "PUSH_I64 42\nREF_SET 0 0\nREF_GET 0 0\nRET"};
    for(unsigned i=0;i<2;i++) {
        NvmModule *m=call_fixture("CALL_REF 1 0",helpers[i]);
        NvmAffineAnalysis result=nvm_affine_analyze_function(m,0);
        if(!result.ok)fprintf(stderr,"%s\n",result.message);
        CHECK(result.ok);
        CHECK(nvm_verify_owned_module(m).ok);
        nvm_module_free(m);
    }
    const char *invalid[]={"LOAD_LOCAL 0\nAGG_GET 0\nRET", "OWN_MOVE_LOCAL 0\nRET",
        "REGION_BEGIN\nREF_GET 0 0\nRET", "CALL_REF 1 0\nRET"};
    for(unsigned i=0;i<4;i++) {
        NvmModule *m=call_fixture("CALL_REF 1 0",invalid[i]);
        CHECK(!nvm_affine_analyze_function(m,0).ok);nvm_module_free(m);
    }
    NvmModule *m=call_fixture("CALL_REF 0 0","REF_GET 0 0\nRET");
    CHECK(!nvm_affine_analyze_function(m,0).ok);nvm_module_free(m);
    m=call_fixture("CALL_REF 1 1","REF_GET 0 0\nRET");
    CHECK(!nvm_affine_analyze_function(m,0).ok);nvm_module_free(m);
    m=call_fixture("CALL_REF 1 0","REF_GET 0 0\nRET");
    slot(m->ownership_data+104,TAG_STRUCT,2,2);
    CHECK(!nvm_affine_analyze_function(m,0).ok);nvm_module_free(m);
    m=call_fixture_profile("CALL_REF 1 0","PUSH_STR borrowed\nPRINT\nREF_GET 0 0\nRET",true,true);
    NvmAffineAnalysis string=nvm_affine_analyze_function(m,1);
    CHECK(!string.ok&&!strcmp(string.message,
        "I require string literals inside an owned value-call graph"));
    CHECK(!nvm_verify_owned_module(m).ok);
    string=nvm_affine_analyze_function(m,0);
    CHECK(!string.ok&&!strcmp(string.message,
        "I require checked caller authority and a non-escaping helper"));
    char error[256];CHECK(nvm2c_emit(m,error,sizeof(error))==NULL);
    nvm_module_free(m);
    const char *output_helpers[]={
        "PUSH_I64 1\nPRINT\nREF_GET 0 0\nRET",
        "PUSH_I64 1\nPRINTLN\nREF_GET 0 0\nRET"
    };
    for(unsigned i=0;i<2;i++) {
        m=call_fixture("CALL_REF 1 0",output_helpers[i]);
        string=nvm_affine_analyze_function(m,1);
        CHECK(!string.ok&&!strcmp(string.message,
            "I require string output inside an owned value-call graph"));
        CHECK(!nvm_verify_owned_module(m).ok);
        CHECK(nvm2c_emit(m,error,sizeof(error))==NULL);
        nvm_module_free(m);
    }
    printf("%u caller-origin analysis checks passed\n",checks);return 0;
}

#endif
