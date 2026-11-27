/* I retain the original complete owner ARRAY fixture and scalar oracle. */
#define main retained_owner_origin_main
#include "test_owned_array_origins.c"
#undef main
#include "verifier.h"
#include "owned_array_admission.h"
static void depth_pair(NvmModule *m,bool accepted) {
    CHECK(nvm_owned_array_route(m)!=NVM_OWNER_ARRAY_NOT_SELECTED);
    uint16_t expected=0,depth=0;
    NvmVerifyResult scalar=nvm_verify_function_max_stack(m,0,&depth);
    CHECK(scalar.ok==accepted);if(scalar.ok)expected=depth;
    typedef struct {uint32_t before;uint16_t depth;uint32_t after;} Row;
    Row out={0xabc,0xa5a5,0xdef};
    NvmVerifyResult batch=nvm_infer_function_max_stacks(m,&out,1,sizeof out,offsetof(Row,depth));
    CHECK(batch.ok==scalar.ok && out.depth==expected && out.before==0xabc && out.after==0xdef);
    if(!scalar.ok)CHECK(!strcmp(scalar.error_msg,batch.error_msg));
}
int main(void) {
    Function f={"ARR_NEW 3\nSTORE_LOCAL 0\nPUSH_I64 7\nOWN_PACK 0\nLOAD_LOCAL 0\nPUSH_STR value\nOWN_PACK 1\nOWN_STORE_LOCAL 1\n"
        "OWN_UNPACK_LOCAL 1\nPOP\nPOP\nOWN_STORE_LOCAL 2\nOWN_UNPACK_LOCAL 2\nPOP\nPUSH_I64 0\nRET\n",0,3,T(TAG_INT),{T(TAG_ARRAY),OWNER(1),OWNER(0)}};
    NvmModule *m=build(&f,1);depth_pair(m,true);
    uint8_t saved=m->header.magic[0];m->header.magic[0]=0;depth_pair(m,false);
    m->header.magic[0]=saved;depth_pair(m,true);nvm_module_free(m);
    printf("I passed %u private-profile depth and fixture controls.\n",checks);return 0;
}
