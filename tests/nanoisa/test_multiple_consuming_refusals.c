/* I serialize ordinary ownership-policy refusals for verification only. */
#define MULTIPLE_CONSUMING_ALLOC_TEST
#include "test_multiple_consuming_calls.c"
int main(int argc,char **argv) {
    (void)multiple_refusals;(void)artifacts;(void)consuming_verified;
    CHECK(argc==2);
    for(unsigned index=0;index<2;index++) {
        NvmModule *m=multiple_fixture(0,index==0?"OWN_MOVE_LOCAL 1\nOWN_MOVE_LOCAL 0\n":NULL,
            index==1?"PUSH_I64 42\nRET\n":NULL);
        CHECK(!nvm_verify_owned_module(m).ok);
        NvmV2Module v2;size_t size;CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
        CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
        uint8_t *bytes=malloc(size);CHECK(bytes);
        CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
        char path[1024];snprintf(path,sizeof(path),"%s/refused%u.nvm",argv[1],index);
        FILE *file=fopen(path,"wb");CHECK(file);CHECK(fwrite(bytes,1,size,file)==size);CHECK(!fclose(file));
        free(bytes);nvm_v2_module_free(&v2);nvm_module_free(m);
    }
    printf("%u multiple consuming publication-refusal checks passed\n",checks);return 0;
}
