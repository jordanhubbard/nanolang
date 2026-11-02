#define UFE_MAIN retained_union_recipe_controls
#include "/Users/jkh/Src/nanolang-51-platform-qualification/tests/nanoisa/test_union_field_execution.c"
#include "/Users/jkh/Src/nanolang-51-platform-qualification/src/nanoisa/union_field_snapshot_private.h"
int g_argc=0;char **g_argv=NULL;
static void annotate(const char *path,const char *destination) {
    FILE *file=fopen(path,"rb");CHECK(file);CHECK(!fseek(file,0,SEEK_END));
    long size=ftell(file);CHECK(size>0&&(unsigned long)size<=UINT32_MAX);rewind(file);
    uint8_t *bytes=malloc((size_t)size);CHECK(bytes);
    CHECK(fread(bytes,1,(size_t)size,file)==(size_t)size);CHECK(!fclose(file));
    NvmV2Module original_wire;memset(&original_wire,0,sizeof original_wire);
    CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&original_wire)==NVM_V2_OK);
    NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&original_wire,&m)==NVM_V2_OK);
    CHECK(m&&nvm_verify(m).ok);
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
    NvmUnionFieldExecutionPlan *proof=NULL;NvmArrayEligibilityResult result=nvm_prepare_union_field_execution(m,&proof);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"original %s status=%u fn=%u pc=%u: %s\n",path,result.status,result.function,result.pc,result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);execution_rows(proof);proof_public_refusal(m);
    NvmRecordArrayExecutionCounts counts;CHECK(nvm_union_field_execution_counts(proof,&counts));
    printf("original %s instructions=%u functions=%u reserved=%llu work=%llu\n",path,counts.instructions,counts.functions,
        (unsigned long long)counts.peak_bytes_reserved,(unsigned long long)counts.work_reserved);
    NvmUnionFieldImage *image=NULL;CHECK(nvm_union_field_image_copy(proof,&image).status==NVM_ARRAY_ELIGIBLE);nvm_union_field_image_free(image);
    NvmVerifyResult verification=nvm_verify(m);char error[512]={0};char *native=nvm2c_emit(m,error,sizeof error);CHECK(!native&&!verification.ok);
    printf("selected refusal verify=%s native=%s\n",verification.error_msg,error);
    NvmV2Module wire;memset(&wire,0,sizeof wire);CHECK(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
    size_t total=0;CHECK(nvm_v2_module_serialize(&wire,NULL,0,&total)==NVM_V2_OK);uint8_t *saved=malloc(total);CHECK(saved);
    CHECK(nvm_v2_module_serialize(&wire,saved,total,&total)==NVM_V2_OK);FILE *out=fopen(destination,"wb");CHECK(out);CHECK(fwrite(saved,1,total,out)==total);CHECK(!fclose(out));free(saved);nvm_v2_module_free(&wire);
    m->layout_data=NULL;m->layout_size=0;m->ownership_data=NULL;m->ownership_size=0;
    nvm_module_free(m);nvm_v2_module_free(&original_wire);free(bytes);memset(&v,0,sizeof v);
    NvmUnionFieldOrigin row;CHECK(nvm_union_field_execution_origin(proof,0,&row));
#ifdef UFE_ORIGINAL_CHECK
    UFE_ORIGINAL_CHECK(proof);
#endif
    nvm_union_field_execution_free(proof);
}
int main(int argc,char **argv){CHECK(argc==3);annotate(argv[1],argv[2]);printf("PASS %u annotation boundary checks\n",checks);return 0;}
