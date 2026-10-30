#define UFE_MAIN retained_union_recipe_controls
#include "/Users/jkh/Src/nanolang-51-platform-qualification/tests/nanoisa/test_union_field_execution.c"
#include "/Users/jkh/Src/nanolang-51-platform-qualification/src/nanoisa/nvm2c_union_target_private.inc"
int g_argc=0;char **g_argv=NULL;
static void target_original(const char *path) {
 FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long size=ftell(f);CHECK(size>0);rewind(f);
 uint8_t *bytes=malloc((size_t)size);CHECK(bytes);CHECK(fread(bytes,1,(size_t)size,f)==(size_t)size);CHECK(!fclose(f));
 NvmV2Module wire={0};CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&wire)==NVM_V2_OK);
 NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&wire,&m)==NVM_V2_OK);
 Nvm2cUnionTargetResult result=nvm2c_prepare_union_target_source(m);
 printf("target %s status=%u work=%llu scratch=%llu message=%s\n",path,result.eligibility.status,
  (unsigned long long)result.work,(unsigned long long)result.scratch_reserved,result.eligibility.message);fflush(stdout);
 CHECK(result.eligibility.status==NVM_ARRAY_ELIGIBLE);
 NvmUnionFieldExecutionPlan *plan=execution_query(m);NvmUnionFieldImage *image=NULL;
 CHECK(nvm_union_field_image_copy(plan,&image).status==NVM_ARRAY_ELIGIBLE);
 Nvm2cUnionTargetContext context;memset(&context,0xa5,sizeof context);Nvm2cUnionTargetContext prior=context;
 result=nvm2c_union_target(image,UFN_WORK_LIMIT,&context);CHECK(result.eligibility.status==NVM_ARRAY_ELIGIBLE);
 uint64_t exact=result.work;CHECK(context.record_layout==1&&context.union_layout==0);
 context=prior;result=nvm2c_union_target(image,exact-1,&context);
 CHECK(result.eligibility.status==NVM_ARRAY_LIMIT&&!memcmp(&context,&prior,sizeof context));
 result=nvm2c_union_target(image,exact,&context);CHECK(result.eligibility.status==NVM_ARRAY_ELIGIBLE&&result.work==exact);
 context=prior;result=nvm2c_union_target(NULL,UFN_WORK_LIMIT,&context);
 CHECK(result.eligibility.status==NVM_ARRAY_INVALID&&!memcmp(&context,&prior,sizeof context));
 /* Internal image module corruption, restored before independent valid retry. */
 NvmModule *copy=(NvmModule *)nvm_union_field_image_module_internal(image);
 uint32_t saved=copy->function_count;copy->function_count=0;
 result=nvm2c_union_target(image,UFN_WORK_LIMIT,&context);
 CHECK(result.eligibility.status==NVM_ARRAY_INVALID&&!memcmp(&context,&prior,sizeof context));copy->function_count=saved;
 result=nvm2c_union_target(image,UFN_WORK_LIMIT,&context);CHECK(result.eligibility.status==NVM_ARRAY_ELIGIBLE);
 UfnOrigins origins={0};NvmRecordValueOrigins value={0};value.tags=(uint16_t)(1u<<15);
 CHECK(!ufn_origins_value(&origins,value));value.tags=(uint16_t)(1u<<TAG_INT);value.origins=1;CHECK(!ufn_origins_value(&origins,value));
 nvm_union_field_image_free(image);nvm_union_field_execution_free(plan);nvm_module_free(m);nvm_v2_module_free(&wire);free(bytes);
}
static void old_bytes(const char *name) {
 char path[512];snprintf(path,sizeof path,"/private/tmp/nanolang-union-field-baseline-20261022/%s.nvm",name);
 FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long n=ftell(f);CHECK(n>0);rewind(f);
 uint8_t *bytes=malloc((size_t)n);CHECK(bytes);CHECK(fread(bytes,1,(size_t)n,f)==(size_t)n);CHECK(!fclose(f));
 NvmV2Module wire={0};CHECK(nvm_v2_module_deserialize(bytes,(size_t)n,&wire)==NVM_V2_OK);
 NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&wire,&m)==NVM_V2_OK);char error[512]={0};char *text=nvm2c_emit(m,error,sizeof error);CHECK(text);
 snprintf(path,sizeof path,"/private/tmp/nanolang-union-native-baseline-20261022/%s.c",name);
 f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));n=ftell(f);CHECK(n>0);rewind(f);
 char *prior=malloc((size_t)n);CHECK(prior);CHECK(fread(prior,1,(size_t)n,f)==(size_t)n);CHECK(!fclose(f));
 CHECK(strlen(text)==(size_t)n&&!memcmp(text,prior,(size_t)n));printf("old native bytes identical %s %ld\n",name,n);
 free(prior);free(text);nvm_module_free(m);nvm_v2_module_free(&wire);free(bytes);
}
int main(int argc,char **argv) {
 CHECK(argc==7);printf("context=%zu scratch=%zu origins=%zu\n",sizeof(Nvm2cUnionTargetContext),sizeof(Nvm2cUnionTargetScratch),sizeof(UfnOrigins));
 for(int i=1;i<argc;i++)target_original(argv[i]);
 const char *names[]={"seed","c-shadow","canonical","canonical-shadow","raw-program","raw-shadow"};
 for(unsigned i=0;i<6;i++)old_bytes(names[i]);printf("PASS %u target checks six originals\n",checks);return 0;
}
