#define MAP_PUBLIC_MAIN retained_public_main
#include "/Users/jkh/Src/nanolang-51-platform-qualification/tests/nanoisa/test_record_map_public.c"
static bool native_expected;
static void concat_original_module(const char *path) {
    FILE *file=fopen(path,"rb");CHECK(file);CHECK(!fseek(file,0,SEEK_END));long length=ftell(file);CHECK(length>0&&length<1048576);rewind(file);
    uint8_t *bytes=malloc((size_t)length);CHECK(bytes);CHECK(fread(bytes,1,(size_t)length,file)==(size_t)length);CHECK(!fclose(file));
    NvmV2Module decoded;CHECK(nvm_v2_module_deserialize(bytes,(size_t)length,&decoded)==NVM_V2_OK);
    NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&m)==NVM_V2_OK);nvm_v2_module_free(&decoded);free(bytes);CHECK(m);CHECK(nvm_verify(m).ok);
    CHECK(!m->layout_size&&!m->ownership_size&&m->struct_count==1);
    /* I bind the exact unchanged CollectResult fixture declaration only. I do
     * not infer this schema from names or feed it to a public publisher. */
    Input c={0};b32(c.layouts,&c.l,1);b8(c.layouts,&c.l,NVM_V2_LAYOUT_STRUCT);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,3);b32(c.layouts,&c.l,NO);
    const uint8_t fields[]={TAG_ARRAY,TAG_HASHMAP,TAG_BOOL};
    for(unsigned i=0;i<3;i++){b8(c.layouts,&c.l,fields[i]);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,0);b32(c.layouts,&c.l,NO);b32(c.layouts,&c.l,NO);}
    b32(c.ownership,&c.o,3);b32(c.ownership,&c.o,1);b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);while(c.o%4)b8(c.ownership,&c.o,0);
    b32(c.ownership,&c.o,m->function_count);
    for(unsigned i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *fn=&m->functions[i];b16(c.ownership,&c.o,fn->local_count);b16(c.ownership,&c.o,fn->arity);desc(&c,fn->result_tag);
        for(unsigned j=0;j<fn->local_count;j++)desc(&c,j<fn->arity?m->function_param_types[i][j]:TAG_VOID);
    }
    b32(c.ownership,&c.o,4);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,2);
    b16(c.ownership,&c.o,2);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,28);b32(c.ownership,&c.o,1);desc(&c,TAG_STRING);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);
    b16(c.ownership,&c.o,5);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,48);b32(c.ownership,&c.o,2);desc(&c,TAG_INT);desc(&c,TAG_STRING);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b32(c.ownership,&c.o,0);b16(c.ownership,&c.o,1);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);
    m->layout_data=malloc(c.l);m->ownership_data=malloc(c.o);CHECK(m->layout_data&&m->ownership_data);
    memcpy(m->layout_data,c.layouts,c.l);memcpy(m->ownership_data,c.ownership,c.o);m->layout_size=(uint32_t)c.l;m->ownership_size=(uint32_t)c.o;
    NvmRecordMapFrozenContext *p=(void *)(uintptr_t)1;NvmArrayEligibilityResult result=nvm_analyze_record_map_origins(m,&p);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"I refused original private fixture: %s\n",result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);public_check(m,native_expected); if(!native_expected){char error[512]; char *native=nvm2c_emit(m,error,sizeof error); CHECK(!native); CHECK(strstr(error,"string producer")); printf("Expected selected native refusal: %s\n",error);}
#ifdef MAP_ORIGINAL_CHECK
    MAP_ORIGINAL_CHECK(m);
#endif
    nvm_module_free(m);
    uint8_t key,value;CHECK(nvm_record_map_frozen_field(p,0,1,&key,&value)&&key==TAG_STRING&&value==TAG_INT);nvm_record_map_frozen_free(p);
    printf("I copied original fixture proof from %s; public transport and consumers qualified.\n",path);
}
int main(int argc,char **argv){CHECK(argc==9);for(int i=1;i<argc;i++){native_expected=i<=4;concat_original_module(argv[i]);}printf("PASS %u annotated baseline checks\n",checks);return 0;}
