/* First private native publication gate; original code bytes remain unchanged. */
#include "../../src/nanoisa/nvm2c_map_private.h"
#include "../../src/nanovm/record_map_runtime_private.h"
#include "../../src/nanoisa/record_map_snapshot_private.h"
static void native_original(const NvmModule *);
#define MAP_ORIGINAL_CHECK native_original
#define MAP_ORIGIN_MAIN native_retained_origin_controls
#include "test_record_map_origins.c"
#ifdef MAP_NATIVE_WHITEBOX
#ifdef MAP_NATIVE_ALLOC
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#endif
#include "../../src/nanoisa/nvm2c.c"
#ifdef MAP_NATIVE_ALLOC
#undef malloc
#undef calloc
#undef realloc
#undef free
#endif
#endif
static const char *native_output_directory;
static NvmRecordMapImage *native_images[4];
static char *native_texts[4];
static unsigned native_outputs;
int g_argc;char **g_argv;
static void native_original(const NvmModule *m) {
    char error[512]={0};char *source=nvm2c_emit_record_map_private(m,error,sizeof error);
    if(!source)fprintf(stderr,"private native original %u: %s\n",native_outputs,error);
    CHECK(source);char *public_source=nvm2c_emit(m,error,sizeof error);CHECK(public_source&&!strcmp(public_source,source));free(public_source);
    NvmRecordMapExecutionPlan *plan=NULL;
    CHECK(nvm_prepare_record_map_execution(m,&plan).status==NVM_ARRAY_ELIGIBLE);
    CHECK(nvm_record_map_image_copy(plan,&native_images[native_outputs]).status==NVM_ARRAY_ELIGIBLE);
    nvm_record_map_execution_free(plan);native_texts[native_outputs]=source;
    char path[4096];int length=snprintf(path,sizeof path,"%s/original-%u.c",native_output_directory,native_outputs++);
    CHECK(length>0&&(size_t)length<sizeof path);FILE *f=fopen(path,"wb");CHECK(f);
    size_t bytes=strlen(source);CHECK(fwrite(source,1,bytes,f)==bytes);CHECK(!fclose(f));
}
static void native_integer(Input *c,int64_t value) {
    op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)b8(c->code,&c->n,(uint8_t)((uint64_t)value>>(8u*i)));
}
static void native_record(Input *c,unsigned ordinal) {
    op(c,OP_AGG_PACK);b8(c->code,&c->n,AGG_RECORD);b32(c->code,&c->n,ordinal);
    b16(c->code,&c->n,0);b16(c->code,&c->n,1);
}
static void native_map_value(Input *c,int64_t value) {
    map_new(c,TAG_STRING,TAG_INT);scalar(c,TAG_STRING);native_integer(c,value);op(c,OP_HM_SET);
}
static void native_expect(Input *c,int64_t value) {
    native_integer(c,value);op(c,OP_EQ);op(c,OP_ASSERT);
}
static void native_fixture(Input *c,const char *name,VmResult expected) {
    NvmRecordMapExecutionPlan *plan=NULL;VmRecordMapPrivate *vm=NULL;
    NvmArrayEligibilityResult proof=nvm_prepare_record_map_execution(&c->m,&plan);
    if(proof.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"native fixture %s proof: %s\n",name,proof.message);
    CHECK(proof.status==NVM_ARRAY_ELIGIBLE);
    CHECK(vm_record_map_private_create(plan,&vm).status==NVM_ARRAY_ELIGIBLE);
    nvm_record_map_execution_free(plan);CHECK(vm_record_map_private_run(vm)==expected);
    vm_record_map_private_destroy(vm);map_public_transport(&c->m,true);
    char error[512]={0};char *text=nvm2c_emit_record_map_private(&c->m,error,sizeof error);
    if(!text)fprintf(stderr,"native fixture %s output: %s\n",name,error);
    CHECK(text);char *public_text=nvm2c_emit(&c->m,error,sizeof error);CHECK(public_text&&!strcmp(public_text,text));free(public_text);char path[4096];CHECK(snprintf(path,sizeof path,"%s/%s.c",native_output_directory,name)>0);
    FILE *f=fopen(path,"wb");CHECK(f);size_t bytes=strlen(text);CHECK(fwrite(text,1,bytes,f)==bytes);CHECK(!fclose(f));free(text);
}
static void native_semantics(void) {
    Input c;init(&c,TAG_INT,false);native_map_value(&c,7);native_record(&c,0);arg16(&c,OP_STORE_LOCAL,0);
    arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STORE_LOCAL,1);
    arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_AGG_GET,0);arg16(&c,OP_STORE_LOCAL,2);
    arg16(&c,OP_LOAD_LOCAL,0);native_map_value(&c,9);arg16(&c,OP_AGG_SET,0);op(&c,OP_POP);
    arg16(&c,OP_LOAD_LOCAL,1);arg16(&c,OP_AGG_GET,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);native_expect(&c,9);
    arg16(&c,OP_LOAD_LOCAL,2);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);native_expect(&c,7);
    map_finish(&c);native_fixture(&c,"alias-replacement",VM_OK);
    for(unsigned wrong=0;wrong<2;wrong++) {
        init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);native_record(&c,0);arg16(&c,OP_AGG_GET,0);
        scalar(&c,TAG_STRING);op(&c,OP_HM_GET);op(&c,OP_PUSH_VOID);op(&c,OP_EQ);
        if(wrong)op(&c,OP_BOOL_NOT);
        op(&c,OP_ASSERT);map_finish(&c);
        native_fixture(&c,wrong?"missing-assert":"missing-void",wrong?VM_ERR_ASSERT_FAILED:VM_OK);
    }
    init(&c,TAG_INT,false);map_finish(&c);native_fixture(&c,"unused-declaration",VM_OK);
    init(&c,TAG_INT,false);arg32(&c,OP_LOAD_GLOBAL,0);arg16(&c,OP_AGG_GET,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);native_expect(&c,17);map_finish(&c);
    c.m.function_count=2;c.fn[1].name_idx=3;c.fn[1].code_offset=(uint32_t)c.n;
    c.fn[1].result_count=1;c.fn[1].result_tag=TAG_INT;
    native_map_value(&c,17);native_record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);native_integer(&c,81);op(&c,OP_RET);
    c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);native_fixture(&c,"initializer",VM_OK);
    init(&c,TAG_INT,false);native_map_value(&c,37);native_record(&c,0);arg16(&c,OP_STORE_LOCAL,0);
    for(unsigned i=0;i<17;i++)arg16(&c,OP_LOAD_LOCAL,0);
    arg32(&c,OP_TAIL_CALL,1);c.fn[0].code_length=(uint32_t)c.n;c.m.function_count=2;
    c.fn[1].name_idx=2;c.fn[1].code_offset=(uint32_t)c.n;c.fn[1].arity=17;c.fn[1].local_count=17;
    c.fn[1].result_count=1;c.fn[1].result_tag=TAG_INT;uint8_t params[17];memset(params,TAG_STRUCT,sizeof params);
    c.param_rows[1]=params;c.m.function_param_types=c.param_rows;
    arg16(&c,OP_LOAD_LOCAL,16);arg16(&c,OP_AGG_GET,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);native_expect(&c,37);
    scalar(&c,TAG_INT);op(&c,OP_RET);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);
    native_fixture(&c,"tail-duplicates",VM_OK);
}
static void native_nonzero_layout(void) {
    Input c;init(&c,TAG_INT,false);native_integer(&c,11);native_map_value(&c,9);array(&c,TAG_INT);
    op(&c,OP_AGG_PACK);b8(c.code,&c.n,AGG_RECORD);size_t ordinal_at=c.n;b32(c.code,&c.n,1);b16(c.code,&c.n,0);b16(c.code,&c.n,3);
    arg16(&c,OP_STORE_LOCAL,0);arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STORE_LOCAL,1);
    arg16(&c,OP_LOAD_LOCAL,0);native_map_value(&c,12);arg16(&c,OP_AGG_SET,1);op(&c,OP_POP);
    arg16(&c,OP_LOAD_LOCAL,1);arg16(&c,OP_AGG_GET,0);native_expect(&c,11);
    arg16(&c,OP_LOAD_LOCAL,1);arg16(&c,OP_AGG_GET,2);op(&c,OP_ARR_LEN);native_expect(&c,0);
    arg16(&c,OP_LOAD_LOCAL,1);arg16(&c,OP_AGG_GET,1);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);native_expect(&c,12);
    map_finish(&c);c.l=c.o=0;b32(c.layouts,&c.l,2);
    for(unsigned layout=0;layout<2;layout++) {
        b8(c.layouts,&c.l,NVM_V2_LAYOUT_STRUCT);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,layout?3:1);b32(c.layouts,&c.l,NO);
        const uint8_t tags[]={TAG_INT,TAG_HASHMAP,TAG_ARRAY};
        for(unsigned f=0;f<(layout?3u:1u);f++){b8(c.layouts,&c.l,layout?tags[f]:TAG_HASHMAP);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,0);b32(c.layouts,&c.l,NO);b32(c.layouts,&c.l,NO);}
    }
    b32(c.ownership,&c.o,3);b32(c.ownership,&c.o,2);b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);b16(c.ownership,&c.o,0);
    b32(c.ownership,&c.o,1);b16(c.ownership,&c.o,c.fn[0].local_count);b16(c.ownership,&c.o,0);desc(&c,TAG_INT);
    for(unsigned i=0;i<c.fn[0].local_count;i++)desc(&c,TAG_VOID);
    b32(c.ownership,&c.o,4);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,2);
    b16(c.ownership,&c.o,2);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,28);b32(c.ownership,&c.o,1);desc(&c,TAG_INT);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,1);b16(c.ownership,&c.o,2);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);
    b16(c.ownership,&c.o,5);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,60);b32(c.ownership,&c.o,2);desc(&c,TAG_INT);desc(&c,TAG_STRING);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,2);
    for(unsigned i=0;i<2;i++){b32(c.ownership,&c.o,i);b16(c.ownership,&c.o,(uint16_t)i);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);}
    c.m.layout_size=(uint32_t)c.l;c.m.ownership_size=(uint32_t)c.o;
    native_fixture(&c,"nonzero-layout",VM_OK);
    c.code[ordinal_at+2]=1;NvmRecordMapExecutionPlan *sentinel=(void *)(uintptr_t)1;
    CHECK(nvm_prepare_record_map_execution(&c.m,&sentinel).status!=NVM_ARRAY_ELIGIBLE);CHECK(sentinel==(void *)(uintptr_t)1);
    char error[512];CHECK(!nvm2c_emit_record_map_private(&c.m,error,sizeof error));
}
static void native_growth(void) {
    Input c;init(&c,TAG_INT,false);char storage[24][16];char *strings[24];uint32_t lengths[24];
    for(unsigned i=0;i<24;i++){CHECK(snprintf(storage[i],sizeof storage[i],"key%u",i)>0);strings[i]=storage[i];lengths[i]=(uint32_t)strlen(strings[i]);}
    strings[0]="main";lengths[0]=4;c.m.strings=strings;c.m.string_lengths=lengths;c.m.string_count=24;
    map_new(&c,TAG_STRING,TAG_INT);
    for(unsigned i=1;i<=20;i++){arg32(&c,OP_PUSH_STR,i);native_integer(&c,i);op(&c,OP_HM_SET);}
    native_record(&c,0);arg16(&c,OP_AGG_GET,0);op(&c,OP_HM_LEN);native_expect(&c,20);map_finish(&c);
    native_fixture(&c,"growth",VM_OK);
}
static void native_concat_lifetime(bool empty) {
    Input c;init(&c,TAG_INT,false);c.strings[1]=empty?"":"\xce\xba";
    c.lengths[1]=(uint32_t)strlen(c.strings[1]);
    map_new(&c,TAG_STRING,TAG_INT);native_record(&c,0);arg16(&c,OP_STORE_LOCAL,0);
    for(unsigned value=7;value<=9;value+=2) {
        arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_AGG_GET,0);
        arg32(&c,OP_CALL,1);arg16(&c,OP_STORE_LOCAL,1);arg16(&c,OP_LOAD_LOCAL,1);
        native_integer(&c,value);op(&c,OP_HM_SET);op(&c,OP_POP);
    }
    arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_AGG_GET,0);op(&c,OP_HM_LEN);native_expect(&c,1);
    arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_AGG_GET,0);arg32(&c,OP_CALL,1);
    op(&c,OP_HM_GET);native_expect(&c,9);map_finish(&c);
    c.m.function_count=3;c.fn[1].name_idx=2;c.fn[1].code_offset=(uint32_t)c.n;
    c.fn[1].result_count=1;c.fn[1].result_tag=TAG_STRING;c.fn[1].local_count=1;
    scalar(&c,TAG_STRING);scalar(&c,TAG_STRING);op(&c,OP_STR_CONCAT);
    arg16(&c,OP_STORE_LOCAL,0);arg16(&c,OP_LOAD_LOCAL,0);arg32(&c,OP_TAIL_CALL,2);
    c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;
    c.fn[2].name_idx=1;c.fn[2].code_offset=(uint32_t)c.n;
    c.fn[2].result_count=1;c.fn[2].result_tag=TAG_STRING;c.fn[2].arity=c.fn[2].local_count=1;
    uint8_t parameter=TAG_STRING;c.param_rows[2]=&parameter;c.m.function_param_types=c.param_rows;
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);c.fn[2].code_length=(uint32_t)c.n-c.fn[2].code_offset;
    map_authority(&c);native_fixture(&c,empty?"concat-empty":"concat-lifetime",VM_OK);
}
static void native_target_refusal(Input *c,const char *diagnostic,bool public_eligible) {
    NvmRecordMapExecutionPlan *plan=NULL;VmRecordMapPrivate *vm=NULL;
    CHECK(nvm_prepare_record_map_execution(&c->m,&plan).status==NVM_ARRAY_ELIGIBLE);
    CHECK(vm_record_map_private_create(plan,&vm).status==NVM_ARRAY_ELIGIBLE);nvm_record_map_execution_free(plan);
    CHECK(vm_record_map_private_run(vm)==VM_OK);vm_record_map_private_destroy(vm);
    char error[512]={0};CHECK(!nvm2c_emit_record_map_private(&c->m,error,sizeof error));CHECK(strstr(error,diagnostic));map_public_transport(&c->m,public_eligible);
    CHECK(!nvm2c_emit(&c->m,error,sizeof error));CHECK(strstr(error,public_eligible?diagnostic:"zero-argument INT"));
}
static void native_target_negatives(void) {
    Input c;init(&c,TAG_INT,false);native_integer(&c,12);op(&c,OP_CAST_STRING);op(&c,OP_POP);map_finish(&c);
    native_target_refusal(&c,"string producer",true);
    init(&c,TAG_INT,false);c.strings[1]="x\0a";c.lengths[1]=3;native_map_value(&c,7);native_record(&c,0);op(&c,OP_POP);map_finish(&c);
    native_target_refusal(&c,"NUL-free",true);
    init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);native_record(&c,0);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;c.fn[0].result_tag=TAG_STRUCT;map_authority(&c);
    native_target_refusal(&c,"INT native map entry",false);
    init(&c,TAG_INT,false);map_finish(&c);c.m.function_count=2;c.fn[1].name_idx=3;c.fn[1].code_offset=(uint32_t)c.n;
    c.fn[1].result_count=1;c.fn[1].result_tag=TAG_STRUCT;map_new(&c,TAG_STRING,TAG_INT);native_record(&c,0);op(&c,OP_RET);
    c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);native_target_refusal(&c,"VOID/INT native map initializer",false);
    map_basic(&c);native_target_refusal(&c,"STRUCT transport",true);
    init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);scalar(&c,TAG_STRING);op(&c,OP_HM_DELETE);op(&c,OP_POP);map_finish(&c);
    NvmRecordMapExecutionPlan *sentinel=(void *)(uintptr_t)1;
    CHECK(nvm_prepare_record_map_execution(&c.m,&sentinel).status!=NVM_ARRAY_ELIGIBLE);CHECK(sentinel==(void *)(uintptr_t)1);
    char error[512];CHECK(!nvm2c_emit_record_map_private(&c.m,error,sizeof error));
}
#ifndef MAP_NATIVE_MAIN
#define MAP_NATIVE_MAIN main
#endif
int MAP_NATIVE_MAIN(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);CHECK(argc==2||argc==6);native_output_directory=argv[1];
    for(int i=2;i<argc;i++)map_original_module(argv[i]);
    CHECK(native_outputs==(unsigned)(argc-2));
    for(unsigned i=0;i<native_outputs;i++) {
#ifdef MAP_NATIVE_WHITEBOX
        char error[512]={0};char *copy=nvm2c_map_image_emit(native_images[i],error,sizeof error);
        CHECK(copy&&!strcmp(copy,native_texts[i]));free(copy);
        const NvmModule *snapshot=nvm_record_map_image_borrow(native_images[i]);uint8_t old=snapshot->code[0];
        snapshot->code[0]^=1;error[0]=0;CHECK(!nvm2c_map_image_emit(native_images[i],error,sizeof error));CHECK(error[0]);snapshot->code[0]=old;
        copy=nvm2c_map_image_emit(native_images[i],error,sizeof error);CHECK(copy&&!strcmp(copy,native_texts[i]));free(copy);
#endif
        nvm_record_map_image_free(native_images[i]);free(native_texts[i]);
    }
    native_semantics();native_nonzero_layout();native_growth();native_concat_lifetime(false);native_concat_lifetime(true);native_target_negatives();printf("PASS %u private native publication checks; public boundaries checked.\n",checks);return 0;
}
