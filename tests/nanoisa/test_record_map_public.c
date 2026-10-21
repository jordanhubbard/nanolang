/* Actual exported consumer APIs; source publication remains outside this gate. */
#define MAP_ORIGIN_MAIN retained_map_origin_tests
#include "test_record_map_origins.c"
#include "../../src/nanovm/vm.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/disassembler.h"
#include "../../src/nanoisa/record_map_admission.h"
int g_argc;char **g_argv;
static unsigned native_outputs;
static __attribute__((noinline)) void public_execute(const NvmModule *m) {
    VmState vm;vm_init(&vm,m);
    for(unsigned repeat=0;repeat<2;repeat++) {
        VmResult r=vm_execute(&vm);
        if(r!=VM_OK)fprintf(stderr,"public execute %u: %s\n",r,vm.error_msg);
        CHECK(r==VM_OK);CHECK(vm.stack_size==1&&!vm.frame_count);
        CHECK(vm_get_result(&vm).tag==TAG_INT&&vm_get_result(&vm).as.i64==0);
        CHECK(vm_execute(&vm)==VM_ERR_TYPE_ERROR);CHECK(vm.stack_size==1);
        NanoValue result=vm.stack[--vm.stack_size];vm_release(&vm.heap,result);
    }
    NanoValue unchanged=val_int(91);CHECK(vm_invoke(&vm,0,NULL,0,&unchanged)==VM_ERR_TYPE_ERROR);
    CHECK(unchanged.tag==TAG_INT&&unchanged.as.i64==91);
    CHECK(vm_core_execute(&vm).type==TRAP_ERROR);CHECK(!vm.stack_size&&!vm.frame_count);
    vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
static void public_bad_directory(const uint8_t *bytes,size_t size) {
    NvmV2Header h;CHECK(nvm_v2_read_header(bytes,size,&h)==NVM_V2_OK);CHECK(h.section_count>2);
    uint8_t *bad=malloc(size);CHECK(bad);
    for(unsigned mode=0;mode<2;mode++) {
        memcpy(bad,bytes,size);size_t row=h.header_size+(h.section_count-1)*NVM_V2_SECTION_ENTRY_SIZE;
        size_t pos=row;
        if(mode==0){pos=row+8;for(unsigned i=0;i<8;i++)bad[pos+i]=255;}
        else {b32(bad,&pos,NVM_V2_SECTION_LAYOUTS);pos=row+8;for(unsigned i=0;i<8;i++)bad[pos+i]=(uint8_t)(i?0:32);pos=row+16;for(unsigned i=0;i<8;i++)bad[pos+i]=(uint8_t)(i?0:4);}
        NvmV2Module prior;memset(&prior,0xa5,sizeof prior);NvmV2Module output=prior;
        CHECK(nvm_v2_module_deserialize(bad,size,&output)!=NVM_V2_OK);
        CHECK(!memcmp(&output,&prior,sizeof prior));
    }
    free(bad);
}
static void public_check(const NvmModule *m,bool native) {
    CHECK(nvm_record_map_route(m)==NVM_RECORD_MAP_PRESENT);
    NvmVerifyResult verified=nvm_verify(m);if(!verified.ok)fprintf(stderr,"public verify: %s\n",verified.error_msg);
    CHECK(verified.ok);public_execute(m);
    CHECK(nvm_verify_profile(m,NVM_PROFILE_GENERAL).ok);CHECK(!nvm_verify_profile(m,NVM_PROFILE_CLOSED_SCALAR).ok);
    CHECK(nvm_verify_linked(m,NULL,0).ok);const NvmModule *linked[]={m};CHECK(!nvm_verify_linked(m,linked,1).ok);
    CHECK(m->function_count<=256);uint16_t depths[256]={0};
    CHECK(nvm_verify_declared_max_stacks(m,depths,m->function_count).ok);
    for(uint32_t i=0;i<m->function_count;i++){CHECK(nvm_verify_function(m,i).ok);CHECK(nvm_verify_function_max_stack(m,i,&depths[i]).ok);}
    CHECK(nvm_verify_declared_max_stacks(m,depths,m->function_count).ok);
    uint16_t sentinel=65535;CHECK(!nvm_verify_function_max_stack(m,m->function_count,&sentinel).ok);CHECK(sentinel==65535);
    uint32_t legacy_size=77;CHECK(!nvm_serialize(m,&legacy_size)&&legacy_size==0);
    char *assembly=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(assembly);AsmResult assembled;
    NvmModule *text_copy=asm_assemble(assembly,&assembled);
    if(!text_copy)fprintf(stderr,"public assembly: %s\n",assembled.message);
    CHECK(text_copy&&assembled.error==ASM_OK);free(assembly);
    CHECK(text_copy->layout_size==m->layout_size&&!memcmp(text_copy->layout_data,m->layout_data,m->layout_size));
    CHECK(text_copy->ownership_size==m->ownership_size&&!memcmp(text_copy->ownership_data,m->ownership_data,m->ownership_size));
    public_execute(text_copy);nvm_module_free(text_copy);
    NvmV2Module wire;CHECK(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
    size_t size=0;CHECK(nvm_v2_module_serialize(&wire,NULL,0,&size)==NVM_V2_OK);CHECK(size>40);
    uint8_t *bytes=malloc(size);CHECK(bytes);CHECK(nvm_v2_module_serialize(&wire,bytes,size,NULL)==NVM_V2_OK);
    public_bad_directory(bytes,size);
    NvmV2Module decoded;CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);CHECK(nvm_verify(copy).ok);
    CHECK(copy->layout_size==m->layout_size&&!memcmp(copy->layout_data,m->layout_data,m->layout_size));
    CHECK(copy->ownership_size==m->ownership_size&&!memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
    nvm_v2_module_free(&decoded);nvm_v2_module_free(&wire);free(bytes);public_execute(copy);nvm_module_free(copy);
    char error[512];char *c=nvm2c_emit(m,error,sizeof error);
    if(native&&!c)fprintf(stderr,"public native: %s\n",error);
    CHECK(native?c!=NULL:c==NULL);
    if(c) {const char *directory=getenv("RECORD_MAP_PUBLIC_OUTPUT");CHECK(directory);char path[4096];CHECK(snprintf(path,sizeof path,"%s/original-%u.c",directory,native_outputs++)>0);FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(c,1,strlen(c),f)==strlen(c));CHECK(!fclose(f));free(c);}
}
static void public_original_module(const char *path) {
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
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);public_check(m,true);
#ifdef MAP_ORIGINAL_CHECK
    MAP_ORIGINAL_CHECK(m);
#endif
    nvm_module_free(m);
    uint8_t key,value;CHECK(nvm_record_map_frozen_field(p,0,1,&key,&value)&&key==TAG_STRING&&value==TAG_INT);nvm_record_map_frozen_free(p);
    printf("I copied original fixture proof from %s; public transport and consumers qualified.\n",path);
}
static void borrowed_integer(Input *c,int64_t value) {
    op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)b8(c->code,&c->n,(uint8_t)((uint64_t)value>>(8*i)));
}
static __attribute__((noinline)) void borrowed_tree(Input *c,unsigned records,unsigned depth,uint8_t leaf) {
    init(c,TAG_INT,false);c->m.struct_count=records;c->fn[0].result_tag=TAG_INT;
    for(unsigned i=1;i<depth;i++)array(c,TAG_ARRAY);
    array(c,leaf);if(leaf==TAG_U8)borrowed_integer(c,300);else scalar(c,leaf);op(c,OP_ARR_PUSH);
    for(unsigned i=1;i<depth;i++)op(c,OP_ARR_PUSH);
    map_new(c,TAG_STRING,TAG_INT);scalar(c,TAG_STRING);scalar(c,TAG_INT);op(c,OP_HM_SET);
    op(c,OP_STRUCT_LITERAL);b32(c->code,&c->n,0);b16(c->code,&c->n,2);
    for(unsigned i=1;i<records;i++)record(c,i);
    arg32(c,OP_STORE_GLOBAL,0);scalar(c,TAG_INT);op(c,OP_RET);c->fn[0].code_length=(uint32_t)c->n;
    b32(c->layouts,&c->l,records);
    for(unsigned i=0;i<records;i++) {
        b8(c->layouts,&c->l,NVM_V2_LAYOUT_STRUCT);b8(c->layouts,&c->l,0);b16(c->layouts,&c->l,i?1:2);b32(c->layouts,&c->l,NO);
        for(unsigned field=0;field<(i?1u:2u);field++) {
            b8(c->layouts,&c->l,i?TAG_STRUCT:field?TAG_HASHMAP:TAG_ARRAY);b8(c->layouts,&c->l,0);b16(c->layouts,&c->l,0);
            b32(c->layouts,&c->l,i?i-1:NO);b32(c->layouts,&c->l,NO);
        }
    }
    b32(c->ownership,&c->o,3);b32(c->ownership,&c->o,records);
    for(unsigned i=0;i<records;i++)b8(c->ownership,&c->o,NVM_LAYOUT_COMPLETE);
    while(c->o%4)b8(c->ownership,&c->o,0);
    b32(c->ownership,&c->o,1);b16(c->ownership,&c->o,c->fn[0].local_count);b16(c->ownership,&c->o,0);desc(c,TAG_INT);
    for(unsigned i=0;i<c->fn[0].local_count;i++)desc(c,TAG_VOID);
    b32(c->ownership,&c->o,4);b32(c->ownership,&c->o,0);b32(c->ownership,&c->o,2);
    b16(c->ownership,&c->o,2);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,20+depth*8);b32(c->ownership,&c->o,depth);
    for(unsigned i=0;i<depth;i++){b8(c->ownership,&c->o,i+1==depth?leaf:TAG_ARRAY);b8(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,i+1==depth?NO:i+1);}
    b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);
    b16(c->ownership,&c->o,5);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,48);b32(c->ownership,&c->o,2);desc(c,TAG_INT);desc(c,TAG_STRING);
    b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,0);b32(c->ownership,&c->o,1);
    b32(c->ownership,&c->o,0);b16(c->ownership,&c->o,1);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);
    c->m.layout_size=(uint32_t)c->l;c->m.ownership_size=(uint32_t)c->o;c->m.code_size=(uint32_t)c->n;
}
static __attribute__((noinline)) void borrowed_source_global(const char *path) {
    FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long size=ftell(f);CHECK(size>0&&size<1048576);rewind(f);
    uint8_t *bytes=malloc((size_t)size);CHECK(bytes&&fread(bytes,1,(size_t)size,f)==(size_t)size);CHECK(!fclose(f));
    NvmV2Module v;NvmModule *m=NULL;CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&v)==NVM_V2_OK);
    CHECK(nvm_v2_to_nvm_module(&v,&m)==NVM_V2_OK);nvm_v2_module_free(&v);free(bytes);CHECK(m&&nvm_verify(m).ok);
    CHECK(m->struct_count==1&&!m->layout_size&&!m->ownership_size);
    Input c={0};b32(c.layouts,&c.l,1);b8(c.layouts,&c.l,NVM_V2_LAYOUT_STRUCT);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,1);b32(c.layouts,&c.l,NO);
    b8(c.layouts,&c.l,TAG_HASHMAP);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,0);b32(c.layouts,&c.l,NO);b32(c.layouts,&c.l,NO);
    b32(c.ownership,&c.o,3);b32(c.ownership,&c.o,1);b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);while(c.o%4)b8(c.ownership,&c.o,0);
    b32(c.ownership,&c.o,m->function_count);
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *fn=&m->functions[i];b16(c.ownership,&c.o,fn->local_count);b16(c.ownership,&c.o,fn->arity);desc(&c,fn->result_tag);
        for(uint16_t j=0;j<fn->local_count;j++)desc(&c,j<fn->arity?m->function_param_types[i][j]:TAG_VOID);
    }
    b32(c.ownership,&c.o,4);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b16(c.ownership,&c.o,5);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,48);b32(c.ownership,&c.o,2);desc(&c,TAG_INT);desc(&c,TAG_STRING);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b32(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);
    m->layout_data=malloc(c.l);m->ownership_data=malloc(c.o);CHECK(m->layout_data&&m->ownership_data);
    memcpy(m->layout_data,c.layouts,c.l);memcpy(m->ownership_data,c.ownership,c.o);m->layout_size=(uint32_t)c.l;m->ownership_size=(uint32_t)c.o;
    public_check(m,true);nvm_module_free(m);printf("I checked real source-global input %s.\n",path);
}
static __attribute__((noinline)) void borrowed_u8_source(const char *path) {
    FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long size=ftell(f);CHECK(size>0&&size<1048576);rewind(f);
    uint8_t *bytes=malloc((size_t)size);CHECK(bytes&&fread(bytes,1,(size_t)size,f)==(size_t)size);CHECK(!fclose(f));
    NvmV2Module v;NvmModule *m=NULL;CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&v)==NVM_V2_OK);
    CHECK(nvm_v2_to_nvm_module(&v,&m)==NVM_V2_OK);nvm_v2_module_free(&v);free(bytes);CHECK(m&&nvm_verify(m).ok);
    CHECK(m->struct_count==1&&!m->layout_size&&!m->ownership_size);
    Input shape;borrowed_tree(&shape,1,2,TAG_U8);Input c={0};
    memcpy(c.layouts,shape.layouts,shape.l);c.l=shape.l;
    b32(c.ownership,&c.o,3);b32(c.ownership,&c.o,1);b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);while(c.o%4)b8(c.ownership,&c.o,0);
    b32(c.ownership,&c.o,m->function_count);
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *fn=&m->functions[i];b16(c.ownership,&c.o,fn->local_count);b16(c.ownership,&c.o,fn->arity);desc(&c,fn->result_tag);
        for(uint16_t j=0;j<fn->local_count;j++)desc(&c,j<fn->arity?m->function_param_types[i][j]:TAG_VOID);
    }
    b32(c.ownership,&c.o,4);b32(c.ownership,&c.o,0);
    /* Exact two-row ARRAY and one STRING→INT pair extension from the same
     * field schema; function descriptors above use this original producer. */
    size_t extensions=4+8+(20+2*8)+8+48;
    CHECK(shape.o>=extensions&&c.o+extensions<sizeof c.ownership);
    memcpy(c.ownership+c.o,shape.ownership+shape.o-extensions,extensions);c.o+=extensions;
    m->layout_data=malloc(c.l);m->ownership_data=malloc(c.o);CHECK(m->layout_data&&m->ownership_data);
    memcpy(m->layout_data,c.layouts,c.l);memcpy(m->ownership_data,c.ownership,c.o);
    m->layout_size=(uint32_t)c.l;m->ownership_size=(uint32_t)c.o;
    public_check(m,true);nvm_module_free(m);printf("I executed exact source U8/map input %s.\n",path);
}

static void public_native_boundaries(void) {
    for(unsigned mode=0;mode<2;mode++) {
        Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);scalar(&c,TAG_STRING);
        if(mode){scalar(&c,TAG_STRING);op(&c,OP_STR_CONCAT);}
        else {c.strings[1]="a\0b";c.lengths[1]=3;}
        scalar(&c,TAG_INT);op(&c,OP_HM_SET);op(&c,OP_AGG_PACK);b8(c.code,&c.n,AGG_RECORD);b32(c.code,&c.n,0);b16(c.code,&c.n,0);b16(c.code,&c.n,1);
        op(&c,OP_POP);map_finish(&c);public_check(&c.m,false);
        char error[512];CHECK(!nvm2c_emit(&c.m,error,sizeof error));
        CHECK(strstr(error,mode?"string producer":"NUL-free")!=NULL);
    }
}
static void public_route_boundaries(void) {
    Input c;init(&c,TAG_INT,false);scalar(&c,TAG_INT);op(&c,OP_RET);c.fn[0].code_length=c.m.code_size=(uint32_t)c.n;
    size_t bytes=4+257*8;uint8_t *layouts=calloc(1,bytes);CHECK(layouts);size_t pos=0;b32(layouts,&pos,257);
    for(unsigned i=0;i<257;i++){b8(layouts,&pos,NVM_V2_LAYOUT_STRUCT);b8(layouts,&pos,0);b16(layouts,&pos,0);b32(layouts,&pos,NO);}
    c.m.layout_data=layouts;c.m.layout_size=(uint32_t)bytes;c.m.ownership_data=NULL;c.m.ownership_size=0;c.m.struct_count=257;
    CHECK(nvm_record_map_route(&c.m)==NVM_RECORD_MAP_ABSENT);CHECK(nvm_verify(&c.m).ok);
    free(layouts);c.m.layout_data=NULL;c.m.layout_size=0;c.m.struct_count=0;
    /* The selected marker follows the bounded framing scan. Full old validation
     * still refuses this unsupported metadata; DEFER cannot authorize dispatch. */
    bytes=12+257*(4+8193*8)+4+4+4+8+4;uint8_t *ownership=calloc(1,bytes);CHECK(ownership);pos=0;
    b32(ownership,&pos,3);b32(ownership,&pos,0);b32(ownership,&pos,257);
    for(unsigned i=0;i<257;i++){b16(ownership,&pos,8192);b16(ownership,&pos,0);pos+=8193*8;}
    b32(ownership,&pos,4);b32(ownership,&pos,0);b32(ownership,&pos,1);b16(ownership,&pos,5);b16(ownership,&pos,1);b32(ownership,&pos,4);b32(ownership,&pos,0);
    CHECK(pos==bytes&&pos>16*1024*1024);c.m.ownership_data=ownership;c.m.ownership_size=(uint32_t)bytes;
    CHECK(nvm_record_map_route(&c.m)==NVM_RECORD_MAP_DEFER);CHECK(!nvm_verify(&c.m).ok);
    char error[256];CHECK(!nvm2c_emit(&c.m,error,sizeof error));
    VmState *vm=malloc(sizeof *vm);CHECK(vm);vm_init(vm,&c.m);CHECK(vm_execute(vm)!=VM_OK);CHECK(!vm->stack_size&&!vm->frame_count);vm_destroy(vm);free(vm);
    ownership[0]=255;CHECK(nvm_record_map_route(&c.m)==NVM_RECORD_MAP_DEFER);CHECK(!nvm_verify(&c.m).ok);free(ownership);
}

static __attribute__((noinline)) void public_global_failures(void) {
    for(unsigned assertion=0;assertion<2;assertion++) {
        Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);scalar(&c,TAG_STRING);scalar(&c,TAG_INT);op(&c,OP_HM_SET);
        record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);
        if(assertion){op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,0);op(&c,OP_ASSERT);}
        map_finish(&c);CHECK(nvm_verify(&c.m).ok);VmState vm;vm_init(&vm,&c.m);
        CHECK(vm_execute(&vm)==(assertion?VM_ERR_ASSERT_FAILED:VM_OK));
        CHECK(!vm.frame_count&&vm.stack_size==(assertion?0u:1u));
        if(!assertion)vm.stack_size--;
        CHECK(vm.global_count==1&&vm.globals[0].tag==TAG_STRUCT);
        NanoValue alias=vm.globals[0];vm_retain(&vm.heap,alias);
        VmHashMap *map=alias.as.sval->fields[0].as.hashmap;uint32_t bucket=0;
        while(bucket<map->bucket_count&&map->entries[bucket].state!=1)bucket++;
        CHECK(bucket<map->bucket_count);NanoValue saved=map->entries[bucket].value;
        CHECK(saved.tag==TAG_INT&&saved.as.i64==0);
        if(!assertion) {
            map->entries[bucket].value=val_bool(true);void *identity=vm.globals[0].as.obj;
            CHECK(vm_execute(&vm)==VM_ERR_TYPE_ERROR);CHECK(!vm.stack_size&&!vm.frame_count&&vm.globals[0].as.obj==identity);
            CHECK(map->entries[bucket].value.tag==TAG_BOOL);map->entries[bucket].value=saved;
            CHECK(vm_execute(&vm)==VM_OK);CHECK(vm_get_result(&vm).as.i64==0);vm.stack_size--;
        } else {
            CHECK(vm.last_error==VM_ERR_ASSERT_FAILED);CHECK(vm_execute(&vm)==VM_ERR_ASSERT_FAILED);
            CHECK(vm.last_error==VM_ERR_ASSERT_FAILED&&!vm.stack_size&&!vm.frame_count);
        }
        CHECK(alias.as.obj!=vm.globals[0].as.obj);CHECK(map->entries[bucket].value.tag==TAG_INT&&map->entries[bucket].value.as.i64==0);
        vm_release(&vm.heap,alias);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
    }
}

#ifndef MAP_PUBLIC_MAIN
#define MAP_PUBLIC_MAIN main
#endif
int MAP_PUBLIC_MAIN(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);CHECK(argc==1||argc==5||argc==9||argc==13);
    Input c;map_basic(&c);public_check(&c.m,false);public_native_boundaries();public_route_boundaries();public_global_failures();
    for(int i=1;i<argc&&i<=4;i++)public_original_module(argv[i]);
    for(int i=5;i<argc&&i<=8;i++)borrowed_source_global(argv[i]);
    for(int i=9;i<argc;i++)borrowed_u8_source(argv[i]);
    printf("PASS %u actual public map consumer checks\n",checks);return 0;
}
