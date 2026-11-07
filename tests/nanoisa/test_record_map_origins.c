/* I qualify private map proof facts only; public transport is checked separately. */
#define main old_record_array_controls_main
#include "test_record_array_origins.c"
#undef main
#include "../../src/nanoisa/managed_record_map_origins.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanoisa/nvm2c.h"
static void map_authority(Input *c) {
    c->l=c->o=0;b32(c->layouts,&c->l,2);
    for(unsigned i=0;i<2;i++) {
        b8(c->layouts,&c->l,NVM_V2_LAYOUT_STRUCT);b8(c->layouts,&c->l,0);
        b16(c->layouts,&c->l,1);b32(c->layouts,&c->l,NO);
        b8(c->layouts,&c->l,TAG_HASHMAP);b8(c->layouts,&c->l,0);b16(c->layouts,&c->l,0);
        b32(c->layouts,&c->l,NO);b32(c->layouts,&c->l,NO);
    }
    b32(c->ownership,&c->o,3);b32(c->ownership,&c->o,2);
    b8(c->ownership,&c->o,NVM_LAYOUT_COMPLETE);b8(c->ownership,&c->o,NVM_LAYOUT_COMPLETE);
    b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,c->m.function_count);
    for(unsigned i=0;i<c->m.function_count;i++) {
        b16(c->ownership,&c->o,c->fn[i].local_count);b16(c->ownership,&c->o,c->fn[i].arity);desc(c,c->fn[i].result_tag);
        for(unsigned j=0;j<c->fn[i].local_count;j++)desc(c,j<c->fn[i].arity?c->param_rows[i][j]:TAG_VOID);
    }
    b32(c->ownership,&c->o,4);b32(c->ownership,&c->o,0);b32(c->ownership,&c->o,1);
    b16(c->ownership,&c->o,5);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,60);
    b32(c->ownership,&c->o,2);desc(c,TAG_INT);desc(c,TAG_STRING);
    b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,0);
    b32(c->ownership,&c->o,2);
    for(unsigned i=0;i<2;i++) {b32(c->ownership,&c->o,i);b16(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);}
    c->m.layout_size=(uint32_t)c->l;c->m.ownership_size=(uint32_t)c->o;c->m.code_size=(uint32_t)c->n;
}
static void map_new(Input *c,uint8_t key,uint8_t val) {op(c,OP_HM_NEW);b8(c->code,&c->n,key);b8(c->code,&c->n,val);}
static void map_finish(Input *c) {scalar(c,TAG_INT);op(c,OP_RET);c->fn[0].code_length=(uint32_t)c->n;map_authority(c);}
static NvmRecordMapFrozenContext *map_query(Input *c,NvmArrayEligibilityStatus expected) {
    Input saved=*c;NvmRecordMapFrozenContext *p=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult r=nvm_analyze_record_map_origins(&c->m,&p);
    if(r.status!=expected)fprintf(stderr,"I expected map proof%u got%u: %s\n",expected,r.status,r.message);
    CHECK(r.status==expected);CHECK(!memcmp(c,&saved,sizeof saved));
    CHECK(expected==NVM_ARRAY_ELIGIBLE?p!=(void *)(uintptr_t)1:p==(void *)(uintptr_t)1);return p;
}
static void map_basic(Input *c) {
    init(c,TAG_INT,false);map_new(c,TAG_STRING,TAG_INT);
    scalar(c,TAG_STRING);scalar(c,TAG_INT);op(c,OP_HM_SET);record(c,0);
    arg16(c,OP_STORE_LOCAL,0);arg16(c,OP_LOAD_LOCAL,0);arg16(c,OP_STRUCT_GET,0);
    arg16(c,OP_STORE_LOCAL,1);arg16(c,OP_LOAD_LOCAL,1);scalar(c,TAG_STRING);op(c,OP_HM_HAS);op(c,OP_ASSERT);
    arg16(c,OP_LOAD_LOCAL,1);scalar(c,TAG_STRING);op(c,OP_HM_GET);op(c,OP_CAST_INT);op(c,OP_POP);
    arg16(c,OP_LOAD_LOCAL,1);op(c,OP_HM_LEN);op(c,OP_POP);map_finish(c);
}
/* Explicit fixture oracle: only these named roots are public-eligible. Native
 * target intersection is checked at named native/public fixture sites. */
static void map_public_transport(const NvmModule *m,bool expected) {
    CHECK(nvm_verify(m).ok==expected);
    NvmV2Module encoded;NvmV2Result status=nvm_v2_from_nvm_module(m,&encoded);
    CHECK((status==NVM_V2_OK)==expected);if(status==NVM_V2_OK)nvm_v2_module_free(&encoded);
}

static void map_copied(void) {
    Input c;map_basic(&c);map_public_transport(&c.m,true);
    NvmRecordMapFrozenContext *p=map_query(&c,NVM_ARRAY_ELIGIBLE);NvmRecordMapCounts n;
    CHECK(nvm_record_map_frozen_counts(p,&n));CHECK(n.origins==2&&n.map_origins==1&&n.fields==1&&n.checked_map_operations==4);
    uint8_t key=99,value=99;CHECK(nvm_record_map_frozen_origin(p,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
    CHECK(!nvm_record_map_frozen_origin(p,1,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
    CHECK(nvm_record_map_frozen_field(p,1,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
    CHECK(!nvm_record_map_frozen_field(p,1,1,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
    CHECK(!nvm_record_map_frozen_origin(p,0,&key,&key)&&key==TAG_STRING);
    CHECK(!nvm_record_map_frozen_field(p,0,0,&value,&value)&&value==TAG_INT);
    NvmRecordValueOrigins field;CHECK(nvm_record_map_field_value(p,0,&field));CHECK(field.tags==MASK(TAG_HASHMAP)&&field.origins==1&&!field.unknown);
    NvmRecordHeapOrigin origin;CHECK(nvm_record_map_heap_origin(p,0,&origin)&&origin.kind==NVM_RECORD_MAP_ORIGIN);
    memset(c.code,0,sizeof c.code);memset(c.layouts,0,sizeof c.layouts);memset(c.ownership,0,sizeof c.ownership);
    CHECK(nvm_record_map_frozen_field(p,0,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
    nvm_record_map_frozen_free(p);
}
static void map_refusals(void) {
    Input c;
    for(unsigned bad=0;bad<6;bad++) {
        init(&c,TAG_INT,false);map_new(&c,bad==0?TAG_INT:TAG_STRING,bad==1?TAG_STRING:TAG_INT);
        if(bad>=2) {
            scalar(&c,bad==2?TAG_INT:TAG_STRING);scalar(&c,bad==3?TAG_BOOL:TAG_INT);op(&c,OP_HM_SET);
            if(bad==4){scalar(&c,TAG_BOOL);op(&c,OP_HM_DELETE);} /* DELETE requires a STRING key. */
            if(bad==5){op(&c,OP_POP);scalar(&c,TAG_INT);}
        }
        record(&c,0);op(&c,OP_POP);map_finish(&c);map_query(&c,NVM_ARRAY_UNRESOLVED);
    }
    /* The discarded historical alias still retains its invalid write. */
    init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);arg16(&c,OP_STORE_LOCAL,0);
    arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_STRING);scalar(&c,TAG_BOOL);op(&c,OP_HM_SET);op(&c,OP_POP);
    map_new(&c,TAG_STRING,TAG_INT);record(&c,0);op(&c,OP_POP);map_finish(&c);map_query(&c,NVM_ARRAY_UNRESOLVED);
}
static void map_calls(void) {
    Input c;init(&c,TAG_INT,false);c.m.function_count=2;
    map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg16(&c,OP_STORE_LOCAL,0);
    arg16(&c,OP_LOAD_LOCAL,0);arg32(&c,OP_CALL,1);arg16(&c,OP_STRUCT_GET,0);op(&c,OP_HM_LEN);op(&c,OP_POP);
    scalar(&c,TAG_INT);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;
    c.fn[1].code_offset=(uint32_t)c.n;c.fn[1].arity=1;c.fn[1].local_count=1;c.fn[1].result_tag=TAG_STRUCT;c.fn[1].result_count=1;
    c.params[0]=TAG_STRUCT;c.param_rows[1]=c.params;c.m.function_param_types=c.param_rows;
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;
    map_authority(&c);NvmRecordMapFrozenContext *p=map_query(&c,NVM_ARRAY_ELIGIBLE);nvm_record_map_frozen_free(p);
}
static void map_origin_limits(void) {
    for(unsigned extra=0;extra<2;extra++) {
        Input c;init(&c,TAG_INT,false);
        for(unsigned i=0;i<62+extra;i++){map_new(&c,TAG_STRING,TAG_INT);op(&c,OP_POP);}
        map_new(&c,TAG_STRING,TAG_INT);record(&c,0);op(&c,OP_POP);map_finish(&c);
        NvmRecordMapFrozenContext *p=map_query(&c,extra?NVM_ARRAY_LIMIT:NVM_ARRAY_ELIGIBLE);
        if(!extra){NvmRecordMapCounts n;CHECK(nvm_record_map_frozen_counts(p,&n)&&n.origins==64&&n.map_origins==63);nvm_record_map_frozen_free(p);}
    }
}
static void map_global_join(void) {
    for(unsigned bad=0;bad<2;bad++) {
        Input c;init(&c,TAG_INT,false);c.m.function_count=2;
        map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);
        arg32(&c,OP_CALL,1);arg32(&c,OP_LOAD_GLOBAL,0);arg16(&c,OP_STRUCT_GET,0);
        scalar(&c,TAG_STRING);op(&c,OP_HM_GET);op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,TAG_VOID);op(&c,OP_POP);
        scalar(&c,TAG_INT);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;
        c.fn[1].code_offset=(uint32_t)c.n;c.fn[1].local_count=1;c.fn[1].result_tag=TAG_VOID;
        arg32(&c,OP_LOAD_GLOBAL,0);arg16(&c,OP_STRUCT_GET,0);arg16(&c,OP_STORE_LOCAL,0);
        /* Replace the record's map, then mutate its detached alias. */
        arg32(&c,OP_LOAD_GLOBAL,0);map_new(&c,TAG_STRING,TAG_INT);arg16(&c,OP_STRUCT_SET,0);op(&c,OP_POP);
        arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_STRING);scalar(&c,bad?TAG_BOOL:TAG_INT);op(&c,OP_HM_SET);op(&c,OP_POP);op(&c,OP_RET);
        c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);
        NvmRecordMapFrozenContext *p=map_query(&c,bad?NVM_ARRAY_UNRESOLVED:NVM_ARRAY_ELIGIBLE);
        if(!bad){NvmRecordValueOrigins field;CHECK(nvm_record_map_field_value(p,0,&field));CHECK(field.origins==5&&field.tags==MASK(TAG_HASHMAP));nvm_record_map_frozen_free(p);}
    }
    Input c;init(&c,TAG_INT,false);arg32(&c,OP_LOAD_GLOBAL,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);op(&c,OP_POP);map_finish(&c);map_query(&c,NVM_ARRAY_UNRESOLVED);
}
static void map_branch_loop(void) {
    Input c;init(&c,TAG_INT,false);scalar(&c,TAG_BOOL);size_t branch=c.n;arg32(&c,OP_JMP_FALSE,0);
    map_new(&c,TAG_STRING,TAG_INT);arg16(&c,OP_STORE_LOCAL,0);size_t jump=c.n;arg32(&c,OP_JMP,0);
    size_t at=branch+1;b32(c.code,&at,(uint32_t)(c.n-branch));
    map_new(&c,TAG_STRING,TAG_INT);arg16(&c,OP_STORE_LOCAL,0);at=jump+1;b32(c.code,&at,(uint32_t)(c.n-jump));
    size_t header=c.n;scalar(&c,TAG_BOOL);size_t loop_exit=c.n;arg32(&c,OP_JMP_FALSE,0);
    arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_STRING);scalar(&c,TAG_INT);op(&c,OP_HM_SET);op(&c,OP_POP);
    size_t back=c.n;arg32(&c,OP_JMP,(uint32_t)(header-back));at=loop_exit+1;b32(c.code,&at,(uint32_t)(c.n-loop_exit));
    arg16(&c,OP_LOAD_LOCAL,0);record(&c,0);op(&c,OP_POP);map_finish(&c);
    NvmRecordMapFrozenContext *p=map_query(&c,NVM_ARRAY_ELIGIBLE);NvmRecordValueOrigins field;
    CHECK(nvm_record_map_field_value(p,0,&field)&&field.origins==3&&field.tags==MASK(TAG_HASHMAP));nvm_record_map_frozen_free(p);
    /* No GLOBAL declaration may be bypassed by storing a standalone map. */
    init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);arg32(&c,OP_STORE_GLOBAL,0);map_finish(&c);
    NvmModule original=c.m;original.layout_data=NULL;original.layout_size=0;original.ownership_data=NULL;original.ownership_size=0;
    CHECK(nvm_verify(&original).ok);map_query(&c,NVM_ARRAY_UNRESOLVED);
}
static void map_absence_proof(void) {
#ifdef RA_WHITEBOX
    Input c;map_basic(&c);
    /* The result union is retained at the actual original instruction state.
     * No HAS-to-GET presence fact or literal-key interpretation is introduced. */
    Analysis a={0};NvmRecordArrayStructure st={0};NvmRecordMapAnalysisContext maps={0};
    a.module=&c.m;a.structure=&st;a.report.origin_count=1;a.origin_kind[0]=NVM_RECORD_MAP_ORIGIN;
    Function *f=&a.functions[0];VmDecodedInstruction ins[3]={0};Value states[3*2]={{0}};
    uint16_t depths[3]={2,0,0};uint8_t seen[3]={1,0,0},queued[3]={0};uint32_t queue[3]={0};
    f->states=states;f->depths=depths;f->seen=seen;f->queued=queued;f->queue=queue;f->stride=2;f->stack=2;
    f->decoded.instructions=ins;f->decoded.instruction_count=2;
    ins[0].instruction.opcode=OP_HM_GET;ins[1].instruction.opcode=OP_RET;
    states[0]=tag(TAG_HASHMAP);states[0].origins=1;states[1]=tag(TAG_STRING);
    CHECK(walk_map(&a,0,&maps));CHECK(f->result.tags==(MASK(TAG_INT)|MASK(TAG_VOID))&&!f->result.unknown);
#endif
}
static void map_allocations(void) {
#ifdef RA_WHITEBOX
    Input c;map_basic(&c);CHECK(!ra_live&&!ra_bytes);ra_calls=ra_peak=0;
    NvmRecordMapFrozenContext *p=map_query(&c,NVM_ARRAY_ELIGIBLE);size_t count=ra_calls;
    NvmRecordMapCounts n;CHECK(nvm_record_map_frozen_counts(p,&n));CHECK(ra_peak<=n.peak_bytes_reserved);
    nvm_record_map_frozen_free(p);CHECK(!ra_live&&!ra_bytes);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<count;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)mode;p=(void *)(uintptr_t)1;
        NvmArrayEligibilityResult r=nvm_analyze_record_map_origins(&c.m,&p);
        CHECK(r.status!=NVM_ARRAY_ELIGIBLE&&p==(void *)(uintptr_t)1);CHECK(!ra_live&&!ra_bytes);
        ra_fail=SIZE_MAX;ra_persistent=0;p=map_query(&c,NVM_ARRAY_ELIGIBLE);nvm_record_map_frozen_free(p);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I retained %zu allocation prefixes and independent recovery.\n",count);
#endif
}
static void map_original_module(const char *path) {
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
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);map_public_transport(m,true);
#ifdef MAP_ORIGINAL_CHECK
    MAP_ORIGINAL_CHECK(m);
#endif
    nvm_module_free(m);
    uint8_t key,value;CHECK(nvm_record_map_frozen_field(p,0,1,&key,&value)&&key==TAG_STRING&&value==TAG_INT);nvm_record_map_frozen_free(p);
    printf("I copied original fixture proof from %s; public boundaries checked.\n",path);
}
#ifndef MAP_ORIGIN_MAIN
#define MAP_ORIGIN_MAIN main
#endif
int MAP_ORIGIN_MAIN(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);map_copied();map_refusals();map_calls();map_global_join();map_origin_limits();map_branch_loop();map_absence_proof();map_allocations();
    for(int i=1;i<argc;i++)map_original_module(argv[i]);
    printf("PASS %u map origin checks; public boundaries checked\n",checks);return 0;
}
