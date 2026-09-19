/* I qualify bounded shared-DAG queries and allocation-failure atomicity. */
#define main prior_descriptor_main
#include "test_owned_result_descriptors.c"
#undef main
static unsigned allocations;static bool fail_query;
void *nested_query_calloc(size_t n,size_t s){allocations++;return fail_query?NULL:calloc(n,s);}
static NvmModule *dag(unsigned count,unsigned width,bool floating,bool ordinary_child) {
    NvmModule *m=fixture(TAG_STRUCT,0,2);m->struct_count=count;
    NvmV2Layout *rows=calloc(count,sizeof(*rows));CHECK(rows);
    for(unsigned i=0;i<count;i++) {
        unsigned fields=i==count-1?width:i?2:1;
        rows[i]=(NvmV2Layout){NVM_V2_LAYOUT_STRUCT,(uint16_t)fields,NVM_V2_NO_INDEX,calloc(fields,sizeof(NvmV2LayoutField))};CHECK(rows[i].fields);
        for(unsigned f=0;f<fields;f++) rows[i].fields[f]=i?
            (NvmV2LayoutField){TAG_STRUCT,(i==count-1&&f==0)?0:i-1,NVM_V2_NO_INDEX}:
            (NvmV2LayoutField){floating?TAG_FLOAT:TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    }
    NvmV2Layouts layouts={rows,count};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);nvm_v2_layouts_free(&layouts);
    free(m->ownership_data);unsigned header=(8+count+3)&~3u;
    m->ownership_size=header+4+24+4;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,2);word(p+4,count);memset(p+8,3,count);if(ordinary_child)p[8]=NVM_LAYOUT_COMPLETE;
    word(p+header,2); /* I write descriptors without importing runtime fixture state. */
    p[header+8]=TAG_INT;word(p+header+12,NVM_V2_NO_INDEX);
    p[header+20]=TAG_STRUCT;word(p+header+24,count-1);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);return m;
}
static void inspect(NvmModule *m,bool accepted,bool fault,unsigned fields) {
    NvmAffineState *state=nvm_affine_state_create(m,1,0);CHECK(state);
    NvmAffineType type={TAG_BOOL,123};uint16_t count=456;allocations=0;fail_query=fault;
    CHECK(nvm_affine_value_result(state,&type,&count)==accepted);fail_query=false;
    CHECK(allocations==1);
    if(accepted)CHECK(type.tag==TAG_STRUCT&&type.layout==m->struct_count-1&&count==fields);
    else CHECK(type.tag==TAG_BOOL&&type.layout==123&&count==456);
    nvm_affine_state_free(state);
}
int main(void) {
    NvmModule *m=dag(32,2,false,false);inspect(m,true,false,2);inspect(m,false,true,2);nvm_module_free(m);
    m=dag(33,2,false,false);inspect(m,false,false,2);nvm_module_free(m);
    m=dag(2,256,false,false);inspect(m,true,false,256);nvm_module_free(m);
    m=dag(2,257,false,false);inspect(m,false,false,257);nvm_module_free(m);
    /* I preserve STRING result leaves and unchanged query outputs on allocation refusal. */
    m=dag(3,2,false,false);NvmV2Layouts strings={0};
    CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&strings)==NVM_V2_OK);
    strings.items[0].fields[0].type_tag=TAG_STRING;
    CHECK(nvm_retain_layouts(m,&strings)==NVM_V2_OK);nvm_v2_layouts_free(&strings);
    inspect(m,true,false,2);inspect(m,false,true,2);nvm_module_free(m);
    m=dag(3,2,true,false);inspect(m,false,false,2);nvm_module_free(m);
    m=dag(3,2,false,true);inspect(m,false,false,2);nvm_module_free(m);
    const uint8_t tags[]={TAG_VOID,TAG_INT,TAG_BOOL,TAG_U8,TAG_STRUCT};
    for(unsigned i=0;i<sizeof(tags);i++) {
        m=fixture(tags[i],tags[i]==TAG_STRUCT?0:NVM_V2_NO_INDEX,2);
        NvmAffineState *state=nvm_affine_state_create(m,1,0);CHECK(state);
        NvmAffineType type={0};uint16_t fields=123;allocations=0;fail_query=true;
        CHECK(nvm_affine_value_result(state,&type,&fields));CHECK(allocations==0);fail_query=false;
        nvm_affine_state_free(state);nvm_module_free(m);
    }
    printf("%u nested descriptor DAG/depth/allocation checks passed\n",checks);return 0;
}
