/* I build exact private WebSocket metadata for nominal and flow tests. */
#ifndef TEST_WEBSOCKET_NOMINAL_FIXTURE_H
#define TEST_WEBSOCKET_NOMINAL_FIXTURE_H
static void wr16(uint8_t *p,uint16_t n){p[0]=(uint8_t)n;p[1]=(uint8_t)(n>>8);}
static void wr32(uint8_t *p,uint32_t n){for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(n>>(8*i));}
typedef struct {
    NvmModule module;NvmWebSocketNominalBindings bindings;
    char names[96][128];char *strings[96];uint32_t lengths[96];
    NvmImportEntry imports[4];uint8_t params[4][3],*param_rows[4];
    NvmFunctionEntry function;uint8_t function_params[3],*function_param_rows[1];
    uint8_t layouts[1024],ownership[256],service[104];size_t offsets[7];
} Fixture;
static uint32_t name(Fixture *f,const char *text) {
    uint32_t n=f->module.string_count++;CHECK(n<96 && strlen(text)<128);
    strcpy(f->names[n],text);f->strings[n]=f->names[n];f->lengths[n]=(uint32_t)strlen(text);return n;
}
static uint8_t tag(const char *id) {
    if(!id)return TAG_VOID;
    if(!strcmp(id,"nsi:core/int"))return TAG_INT;
    if(!strcmp(id,"nsi:core/bool"))return TAG_BOOL;
    if(!strcmp(id,"nsi:core/string"))return TAG_STRING;
    for(unsigned i=0;i<7;i++)if(!strcmp(id,nl_websocket_catalog_type(i)->id))return i<3?TAG_STRUCT:TAG_UNION;
    CHECK(false);return 0;
}
static uint32_t nested(Fixture *f,const char *id) {
    for(unsigned i=0;id && i<7;i++)if(!strcmp(id,nl_websocket_catalog_type(i)->id))return f->bindings.layouts[i];
    return NVM_V2_NO_INDEX;
}
static void descriptor(uint8_t *p,uint8_t t,uint8_t mode,uint32_t layout) {
    p[0]=t;p[1]=mode;p[2]=p[3]=0;wr32(p+4,layout);
}
static void fixture(Fixture *f,bool permute) {
    memset(f,0,sizeof *f);NvmModule *m=&f->module;
    m->strings=f->strings;m->string_lengths=f->lengths;
    const unsigned order[]={3,2,1,6,5,7,4},imports[]={2,0,3,1};
    for(unsigned i=0;i<7;i++)f->bindings.layouts[i]=permute?order[i]:i+1;
    for(unsigned i=0;i<4;i++) {
        unsigned index=permute?imports[i]:i;f->bindings.imports[i]=index;
        const NlServicePlanMethod *method=nl_websocket_catalog_method(i);
        f->imports[index]=(NvmImportEntry){name(f,nl_websocket_catalog_interface()),name(f,method->id),(uint16_t)(method->param_count-1),TAG_UNION,NVM_IMPORT_SERVICE};
        for(unsigned j=0;j<method->param_count-1;j++)f->params[index][j]=tag(method->params[j].type_id);
        f->param_rows[index]=f->params[index];
    }
    m->imports=f->imports;m->import_count=4;m->import_param_types=f->param_rows;
    wr32(f->layouts,8);size_t at=4;
    /* I retain one same-shaped ordinary string record without catalog identity. */
    f->layouts[at]=NVM_V2_LAYOUT_STRUCT;wr16(f->layouts+at+2,2);wr32(f->layouts+at+4,name(f,"ordinary.Message"));at+=8;
    descriptor(f->layouts+at,TAG_BOOL,0,NVM_V2_NO_INDEX);wr32(f->layouts+at+8,name(f,"ordinary.binary"));at+=12;
    descriptor(f->layouts+at,TAG_STRING,0,NVM_V2_NO_INDEX);wr32(f->layouts+at+8,name(f,"ordinary.data"));at+=12;
    for(unsigned global=1;global<8;global++) {
        unsigned ordinal=0;while(f->bindings.layouts[ordinal]!=global)ordinal++;
        const NlServicePlanType *t=nl_websocket_catalog_type(ordinal);f->offsets[ordinal]=at;
        f->layouts[at]=ordinal<3?NVM_V2_LAYOUT_STRUCT:NVM_V2_LAYOUT_UNION;
        wr16(f->layouts+at+2,(uint16_t)t->member_count);wr32(f->layouts+at+4,name(f,t->id));at+=8;
        for(unsigned j=0;j<t->member_count;j++) {
            descriptor(f->layouts+at,tag(t->members[j].type_id),0,nested(f,t->members[j].type_id));
            wr32(f->layouts+at+8,name(f,t->members[j].id));at+=12;
        }
    }
    m->layout_data=f->layouts;m->layout_size=(uint32_t)at;m->struct_count=4;m->union_count=4;
    f->function=(NvmFunctionEntry){.arity=3,.local_count=4,.result_count=1,.result_tag=TAG_INT};
    m->functions=&f->function;m->function_count=1;
    f->function_params[0]=TAG_STRING;f->function_params[1]=TAG_INT;f->function_params[2]=TAG_STRUCT;
    f->function_param_rows[0]=f->function_params;m->function_param_types=f->function_param_rows;
    uint8_t *o=f->ownership;wr32(o,1);wr32(o+4,8);
    for(unsigned i=0;i<7;i++)o[8+f->bindings.layouts[i]]=(i==0 || i==3)?3:1;
    wr32(o+16,1);wr16(o+20,4);wr16(o+22,3);
    descriptor(o+24,TAG_INT,0,NVM_V2_NO_INDEX);
    descriptor(o+32,TAG_STRING,0,NVM_V2_NO_INDEX);
    descriptor(o+40,TAG_INT,0,NVM_V2_NO_INDEX);
    descriptor(o+48,TAG_STRUCT,2,f->bindings.layouts[0]);
    descriptor(o+56,TAG_STRUCT,0,f->bindings.layouts[2]);
    m->ownership_data=o;m->ownership_size=64;
    size_t size=0;CHECK(nvm_websocket_nominal_encode(&f->bindings,f->service,sizeof f->service,&size)==NVM_SERVICE_OK && size==104);
    m->service_data=f->service;m->service_size=104;
}
#endif
