/* I share exact retained union fixtures across analysis and execution checks. */
static void word(uint8_t *p,uint32_t v) {for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i));}
static void slot(uint8_t *p,uint8_t tag,uint8_t mode) {
    p[0]=tag;p[1]=mode;word(p+4,(tag==TAG_STRUCT || tag==TAG_UNION)?0:NVM_V2_NO_INDEX);
}
static const char *tag_name(uint8_t tag) {
    switch(tag) {
    case TAG_STRUCT:return "struct";case TAG_UNION:return "union";case TAG_INT:return "int";
    case TAG_BOOL:return "bool";case TAG_FLOAT:return "float";
    default:return "void";
    }
}
static NvmModule *union_fixture(const char *body,uint16_t params,uint16_t locals,
                                uint8_t result) {
    char parameters[256]={0};size_t parameter_bytes=0;
    if (params) {
        int wrote=snprintf(parameters,sizeof(parameters),".parameters 0");
        CHECK(wrote>0 && (size_t)wrote<sizeof(parameters));parameter_bytes=(size_t)wrote;
        for (uint16_t i=0;i<params;i++) {
            wrote=snprintf(parameters+parameter_bytes,sizeof(parameters)-parameter_bytes," union");
            CHECK(wrote>0 && (size_t)wrote<sizeof(parameters)-parameter_bytes);
            parameter_bytes+=(size_t)wrote;
        }
        CHECK(parameter_bytes+1<sizeof(parameters));parameters[parameter_bytes++]='\n';
        parameters[parameter_bytes]='\0';
    }
    char source[8192];int used=snprintf(source,sizeof(source),
        ".types 0 0 1\n.entry 0\n.string \"text\"\n.function inspect %u %u 0 %s %u\n%s.end\n%s",
        params,locals,tag_name(result),result!=TAG_VOID,body,parameters);
    CHECK(used>0 && (size_t)used<sizeof(source));
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m) {
        fprintf(stderr,"%s\n",assembled.message);
    }
    CHECK(m);
    uint32_t identity=nvm_add_string(m,"Choice<int,string>",18);
    uint32_t v0=nvm_add_string(m,"IntValue",8),v1=nvm_add_string(m,"TextPair",8);
    uint32_t v2=nvm_add_string(m,"Empty",5),f0=nvm_add_string(m,"value",5);
    uint32_t f1=nvm_add_string(m,"left",4),f2=nvm_add_string(m,"right",5);
    CHECK(identity!=UINT32_MAX && v0!=UINT32_MAX && v1!=UINT32_MAX &&
          v2!=UINT32_MAX && f0!=UINT32_MAX && f1!=UINT32_MAX && f2!=UINT32_MAX);
    NvmV2LayoutField fields[]={{TAG_INT,NVM_V2_NO_INDEX,f0},
        {TAG_STRING,NVM_V2_NO_INDEX,f1},{TAG_BOOL,NVM_V2_NO_INDEX,f2}};
    NvmV2Layout layout={NVM_V2_LAYOUT_UNION,3,identity,fields};
    NvmV2Layouts layouts={&layout,1};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    uint32_t at=28+8u*locals;
    m->ownership_size=at+56;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,NVM_OWNERSHIP_EXTENSION_VERSION);word(p+4,1);
    word(p+12,1);p[16]=(uint8_t)locals;p[17]=(uint8_t)(locals>>8);
    p[18]=(uint8_t)params;p[19]=(uint8_t)(params>>8);slot(p+20,result,0);
    for(uint16_t i=0;i<locals;i++)slot(p+28+8*i,TAG_UNION,0);
    word(p+at,4);word(p+at+4,0);word(p+at+8,1);
    p[at+12]=NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS;
    p[at+14]=NVM_OWNERSHIP_EXTENSION_REVISION_1;word(p+at+16,36);
    word(p+at+20,1);word(p+at+24,0);p[at+28]=3;
    word(p+at+32,v0);p[at+38]=1;
    word(p+at+40,v1);p[at+44]=1;p[at+46]=2;
    word(p+at+48,v2);p[at+52]=3;
    bool needs=false;NvmV2Result contract=nvm_ownership_contracts_validate(m,&needs);
    if(contract!=NVM_V2_OK)fprintf(stderr,"union ownership contract: %d\n",contract);
    CHECK(contract==NVM_V2_OK && needs);
    return m;
}
static NvmModule *owned_union_fixture(const char *body,uint16_t params,uint8_t result) {
    NvmModule *m=union_fixture(body,params,3,result);
    NvmV2Layouts old={0};CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&old)==NVM_V2_OK);
    uint32_t name=nvm_add_string(m,"Handle",6);
    NvmV2LayoutField fd={TAG_INT,NVM_V2_NO_INDEX,old.items[0].fields[0].name_idx};
    old.items[0].fields[0].type_tag=TAG_STRUCT;old.items[0].fields[0].nested_idx=0;
    old.items[0].fields[1].type_tag=TAG_INT;
    NvmV2Layout items[]={{NVM_V2_LAYOUT_STRUCT,1,name,&fd},old.items[0]};
    NvmV2Layouts layouts={items,2};m->struct_count=1;
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);nvm_v2_layouts_free(&old);
    uint8_t *p=m->ownership_data;word(p,NVM_OWNERSHIP_UNION_GRAPH_VERSION);word(p+4,2);
    p[8]=p[9]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;
    slot(p+28,TAG_UNION,0);word(p+32,1);
    slot(p+36,TAG_STRUCT,0);slot(p+44,TAG_UNION,0);word(p+48,1);
    /* The union extension follows the three local descriptors and names layout 1. */
    word(p+76,1);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    return m;
}
