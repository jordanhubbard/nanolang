/* I retain collection identities without granting collection execution. */
static void check_effect_array_contracts(void) {
    AsmResult error;
    NvmModule *m=asm_assemble_unverified(".string op \"Collection.pass\"\n.entry 0\n.function main 0 0 0 int 1\nPERFORM op 0\nPOP\nPUSH_I64 0\nRET\n.end\n",&error);
    CHECK(m);uint32_t key=nvm_add_string(m,"Collection.pass",15);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout records[]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field},
                          {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field}};
    NvmV2Layouts layouts={records,2};m->struct_count=2;
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    uint8_t *data=calloc(4096,1);CHECK(data);m->ownership_data=data;m->ownership_size=104;
    word(data,0,3);word(data,4,2);data[8]=3;data[9]=1;
    word(data,12,1);slot(data,20,TAG_INT,0,NVM_V2_NO_INDEX);
    word(data,28,4);word(data,36,1);half(data,40,NVM_OWNERSHIP_EXTENSION_EFFECTS);
    half(data,42,NVM_OWNERSHIP_EFFECT_ARRAY_REVISION);word(data,44,56);word(data,48,4);
    slot(data,52,TAG_INT,0,NVM_V2_NO_INDEX);slot(data,60,TAG_STRUCT,0,0);
    slot(data,68,TAG_STRUCT,0,1);slot(data,76,TAG_ARRAY,0,0);
    word(data,84,1);word(data,88,key);slot(data,96,TAG_ARRAY,0,3);
    check_status(m,true,true);
    NvmOwnershipEffect effect;NvmOwnershipEffectTypes types;
    CHECK(nvm_ownership_effect(m,key,&effect)==NVM_V2_OK);
    CHECK(effect.parameter_count==0 && effect.result.tag==TAG_ARRAY && effect.result.layout==3);
    CHECK(nvm_ownership_effect_types(m,&types)==NVM_V2_OK && types.count==4);
    CHECK(types.items[1].tag==TAG_STRUCT && types.items[1].layout==0);
    CHECK(types.items[2].tag==TAG_STRUCT && types.items[2].layout==1);
    CHECK(types.items[3].tag==TAG_ARRAY && types.items[3].layout==0);
    /* The argument count agrees; my array result still needs flow authority. */
    NvmAffineAnalysis analysis=nvm_affine_analyze_function(m,0);
    CHECK(!analysis.ok && strstr(analysis.message,"handler-aware array flow"));
    NvmAffineState *state=nvm_affine_state_create(m,0,0);CHECK(state);
    CHECK(!nvm_affine_handler_enter(state,key,0));nvm_affine_state_free(state);
    CHECK(!nvm_verify(m).ok);
    char diagnostic[256];CHECK(!nvm2c_emit(m,diagnostic,sizeof(diagnostic)));
    /* I also retain two different nominal element identities in parameters. */
    half(data,92,2);m->ownership_size=120;word(data,44,72);
    slot(data,104,TAG_ARRAY,0,1);slot(data,112,TAG_ARRAY,0,2);
    CHECK(nvm_ownership_effect(m,key,&effect)==NVM_V2_OK);
    CHECK(effect.parameters[0].layout==1 && effect.parameters[1].layout==2);
    size_t length;uint8_t *bytes=wire(m,&length,NULL);NvmV2Module decoded;
    CHECK(nvm_v2_module_deserialize(bytes,length,&decoded)==NVM_V2_OK);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);
    CHECK(copy->ownership_size==120 && !memcmp(copy->ownership_data,data,120));
    CHECK(nvm_ownership_effect_types(copy,&types)==NVM_V2_OK && types.count==4);
    nvm_module_free(copy);nvm_v2_module_free(&decoded);free(bytes);
    char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);
    copy=asm_assemble_unverified(text,&error);CHECK(copy);
    CHECK(copy->ownership_size==120 && !memcmp(copy->ownership_data,data,120));
    CHECK(nvm_ownership_effect(copy,key,&effect)==NVM_V2_OK);
    nvm_module_free(copy);free(text);
    NvmOwnershipEffect prior;NvmOwnershipEffectTypes prior_types;
    memset(&effect,0xa5,sizeof effect);prior=effect;
    memset(&types,0xa5,sizeof types);prior_types=types;
    for(unsigned n=0;n<120;n++) {
        m->ownership_size=n;
        CHECK(nvm_ownership_effect(m,key,&effect)!=NVM_V2_OK);
        CHECK(nvm_ownership_effect_types(m,&types)!=NVM_V2_OK);
        CHECK(!memcmp(&effect,&prior,sizeof effect));CHECK(!memcmp(&types,&prior_types,sizeof types));
    }
    m->ownership_size=120;
    const unsigned offsets[]={42,44,48,52,53,54,56,60,61,62,64,68,69,70,72,76,77,78,80,84,88,92,94,96,97,98,100,104,105,106,108,112,113,114,116};
    for(unsigned i=0;i<sizeof offsets/sizeof *offsets;i++) {
        unsigned at=offsets[i];uint8_t old=data[at];data[at]^=255;
        CHECK(nvm_ownership_effect(m,key,&effect)!=NVM_V2_OK);
        CHECK(nvm_ownership_effect_types(m,&types)!=NVM_V2_OK);
        CHECK(!memcmp(&effect,&prior,sizeof effect));CHECK(!memcmp(&types,&prior_types,sizeof types));
        data[at]=old;
    }
    /* I refuse self/forward references even in unused rows. */
    word(data,80,3);CHECK(nvm_ownership_effect_types(m,&types)!=NVM_V2_OK);
    CHECK(!memcmp(&types,&prior_types,sizeof types));word(data,80,4);
    CHECK(nvm_ownership_effect_types(m,&types)!=NVM_V2_OK);word(data,80,0);
    data[52]=TAG_VOID;CHECK(nvm_ownership_effect_types(m,&types)!=NVM_V2_OK);data[52]=TAG_INT;
    data[9]=0;CHECK(nvm_ownership_effect_types(m,&types)!=NVM_V2_OK);data[9]=1;
    half(data,42,1);CHECK(nvm_ownership_effect_types(m,&types)!=NVM_V2_OK);half(data,42,2);
    CHECK(!memcmp(&types,&prior_types,sizeof types));
    half(data,40,NVM_OWNERSHIP_EXTENSION_GLOBALS);
    CHECK(nvm_ownership_effect_types(m,&types)==NVM_V2_ERR_FORMAT_VERSION);
    CHECK(!memcmp(&types,&prior_types,sizeof types));
    half(data,40,NVM_OWNERSHIP_EXTENSION_EFFECTS);
    CHECK(nvm_ownership_effect_types(NULL,&types)!=NVM_V2_OK);
    CHECK(nvm_ownership_effect_types(m,NULL)!=NVM_V2_OK);
    /* I check the entire 256-row bound without recursion or hidden depth limits. */
    memset(data+48,0,4096-48);word(data,48,256);
    slot(data,52,TAG_STRING,0,NVM_V2_NO_INDEX);
    for(unsigned i=1;i<256;i++)slot(data,52+8*i,TAG_ARRAY,0,i-1);
    unsigned operations=52+8*256;
    word(data,operations,1);word(data,operations+4,key);
    slot(data,operations+12,TAG_ARRAY,0,255);
    m->ownership_size=operations+20;word(data,44,m->ownership_size-48);
    CHECK(nvm_ownership_effect_types(m,&types)==NVM_V2_OK && types.count==256);
    CHECK(types.items[255].tag==TAG_ARRAY && types.items[255].layout==254);
    CHECK(nvm_ownership_effect(m,key,&effect)==NVM_V2_OK && effect.result.layout==255);
    prior_types=types;word(data,48,257);
    CHECK(nvm_ownership_effect_types(m,&types)==NVM_V2_ERR_INDEX_RANGE);
    CHECK(!memcmp(&types,&prior_types,sizeof types));word(data,48,0);
    CHECK(nvm_ownership_effect_types(m,&types)==NVM_V2_ERR_INDEX_RANGE);
    nvm_module_free(m);
}
