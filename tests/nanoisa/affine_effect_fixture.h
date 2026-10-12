/* I check affine effect edges while runtime admission remains separate. */
static NvmModule *effect_fixture(const char *body,const char *helper) {
    char source[16384];
    int used=snprintf(source,sizeof(source),
        ".string pass \"Transfer.pass\"\n.string tick \"Transfer.tick\"\n.entry 0\n"
        ".function main 0 6 0 int 1\n%s.end\n"
        ".function send 1 1 0 struct 1\n%s.end\n.parameters 1 struct\n"
        ".function forward 1 1 0 struct 1\nOWN_MOVE_LOCAL 0\nCALL 1\nRET\n.end\n.parameters 2 struct\n",
        body,helper?helper:"OWN_MOVE_LOCAL 0\nRET\n");
    CHECK(used>0 && (size_t)used<sizeof(source));
    AsmResult error;NvmModule *m=asm_assemble_unverified(source,&error);
    if (!m) fprintf(stderr,"%s\n",error.message);
    CHECK(m);
    uint32_t pass=nvm_add_string(m,"Transfer.pass",13),tick=nvm_add_string(m,"Transfer.tick",13);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout records[]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field},
                          {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field}};
    NvmV2Layouts layouts={records,2};m->struct_count=2;
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=188;m->ownership_data=calloc(188,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,3);word(p+4,2);p[8]=p[9]=3;word(p+12,3);
    p[16]=6;slot(p+20,TAG_INT,0);
    for (unsigned i=0;i<6;i++) slot(p+28+8*i,i==3?TAG_INT:TAG_STRUCT,0);
    word(p+64,1); /* local 4: distinct OtherHandle */
    p[76]=1;p[78]=1;slot(p+80,TAG_STRUCT,0);slot(p+88,TAG_STRUCT,0);
    p[96]=1;p[98]=1;slot(p+100,TAG_STRUCT,0);slot(p+108,TAG_STRUCT,0);
    word(p+116,4);word(p+124,1);p[128]=NVM_OWNERSHIP_EXTENSION_EFFECTS;p[130]=1;
    word(p+132,52);word(p+136,2);
    word(p+140,pass);p[144]=1;slot(p+148,TAG_STRUCT,0);slot(p+156,TAG_STRUCT,0);
    word(p+164,tick);p[168]=1;slot(p+172,TAG_INT,0);slot(p+180,TAG_INT,0);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    return m;
}
static void effect_case(const char *body,const char *helper,bool accepted,const char *reason) {
    NvmModule *m=effect_fixture(body,helper);
    NvmAffineAnalysis result=nvm_affine_analyze_function(m,0);
    if (result.ok!=accepted || (reason && !strstr(result.message,reason))) fprintf(stderr,"effect analysis at %u: %s\n%s",result.byte_offset,result.message,body);
    CHECK(result.ok==accepted);if (reason) CHECK(strstr(result.message,reason));
    /* I preserve the runtime gate while establishing instruction transitions. */
    CHECK(!nvm_verify_owned_module(m).ok);
    nvm_module_free(m);
}
#define EFFECT_START "HANDLER_PUSH pass arm 2 1\nPUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 0\n"
#define EFFECT_SEND "OWN_MOVE_LOCAL 0\nPERFORM pass 1\nOWN_STORE_LOCAL 1\nHANDLER_POP 1\n"
#define EFFECT_END "OWN_UNPACK_LOCAL 1\nSTORE_LOCAL 3\nLOAD_LOCAL 3\nRET\narm:\n"
#define EFFECT_ARM "OWN_MOVE_LOCAL 2\nEFFECT_RESUME\n"
static void effect_edges(void) {
    effect_case(EFFECT_START EFFECT_SEND EFFECT_END EFFECT_ARM,NULL,true,NULL);
    effect_case(EFFECT_START EFFECT_SEND EFFECT_END "OWN_UNPACK_LOCAL 2\nRET\n",NULL,true,NULL);
    effect_case(EFFECT_START EFFECT_SEND EFFECT_END "PUSH_I64 1\nOWN_PACK 0\nEFFECT_RESUME\n",NULL,false,"resumption");
    effect_case(EFFECT_START EFFECT_SEND EFFECT_END "OWN_UNPACK_LOCAL 2\nPOP\nPUSH_I64 7\nEFFECT_RESUME\n",NULL,false,"resumption");
    effect_case(EFFECT_START EFFECT_SEND EFFECT_END "OWN_MOVE_LOCAL 2\nOWN_STORE_LOCAL 5\nOWN_MOVE_LOCAL 2\nEFFECT_RESUME\n",NULL,false,"explicit move");
    effect_case(EFFECT_START EFFECT_SEND EFFECT_END "PUSH_I64 1\nOWN_PACK 0\nOWN_STORE_LOCAL 0\n" EFFECT_ARM,NULL,false,"owner destination");
    effect_case(EFFECT_START "PUSH_I64 1\nOWN_PACK 0\nOWN_STORE_LOCAL 5\n" EFFECT_SEND EFFECT_END EFFECT_ARM,NULL,false,"local owner");
    effect_case(EFFECT_START "PUSH_I64 1\nOWN_PACK 0\n" EFFECT_SEND EFFECT_END EFFECT_ARM,NULL,false,"staged owner");
    effect_case(EFFECT_START "PUSH_I64 1\nOWN_PACK 1\nPERFORM pass 1\nRET\narm:\n" EFFECT_ARM,NULL,false,"argument ownership");
    effect_case("HANDLER_PUSH pass arm 4 1\nPUSH_I64 0\nRET\narm:\nOWN_MOVE_LOCAL 4\nEFFECT_RESUME\n",NULL,false,"parameter types");
    effect_case("HANDLER_POP 1\nPUSH_I64 0\nRET\n",NULL,false,"absent or inherited");
    effect_case(EFFECT_START EFFECT_SEND EFFECT_END "HANDLER_POP 1\n" EFFECT_ARM,NULL,false,"absent or inherited");
    effect_case("PUSH_I64 1\nOWN_PACK 0\nEFFECT_RESUME\n",NULL,false,"resumption");
    effect_case(EFFECT_START EFFECT_SEND EFFECT_END "OWN_UNPACK_LOCAL 2\nPOP\nHALT\n",NULL,false,"terminal invariant");
    effect_case("PUSH_I64 1\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nHALT\n",NULL,false,"terminal invariant");
    effect_case("PUSH_BOOL 1\nJMP_TRUE skip\nHANDLER_PUSH pass arm 2 1\nskip:\nPUSH_I64 0\nRET\narm:\n" EFFECT_ARM,NULL,false,"handler context");
    const char *escaping="OWN_MOVE_LOCAL 0\nPERFORM pass 1\nRET\n";
    const char *contained="HANDLER_PUSH pass inner 0 1\nOWN_MOVE_LOCAL 0\nPERFORM pass 1\nHANDLER_POP 1\nRET\ninner:\nOWN_MOVE_LOCAL 0\nEFFECT_RESUME\n";
    effect_case(EFFECT_START "OWN_MOVE_LOCAL 0\nCALL 1\nOWN_STORE_LOCAL 1\nHANDLER_POP 1\n" EFFECT_END EFFECT_ARM,escaping,true,NULL);
    const char *held=EFFECT_START "PUSH_I64 1\nOWN_PACK 0\nOWN_STORE_LOCAL 5\n"
        "OWN_MOVE_LOCAL 0\nCALL 1\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 5\nPOP\nHANDLER_POP 1\n" EFFECT_END EFFECT_ARM;
    effect_case(held,escaping,false,"escaping effect call");
    effect_case(held,contained,true,NULL);
    effect_case(held,NULL,true,NULL);
    const char *transitive=EFFECT_START "PUSH_I64 1\nOWN_PACK 0\nOWN_STORE_LOCAL 5\n"
        "OWN_MOVE_LOCAL 0\nCALL 2\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 5\nPOP\nHANDLER_POP 1\n" EFFECT_END EFFECT_ARM;
    effect_case(transitive,escaping,false,"escaping effect call");
    effect_case(transitive,contained,true,NULL);
    const char *partial="HANDLER_PUSH pass inner 0 1\nOWN_UNPACK_LOCAL 0\nPERFORM tick 1\nOWN_PACK 0\nHANDLER_POP 1\nRET\ninner:\nOWN_MOVE_LOCAL 0\nEFFECT_RESUME\n";
    effect_case(transitive,partial,false,"escaping effect call");
    effect_case(EFFECT_START EFFECT_SEND EFFECT_END
        "HANDLER_PUSH pass nested 5 1\nOWN_MOVE_LOCAL 2\nPERFORM pass 1\nHANDLER_POP 1\nEFFECT_RESUME\n"
        "nested:\nOWN_MOVE_LOCAL 5\nEFFECT_RESUME\n",NULL,true,NULL);
    effect_case("PUSH_I64 4\nSTORE_LOCAL 3\nHANDLER_PUSH pass arm 5 1\n"
        "PUSH_I64 7\nOWN_PACK 0\nPERFORM pass 1\nOWN_STORE_LOCAL 1\nHANDLER_POP 1\n" EFFECT_END
        "LOAD_LOCAL 3\nPOP\nOWN_MOVE_LOCAL 5\nEFFECT_RESUME\n",NULL,true,NULL);
    {
        NvmModule *m=effect_fixture("HANDLER_PUSH tick arm 3 0\nPERFORM tick 0\nPOP\nHANDLER_POP 1\nPUSH_I64 0\nRET\narm:\nPUSH_VOID\nEFFECT_RESUME\n",NULL);
        m->ownership_size=180;word(m->ownership_data+132,44);m->ownership_data[168]=0;
        slot(m->ownership_data+172,TAG_VOID,0);
        CHECK(nvm_affine_analyze_function(m,0).ok);
        CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    }
    for (unsigned depth=64;depth<=65;depth++) {
        char body[8192];size_t used=0;
        for (unsigned i=0;i<depth;i++)
            used+=(size_t)snprintf(body+used,sizeof(body)-used,"HANDLER_PUSH pass arm%u 2 1\n",i);
        used+=(size_t)snprintf(body+used,sizeof(body)-used,"PUSH_I64 0\nRET\n");
        for (unsigned i=0;i<depth;i++)
            used+=(size_t)snprintf(body+used,sizeof(body)-used,"arm%u:\nOWN_MOVE_LOCAL 2\nEFFECT_RESUME\n",i);
        CHECK(used<sizeof(body));
        effect_case(body,NULL,depth==64,depth==64?NULL:"handler depth");
    }
    {
        NvmModule *m=effect_fixture(EFFECT_START EFFECT_SEND EFFECT_END EFFECT_ARM,NULL);
        uint32_t count=0;const NvmFunctionEntry *function=&m->functions[0];
        for (uint32_t offset=0;offset<function->code_length;count++) {
            DecodedInstruction instruction;
            uint32_t size=isa_decode(m->code+function->code_offset+offset,function->code_length-offset,&instruction);
            CHECK(size);offset+=size;
        }
        NvmAffineInstructionFact *facts=malloc(count*sizeof(*facts));CHECK(facts);
        memset(facts,0xa5,count*sizeof(*facts));
        CHECK(nvm_affine_analyze_instructions(m,0,facts,count).ok);
        CHECK(facts[count-1].reachable && facts[count-1].stack_depth==1 && facts[count-1].top_tag==TAG_STRUCT);
        memset(facts,0xa5,count*sizeof(*facts));
        CHECK(!nvm_affine_analyze_instructions(m,0,facts,count-1).ok);
        for (size_t i=0;i<count*sizeof(*facts);i++) CHECK(((unsigned char*)facts)[i]==0xa5);
        free(facts);nvm_module_free(m);
    }
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
    NvmModule *m=effect_fixture(EFFECT_START EFFECT_SEND EFFECT_END EFFECT_ARM,NULL);
    allocation_attempts=0;NvmAffineAnalysis result=nvm_affine_analyze_function(m,0);CHECK(result.ok);
    unsigned attempts=allocation_attempts;
    for (unsigned i=1;i<=attempts;i++) {
        allocation_attempts=0;fail_at=i;result=nvm_affine_analyze_function(m,0);fail_at=0;
        CHECK(!result.ok);
    }
    nvm_module_free(m);
#endif
}
#undef EFFECT_START
#undef EFFECT_SEND
#undef EFFECT_END
#undef EFFECT_ARM
