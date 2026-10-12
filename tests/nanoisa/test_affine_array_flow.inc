/* I qualify instruction facts without opening the collection runtime gate. */
static void array_flow_slot(uint8_t *p,uint8_t tag,uint32_t layout) {
    slot(p,tag,0);word(p+4,layout);
}
static NvmModule *array_flow_fixture(const char *body,const char *helper,bool owned_helper) {
    char source[16384];
    int n=snprintf(source,sizeof source,
        ".string pass \"Array.pass\"\n.string owner \"Array.owner\"\n.entry 0\n"
        ".function main 0 7 0 int 1\n%s.end\n.function relay %u 1 0 array 1\n%s.end\n%s",
        body,owned_helper?0:1,helper?helper:"LOAD_LOCAL 0\nRET\n",owned_helper?"":".parameters 1 array\n");
    CHECK(n>0 && (size_t)n<sizeof source);AsmResult error;
    NvmModule *m=asm_assemble_unverified(source,&error);if(!m)fprintf(stderr,"%s\n",error.message);CHECK(m);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout records[]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field},
                          {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field}};
    NvmV2Layouts layouts={records,2};m->struct_count=2;CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    uint8_t *p=calloc(228,1);CHECK(p);m->ownership_data=p;m->ownership_size=228;
    word(p,3);word(p+4,2);p[8]=p[9]=3;word(p+12,2);p[16]=7;
    array_flow_slot(p+20,TAG_INT,NVM_V2_NO_INDEX);
    for(unsigned i=0;i<6;i++)array_flow_slot(p+28+8*i,TAG_ARRAY,i);
    array_flow_slot(p+76,TAG_INT,NVM_V2_NO_INDEX);
    p[84]=1;p[86]=owned_helper?0:1;array_flow_slot(p+88,TAG_ARRAY,owned_helper?3:0);
    array_flow_slot(p+96,TAG_ARRAY,owned_helper?3:0);
    word(p+104,4);word(p+112,1);p[116]=NVM_OWNERSHIP_EXTENSION_EFFECTS;p[118]=2;word(p+120,104);
    word(p+124,6);array_flow_slot(p+128,TAG_INT,NVM_V2_NO_INDEX);
    array_flow_slot(p+136,TAG_BOOL,NVM_V2_NO_INDEX);array_flow_slot(p+144,TAG_ARRAY,0);
    array_flow_slot(p+152,TAG_STRUCT,0);array_flow_slot(p+160,TAG_STRUCT,1);
    array_flow_slot(p+168,TAG_STRING,NVM_V2_NO_INDEX);
    word(p+176,2);word(p+180,0);p[184]=1;array_flow_slot(p+188,TAG_ARRAY,0);array_flow_slot(p+196,TAG_ARRAY,0);
    word(p+204,1);p[208]=1;array_flow_slot(p+212,TAG_ARRAY,3);array_flow_slot(p+220,TAG_ARRAY,3);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);return m;
}
static void array_flow_case(const char *body,const char *helper,bool owned_helper,bool accepted,const char *reason) {
    NvmModule *m=array_flow_fixture(body,helper,owned_helper);
    NvmAffineAnalysis result=nvm_affine_analyze_function(m,owned_helper?1:0);
    if(result.ok!=accepted || (reason && !strstr(result.message,reason)))fprintf(stderr,"array flow: %s\n%s\n%s\n",result.message,body,helper?helper:"");
    CHECK(result.ok==accepted);if(reason)CHECK(strstr(result.message,reason));
    VmDecodedFunction decoded={0};char decode_error[VM_DECODE_ERROR_SIZE];
    CHECK(vm_decode_function(m,owned_helper?1:0,&decoded,decode_error));
    size_t bytes=decoded.instruction_count*sizeof(NvmAffineInstructionFact);
    NvmAffineInstructionFact *facts=malloc(bytes),*prior=malloc(bytes);CHECK(facts && prior);
    memset(facts,0xa5,bytes);memcpy(prior,facts,bytes);
    CHECK(nvm_affine_analyze_instructions(m,owned_helper?1:0,facts,decoded.instruction_count).ok==accepted);
    if(!accepted)CHECK(!memcmp(facts,prior,bytes));
    free(facts);free(prior);vm_decoded_function_free(&decoded);
    CHECK(!nvm_verify_owned_module(m).ok);CHECK(!nvm_verify(m).ok);
    char message[256];CHECK(!nvm2c_emit(m,message,sizeof message));nvm_module_free(m);
}
#define ARRAY_FLOW_START "HANDLER_PUSH pass arm 0 1\nPUSH_I64 7\nARR_LITERAL 1 1\nPERFORM pass 1\nHANDLER_POP 1\n"
#define ARRAY_FLOW_END "PUSH_I64 0\nARR_GET\nRET\narm:\nLOAD_LOCAL 0\nEFFECT_RESUME\n"
static void array_flow_edges(void) {
    array_flow_case(ARRAY_FLOW_START ARRAY_FLOW_END,NULL,false,true,NULL);
    array_flow_case(ARRAY_FLOW_START "CALL 1\n" ARRAY_FLOW_END,NULL,false,true,NULL);
    array_flow_case("PUSH_I64 7\nARR_LITERAL 1 1\nARR_LITERAL 7 1\nSTORE_LOCAL 2\nLOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\nRET\n",NULL,false,true,NULL);
    array_flow_case("ARR_NEW 1\nPUSH_I64 7\nARR_PUSH\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 8\nARR_SET\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nARR_POP\nRET\n",NULL,false,true,NULL);
    array_flow_case("PUSH_I64 7\nARR_LITERAL 1 1\nPUSH_I64 0\nARR_REMOVE\nARR_LEN\nRET\n",NULL,false,true,NULL);
    array_flow_case("HANDLER_PUSH pass arm 0 1\nPUSH_I64 7\nARR_LITERAL 1 1\nPERFORM pass 1\nPOP\nHANDLER_POP 1\nPUSH_I64 0\nRET\narm:\nPUSH_I64 7\nRET\n",NULL,false,true,NULL);
    array_flow_case("ARR_NEW 4\nPUSH_I64 7\nARR_PUSH\nPOP\nPUSH_I64 0\nRET\n",NULL,false,false,"exact array element");
    array_flow_case("PUSH_I64 7\nPUSH_BOOL 1\nARR_LITERAL 1 2\nPOP\nPUSH_I64 0\nRET\n",NULL,false,false,"literal element ownership");
    array_flow_case("ARR_NEW 4\nPERFORM pass 1\nPOP\nPUSH_I64 0\nRET\n",NULL,false,false,"argument ownership");
    array_flow_case("ARR_NEW 4\nCALL 1\nPOP\nPUSH_I64 0\nRET\n",NULL,false,false,"argument types");
    array_flow_case("ARR_NEW 4\nSTORE_LOCAL 0\nPUSH_I64 0\nRET\n",NULL,false,false,"local type");
    array_flow_case("ARR_NEW 8\nPOP\nPUSH_I64 0\nRET\n",NULL,false,false,"unambiguous");
    array_flow_case("PUSH_BOOL 1\nJMP_TRUE other\nARR_NEW 1\nJMP join\nother:\nARR_NEW 4\njoin:\nPOP\nPUSH_I64 0\nRET\n",NULL,false,false,"join");
    array_flow_case("ARR_NEW 1\nPUSH_BOOL 1\nARR_GET\nRET\n",NULL,false,false,"integer array index");
    array_flow_case(ARRAY_FLOW_START "CALL 1\n" ARRAY_FLOW_END,"ARR_NEW 4\nRET\n",false,false,"helper owner resolution");
    array_flow_case("HANDLER_PUSH pass arm 1 1\nPUSH_I64 0\nRET\narm:\nLOAD_LOCAL 1\nEFFECT_RESUME\n",NULL,false,false,"handler parameter");
    array_flow_case(ARRAY_FLOW_START "PUSH_I64 0\nARR_GET\nRET\narm:\nARR_NEW 4\nEFFECT_RESUME\n",NULL,false,false,"resumption");
    const char *construct="PUSH_I64 7\nOWN_PACK 0\nARR_LITERAL 8 1\n";
    char helper[2048];snprintf(helper,sizeof helper,"%sOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nARR_LEN\nPOP\nOWN_MOVE_LOCAL 0\nPUSH_I64 8\nOWN_PACK 0\nARR_PUSH\nRET\n",construct);
    array_flow_case("PUSH_I64 7\nRET\n",helper,true,true,NULL);
    snprintf(helper,sizeof helper,"HANDLER_PUSH owner arm 0 1\n%sPERFORM owner 1\nHANDLER_POP 1\nRET\narm:\nOWN_MOVE_LOCAL 0\nEFFECT_RESUME\n",construct);
    array_flow_case("PUSH_I64 7\nRET\n",helper,true,true,NULL);
    snprintf(helper,sizeof helper,"%sOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nOWN_MOVE_LOCAL 0\nOWN_STORE_LOCAL 0\nARR_LEN\nPOP\nOWN_MOVE_LOCAL 0\nRET\n",construct);
    array_flow_case("PUSH_I64 7\nRET\n",helper,true,false,"owner observation");
    snprintf(helper,sizeof helper,"%sARR_POP\nRET\n",construct);
    array_flow_case("PUSH_I64 7\nRET\n",helper,true,false,"resource-element");
    snprintf(helper,sizeof helper,"%sDUP\nRET\n",construct);
    array_flow_case("PUSH_I64 7\nRET\n",helper,true,false,"duplicate");
    snprintf(helper,sizeof helper,"%sOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nOWN_MOVE_LOCAL 0\nSWAP\nARR_LEN\nRET\n",construct);
    array_flow_case("PUSH_I64 7\nRET\n",helper,true,false,"owner observation");
    snprintf(helper,sizeof helper,"%sPUSH_I64 8\nOWN_PACK 1\nARR_PUSH\nRET\n",construct);
    array_flow_case("PUSH_I64 7\nRET\n",helper,true,false,"exact array element");
}

static void array_flow_facts(void) {
    NvmModule *m=array_flow_fixture("PUSH_BOOL 1\nJMP_TRUE other\nARR_NEW 1\nJMP join\nother:\nARR_NEW 1\nCALL 1\njoin:\nARR_LEN\nRET\n",NULL,false);
    array_flow_slot(m->ownership_data+160,TAG_INT,NVM_V2_NO_INDEX);
    array_flow_slot(m->ownership_data+88,TAG_ARRAY,4);array_flow_slot(m->ownership_data+96,TAG_ARRAY,4);
    CHECK(nvm_affine_analyze_function(m,0).ok);nvm_module_free(m);
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
    m=array_flow_fixture(ARRAY_FLOW_START "CALL 1\n" ARRAY_FLOW_END,NULL,false);
    VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];CHECK(vm_decode_function(m,0,&decoded,error));
    size_t bytes=decoded.instruction_count*sizeof(NvmAffineInstructionFact);
    NvmAffineInstructionFact *facts=malloc(bytes),*prior=malloc(bytes);CHECK(facts && prior);memset(prior,0xa5,bytes);
    unsigned failure;
    for(failure=1;failure<4096;failure++) {
        memcpy(facts,prior,bytes);allocation_attempts=0;fail_at=failure;
        NvmAffineAnalysis result=nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count);fail_at=0;
        if(result.ok){CHECK(allocation_attempts<failure);break;}
        CHECK(!memcmp(facts,prior,bytes));CHECK(nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count).ok);
    }
    CHECK(failure<4096);free(facts);free(prior);vm_decoded_function_free(&decoded);nvm_module_free(m);
#endif
}
