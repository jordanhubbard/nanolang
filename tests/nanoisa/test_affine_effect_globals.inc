/* I exercise actual perform-time state, resumption and lexical destinations. */
static NvmModule *effect_global_fixture(const char *body,const char *helper,bool mutable) {
    NvmModule *m=effect_fixture(body,helper);
    uint8_t *p=realloc(m->ownership_data,m->ownership_size+20);CHECK(p);
    m->ownership_data=p;m->ownership_size+=20;
    memmove(p+148,p+128,60);memset(p+128,0,20);word(p+124,2);
    p[128]=NVM_OWNERSHIP_EXTENSION_GLOBALS;p[130]=1;word(p+132,12);word(p+136,1);
    slot(p+140,TAG_INT,mutable?1:0);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    return m;
}
static void effect_global_case(const char *body,const char *helper,bool mutable,bool accepted) {
    NvmModule *m=effect_global_fixture(body,helper,mutable);
    NvmAffineAnalysis result=nvm_affine_analyze_function(m,0);
    if(result.ok!=accepted || (!accepted && !strstr(result.message,"global initialization")))fprintf(stderr,"effect/global at %u: %s\n%s",result.byte_offset,result.message,body);
    CHECK(result.ok==accepted);
    if(!accepted)CHECK(strstr(result.message,"global initialization"));
    CHECK(nvm_verify_owned_module(m).ok==accepted);
    char error[256];char *native=nvm2c_emit(m,error,sizeof error);
    if((native!=NULL)!=accepted)fprintf(stderr,"effect/global native: %s\n",error);
    CHECK((native!=NULL)==accepted);free(native);
    VmDecodedFunction decoded={0};CHECK(vm_decode_function(m,0,&decoded,error));
    size_t bytes=decoded.instruction_count*sizeof(NvmAffineInstructionFact);
    NvmAffineInstructionFact *facts=malloc(bytes),*before=malloc(bytes);CHECK(facts && before);
    memset(facts,0xa5,bytes);memcpy(before,facts,bytes);
    CHECK(nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count).ok==accepted);
    if(!accepted)CHECK(!memcmp(facts,before,bytes));
    free(facts);free(before);vm_decoded_function_free(&decoded);nvm_module_free(m);
}
#define EG_INSTALL "HANDLER_PUSH tick arm 3 1\n"
#define EG_PERFORM "PUSH_I64 7\nPERFORM tick 1\nPOP\nHANDLER_POP 1\n"
#define EG_FINISH "LOAD_GLOBAL 0\nRET\narm:\n"
#define EG_WRITE_RESUME "LOAD_LOCAL 3\nSTORE_GLOBAL 0\nLOAD_LOCAL 3\nEFFECT_RESUME\n"
static void check_effect_globals(void) {
    effect_global_case(EG_INSTALL EG_PERFORM EG_FINISH EG_WRITE_RESUME,NULL,false,true);
    effect_global_case(EG_INSTALL "PUSH_I64 7\nSTORE_GLOBAL 0\n" EG_PERFORM EG_FINISH
        "LOAD_GLOBAL 0\nEFFECT_RESUME\n",NULL,false,true);
    effect_global_case(EG_INSTALL EG_PERFORM EG_FINISH "LOAD_GLOBAL 0\nEFFECT_RESUME\n",NULL,true,false);
    effect_global_case(EG_INSTALL "PUSH_I64 1\nSTORE_GLOBAL 0\n" EG_PERFORM EG_FINISH EG_WRITE_RESUME,NULL,false,false);
    effect_global_case(EG_INSTALL EG_PERFORM EG_FINISH "PUSH_I64 7\nRET\n",NULL,false,true);
    effect_global_case(EG_INSTALL EG_PERFORM EG_FINISH
        "PUSH_BOOL 0\nJMP_FALSE resume\nLOAD_LOCAL 3\nSTORE_GLOBAL 0\nresume:\nLOAD_LOCAL 3\nEFFECT_RESUME\n",NULL,true,false);
    effect_global_case(EG_INSTALL EG_PERFORM EG_FINISH
        "PUSH_BOOL 0\nJMP_FALSE leave\n" EG_WRITE_RESUME "leave:\nPUSH_I64 7\nRET\n",NULL,false,true);
    effect_global_case(EG_INSTALL "PUSH_I64 7\nOWN_PACK 0\nCALL 2\nOWN_STORE_LOCAL 1\n"
        "HANDLER_POP 1\nOWN_UNPACK_LOCAL 1\nPOP\nLOAD_GLOBAL 0\nRET\narm:\nPUSH_I64 7\nRET\n",
        "OWN_UNPACK_LOCAL 0\nPERFORM tick 1\nPOP\nLOAD_GLOBAL 0\nOWN_PACK 0\nRET\n",false,true);
    effect_global_case("PUSH_I64 7\nOWN_PACK 0\nCALL 1\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nPOP\nLOAD_GLOBAL 0\nRET\n",
        "HANDLER_PUSH pass inner 0 1\nOWN_MOVE_LOCAL 0\nPERFORM pass 1\nOWN_STORE_LOCAL 0\n"
        "HANDLER_POP 1\nLOAD_GLOBAL 0\nPOP\nOWN_MOVE_LOCAL 0\nRET\ninner:\n"
        "OWN_UNPACK_LOCAL 0\nSTORE_GLOBAL 0\nPUSH_I64 7\nOWN_PACK 0\nRET\n",false,true);
    effect_global_case(EG_INSTALL "loop:\nPUSH_I64 7\nPERFORM tick 1\nPOP\n"
        "PUSH_BOOL 0\nJMP_TRUE done\nJMP loop\ndone:\nHANDLER_POP 1\n" EG_FINISH EG_WRITE_RESUME,NULL,false,false);
    effect_global_case(EG_INSTALL EG_PERFORM EG_FINISH
        "LOAD_LOCAL 3\nPUSH_I64 0\nEQ\nJMP_TRUE base\nLOAD_LOCAL 3\nPUSH_I64 1\nSUB\nPERFORM tick 1\nEFFECT_RESUME\n"
        "base:\nPUSH_I64 7\nSTORE_GLOBAL 0\nPUSH_I64 7\nEFFECT_RESUME\n",NULL,false,true);
    {

 const char *helper="OWN_UNPACK_LOCAL 0\nPERFORM tick 1\nOWN_PACK 0\nRET\n";
 const char *body="HANDLER_PUSH tick write_arm 3 1\nPUSH_I64 7\nOWN_PACK 0\nCALL 2\nOWN_STORE_LOCAL 1\nHANDLER_POP 1\n"
  "HANDLER_PUSH tick read_arm 3 1\nOWN_MOVE_LOCAL 1\nCALL 1\nOWN_STORE_LOCAL 1\nHANDLER_POP 1\nOWN_UNPACK_LOCAL 1\nRET\n"
  "write_arm:\nLOAD_LOCAL 3\nSTORE_GLOBAL 0\nLOAD_LOCAL 3\nEFFECT_RESUME\nread_arm:\nLOAD_GLOBAL 0\nEFFECT_RESUME\n";
 effect_global_case(body,helper,false,true);
 effect_global_case("HANDLER_PUSH tick read_arm 3 1\nPUSH_I64 7\nOWN_PACK 0\nCALL 1\nOWN_STORE_LOCAL 1\nHANDLER_POP 1\nOWN_UNPACK_LOCAL 1\nRET\nread_arm:\nLOAD_GLOBAL 0\nEFFECT_RESUME\n",helper,false,false);
 NvmModule *m=effect_global_fixture("PUSH_I64 7\nSTORE_GLOBAL 0\n" EG_INSTALL EG_PERFORM
  "LOAD_GLOBAL 1\nRET\narm:\nLOAD_GLOBAL 0\nSTORE_GLOBAL 1\nLOAD_GLOBAL 1\nEFFECT_RESUME\n",NULL,false);
 uint8_t *p=realloc(m->ownership_data,m->ownership_size+8);CHECK(p);m->ownership_data=p;m->ownership_size+=8;
 memmove(p+156,p+148,60);memset(p+148,0,8);word(p+132,20);word(p+136,2);slot(p+148,TAG_INT,0);
 CHECK(nvm_affine_analyze_function(m,0).ok);CHECK(nvm_verify_owned_module(m).ok);
 char error[256];char *native=nvm2c_emit(m,error,sizeof error);CHECK(native);free(native);nvm_module_free(m);
    }
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
    NvmModule *m=effect_global_fixture(EG_INSTALL EG_PERFORM EG_FINISH EG_WRITE_RESUME,NULL,false);
    VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
    CHECK(vm_decode_function(m,0,&decoded,error));
    size_t bytes=decoded.instruction_count*sizeof(NvmAffineInstructionFact);
    NvmAffineInstructionFact *facts=malloc(bytes),*before=malloc(bytes);CHECK(facts && before);
    memset(before,0xa5,bytes);
    unsigned failure;
    for(failure=1;failure<4096;failure++) {
        memcpy(facts,before,bytes);allocation_attempts=0;fail_at=failure;
        NvmAffineAnalysis result=nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count);
        fail_at=0;
        if(result.ok){CHECK(allocation_attempts<failure);break;}
        CHECK(!memcmp(facts,before,bytes));
        CHECK(nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count).ok);
    }
    CHECK(failure<4096);
    memcpy(facts,before,bytes);forced_effect_limit=1;
    NvmAffineAnalysis limited=nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count);
    forced_effect_limit=0;
    CHECK(!limited.ok && strstr(limited.message,"effect/global"));
    CHECK(!memcmp(facts,before,bytes));
    CHECK(nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count).ok);
    free(facts);free(before);vm_decoded_function_free(&decoded);nvm_module_free(m);
#endif

}
static void effect_global_runtime_artifacts(const char *directory) {
    effect_save(directory,"global_resume",effect_global_fixture(EG_INSTALL EG_PERFORM EG_FINISH EG_WRITE_RESUME,NULL,false));
    const char *prefix="PUSH_STR pass\nSTORE_GLOBAL 0\n" EG_INSTALL EG_PERFORM;
    const char *body="LOAD_GLOBAL 0\nPUSH_STR tick\nEQ\nASSERT\nPUSH_I64 7\nRET\narm:\n"
        "PUSH_STR tick\nSTORE_GLOBAL 0\nPUSH_I64 7\nEFFECT_RESUME\n";
    char source[2048];snprintf(source,sizeof source,"%s%s",prefix,body);
    NvmModule *m=effect_global_fixture(source,NULL,true);slot(m->ownership_data+140,TAG_STRING,1);
    effect_save(directory,"global_string",m);
    snprintf(source,sizeof source,"%sPUSH_BOOL 0\nASSERT\nPUSH_I64 0\nRET\narm:\n"
        "PUSH_STR tick\nSTORE_GLOBAL 0\nPUSH_I64 7\nRET\n",prefix);
    m=effect_global_fixture(source,NULL,true);slot(m->ownership_data+140,TAG_STRING,1);
    effect_save(directory,"global_return",m);
    snprintf(source,sizeof source,"%sPUSH_I64 7\nRET\narm:\n"
        "PUSH_STR tick\nSTORE_GLOBAL 0\nPUSH_BOOL 0\nASSERT\nPUSH_I64 7\nEFFECT_RESUME\n",prefix);
    m=effect_global_fixture(source,NULL,true);slot(m->ownership_data+140,TAG_STRING,1);
    effect_save(directory,"global_assertion",m);
    m=effect_global_fixture("PUSH_STR tick\nSTORE_GLOBAL 0\nPUSH_I64 7\nOWN_PACK 0\nPERFORM pass 1\n"
        "OWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nRET\n",NULL,true);
    slot(m->ownership_data+140,TAG_STRING,1);effect_save(directory,"global_unhandled",m);
    effect_save(directory,"global_recursive",effect_global_fixture(EG_INSTALL EG_PERFORM EG_FINISH
        "LOAD_LOCAL 3\nPUSH_I64 0\nEQ\nJMP_TRUE base\nLOAD_LOCAL 3\nPUSH_I64 1\nSUB\nPERFORM tick 1\nEFFECT_RESUME\n"
        "base:\nPUSH_I64 7\nSTORE_GLOBAL 0\nPUSH_I64 7\nEFFECT_RESUME\n",NULL,false));
}
