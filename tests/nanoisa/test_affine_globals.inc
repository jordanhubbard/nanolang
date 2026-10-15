/* I exercise global flow in the ordinary and allocation-fault bytecode runs. */
#include "../../src/nanovm/vm_decode.h"
#include "owned_global_fixture.h"
static void global_case(const char *body,const char *helper,const char *nested,
                        uint8_t tag,bool mutable,bool accepted,const char *reason) {
    NvmModule *m=global_fixture(body,helper,nested,tag,mutable);
    uint8_t *before=malloc(m->ownership_size);CHECK(before);
    memcpy(before,m->ownership_data,m->ownership_size);
    NvmAffineAnalysis got=nvm_affine_analyze_function(m,0);
    if (got.ok!=accepted) fprintf(stderr,"global flow: %s at %u\n%s",got.message,got.byte_offset,body);
    CHECK(got.ok==accepted);if (reason) CHECK(strstr(got.message,reason));
    CHECK(!memcmp(before,m->ownership_data,m->ownership_size));free(before);
    CHECK(nvm_verify(m).ok==accepted);char diagnostic[256];
    char *native=nvm2c_emit(m,diagnostic,sizeof(diagnostic));CHECK((native!=NULL)==accepted);free(native);
    VmDecodedFunction decoded={0};char detail[VM_DECODE_ERROR_SIZE];
    CHECK(vm_decode_function(m,0,&decoded,detail));
    size_t size=decoded.instruction_count*sizeof(NvmAffineInstructionFact);
    NvmAffineInstructionFact *facts=malloc(size),*sentinel=malloc(size);CHECK(facts && sentinel);
    memset(facts,0xa5,size);memcpy(sentinel,facts,size);
    CHECK(nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count).ok==accepted);
    if (!accepted) CHECK(!memcmp(facts,sentinel,size));
    free(facts);free(sentinel);vm_decoded_function_free(&decoded);nvm_module_free(m);
}
static void check_global_flow(void) {
    const char *read="LOAD_GLOBAL 0\nRET\n";
    const char *write="PUSH_I64 7\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n";
    const char *increment="LOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n";
    global_case("PUSH_I64 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,true,true,NULL);
    global_case(read,NULL,NULL,TAG_INT,true,false,"global initialization");
    global_case("PUSH_I64 0\nSTORE_GLOBAL 1\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,true,false,"global initialization");
    global_case("PUSH_I64 0\nRET\nLOAD_GLOBAL 2\nRET\n",NULL,NULL,TAG_INT,true,false,"declared global");
    global_case("LOAD_GLOBAL 2\nRET\n",NULL,NULL,TAG_INT,true,false,"declared global");
    global_case("PUSH_I64 1\nSTORE_GLOBAL 2\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_INT,true,false,"declared global");
    global_case("PUSH_BOOL 1\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_INT,true,false,"exact global type");
    global_case("STORE_GLOBAL 0\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_INT,true,false,"value for a global store");
    global_case("PUSH_I64 1\nSTORE_GLOBAL 0\nPUSH_I64 2\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,true,true,NULL);
    global_case("PUSH_I64 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,false,true,NULL);
    global_case("PUSH_I64 1\nSTORE_GLOBAL 0\nPUSH_I64 2\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,false,false,"immutable store");
    global_case("PUSH_BOOL 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nASSERT\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_BOOL,true,true,NULL);
    global_case("PUSH_U8 42\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPOP\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_U8,true,true,NULL);
    global_case("PUSH_F64 2.0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_F64 2.0\nF64_EQ\nASSERT\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_FLOAT,true,true,NULL);
    global_case("PUSH_STR 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_STR 0\nEQ\nASSERT\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_STRING,true,true,NULL);
    global_case("LOAD_GLOBAL 0\nPOP\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_STRUCT,true,false,"global owner transfers");
    global_case("PUSH_I64 7\nOWN_PACK 0\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_STRUCT,true,false,"global owner transfers");
    global_case("PUSH_I64 7\nOWN_PACK 0\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_INT,true,false,"exact global type");
    /* Both predecessor orders and delayed joins must preserve missing writes. */
    for (unsigned order=0;order<2;order++) {
        char body[2048];const char *init="PUSH_I64 1\nSTORE_GLOBAL 0\n";
        const char *delay="NOP\nNOP\nNOP\nNOP\nNOP\nNOP\n";
        snprintf(body,sizeof(body),"PUSH_BOOL 0\nJMP_FALSE other\n%sJMP join\nother:\n%sjoin:\nLOAD_GLOBAL 0\nRET\n",
                 order?delay:init,order?init:delay);
        global_case(body,NULL,NULL,TAG_INT,true,false,"global initialization");
    }
    global_case("PUSH_BOOL 0\nJMP_FALSE other\nPUSH_I64 1\nSTORE_GLOBAL 0\nJMP join\nother:\nPUSH_I64 2\nSTORE_GLOBAL 0\njoin:\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,false,true,NULL);
    global_case("loop:\nPUSH_BOOL 0\nJMP_FALSE done\nPUSH_I64 1\nSTORE_GLOBAL 0\nJMP loop\ndone:\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,true,false,"global initialization");
    global_case("PUSH_I64 0\nSTORE_GLOBAL 0\nloop:\nPUSH_BOOL 0\nJMP_FALSE done\nLOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\nJMP loop\ndone:\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,true,true,NULL);
    global_case("loop:\nPUSH_BOOL 0\nJMP_FALSE done\nPUSH_I64 1\nSTORE_GLOBAL 0\nJMP loop\ndone:\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_INT,false,false,"immutable store");
    global_case("CALL 1\nRET\n",read,NULL,TAG_INT,true,false,"global initialization");
    global_case("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\nCALL 1\nRET\n",increment,NULL,TAG_INT,true,true,NULL);
    global_case("CALL 1\nPOP\nLOAD_GLOBAL 0\nRET\n",write,NULL,TAG_INT,false,true,NULL);
    global_case("CALL 1\nPOP\nCALL 1\nPOP\nLOAD_GLOBAL 0\nRET\n",write,NULL,TAG_INT,false,false,"immutable store");
    global_case("CALL 1\nRET\n","CALL 2\nRET\n",read,TAG_INT,true,false,"global initialization");
    global_case("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nRET\n","CALL 2\nRET\n",increment,TAG_INT,true,true,NULL);
    global_case("CALL 1\nPOP\nLOAD_GLOBAL 0\nRET\n","CALL 2\nRET\n",write,TAG_INT,false,true,NULL);
    global_case("PUSH_BOOL 0\nJMP_FALSE skip\nCALL 1\nPOP\nskip:\nLOAD_GLOBAL 0\nRET\n",write,NULL,TAG_INT,true,false,"global initialization");
    global_case("CALL 1\nPOP\nLOAD_GLOBAL 0\nRET\n","PUSH_BOOL 0\nJMP_FALSE skip\nPUSH_I64 7\nSTORE_GLOBAL 0\nskip:\nPUSH_I64 0\nRET\n",NULL,TAG_INT,true,false,"global initialization");
    global_case("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nRET\n","CALL 1\nRET\n",NULL,TAG_INT,true,false,"acyclic value graph");
    global_case("CALL 1\nPOP\nLOAD_GLOBAL 0\nRET\n","PUSH_BOOL 0\nJMP_FALSE second\nPUSH_I64 1\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\nsecond:\nPUSH_I64 2\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n",NULL,TAG_INT,false,true,NULL);
    global_case("CALL 1\nPOP\nLOAD_GLOBAL 0\nRET\n","PUSH_BOOL 0\nJMP_FALSE second\nPUSH_I64 1\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\nsecond:\nPUSH_I64 0\nRET\n",NULL,TAG_INT,false,false,"global initialization");
    /* A helper query permits required initialized input; the entry query must
     * still reject calling that helper before initialization. */
    NvmModule *conditional=global_fixture("CALL 1\nRET\n",read,NULL,TAG_INT,true);
    CHECK(nvm_affine_analyze_function(conditional,1).ok);
    CHECK(!nvm_affine_analyze_function(conditional,0).ok);nvm_module_free(conditional);
    /* Identical union payload shapes retain distinct global identities. A
     * global load loses constructor-arm knowledge and requires MATCH_TAG. */
    for (unsigned wrong=0;wrong<2;wrong++) {
        char body[512];snprintf(body,sizeof(body),
            "PUSH_I64 7\nAGG_PACK 1 %u 0 1\nSTORE_GLOBAL 0\n"
            "LOAD_GLOBAL 0\nMATCH_TAG 0 some\nPOP\nPUSH_I64 0\nRET\n"
            "some:\nAGG_GET 0\nRET\n",wrong?0u:1u);
        NvmModule *u=union_fixture(body,0,0,TAG_INT);
        NvmV2Layouts old={0};CHECK(nvm_v2_layouts_decode(u->layout_data,u->layout_size,&old)==NVM_V2_OK);
        NvmV2Layout pair[]={old.items[0],old.items[0]};
        pair[1].name_idx=nvm_add_string(u,"OtherChoice",11);
        NvmV2Layouts both={pair,2};u->union_count=2;
        CHECK(nvm_retain_layouts(u,&both)==NVM_V2_OK);nvm_v2_layouts_free(&old);
        uint8_t *p=realloc(u->ownership_data,136);CHECK(p);u->ownership_data=p;u->ownership_size=136;
        memset(p+84,0,52);word(p+4,2);p[8]=p[9]=NVM_LAYOUT_COMPLETE;
        word(p+36,2);word(p+44,68);word(p+48,2);
        memcpy(p+84,p+52,32);word(p+84,1);
        p[116]=NVM_OWNERSHIP_EXTENSION_GLOBALS;p[118]=1;word(p+120,12);word(p+124,1);
        slot(p+128,TAG_UNION,1);word(p+132,1);
        bool needs=false;CHECK(nvm_ownership_contracts_validate(u,&needs)==NVM_V2_OK && needs);
        NvmAffineAnalysis result=nvm_affine_analyze_function(u,0);
        if (result.ok!=(wrong==0)) fprintf(stderr,"global union: %s\n",result.message);
        CHECK(result.ok==(wrong==0));
        if (wrong) CHECK(strstr(result.message,"exact global type"));
        CHECK(nvm_verify(u).ok==(wrong==0));nvm_module_free(u);
    }
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
    NvmModule *m=global_fixture("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nRET\n",increment,NULL,TAG_INT,true);
    for (unsigned failure=1;;failure++) {
        allocation_attempts=0;fail_at=failure;
        NvmAffineAnalysis result=nvm_affine_analyze_function(m,0);
        fail_at=0;
        if (result.ok) {CHECK(allocation_attempts<failure);break;}
        CHECK(failure<1000 && result.message[0]);CHECK(nvm_affine_analyze_function(m,0).ok);
    }
    VmDecodedFunction decoded={0};char detail[VM_DECODE_ERROR_SIZE];
    CHECK(vm_decode_function(m,0,&decoded,detail));
    NvmAffineInstructionFact facts[32],sentinel[32];CHECK(decoded.instruction_count<32);
    memset(sentinel,0xa5,sizeof(sentinel));
    for (unsigned failure=1;;failure++) {
        memcpy(facts,sentinel,sizeof(facts));allocation_attempts=0;fail_at=failure;
        NvmAffineAnalysis result=nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count);
        fail_at=0;
        if (result.ok) {CHECK(allocation_attempts<failure);break;}
        CHECK(failure<1000 && result.message[0]);CHECK(!memcmp(facts,sentinel,sizeof(facts)));
        CHECK(nvm_affine_analyze_instructions(m,0,facts,decoded.instruction_count).ok);
    }
    vm_decoded_function_free(&decoded);
    forced_visit_limit=1;CHECK(!nvm_affine_analyze_function(m,0).ok);forced_visit_limit=0;
    CHECK(nvm_affine_analyze_function(m,0).ok);nvm_module_free(m);
#endif
}
