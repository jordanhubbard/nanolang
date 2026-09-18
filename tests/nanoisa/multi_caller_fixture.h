/* I retain two functions and use existing parameter descriptors, not a new wire table. */
static NvmModule *multi_fixture(const char *body,const char *helper,uint16_t count,
                               const uint8_t *modes) {
    char source[24576],parameters[128]=".parameters 1";
    for(uint16_t p=0;p<count;p++)strcat(parameters," struct");
    snprintf(source,sizeof(source),".types 3 0 0\n.entry 0\n.function main 0 16 0 int 1\n%s\n.end\n"
        ".function helper %u 10 0 int 1\n%s\n.end\n%s\n",body,count,helper,parameters);
    AsmResult error;NvmModule *m=asm_assemble_unverified(source,&error);
    if(!m)fprintf(stderr,"%s\n",error.message);
    CHECK(m);
    NvmV2LayoutField scalar={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField fields[2]={{TAG_STRUCT,0,NVM_V2_NO_INDEX},{TAG_STRUCT,0,NVM_V2_NO_INDEX}};
    NvmV2Layout layouts[3]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&scalar},
        {NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,fields},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&scalar}};
    NvmV2Layouts all={layouts,3};CHECK(nvm_retain_layouts(m,&all)==NVM_V2_OK);
    m->ownership_size=276;m->ownership_data=calloc(276,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,2);word(p+4,3);p[8]=p[9]=p[10]=3;word(p+12,2);
    p[16]=16;slot(p+20,TAG_INT,0,NVM_V2_NO_INDEX);
    for(uint16_t i=0;i<16;i++) {
        bool record=i>=2 && i<=11;
        slot(p+28+i*8,record?TAG_STRUCT:TAG_INT,0,
            record?(i==10?1:i==11?2:0):NVM_V2_NO_INDEX);
    }
    p[156]=10;p[158]=(uint8_t)count;slot(p+160,TAG_INT,0,NVM_V2_NO_INDEX);
    for(uint16_t i=0;i<10;i++)slot(p+168+i*8,i<count?TAG_STRUCT:TAG_INT,
        i<count?modes[i]:0,i<count?0:NVM_V2_NO_INDEX);
    word(p+248,3);
    for(unsigned i=0;i<3;i++){p[252+8*i]=1;p[256+8*i]=i==1?1:0;}
    bool needs;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK);
    return m;
}
#define TWO_ROOTS "PUSH_I64 10\nOWN_PACK 0\nOWN_STORE_LOCAL 2\nPUSH_I64 32\nOWN_PACK 0\nOWN_STORE_LOCAL 3\n"
#define TWO_SUM "REGION_END\nOWN_UNPACK_LOCAL 2\nOWN_UNPACK_LOCAL 3\nADD\nRET"
#define TWO_BORROWS "REGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 2\nBORROW_LOCAL_EXCLUSIVE 1 3\n"
#define PAIR_ROOT TWO_ROOTS "OWN_MOVE_LOCAL 2\nOWN_MOVE_LOCAL 3\nOWN_PACK 1\nOWN_STORE_LOCAL 10\n"
#define PAIR_SUM "REGION_END\nOWN_UNPACK_LOCAL 10\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 2\nOWN_UNPACK_LOCAL 2\nOWN_UNPACK_LOCAL 3\nADD\nRET"
#define WRITE_BOTH "PUSH_I64 42\nREF_SET 0 0\nPUSH_I64 7\nREF_SET 1 0\nREF_GET 0 0\nREF_GET 1 0\nADD\nRET"
#define READ_BOTH "REF_GET 0 0\nREF_GET 1 0\nADD\nRET"
