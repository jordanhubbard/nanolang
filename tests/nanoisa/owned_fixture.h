/* I share nominal test declarations between wire and execution gates. */
static void word(uint8_t *p,uint32_t v) { for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i)); }
static void slot(uint8_t *p,uint8_t tag,uint32_t layout) {p[0]=tag;word(p+4,layout);}
static NvmModule *fixture(const char *body,bool parameter,bool record_result) {
    char source[8192];snprintf(source,sizeof(source),
        ".types 3 0 0\n.entry 0\n.function main %u 5 0 %s 1\n%s.end\n%s",
        parameter?1:0,record_result?"struct":"int",body,parameter?".parameters 0 struct\n":"");
    AsmResult error;NvmModule *m=asm_assemble_unverified(source,&error);
    if(!m)fprintf(stderr,"%s\n",error.message);
    CHECK(m);
    NvmV2LayoutField fd={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField children[2]={{TAG_STRUCT,0,NVM_V2_NO_INDEX},{TAG_STRUCT,0,NVM_V2_NO_INDEX}};
    NvmV2Layout records[3]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&fd},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&fd},
        {NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,children}};
    NvmV2Layouts layouts={records,3};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=68;m->ownership_data=calloc(68,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,1);word(p+4,3);p[8]=p[9]=p[10]=3;word(p+12,1);
    p[16]=5;p[18]=parameter?1:0;
    slot(p+20,record_result?TAG_STRUCT:TAG_INT,record_result?0:NVM_V2_NO_INDEX);
    slot(p+28,TAG_STRUCT,0);slot(p+36,TAG_STRUCT,1);slot(p+44,TAG_STRUCT,2);
    slot(p+52,TAG_STRUCT,0);slot(p+60,TAG_INT,NVM_V2_NO_INDEX);
    return m;
}
