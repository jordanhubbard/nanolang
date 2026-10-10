/* Shared typed-global modules for analysis and paired runtime checks. */
static void global_word(uint8_t *p,uint32_t value) {
    for (unsigned i=0;i<4;i++) p[i]=(uint8_t)(value>>(8*i));
}
static void global_slot(uint8_t *p,uint8_t tag,uint8_t mutable) {
    p[0]=tag;p[1]=mutable;global_word(p+4,(tag==TAG_STRUCT || tag==TAG_UNION)?0:NVM_V2_NO_INDEX);
}
static NvmModule *global_fixture(const char *body,const char *helper,const char *nested,
                                 uint8_t tag,bool mutable) {
    char source[8192];
    int length=snprintf(source,sizeof(source),
        ".types 1 0 0\n.entry 0\n.string \"value\"\n"
        ".function main 0 0 0 int 1\n%s.end\n%s%s%s%s%s%s",
        body,helper?".function helper 0 0 0 int 1\n":"",helper?helper:"",helper?".end\n":"",
        nested?".function nested 0 0 0 int 1\n":"",nested?nested:"",nested?".end\n":"");
    CHECK(length>0 && (size_t)length<sizeof(source));
    AsmResult error;NvmModule *m=asm_assemble_unverified(source,&error);
    if (!m) fprintf(stderr,"global fixture: %s\n",error.message);
    CHECK(m);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout record={NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field};
    NvmV2Layouts layouts={&record,1};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=40+12*m->function_count+16;
    m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;global_word(p,3);global_word(p+4,1);p[8]=3;
    global_word(p+12,m->function_count);unsigned at=16;
    for (uint32_t f=0;f<m->function_count;f++,at+=12) global_slot(p+at+4,TAG_INT,0);
    global_word(p+at,4);global_word(p+at+8,1);p[at+12]=NVM_OWNERSHIP_EXTENSION_GLOBALS;
    p[at+14]=1;global_word(p+at+16,20);global_word(p+at+20,2);
    global_slot(p+at+24,tag,mutable);global_slot(p+at+32,TAG_INT,1);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    return m;
}
