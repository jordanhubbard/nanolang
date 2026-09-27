/* I construct identical scalar-global declarations for flow and execution. */
static const char *body="PUSH_I64 7\nOWN_PACK 0\nAGG_PACK 1 0 0 1\n"
    "OWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nMATCH_TAG 0 selected\nPOP\nPUSH_BOOL 0\nASSERT\nHALT\n"
    "selected:\nPOP\nOWN_UNPACK_VARIANT 0 0 1\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nRET\n";
static size_t attach(NvmModule *m,const uint8_t *tags,uint32_t count) {
    size_t at=m->ownership_size;uint32_t bytes=4+4*count;
    uint8_t *data=realloc(m->ownership_data,at+8+bytes);CHECK(data);
    m->ownership_data=data;m->ownership_size=(uint32_t)(at+8+bytes);
    memset(data+at,0,8+bytes);
    /* My entry has three locals; each optional zero-local helper adds 12 bytes. */
    word(data+60+(m->function_count-1)*12,2);data[at]=NVM_OWNERSHIP_EXTENSION_SCALAR_GLOBALS;
    data[at+2]=NVM_OWNERSHIP_EXTENSION_REVISION_1;word(data+at+4,bytes);word(data+at+8,count);
    for(uint32_t i=0;i<count;i++)data[at+12+4*i]=tags[i];
    return at;
}
static void helper(NvmModule *m,const char *code) {
    char source[1024];snprintf(source,sizeof(source),".function change 0 0 0 int 1\n%s.end\n",code);
    AsmResult assembled;NvmModule *h=asm_assemble_unverified(source,&assembled);CHECK(h);
    NvmFunctionEntry entry=h->functions[0];entry.name_idx=nvm_add_string(m,"change",6);
    entry.code_offset=nvm_append_code(m,h->code,h->code_size);CHECK(entry.code_offset!=UINT32_MAX);
    CHECK(nvm_add_function(m,&entry)==1);nvm_module_free(h);
    uint32_t size=m->ownership_size;uint8_t *data=realloc(m->ownership_data,size+12);CHECK(data);
    m->ownership_data=data;m->ownership_size=size+12;
    memmove(data+64,data+52,size-52);memset(data+52,0,12);slot(data+56,TAG_INT,0);word(data+12,2);
}
