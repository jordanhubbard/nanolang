/* My leaf/pair/wrapper tree distinguishes depth, siblings and nominal layouts. */
#define NESTED_START "PUSH_I64 10\nOWN_PACK 0\nPUSH_I64 32\nOWN_PACK 0\nOWN_PACK 2\nOWN_PACK 3\nOWN_STORE_LOCAL 2\nREGION_BEGIN\n"
#define NESTED_FINISH "REGION_END\nOWN_UNPACK_LOCAL 2\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nOWN_UNPACK_LOCAL 3\nADD\nRET\n"
static NvmModule *nested_fixture(const char *body) {
    NvmModule *m=fixture(body,false,false);
    NvmV2Layouts old={0};CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&old)==NVM_V2_OK);
    NvmV2Layout items[4];memcpy(items,old.items,3*sizeof(*items));
    NvmV2LayoutField wrapper={TAG_STRUCT,2,NVM_V2_NO_INDEX};
    items[3]=(NvmV2Layout){NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&wrapper};
    NvmV2Layouts layouts={items,4};m->struct_count=4;
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);nvm_v2_layouts_free(&old);
    uint8_t *data=calloc(112,1);CHECK(data);memcpy(data,m->ownership_data,68);
    free(m->ownership_data);m->ownership_data=data;m->ownership_size=112;
    word(data,NVM_OWNERSHIP_PATH_VERSION);word(data+4,4);data[11]=3;
    word(data+40,2);word(data+48,3); /* local1 pair; local2 wrapper */
    word(data+68,5); /* paths [0,0], [0,1], [0,0], [1], [0] */
    data[72]=2;data[80]=2;data[86]=1;data[88]=2;data[96]=1;data[100]=1;data[104]=1;
    bool needs;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    return m;
}

#define NESTED_ALLOCATION_BODY NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_SHARED 1 0\nPUSH_I64 5\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nREF_GET 1 0\nPOP\nOWN_UNPACK_LOCAL 0\nPOP\nREGION_END\n" NESTED_FINISH
