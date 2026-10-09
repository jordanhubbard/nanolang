/* I reuse the unchanged paired runtime/refusal checks for selected unions. */
#define main owned_record_test_main
#include "test_owned_runtime.c"
#undef main
static void uword(uint8_t *p,uint32_t v){for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i));}
static void uslot(uint8_t *p,uint8_t tag,uint32_t layout){p[0]=tag;uword(p+4,layout);}
static NvmModule *union_module(const char *body,unsigned helper) {
    char callee[256]={0};
    if(helper)snprintf(callee,sizeof callee,".function identity 1 10 0 %s 1\nOWN_MOVE_LOCAL 0\nRET\n.end\n.parameters 1 %s\n",
                      helper==3?"struct":"union",helper==3?"struct":"union");
    char assembly[8192];snprintf(assembly,sizeof assembly,
        ".types 2 0 2\n.entry 0\n.function main 0 10 0 int 1\n%s.end\n%s",body,callee);
    AsmResult result;NvmModule *m=asm_assemble_unverified(assembly,&result);
    if(!m)fprintf(stderr,"%s\n",result.message);CHECK(m);
    uint32_t handle=nvm_add_string(m,"Handle",6),choice=nvm_add_string(m,"Choice",6);
    uint32_t outer=nvm_add_string(m,"Outer",5),some=nvm_add_string(m,"Some",4);
    uint32_t empty=nvm_add_string(m,"Empty",5),pair=nvm_add_string(m,"Pair",4);
    uint32_t field=nvm_add_string(m,"value",5),box=nvm_add_string(m,"BoxChoice",9);
    NvmV2LayoutField fd={TAG_INT,NVM_V2_NO_INDEX,field};
    NvmV2LayoutField owners[]={{TAG_STRUCT,0,field},{TAG_STRUCT,0,field},{TAG_STRUCT,0,field}};
    NvmV2LayoutField nested={TAG_UNION,1,field};
    NvmV2Layout items[]={{NVM_V2_LAYOUT_STRUCT,1,handle,&fd},
        {NVM_V2_LAYOUT_UNION,3,choice,owners},{NVM_V2_LAYOUT_UNION,1,outer,&nested},
        {NVM_V2_LAYOUT_STRUCT,1,box,&nested}};
    NvmV2Layouts layouts={items,4};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=180+(helper?92:0);m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;uword(p,3);uword(p+4,4);p[8]=p[9]=p[10]=p[11]=3;uword(p+12,helper?2:1);
    unsigned at=16;
    for(unsigned f=0;f<(helper?2u:1u);f++) {
        unsigned count=10;p[at]=(uint8_t)count;p[at+2]=(uint8_t)f;
        uslot(p+at+4,f?(helper==3?TAG_STRUCT:TAG_UNION):TAG_INT,f?helper:NVM_V2_NO_INDEX);
        for(unsigned i=0;i<count;i++)uslot(p+at+12+8*i,
            i==0?(f?(helper==3?TAG_STRUCT:TAG_UNION):TAG_INT):i<3?TAG_STRUCT:i<7?TAG_UNION:i==7?TAG_BOOL:TAG_STRUCT,
            i==0?(f?helper:NVM_V2_NO_INDEX):i<3?0:i<5?1:i<7?2:i==7?NVM_V2_NO_INDEX:3);
        at+=12+8*count;
    }
    uword(p+at,4);uword(p+at+8,1);p[at+12]=1;p[at+14]=1;uword(p+at+16,52);uword(p+at+20,2);
    uword(p+at+24,1);p[at+28]=3;
    uword(p+at+32,some);p[at+38]=1;
    uword(p+at+40,empty);p[at+44]=1;
    uword(p+at+48,pair);p[at+52]=1;p[at+54]=2;
    uword(p+at+56,2);p[at+60]=1;uword(p+at+64,some);p[at+70]=1;
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    return m;
}
static const char *match_choice=
    "MATCH_TAG 0 some\nMATCH_TAG 1 empty\n"
    "OWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nOWN_STORE_LOCAL 2\nOWN_STORE_LOCAL 1\n"
    "OWN_UNPACK_LOCAL 1\nOWN_UNPACK_LOCAL 2\nADD\nRET\n"
    "some:\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nRET\n"
    "empty:\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nPUSH_I64 0\nRET\n";
#ifndef OWNED_UNION_NO_MAIN
int main(int argc,char **argv) {
    CHECK(argc==2);unsigned number=0;
    const char *constructors[]={"PUSH_I64 42\nOWN_PACK 0\nAGG_PACK 1 0 0 1\n",
        "AGG_PACK 1 0 1 0\n",
        "PUSH_I64 10\nOWN_PACK 0\nPUSH_I64 32\nOWN_PACK 0\nAGG_PACK 1 0 2 2\n"};
    for(unsigned kind=0;kind<3;kind++) for(unsigned route=0;route<5;route++) {
        char body[4096];snprintf(body,sizeof body,"%s%s%s",constructors[kind],
            route==0?"OWN_STORE_LOCAL 3\nOWN_MOVE_LOCAL 3\nOWN_STORE_LOCAL 4\nOWN_MOVE_LOCAL 4\n":
            route==1?"AGG_PACK 1 1 0 1\nOWN_STORE_LOCAL 5\nOWN_MOVE_LOCAL 5\nOWN_STORE_LOCAL 6\nOWN_UNPACK_LOCAL 6\n":
            route==2?"CALL 1\n":
            route==3?"OWN_PACK 3\nCALL 1\nOWN_STORE_LOCAL 8\nOWN_UNPACK_LOCAL 8\n":
                     "AGG_PACK 1 1 0 1\nCALL 1\nMATCH_TAG 0 outer_arm\nHALT\nouter_arm:\nOWN_STORE_LOCAL 5\nOWN_UNPACK_LOCAL 5\n",match_choice);
        execute_module(union_module(body,route==2?1:route==3?3:route==4?2:0),kind==1?0:42,argv[1],number++,TAG_INT);
    }
    for(unsigned branch=0;branch<2;branch++) {
        char body[4096];snprintf(body,sizeof body,
            "PUSH_BOOL %u\nJMP_FALSE none\n%sJMP joined\nnone:\n%sjoined:\n%s",
            branch,constructors[0],constructors[1],match_choice);
        execute_module(union_module(body,false),branch?42:0,argv[1],number++,TAG_INT);
    }
    /* The loop keeps a scalar count and discharges every aggregate each time. */
    execute_module(union_module("PUSH_I64 0\nSTORE_LOCAL 0\nloop:\nLOAD_LOCAL 0\nOWN_PACK 0\n"
        "AGG_PACK 1 0 0 1\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nOWN_STORE_LOCAL 1\n"
        "OWN_UNPACK_LOCAL 1\nPUSH_I64 1\nADD\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 20\nLT\n"
        "JMP_TRUE loop\nLOAD_LOCAL 0\nRET\n",false),20,argv[1],number++,TAG_INT);
    const char *invalid[]={
        "AGG_PACK 1 0 1 0\nDUP\nPOP\nPOP\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nPOP\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nOWN_STORE_LOCAL 3\nLOAD_LOCAL 3\nPOP\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nCALL 1\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nOWN_STORE_LOCAL 3\nOWN_MOVE_LOCAL 3\nOWN_MOVE_LOCAL 3\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nOWN_STORE_LOCAL 3\nAGG_PACK 1 0 1 0\nOWN_STORE_LOCAL 3\nPUSH_I64 0\nRET\n",
        "PUSH_I64 9\nAGG_PACK 1 0 0 1\nOWN_STORE_LOCAL 3\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nAGG_PACK 1 1 0 1\nAGG_PACK 1 1 0 1\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nOWN_STORE_LOCAL 3\nREGION_BEGIN\nBORROW_LOCAL_SHARED 0 3\nREF_GET 0 0\nRET\n",
        "PUSH_BOOL 1\nJMP_FALSE done\nAGG_PACK 1 0 1 0\nOWN_STORE_LOCAL 3\ndone:\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nOWN_STORE_LOCAL 3\nPUSH_I64 0\nRET\n",
        "PUSH_I64 9\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nAGG_GET 0\nPOP\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nAGG_PACK 1 1 0 1\nCALL 1\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 0 0\nPUSH_I64 0\nRET\n",
        "AGG_PACK 1 0 1 0\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nOWN_UNPACK_LOCAL 3\nPUSH_I64 0\nRET\n",
        ("AGG_PACK 1 0 1 0\nCALL 1\nMATCH_TAG 0 selected\nHALT\nselected:\nOWN_STORE_LOCAL 3\n"
        "OWN_UNPACK_LOCAL 3\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nRET\n"),
        "AGG_PACK 1 0 1 0\nAGG_TAG\nRET\n"
    };
    for(unsigned i=0;i<sizeof invalid/sizeof invalid[0];i++)
        refused(union_module(invalid[i],i==3 || i==12 || i==15),argv[1],i);
    printf("%u selected union runtime checks passed\n",checks);return 0;
}

#endif
