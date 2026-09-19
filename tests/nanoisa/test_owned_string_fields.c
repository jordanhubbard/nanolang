/* I retain immutable field roots while moving unique shells. */
#define OWNED_STRING_ALLOC_TEST
#include "test_owned_string_print.c"
#include "mixed_float_proof.h"

static NvmModule *field_fixture(unsigned index) {
    char source[20000]=".string ready \"ready\"\n.string empty \"\"\n.string other \"other\"\n.types 3 0 0\n.entry 0\n";
    append(source,sizeof(source),".function main 0 8 0 int 1\nPUSH_I64 99\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nREGION_BEGIN\nBORROW_LOCAL_SHARED 0 0\n");
    for(unsigned i=0;i<2;i++) append(source,sizeof(source),"CALL 1\nCALL 2\nOWN_STORE_LOCAL 1\nREF_GET 0 0\nPUSH_I64 99\nEQ\nASSERT\nOWN_MOVE_LOCAL 1\nCALL 3\nPUSH_I64 42\nEQ\nASSERT\n");
    append(source,sizeof(source),"REGION_END\nOWN_UNPACK_LOCAL 0\nPOP\nPUSH_I64 42\nRET\n.end\n.function factory 0 8 0 struct 1\nPUSH_BOOL %u\nJMP_FALSE alternate\n",index==1?0:1);
    const char *text=index==1?"empty":"ready";
    for(unsigned arm=0;arm<2;arm++) {
        if(arm)append(source,sizeof(source),"alternate:\n");
        append(source,sizeof(source),"PUSH_I64 42\nOWN_PACK 0\nPUSH_STR %s\n%sOWN_PACK 1\nPUSH_BOOL 1\nOWN_PACK 2\nRET\n",text,index==2?"PUSH_BOOL 0\nASSERT\n":"");
    }
    append(source,sizeof(source),".end\n.function relay 1 8 0 struct 1\nOWN_MOVE_LOCAL 0\nRET\n.end\n.parameters 2 struct\n.function consume 1 8 0 int 1\nOWN_UNPACK_LOCAL 0\nASSERT\nOWN_STORE_LOCAL 1\nLOAD_LOCAL 1\nAGG_GET 1\nDUP\nSTORE_LOCAL 3\nSTORE_LOCAL 4\nLOAD_LOCAL 3\nPUSH_STR other\nNE\nASSERT\nPUSH_STR other\nSTORE_LOCAL 3\nOWN_UNPACK_LOCAL 1\nPUSH_STR %s\nEQ\nASSERT\nOWN_STORE_LOCAL 2\nLOAD_LOCAL 4\nPUSH_STR %s\nSWAP\nEQ\nASSERT\nLOAD_LOCAL 4\nDUP\nPOP\nCALL 4\nPUSH_I64 42\nEQ\nASSERT\n%sOWN_UNPACK_LOCAL 2\nRET\n.end\n.parameters 3 struct\n.function inspect 1 8 0 int 1\nLOAD_LOCAL 0\nPUSH_STR %s\nEQ\nASSERT\nPUSH_I64 42\nRET\n.end\n.parameters 4 string\n",text,text,index==3?"PUSH_BOOL 0\nASSERT\n":"",text);
    AsmResult error;NvmModule *m=asm_assemble_unverified(source,&error);
    if(!m)fprintf(stderr,"%s\n",error.message);
    CHECK(m);
    NvmV2LayoutField leaf={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField bundle[]={{TAG_STRUCT,0,NVM_V2_NO_INDEX},{TAG_STRING,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX}};
    NvmV2LayoutField outer[]={{TAG_STRUCT,1,NVM_V2_NO_INDEX},{TAG_BOOL,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX}};
    NvmV2Layout rows[]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&leaf},{NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,bundle},{NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,outer}};
    NvmV2Layouts layouts={rows,3};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=16+5*76+4;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,2);word(p+4,3);memset(p+8,3,3);word(p+12,5);
    for(unsigned f=0;f<5;f++) {
        unsigned h=16+f*76;p[h]=8;p[h+2]=(uint8_t)m->functions[f].arity;
        slot(p+h+4,f==1||f==2?TAG_STRUCT:TAG_INT,0,f==1||f==2?2:NVM_V2_NO_INDEX);
        for(unsigned l=0;l<8;l++) {
            unsigned nominal=NVM_V2_NO_INDEX;uint8_t tag=TAG_INT;
            if(f==0&&l<2)nominal=l?2:0;
            if((f==2||f==3)&&l==0)nominal=2;
            if(f==3&&l==1)nominal=1;
            if(f==3&&l==2)nominal=0;
            if((f==3&&(l==3||l==4))||(f==4&&l==0))tag=TAG_STRING;
            if(nominal!=NVM_V2_NO_INDEX)tag=TAG_STRUCT;
            slot(p+h+12+l*8,tag,0,nominal);
        }
    }
    return m;
}
static void field_refusals(void) {
    for(unsigned which=0;which<5;which++) {
        NvmModule *m=field_fixture(0);
        if(which<2) {
            NvmV2Layouts rows={0};CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&rows)==NVM_V2_OK);
            rows.items[1].fields[1].type_tag=which?TAG_FLOAT:TAG_ARRAY;
            CHECK(nvm_retain_layouts(m,&rows)==NVM_V2_OK);nvm_v2_layouts_free(&rows);
        } else if(which==2) {
            /* I refuse a borrowed root even when a chosen sibling is scalar. */
            m->ownership_data[16+2*76+12+1]=NVM_REFERENCE_SHARED;
        } else if(which==3) {
            m->functions[4].result_tag=TAG_STRING;
            slot(m->ownership_data+16+4*76+4,TAG_STRING,0,NVM_V2_NO_INDEX);
        } else CHECK(replace_opcode(m,4,OP_EQ,OP_LT,false));
        CHECK(!nvm_verify_owned_module(m).ok);char error[256];CHECK(!nvm2c_emit(m,error,sizeof(error)));
        nvm_module_free(m);
    }
    NvmModule *m=field_fixture(0);consuming_verified(m);
    CHECK(!nvm_verify_profile(m,NVM_PROFILE_CLOSED_SCALAR).ok);
    CHECK(!nvm_verify_profile(m,NVM_PROFILE_CLOSED_LITERAL_STRINGS).ok);
    CHECK(!nvm_verify_profile(m,NVM_PROFILE_CLOSED_MANAGED_STRINGS).ok);
    NvmMixedLayoutView *view=NULL;
    CHECK(nvm_describe_mixed_layouts(m,&view).status!=NVM_RECORD_DESCRIBED);CHECK(!view);
    NvmMixedFloatProof *proof=NULL;
    CHECK(nvm_analyze_mixed_float_origins(m,&proof).status!=NVM_MIXED_SHAPE_PROVED);CHECK(!proof);
    NvmV2Layouts layouts={0};CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&layouts)==NVM_V2_OK);
    uint16_t path[]={0,0};NvmReferencePlace place={1,0,2,path,2,0,NVM_REFERENCE_SHARED};
    CHECK(!nvm_reference_place_valid(&layouts,2,&place));nvm_v2_layouts_free(&layouts);
    nvm_module_free(m);
}
#ifndef OWNED_FIELD_ALLOC_TEST
int main(int argc,char **argv) {
    (void)result_fixture;(void)string_fixture;(void)roundtrip;(void)refusals;(void)missing_instantiated_literal;
    CHECK(argc==2);field_refusals();
    for(unsigned index=0;index<4;index++) {
        NvmModule *m=field_fixture(index);consuming_verified(m);artifacts(m,argv[1],index);
        char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);
        AsmResult error;NvmModule *copy=asm_assemble(text,&error);CHECK(copy);consuming_verified(copy);free(text);nvm_module_free(copy);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<3;repeat++) {
            fprintf(stderr,"string field case=%u api=%u repeat=%u\n",index,api,repeat);
            NanoValue value=val_int(-91);VmResult status=result_api(&vm,api,&value);
            CHECK(status==(index>=2?VM_ERR_ASSERT_FAILED:VM_OK));
            if(status==VM_OK){if(api==1||api==2)value=vm.stack[--vm.stack_size];CHECK(value.tag==TAG_INT&&value.as.i64==42);vm_release(&vm.heap,value);}
            result_clean(&vm,baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u %u 42\n",index,index>=2?2:0);
    }
    printf("%u retained string field checks passed\n",checks);return 0;
}
#endif
