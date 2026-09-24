/* I keep callback target inference independent of executable admission. */
static NvmModule *targets_fixture(unsigned kind,const unsigned order[6]) {
    unsigned id[6];for(unsigned f=0;f<6;f++)id[order[f]]=f;
    char source[8192]=".types 1 0 0\n.entry 0\n";size_t used=strlen(source);
    for(unsigned f=0;f<6;f++) {
        unsigned role=order[f],arity=(role==2 || role==3 || (kind==15 && role>=4))?1:0;
        const char *result=(role==1 || role==2)?"function":kind==14&&role>=4?"struct":"int";
        used+=(size_t)snprintf(source+used,sizeof(source)-used,
            ".function role%u %u 2 %u %s 1\n",role,arity,kind==7&&role==4?1:0,result);
        if(role==0) {
            if(kind==9)used+=(size_t)snprintf(source+used,sizeof(source)-used,"FUNCREF %u\nCALL_INDIRECT 0 1\n",id[1]);
            else used+=(size_t)snprintf(source+used,sizeof(source)-used,"CALL %u\n",id[1]);
            if(kind==11)used+=(size_t)snprintf(source+used,sizeof(source)-used,"DUP\nSWAP\nPOP\n");
            if(kind==10)used+=(size_t)snprintf(source+used,sizeof(source)-used,"FUNCREF %u\nCALL_INDIRECT 1 1\n",id[2]);
            else used+=(size_t)snprintf(source+used,sizeof(source)-used,"CALL %u\n",id[2]);
            used+=(size_t)snprintf(source+used,sizeof(source)-used,"STORE_LOCAL 0\nLOAD_LOCAL 0\nCALL %u\nRET\n",id[3]);
        } else if(role==1) {
            if(kind==1)used+=(size_t)snprintf(source+used,sizeof(source)-used,
                "FUNCREF %u\nSTORE_LOCAL 0\nloop:\nPUSH_BOOL 0\nJMP_FALSE done\nFUNCREF %u\nSTORE_LOCAL 0\nJMP loop\ndone:\nLOAD_LOCAL 0\nRET\n",id[4],id[5]);
            else if(kind==2)used+=(size_t)snprintf(source+used,sizeof(source)-used,
                "FUNCREF %u\nSTORE_LOCAL 0\nFUNCREF %u\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nRET\n",id[4],id[5]);
            else if(kind==3)used+=(size_t)snprintf(source+used,sizeof(source)-used,"PUSH_I64 0\nRET\n");
            else if(kind==6)used+=(size_t)snprintf(source+used,sizeof(source)-used,"FUNCREF 9\nRET\n");
            else if(kind==8)used+=(size_t)snprintf(source+used,sizeof(source)-used,
                "PUSH_BOOL 0\nJMP_FALSE other\nFUNCREF %u\nSTORE_LOCAL 0\nJMP done\nother:\nFUNCREF %u\nSTORE_LOCAL 0\ndone:\nLOAD_LOCAL 0\nRET\n",id[4],id[5]);
            else used+=(size_t)snprintf(source+used,sizeof(source)-used,
                "PUSH_BOOL 0\nJMP_FALSE other\nFUNCREF %u\nJMP done\nother:\nFUNCREF %u\ndone:\nRET\n",id[4],id[5]);
        } else if(role==2)used+=(size_t)snprintf(source+used,sizeof(source)-used,"LOAD_LOCAL 0\nRET\n");
        else if(role==3)used+=(size_t)snprintf(source+used,sizeof(source)-used,
            "%sLOAD_LOCAL 0\nCALL_INDIRECT %u %u\n%sRET\n",
            kind==5?"PUSH_I64 7\n":kind==15?"PUSH_I64 42\nOWN_PACK 0\n":"",
            kind==5||kind==15?1:0,kind==13?0:1,
            kind==14?"OWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\n":kind==13?"PUSH_I64 0\n":"");
        else if(role==4 && kind==4)used+=(size_t)snprintf(source+used,sizeof(source)-used,"FUNCREF %u\nCALL_INDIRECT 0 1\nRET\n",id[4]);
        else if(role==4 && kind==12)used+=(size_t)snprintf(source+used,sizeof(source)-used,"CALL %u\nRET\n",id[4]);
        else if(kind==15)used+=(size_t)snprintf(source+used,sizeof(source)-used,"OWN_UNPACK_LOCAL 0\nRET\n");
        else used+=(size_t)snprintf(source+used,sizeof(source)-used,"PUSH_I64 %u\n%sRET\n",role+37,kind==14?"OWN_PACK 0\n":"");
        used+=(size_t)snprintf(source+used,sizeof(source)-used,".end\n");
        if(arity)used+=(size_t)snprintf(source+used,sizeof(source)-used,".parameters %u %s\n",f,kind==15&&role>=4?"struct":"function");
        CHECK(used<sizeof(source));
    }
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m)fprintf(stderr,"%s\n",assembled.message);CHECK(m);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout layout={NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field};
    NvmV2Layouts layouts={&layout,1};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=16+6*28+4;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,2);word(data+4,1);data[8]=3;word(data+12,6);
    for(unsigned f=0;f<6;f++) {
        unsigned role=order[f],base=16+f*28;data[base]=2;data[base+2]=(role==2 || role==3 || (kind==15 && role>=4))?1:0;
        slot(data+base+4,(role==1 || role==2)?TAG_FUNCTION:kind==14&&role>=4?TAG_STRUCT:TAG_INT,0);
        slot(data+base+12,role<4?TAG_FUNCTION:kind==15?TAG_STRUCT:TAG_INT,0);
        slot(data+base+20,kind==14&&role==3?TAG_STRUCT:TAG_INT,0);
    }
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);return m;
}
static void owned_callback_targets(void) {
    const unsigned orders[][6]={{0,1,2,3,4,5},{0,5,4,3,2,1},{0,3,1,5,2,4},
        {0,2,4,1,5,3},{0,4,3,2,1,5},{0,5,2,4,3,1}};
    CHECK(!nvm_affine_targets_create(NULL));nvm_affine_targets_free(NULL);
    for(unsigned order=0;order<6;order++)for(unsigned kind=0;kind<16;kind++) {
        NvmModule *m=targets_fixture(kind,orders[order]);NvmAffineTargets *plan=nvm_affine_targets_create(m);
        bool accepted=kind<3 || (kind>=8 && kind<12) || kind>=14;CHECK((plan!=NULL)==accepted);
        if(plan) {
            unsigned id[6];for(unsigned f=0;f<6;f++)id[orders[order][f]]=f;
            uint8_t expected=(uint8_t)((1u<<id[5])|(kind==2?0:1u<<id[4]));
            for(unsigned f=0;f<6;f++) {
                VmDecodedFunction code={0};char error[VM_DECODE_ERROR_SIZE];CHECK(vm_decode_function(m,f,&code,error));
                for(uint32_t i=0;i<code.instruction_count;i++) {
                    const VmDecodedInstruction *in=&code.instructions[i];uint8_t mask=99;
                    if(in->instruction.opcode==OP_CALL_INDIRECT) {
                        CHECK(nvm_affine_targets_at(plan,f,in->byte_offset,&mask));
                        CHECK(mask==(f==id[3]?expected:(1u<<id[kind==9?1:2])));
                    } else {CHECK(!nvm_affine_targets_at(plan,f,in->byte_offset,&mask));CHECK(mask==99);}
                }
                vm_decoded_function_free(&code);
            }
            uint8_t mask=99;CHECK(!nvm_affine_targets_at(plan,6,0,&mask));CHECK(mask==99);
            CHECK(!nvm_affine_targets_at(plan,0,UINT32_MAX,&mask));CHECK(mask==99);
            CHECK(!nvm_affine_targets_at(plan,0,0,NULL));
            /* A provenance plan alone must not switch on owned indirect calls. */
            CHECK(!nvm_verify_owned_module(m).ok);
            nvm_affine_targets_free(plan);
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
            if(!order && kind==10)for(unsigned failure=1;;failure++) {
                allocation_attempts=0;fail_at=failure;
                plan=nvm_affine_targets_create(m);fail_at=0;
                if(plan) {CHECK(allocation_attempts<failure);nvm_affine_targets_free(plan);break;}
                CHECK(failure<1000);plan=nvm_affine_targets_create(m);CHECK(plan);nvm_affine_targets_free(plan);
            }
#endif
        }
        nvm_module_free(m);
    }
}
