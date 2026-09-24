/* I keep callback target inference independent of executable admission. */
static NvmModule *targets_fixture(unsigned kind,const unsigned order[6]) {
    unsigned failure=kind/32;kind%=32;
    unsigned id[6];for(unsigned f=0;f<6;f++)id[order[f]]=f;
    char source[8192]=".types 1 0 0\n.entry 0\n";size_t used=strlen(source);
    for(unsigned f=0;f<6;f++) {
        unsigned role=order[f],arity=(role==2 || role==3 || ((kind==15 || kind==16) && role>=4))?1:0;
        const char *result=(role==1 || role==2)?"function":kind==14&&role>=4?"struct":kind==16&&role>=4?"void":"int";
        used+=(size_t)snprintf(source+used,sizeof(source)-used,
            ".function role%u %u 2 %u %s %u\n",role,arity,kind==7&&role==4?1:0,result,kind==16&&role>=4?0:1);
        if(role>=4 && failure==1)used+=(size_t)snprintf(source+used,sizeof(source)-used,"PUSH_BOOL 0\nASSERT\n");
        if(role==0) {
            used+=(size_t)snprintf(source+used,sizeof(source)-used,"PUSH_I64 42\nOWN_PACK 0\nOWN_STORE_LOCAL 1\n");
            if(kind==9)used+=(size_t)snprintf(source+used,sizeof(source)-used,"FUNCREF %u\nCALL_INDIRECT 0 1\n",id[1]);
            else used+=(size_t)snprintf(source+used,sizeof(source)-used,"CALL %u\n",id[1]);
            if(kind==11)used+=(size_t)snprintf(source+used,sizeof(source)-used,"DUP\nSWAP\nPOP\n");
            if(kind==10)used+=(size_t)snprintf(source+used,sizeof(source)-used,"FUNCREF %u\nCALL_INDIRECT 1 1\n",id[2]);
            else used+=(size_t)snprintf(source+used,sizeof(source)-used,"CALL %u\n",id[2]);
            used+=(size_t)snprintf(source+used,sizeof(source)-used,"STORE_LOCAL 0\nLOAD_LOCAL 0\nCALL %u\nOWN_UNPACK_LOCAL 1\nPOP\nRET\n",id[3]);
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
            "%sLOAD_LOCAL 0\nCALL_INDIRECT %u %u\n%s%sRET\n",
            kind==5?"PUSH_I64 7\n":(kind==15||kind==16)?"PUSH_I64 42\nOWN_PACK 0\n":"",
            kind==5||kind==15||kind==16?1:0,kind==13||kind==16?0:1,failure==3?"PUSH_BOOL 0\nASSERT\n":"",
            kind==14?"OWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\n":kind==13?"PUSH_I64 0\n":kind==16?"PUSH_I64 42\n":"");
        else if(role==4 && kind==4)used+=(size_t)snprintf(source+used,sizeof(source)-used,"FUNCREF %u\nCALL_INDIRECT 0 1\nRET\n",id[4]);
        else if(role==4 && kind==12)used+=(size_t)snprintf(source+used,sizeof(source)-used,"CALL %u\nRET\n",id[4]);
        else if(kind==15 || kind==16)used+=(size_t)snprintf(source+used,sizeof(source)-used,"OWN_UNPACK_LOCAL 0\n%s%sRET\n",kind==16?"POP\n":"",failure==2?"PUSH_BOOL 0\nASSERT\n":"");
        else used+=(size_t)snprintf(source+used,sizeof(source)-used,"PUSH_I64 %u\n%s%sRET\n",role+37,kind==14?"OWN_PACK 0\n":"",failure==2?"PUSH_BOOL 0\nASSERT\n":"");
        used+=(size_t)snprintf(source+used,sizeof(source)-used,".end\n");
        if(arity)used+=(size_t)snprintf(source+used,sizeof(source)-used,".parameters %u %s\n",f,(kind==15||kind==16)&&role>=4?"struct":"function");
        CHECK(used<sizeof(source));
    }
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m)fprintf(stderr,"%s\n",assembled.message);
    CHECK(m);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout layout={NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field};
    NvmV2Layouts layouts={&layout,1};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=16+6*28+4;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,2);word(data+4,1);data[8]=3;word(data+12,6);
    for(unsigned f=0;f<6;f++) {
        unsigned role=order[f],base=16+f*28;data[base]=2;data[base+2]=(role==2 || role==3 || ((kind==15 || kind==16) && role>=4))?1:0;
        slot(data+base+4,(role==1 || role==2)?TAG_FUNCTION:kind==14&&role>=4?TAG_STRUCT:kind==16&&role>=4?TAG_VOID:TAG_INT,0);
        slot(data+base+12,role<4?TAG_FUNCTION:(kind==15||kind==16)?TAG_STRUCT:TAG_INT,0);
        slot(data+base+20,!role || (kind==14&&role==3)?TAG_STRUCT:TAG_INT,0);
    }
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);return m;
}
static void owned_callback_target_boundaries(void) {
    for (unsigned count=8;count<=9;count++) for (unsigned reverse=0;reverse<2;reverse++)
        for (unsigned cycle=0;cycle<2;cycle++) {
            unsigned id[9]={0};
            for (unsigned role=1;role<count;role++) id[role]=reverse?count-role:role;
            char source[4096]=".types 1 0 0\n.entry 0\n";size_t used=strlen(source);
            for (unsigned f=0;f<count;f++) {
                unsigned role=f?(reverse?count-f:f):0;
                used+=(size_t)snprintf(source+used,sizeof(source)-used,
                    ".function role%u 0 2 0 int 1\n",role);
                if (!role) used+=(size_t)snprintf(source+used,sizeof(source)-used,
                    "PUSH_I64 9\nOWN_PACK 0\nOWN_STORE_LOCAL 1\n");
                if (role+1<count || cycle) used+=(size_t)snprintf(source+used,sizeof(source)-used,
                    "FUNCREF %u\nCALL_INDIRECT 0 1\n",id[role+1<count?role+1:1]);
                else used+=(size_t)snprintf(source+used,sizeof(source)-used,"PUSH_I64 42\n");
                if (!role) used+=(size_t)snprintf(source+used,sizeof(source)-used,"OWN_UNPACK_LOCAL 1\nPOP\n");
                used+=(size_t)snprintf(source+used,sizeof(source)-used,"RET\n.end\n");
                CHECK(used<sizeof(source));
            }
            AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);CHECK(m);
            NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
            NvmV2Layout layout={NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field};
            NvmV2Layouts layouts={&layout,1};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
            m->ownership_size=16+count*28+4;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
            uint8_t *data=m->ownership_data;word(data,2);word(data+4,1);data[8]=3;word(data+12,count);
            for (unsigned f=0;f<count;f++) {
                unsigned base=16+f*28;data[base]=2;slot(data+base+4,TAG_INT,0);
                slot(data+base+12,TAG_INT,0);slot(data+base+20,f?TAG_INT:TAG_STRUCT,0);
            }
            NvmAffineTargets *plan=nvm_affine_targets_create(m);
            CHECK((plan!=NULL)==(count==8 && !cycle));
            if (plan) {
                CHECK(nvm_verify_owned_module(m).ok);
                for (unsigned role=0;role+1<count;role++) {
                    VmDecodedFunction code={0};char error[VM_DECODE_ERROR_SIZE];
                    CHECK(vm_decode_function(m,id[role],&code,error));
                    bool found=false;
                    for (uint32_t i=0;i<code.instruction_count;i++)
                        if (code.instructions[i].instruction.opcode==OP_CALL_INDIRECT) {
                            uint8_t mask=0;found=true;
                            CHECK(nvm_affine_targets_at(plan,id[role],code.instructions[i].byte_offset,&mask));
                            CHECK(mask==(1u<<id[role+1]));
                        }
                    CHECK(found);vm_decoded_function_free(&code);
                }
                char error[256];char *native=nvm2c_emit(m,error,sizeof(error));CHECK(native);free(native);
            }
            nvm_affine_targets_free(plan);nvm_module_free(m);
        }
}
static void owned_callback_targets(void) {
    owned_callback_target_boundaries();
    const unsigned orders[][6]={{0,1,2,3,4,5},{0,5,4,3,2,1},{0,3,1,5,2,4},
        {0,2,4,1,5,3},{0,4,3,2,1,5},{0,5,2,4,3,1}};
    CHECK(!nvm_affine_targets_create(NULL));nvm_affine_targets_free(NULL);
    for(unsigned order=0;order<6;order++)for(unsigned kind=0;kind<17;kind++) {
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
            /* I separately require exact ownership after provenance inference. */
            NvmVerifyResult verified=nvm_verify_owned_module(m);
            if(!verified.ok)fprintf(stderr,"owned callback kind %u order %u: %s\n",kind,order,verified.error_msg);
            CHECK(verified.ok);
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

static void owned_callback_contract_refusals(void) {
    const unsigned order[6]={0,1,2,3,4,5};
    for(unsigned kind=14;kind<=15;kind++) {
        NvmModule *m=targets_fixture(kind,order);
        NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
        NvmV2Layout items[2]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field},
            {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field}};
        NvmV2Layouts layouts={items,2};m->struct_count=2;
        CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
        word(m->ownership_data+4,2);m->ownership_data[9]=3;
        unsigned base=16+4*28;
        if(kind==14) {
            word(m->ownership_data+base+8,1);
            VmDecodedFunction code={0};char error[VM_DECODE_ERROR_SIZE];CHECK(vm_decode_function(m,4,&code,error));
            bool found=false;
            for(uint32_t i=0;i<code.instruction_count;i++)if(code.instructions[i].instruction.opcode==OP_OWN_PACK) {
                word(m->code+m->functions[4].code_offset+code.instructions[i].byte_offset+1,1);found=true;
            }
            CHECK(found);vm_decoded_function_free(&code);
        } else word(m->ownership_data+base+16,1);
        bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
        NvmAffineTargets *plan=nvm_affine_targets_create(m);CHECK(plan);nvm_affine_targets_free(plan);
        /* Same-shaped resource layouts are not interchangeable, even when
         * target inference and encoded stack counts agree. */
        CHECK(!nvm_affine_analyze_function(m,0).ok);
        CHECK(!nvm_verify_owned_module(m).ok);
        char error[256];char *native=nvm2c_emit(m,error,sizeof(error));CHECK(!native);free(native);
        nvm_module_free(m);
    }
}
