/* I keep same-numbered local references distinct through a value-call DAG. */
#define MULTIPLE_CONSUMING_ALLOC_TEST
#include "test_multiple_consuming_calls.c"
static NvmModule *callback_value_fixture(unsigned kind) {
    char source[4096];
    const char *producer=kind==2 ? "PUSH_I64 1" : kind==3 ? "LOAD_LOCAL 1" :
                         kind==5 ? "FUNCREF 9" : "FUNCREF 1";
    const char *control=kind==1 ? "PUSH_BOOL 0\nASSERT\n" :
        kind==4 ? "LOAD_LOCAL 1\nPUSH_I64 1\nADD\nPOP\n" :
        kind==7 ? "LOAD_LOCAL 1\nFUNCREF 1\nCALL_INDIRECT 1 1\nPOP\n" : "";
    snprintf(source,sizeof(source),
        ".types 3 0 0\n.entry 0\n.function main 0 16 0 int 1\n"
        "PUSH_I64 42\nOWN_PACK 0\nOWN_STORE_LOCAL 0\n"
        "%s\nCALL 1\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nDUP\nPOP\nPOP\n"
        "%sOWN_UNPACK_LOCAL 0\nRET\n.end\n"
        ".function echo 1 16 %u function 1\n%s\nRET\n.end\n.parameters 1 function\n",
        producer,control,kind==6?1:0,kind==8?"PUSH_I64 1":"LOAD_LOCAL 0");
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m)fprintf(stderr,"%s\n",assembled.message);
    CHECK(m);
    NvmModule *layouts=fixture();m->layout_data=layouts->layout_data;layouts->layout_data=NULL;
    m->layout_size=layouts->layout_size;nvm_module_free(layouts);
    m->ownership_size=16+2*140+4;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,2);word(data+4,3);data[8]=data[9]=data[10]=3;word(data+12,2);
    for(unsigned fn=0;fn<2;fn++) {
        unsigned header=16+fn*140,base=header+12;data[header]=16;data[header+2]=fn?1:0;
        slot(data+header+4,fn?TAG_FUNCTION:TAG_INT,0,NVM_V2_NO_INDEX);
        for(unsigned local=0;local<16;local++) {
            uint8_t tag=(!fn && !local)?TAG_STRUCT:((!fn && local==1)||(fn && !local))?TAG_FUNCTION:TAG_INT;
            slot(data+base+8*local,tag,0,tag==TAG_STRUCT?0:NVM_V2_NO_INDEX);
        }
    }
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);
    return m;
}
static NvmModule *graph_fixture(unsigned failure,unsigned functions) {
    if(functions==2)return callback_value_fixture(failure);
    (void)multiple_fixture;(void)multiple_refusals;
    char source[20000]=".types 3 0 0\n.entry 0\n";
    for(unsigned fn=0;fn<functions;fn++) {
        append(source,sizeof(source),".function graph%u %u 16 0 int 1\n",fn,fn>=2?2:0);
        if(!fn) append(source,sizeof(source),"CALL 1\nPOP\nCALL 1\n%sRET\n",failure==functions?"PUSH_BOOL 0\nASSERT\n":"");
        else {
            append(source,sizeof(source),"PUSH_I64 %u\nOWN_PACK 0\nOWN_STORE_LOCAL 14\nREGION_BEGIN\nBORROW_LOCAL_SHARED 0 14\n",10+fn);
            if(fn==1) append(source,sizeof(source),"PUSH_I64 20\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_I64 22\nOWN_PACK 2\nOWN_STORE_LOCAL 1\n");
            if(fn+1<functions) append(source,sizeof(source),"OWN_MOVE_LOCAL 0\nOWN_MOVE_LOCAL 1\nCALL %u\nSTORE_LOCAL 15\n",fn+1);
            else append(source,sizeof(source),"OWN_UNPACK_LOCAL 0\nOWN_UNPACK_LOCAL 1\nADD\nSTORE_LOCAL 15\n");
            if(fn==1) append(source,sizeof(source),
                "PUSH_I64 20\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_I64 22\nOWN_PACK 2\nOWN_STORE_LOCAL 1\n"
                "OWN_MOVE_LOCAL 0\nOWN_MOVE_LOCAL 1\nCALL %u\nSTORE_LOCAL 15\n",functions-1);
            append(source,sizeof(source),"REF_GET 0 0\nPUSH_I64 %u\nEQ\nASSERT\n%sREGION_END\nOWN_UNPACK_LOCAL 14\nPOP\nLOAD_LOCAL 15\nRET\n",10+fn,failure==fn?"PUSH_BOOL 0\nASSERT\n":"");
        }
        append(source,sizeof(source),".end\n");
        if(fn>=2) append(source,sizeof(source),".parameters %u struct struct\n",fn);
    }
    AsmResult result;NvmModule *m=asm_assemble_unverified(source,&result);
    if(!m)fprintf(stderr,"%s\n",result.message);
    CHECK(m);
    NvmModule *layouts=fixture();m->layout_data=layouts->layout_data;layouts->layout_data=NULL;
    m->layout_size=layouts->layout_size;nvm_module_free(layouts);
    m->ownership_size=16+functions*140+4;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,2);word(data+4,3);data[8]=data[9]=data[10]=3;word(data+12,functions);
    for(unsigned fn=0;fn<functions;fn++) {
        unsigned header=16+fn*140,base=header+12;data[header]=16;data[header+2]=fn>=2?2:0;
        slot(data+header+4,TAG_INT,0,NVM_V2_NO_INDEX);
        for(unsigned p=0;p<16;p++) slot(data+base+8*p,p<2||p==14?TAG_STRUCT:TAG_INT,0,p==1?2:p==0||p==14?0:NVM_V2_NO_INDEX);
    }
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);return m;
}
static void graph_refusals(void) {
    /* I admit local carriers, never callable resource fields or borrows. */
    for(unsigned kind=0;kind<3;kind++) {
        NvmModule *m=callback_value_fixture(0);
        if(!kind) {
            NvmV2Layouts layouts={0};
            CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&layouts)==NVM_V2_OK);
            layouts.items[0].fields[0].type_tag=TAG_FUNCTION;
            CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
            nvm_v2_layouts_free(&layouts);
        } else slot(m->ownership_data+16+12+8,TAG_FUNCTION,kind==1?1:0,kind==1?NVM_V2_NO_INDEX:0);
        CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    }
    for(unsigned kind=2;kind<=8;kind++) {
        NvmModule *m=callback_value_fixture(kind);
        CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    }
    for(unsigned kind=0;kind<4;kind++) {
        NvmModule *m=graph_fixture(0,kind==3?9:8);
        if(kind<3) {
            unsigned function=kind==2?7:2;
            VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
            CHECK(vm_decode_function(m,function,&decoded,error));
            if(kind<2) {
                bool found=false;
                for(uint32_t i=0;i<decoded.instruction_count;i++) {
                    VmDecodedInstruction *in=&decoded.instructions[i];
                    if(in->instruction.opcode==OP_CALL) {
                        word(m->code+m->functions[function].code_offset+in->byte_offset+1,kind?0:function);found=true;break;
                    }
                }
                CHECK(found);
            } else {
                /* I retain structurally valid metadata but require exact nominal parameters. */
                slot(m->ownership_data+16+function*140+12,TAG_STRUCT,0,2);
            }
            vm_decoded_function_free(&decoded);
        }
        CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    }
}
static void ordinary_chain(void) {
    char source[8192]=".entry 0\n";
    for(unsigned f=0;f<10;f++) {
        append(source,sizeof(source),".function ordinary%u 0 0 0 int 1\n",f);
        if(f<9)append(source,sizeof(source),"CALL %u\n",f+1);
        else append(source,sizeof(source),"PUSH_I64 42\n");
        append(source,sizeof(source),"RET\n.end\n");
    }
    AsmResult assembled;NvmModule *m=asm_assemble(source,&assembled);CHECK(m);
    VmState vm;vm_init(&vm,m);NanoValue result=val_void();
    CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK);
    CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_destroy(&vm);nvm_module_free(m);
}
#ifndef OWNED_GRAPH_ALLOC_TEST
static void graph_phase(const char *phase,unsigned index,unsigned api,unsigned repeat) {
    if(getenv("NANO_GRAPH_TRACE_PHASES"))
        fprintf(stderr,"graph phase=%s case=%u api=%u repeat=%u\n",phase,index,api,repeat);
}
int main(int argc,char **argv) {
    CHECK(argc==2||argc==3);
    graph_phase("ordinary-start",0,0,0);ordinary_chain();graph_phase("ordinary-done",0,0,0);
    graph_phase("refusals-start",0,0,0);graph_refusals();graph_phase("refusals-done",0,0,0);
    unsigned begin=0,end=12;
    if(argc==3) {
        char *text_end=NULL;unsigned long selected=strtoul(argv[2],&text_end,10);
        CHECK(argv[2][0]&&!*text_end&&selected<12);begin=(unsigned)selected;end=begin+1;
    }
    graph_phase("four-start",0,0,0);
    NvmModule *four=graph_fixture(0,4);consuming_verified(four);nvm_module_free(four);
    graph_phase("four-done",0,0,0);
    for(unsigned index=begin;index<end;index++) {
        bool fails=(index>0&&index<9)||index==11;
        graph_phase("fixture-start",index,0,0);
        NvmModule *m=index>=10?graph_fixture(index==11?1:0,2):graph_fixture(fails?index:0,index==9?4:8);
        graph_phase("verify-start",index,0,0);consuming_verified(m);
        graph_phase("artifacts-start",index,0,0);artifacts(m,argv[1],index);
        graph_phase("artifacts-done",index,0,0);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<2;repeat++) {
            graph_phase("api-start",index,api,repeat);
            NanoValue result=val_void();uint64_t generation=vm.reference_generation;
            VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):api==1?vm_execute(&vm):api==2?vm_call_function(&vm,0,NULL,0):vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
            CHECK(status==(fails?VM_ERR_ASSERT_FAILED:VM_OK));CHECK(vm.reference_generation>generation);
            if(!fails) {if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);}
            CHECK(!vm.stack_size&&!vm.frame_count);CHECK(!vm.references.active&&!vm.callee_references.active);
            for(unsigned frame=0;frame<NVM_OWNED_MAX_FUNCTIONS-2;frame++)CHECK(!vm.value_references[frame].active);
            CHECK(vm.heap.stats.num_objects==baseline);graph_phase("api-done",index,api,repeat);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u %u 42\n",index,fails?2:0);
    }
    printf("%u owned value graph checks passed\n",checks);return 0;
}

#endif
