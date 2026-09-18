/* I keep immutable string views separate from owned record leaves. */
#define OWNED_RESULT_ALLOC_TEST
#include "test_owned_value_results.c"
#include "disassembler.h"

static const char expected_output[] = "before\nbefore\n\nresource closed";

static NvmModule *string_fixture(bool failure) {
    char source[16384];
    snprintf(source,sizeof(source),
        ".string greeting \"before\"\n"
        ".string empty \"\"\n"
        ".string closed \"resource closed\"\n"
        ".types 3 0 0\n.entry 0\n"
        ".function main 0 8 0 int 1\n"
        "PUSH_I64 99\nOWN_PACK 0\nOWN_STORE_LOCAL 0\n"
        "PUSH_STR greeting\nCALL 1\nPUSH_STR greeting\nCALL 1\n"
        "PUSH_STR empty\nCALL 1\nPUSH_STR closed\nPRINT\n%s"
        "REGION_BEGIN\nBORROW_LOCAL_SHARED 0 0\nREF_GET 0 0\nPUSH_I64 99\nEQ\nASSERT\n"
        "REGION_END\nOWN_UNPACK_LOCAL 0\nPOP\nPUSH_I64 42\nRET\n.end\n"
        ".function first 1 8 0 void 0\nLOAD_LOCAL 0\nCALL 2\nRET\n.end\n.parameters 1 string\n"
        ".function second 1 8 0 void 0\nLOAD_LOCAL 0\nCALL 3\nRET\n.end\n.parameters 2 string\n"
        ".function third 1 8 0 void 0\nLOAD_LOCAL 0\nPRINTLN\nRET\n.end\n.parameters 3 string\n",
        failure?"PUSH_BOOL 0\nASSERT\n":"");
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m)fprintf(stderr,"%s\n",assembled.message);
    CHECK(m);
    NvmModule *layouts=fixture();
    m->layout_data=layouts->layout_data;layouts->layout_data=NULL;
    m->layout_size=layouts->layout_size;nvm_module_free(layouts);
    const unsigned functions=4,locals=8,record=12+8*locals;
    m->ownership_size=16+functions*record+4;
    m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,2);word(data+4,3);
    data[8]=data[9]=data[10]=3;word(data+12,functions);
    for(unsigned f=0;f<functions;f++) {
        unsigned header=16+f*record;data[header]=locals;data[header+2]=f?1:0;
        slot(data+header+4,f?TAG_VOID:TAG_INT,0,NVM_V2_NO_INDEX);
        for(unsigned local=0;local<locals;local++) {
            uint8_t tag=!f&&local==0?TAG_STRUCT:f&&local==0?TAG_STRING:TAG_INT;
            slot(data+header+12+8*local,tag,0,tag==TAG_STRUCT?0:NVM_V2_NO_INDEX);
        }
    }
    bool needs=false;
    CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);
    return m;
}

static uint32_t string_index(const NvmModule *m,const char *text) {
    size_t length=strlen(text);
    for(uint32_t i=0;i<m->string_count;i++)
        if(m->string_lengths[i]==length&&!memcmp(m->strings[i],text,length))return i;
    CHECK(false);return UINT32_MAX;
}

static bool replace_opcode(NvmModule *m,uint32_t function,uint8_t from,uint8_t to,
                           bool replace_operands) {
    VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
    CHECK(vm_decode_function(m,function,&decoded,error));bool changed=false;
    for(uint32_t i=0;i<decoded.instruction_count;i++) {
        VmDecodedInstruction *in=&decoded.instructions[i];
        if(in->instruction.opcode!=from)continue;
        uint32_t offset=m->functions[function].code_offset+in->byte_offset;
        m->code[offset]=to;
        if(replace_operands)for(uint32_t b=1;b<in->next_byte_offset-in->byte_offset;b++)
            m->code[offset+b]=OP_NOP;
        changed=true;break;
    }
    vm_decoded_function_free(&decoded);return changed;
}

static void exact_stream(FILE *stream,const char *expected) {
    char actual[256];CHECK(!fflush(stream));long size=ftell(stream);CHECK(size>=0);
    CHECK((size_t)size<sizeof(actual));rewind(stream);
    CHECK(fread(actual,1,(size_t)size,stream)==(size_t)size);actual[size]='\0';
    CHECK((size_t)size==strlen(expected)&&!memcmp(actual,expected,(size_t)size));
}

static void roundtrip(NvmModule *m) {
    char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult assembled;NvmModule *copy=asm_assemble_unverified(text,&assembled);
    if(!copy)fprintf(stderr,"%s\n",assembled.message);
    CHECK(copy);CHECK(copy->string_count==m->string_count);
    for(uint32_t i=0;i<m->string_count;i++)
        CHECK(copy->string_lengths[i]==m->string_lengths[i]&&
              !memcmp(copy->strings[i],m->strings[i],m->string_lengths[i]));
    CHECK(copy->ownership_size==m->ownership_size&&
          !memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
    CHECK(copy->function_count==m->function_count);
    for(uint32_t f=1;f<m->function_count;f++)
        CHECK(copy->function_param_types[f][0]==TAG_STRING);
    CHECK(nvm_verify_owned_module(copy).ok);
    free(text);nvm_module_free(copy);
}

static void refusals(void) {
    NvmModule *m=string_fixture(false);uint32_t greeting=string_index(m,"before");
    char saved=m->strings[greeting][2];m->strings[greeting][2]='\0';
    CHECK(!nvm_verify_owned_module(m).ok);m->strings[greeting][2]=saved;nvm_module_free(m);

    m=string_fixture(false);slot(m->ownership_data+16+76+12,TAG_INT,0,NVM_V2_NO_INDEX);
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=string_fixture(false);m->ownership_data[16+76+13]=1;
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);

    m=string_fixture(false);m->functions[1].result_count=1;m->functions[1].result_tag=TAG_STRING;
    slot(m->ownership_data+16+76+4,TAG_STRING,0,NVM_V2_NO_INDEX);
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);

    m=string_fixture(false);NvmV2Layouts rows={0};
    CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&rows)==NVM_V2_OK);
    rows.items[0].fields[0].type_tag=TAG_STRING;
    CHECK(nvm_retain_layouts(m,&rows)==NVM_V2_OK);nvm_v2_layouts_free(&rows);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);

    m=string_fixture(false);bool changed=false;
    for(uint32_t f=0;f<m->function_count&&!changed;f++) {
        VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
        CHECK(vm_decode_function(m,f,&decoded,error));
        for(uint32_t i=0;i<decoded.instruction_count;i++)
            if(decoded.instructions[i].instruction.opcode==OP_PRINTLN) {
                m->code[m->functions[f].code_offset+decoded.instructions[i].byte_offset]=OP_POP;
                changed=true;break;
            }
        vm_decoded_function_free(&decoded);
    }
    CHECK(changed&&!nvm_verify_owned_module(m).ok);nvm_module_free(m);

    m=string_fixture(false);changed=false;
    for(uint32_t f=0;f<m->function_count&&!changed;f++) {
        VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
        CHECK(vm_decode_function(m,f,&decoded,error));
        for(uint32_t i=0;i<decoded.instruction_count;i++)
            if(decoded.instructions[i].instruction.opcode==OP_PUSH_STR) {
                word(m->code+m->functions[f].code_offset+decoded.instructions[i].byte_offset+1,m->string_count);
                changed=true;break;
            }
        vm_decoded_function_free(&decoded);
    }
    CHECK(changed&&!nvm_verify_owned_module(m).ok);nvm_module_free(m);

    /* I reject both sides of PRINT's one-string stack contract. */
    m=string_fixture(false);CHECK(replace_opcode(m,3,OP_LOAD_LOCAL,OP_NOP,true));
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=string_fixture(false);CHECK(replace_opcode(m,0,OP_PRINT,OP_NOP,false));
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);

    /* A string edge does not widen the retained direct acyclic call graph. */
    m=string_fixture(false);CHECK(replace_opcode(m,1,OP_CALL,OP_TAIL_CALL,false));
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=string_fixture(false);changed=false;
    VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
    CHECK(vm_decode_function(m,1,&decoded,error));
    for(uint32_t i=0;i<decoded.instruction_count;i++) {
        VmDecodedInstruction *in=&decoded.instructions[i];
        if(in->instruction.opcode==OP_CALL) {
            word(m->code+m->functions[1].code_offset+in->byte_offset+1,1);
            changed=true;break;
        }
    }
    vm_decoded_function_free(&decoded);
    CHECK(changed&&!nvm_verify_owned_module(m).ok);nvm_module_free(m);
}

static void missing_instantiated_literal(void) {
    NvmModule *m=string_fixture(false);consuming_verified(m);VmState vm;vm_init(&vm,m);
    uint32_t greeting=string_index(m,"before");CHECK(greeting<vm.module_constants.count);
    VmString *saved=vm.module_constants.strings[greeting];CHECK(saved);
    vm.module_constants.strings[greeting]=NULL;uint64_t generation=vm.reference_generation;
    FILE *output=tmpfile();CHECK(output);vm.output=output;NanoValue result=val_void();
    CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_ERR_TYPE_ERROR);
    CHECK(vm.reference_generation==generation);exact_stream(output,"");
    vm.output=NULL;CHECK(!fclose(output));vm.module_constants.strings[greeting]=saved;
    vm_destroy(&vm);nvm_module_free(m);
}

#ifndef OWNED_STRING_ALLOC_TEST
int main(int argc,char **argv) {
    CHECK(argc==2);(void)result_fixture;refusals();missing_instantiated_literal();
    for(unsigned index=0;index<2;index++) {
        bool failure=index!=0;NvmModule *m=string_fixture(failure);
        consuming_verified(m);roundtrip(m);artifacts(m,argv[1],index);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<2;repeat++) {
            FILE *output=tmpfile();CHECK(output);vm.output=output;
            NanoValue result=val_int(-91);uint64_t generation=vm.reference_generation;
            VmResult status=result_api(&vm,api,&result);
            CHECK(status==(failure?VM_ERR_ASSERT_FAILED:VM_OK));CHECK(vm.reference_generation>generation);
            exact_stream(output,expected_output);vm.output=NULL;CHECK(!fclose(output));
            if(!failure) {
                if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}
                CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);
            }
            result_clean(&vm,baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);
        printf("case %u %u 42\n",index,failure?2:0);
    }
    printf("%u owned string print checks passed\n",checks);return 0;
}
#endif
