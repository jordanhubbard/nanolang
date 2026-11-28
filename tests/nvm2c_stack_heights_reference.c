/* I invoke the actual ordinary shared height/retain helper unchanged. */
#include "../src/nanoisa/verifier.c"
static uint32_t word(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
int main(int argc,char **argv) {
    for(int arg=1;arg<argc;arg++) {
        FILE *file=fopen(argv[arg],"rb");
        if(!file || fseek(file,0,SEEK_END))return 2;
        long size=ftell(file);
        if(size<0 || fseek(file,0,SEEK_SET))return 2;
        unsigned char *bytes=malloc(size?(size_t)size:1);
        if(!bytes || fread(bytes,1,(size_t)size,file)!=(size_t)size || fclose(file))return 2;
        NvmFunctionEntry *functions=NULL;NvmImportEntry *imports=NULL;
        VmDecodedFunction decoded={0};bool valid=false;
        NvmModule module={0};uint32_t current=0;
        if(size>=24) {
            uint32_t start=word(bytes),length=word(bytes+4),count=word(bytes+8);
            current=word(bytes+12);uint32_t locals=word(bytes+16),foreign=word(bytes+20);
            size_t header=24+(size_t)count*8+(size_t)foreign*8;
            if(count && count<=65536 && foreign<=65536 && current<count && locals<=UINT16_MAX &&
               header<=(size_t)size && start<=(size_t)size-header && length<=(size_t)size-header-start) {
                functions=calloc(count,sizeof *functions);imports=calloc(foreign?foreign:1,sizeof *imports);
                if(!functions || !imports)return 2;
                valid=true;
                for(uint32_t i=0;i<count;i++) {
                    uint32_t arity=word(bytes+24+i*8),results=word(bytes+28+i*8);
                    if(arity>UINT16_MAX || results>UINT8_MAX){valid=false;break;}
                    functions[i].arity=(uint16_t)arity;functions[i].result_count=(uint8_t)results;
                }
                for(uint32_t i=0;valid && i<foreign;i++) {
                    uint32_t params=word(bytes+24+count*8+i*8),tag=word(bytes+28+count*8+i*8);
                    if(params>UINT16_MAX || tag>UINT8_MAX){valid=false;break;}
                    imports[i].param_count=(uint16_t)params;imports[i].return_type=(uint8_t)tag;
                }
                functions[current].code_offset=start;functions[current].code_length=length;
                functions[current].local_count=(uint16_t)locals;
                module.functions=functions;module.function_count=count;
                module.imports=imports;module.import_count=foreign;
                module.code=bytes+header;module.code_size=(uint32_t)((size_t)size-header);
                char error[VM_DECODE_ERROR_SIZE];
                if(valid)valid=vm_decode_function(&module,current,&decoded,error);
                for(uint32_t i=0;valid && i<decoded.instruction_count;i++) {
                    const DecodedInstruction *item=&decoded.instructions[i].instruction;
                    if(item->opcode==OP_CALL_EXTERN && item->operands[0].u32>=foreign)valid=false;
                }
            }
        }
        if(!valid)puts("D");
        else {
            uint16_t maximum=0xabcd;
            NvmVerifyResult result=verify_stack_heights(&module,&decoded,current,&maximum);
            NvmVerifyResult no_max=verify_stack_heights(&module,&decoded,current,NULL);
            if(result.ok!=no_max.ok || strcmp(result.error_msg,no_max.error_msg))return 3;
            if(result.ok)printf("1:%u\n",maximum);
            else {if(maximum!=0xabcd)return 3;printf("0:%s\n",result.error_msg);}
        }
        vm_decoded_function_free(&decoded);free(functions);free(imports);free(bytes);
    }
    return 0;
}
