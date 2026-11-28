/* I invoke the actual shared type-rule and fixpoint implementation unchanged. */
#include "../src/nanoisa/verifier_types.c"

static uint32_t word(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
int main(int argc,char **argv) {
    bool trace=argc>1 && !strcmp(argv[1],"--trace");
    for(int argument=trace?2:1;argument<argc;argument++) {
        FILE *file=fopen(argv[argument],"rb");
        if(!file || fseek(file,0,SEEK_END))return 2;
        long size=ftell(file);
        if(size<0 || fseek(file,0,SEEK_SET))return 2;
        unsigned char *bytes=malloc(size?(size_t)size:1);
        if(!bytes || fread(bytes,1,(size_t)size,file)!=(size_t)size || fclose(file))return 2;
        if(size<24){puts("D");free(bytes);continue;}
        uint32_t start=word(bytes),length=word(bytes+4),count=word(bytes+8);
        uint32_t current=word(bytes+12),maximum=word(bytes+16),mode=word(bytes+20);
        if(mode==1) {
            TypeRule rule={.result=TYPE_UNKNOWN};
            bool typed=start<=255 && type_rule_for((uint8_t)start,&rule);
            const InstructionInfo *info=start<=255?isa_get_info((uint8_t)start):NULL;
            printf("R:%u:%u:%u:%u:%u:%u:%u:%s\n",typed,rule.arg_count,rule.result,rule.result_count,
                rule.args[0],rule.args[1],rule.args[2],info?info->name:"UNKNOWN");
            free(bytes);continue;
        }
        if(mode || !count || count>65536 || current>=count || maximum>UINT16_MAX ||
           start>(uint64_t)size-24 || length>(uint64_t)size-24-start) {
            puts("D");free(bytes);continue;
        }
        NvmFunctionEntry *functions=calloc(count,sizeof *functions);
        if(!functions)return 2;
        functions[current].code_offset=start;functions[current].code_length=length;
        NvmModule module={0};module.functions=functions;module.function_count=count;
        module.code=bytes+24;module.code_size=(uint32_t)((size_t)size-24);
        VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
        if(!vm_decode_function(&module,current,&decoded,error))puts("D");
        else {
            NvmVerifyResult result=nvm_verify_function_types(&module,current,&decoded,(uint16_t)maximum,NULL,0);
            if(trace) {
                NvmRecordArrayBudget budget={0};
                NvmVerifyResult observed=verify_types_checked(&module,current,&decoded,(uint16_t)maximum,NULL,0,&budget);
                if(observed.ok!=result.ok || strcmp(observed.error_msg,result.error_msg) || budget.limited || budget.memory || budget.bytes)return 3;
                fprintf(stderr,"%s\t%u\t%u\t%llu\n",argv[argument],decoded.instruction_count,maximum,(unsigned long long)budget.work);
            }
            if(result.ok)puts("1");else printf("0:%s\n",result.error_msg);
        }
        vm_decoded_function_free(&decoded);free(functions);free(bytes);
    }
    return 0;
}
