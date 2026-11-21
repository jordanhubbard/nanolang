/* I use both actual private recipe copying and the shared C height verifier. */
#define main record_recipe_fixture_main
#include "nvm2c_record_instructions_reference.c"
#undef main
#include "../src/nanoisa/verifier.c"

int main(int argc,char **argv) {
    for(int arg=1;arg<argc;arg++) {
        FILE *file=fopen(argv[arg],"rb");
        if(!file || fseek(file,0,SEEK_END))return 2;
        long size=ftell(file);
        if(size<0 || fseek(file,0,SEEK_SET))return 2;
        uint8_t *bytes=malloc(size?(size_t)size:1);
        if(!bytes || fread(bytes,1,(size_t)size,file)!=(size_t)size || fclose(file))return 2;
        NvmRecordArrayExecutionPlan plan={0};NvmFunctionEntry signatures[256]={{0}};
        bool valid=false;uint16_t maximum=UINT16_MAX;
        VmDecodedFunction decoded={0};
        if(size>=24) {
            uint32_t start=word(bytes),length=word(bytes+4),count=word(bytes+8);
            uint32_t current=word(bytes+12),maps=word(bytes+16),locals=word(bytes+20);
            if(count && count<=256 && current<count && maps<=1 && locals<=UINT16_MAX && (uint64_t)size>=24u+(uint64_t)count*8u) {
                size_t header=24+(size_t)count*8;valid=true;
                for(uint32_t i=0;i<count;i++) {
                    uint32_t arity=word(bytes+24+i*8),results=word(bytes+28+i*8);
                    if(arity>UINT16_MAX || results>UINT8_MAX){valid=false;break;}
                    signatures[i].arity=(uint16_t)arity;signatures[i].result_count=(uint8_t)results;
                }
                if(start>(uint64_t)size-header || length>(uint64_t)size-header-start)valid=false;
                if(valid) {
                    signatures[current].code_offset=start;signatures[current].code_length=length;
                    signatures[current].local_count=(uint16_t)locals;
                    plan.snapshot.functions=signatures;plan.snapshot.function_count=count;
                    plan.snapshot.code=bytes+header;plan.snapshot.code_size=(uint32_t)((size_t)size-header);
                    valid=rae_instructions_using(&plan,maps?rme_recipe:rae_old_recipe);
                    if(valid) {
                        char error[VM_DECODE_ERROR_SIZE];
                        valid=vm_decode_function(&plan.snapshot,current,&decoded,error);
                        if(valid)valid=verify_stack_heights(&plan.snapshot,&decoded,current,&maximum).ok;
                    }
                }
            }
        }
        if(valid)printf("1:%u\n",maximum);
        else { if(maximum!=UINT16_MAX)return 3; puts("0"); }
        vm_decoded_function_free(&decoded);free(plan.instructions);free(bytes);
    }
    return 0;
}
