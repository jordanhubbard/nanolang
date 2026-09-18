/* I check ordinary byte-string conversion bits and ownership in one VM. */
#include "../../src/nanovm/vm.h"
#include "../../src/nanoisa/verifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc;
char **g_argv;
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *input = fopen(argv[1], "rb");
    if (!input) return 2;
    const uint8_t code[] = {OP_LOAD_LOCAL,0,0,OP_CAST_FLOAT,OP_RET};
    NvmModule *module = nvm_module_new();
    NvmFunctionEntry fn = {.arity=1,.local_count=1,.result_count=1,.result_tag=TAG_FLOAT};
    fn.name_idx = nvm_add_string(module,"parse",5);
    fn.code_offset = nvm_append_code(module,code,sizeof code);
    fn.code_length = sizeof code;
    nvm_add_function(module,&fn);
    if (!nvm_verify(module).ok) return 3;
    VmState vm; vm_init(&vm,module);
    uint64_t baseline = vm.heap.stats.num_objects;
    for (;;) {
        unsigned char size[4];
        size_t count=fread(size,1,4,input);
        if (!count && feof(input)) break;
        if (count!=4) return 4;
        uint32_t length=(uint32_t)size[0]|((uint32_t)size[1]<<8)|((uint32_t)size[2]<<16)|((uint32_t)size[3]<<24);
        if(length>1048576)return 4;
        char *bytes=malloc((size_t)length+1);
        if(!bytes || fread(bytes,1,length,input)!=length)return 4;
        bytes[length]=0;
        VmString *string=vm_string_new(&vm.heap,bytes,length);free(bytes);
        if(!string)return 5;
        NanoValue value=val_string(string),result=val_void();
        if(vm_invoke(&vm,0,&value,1,&result)!=VM_OK || result.tag!=TAG_FLOAT ||
           vm.frame_count || vm.stack_size || string->header.ref_count!=1)return 6;
        uint64_t bits;memcpy(&bits,&result.as.f64,8);
        printf("%016llx\n",(unsigned long long)bits);
        vm_release(&vm.heap,value);
        if(vm.heap.stats.num_objects!=baseline)return 7;
    }
    fclose(input);vm_destroy(&vm);nvm_module_free(module);return 0;
}
