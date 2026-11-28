/* I expose actual private instruction copying, without its separate analysis. */
#include "../src/nanoisa/managed_array_shapes.c"
#include <inttypes.h>

static uint32_t word(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
int main(int argc, char **argv) {
    for(int arg=1;arg<argc;arg++) {
        FILE *file=fopen(argv[arg],"rb");
        if(!file || fseek(file,0,SEEK_END))return 2;
        long size=ftell(file);
        if(size<0 || fseek(file,0,SEEK_SET))return 2;
        uint8_t *bytes=malloc(size?(size_t)size:1);
        if(!bytes || fread(bytes,1,(size_t)size,file)!=(size_t)size || fclose(file))return 2;
        NvmRecordArrayExecutionPlan plan={0};
        NvmFunctionEntry signatures[256]={{0}};
        bool valid=false;uint32_t current=0;
        if(size>=20) {
            uint32_t start=word(bytes),length=word(bytes+4),count=word(bytes+8);
            current=word(bytes+12);uint32_t maps=word(bytes+16);
            if(count && count<=256 && current<count && maps<=1 && (uint64_t)size>=20u+(uint64_t)count*8u) {
                size_t header=20+(size_t)count*8;
                valid=true;
                for(uint32_t i=0;i<count;i++) {
                    uint32_t arity=word(bytes+20+i*8),results=word(bytes+24+i*8);
                    if(arity>UINT16_MAX || results>UINT8_MAX){valid=false;break;}
                    signatures[i].arity=(uint16_t)arity;signatures[i].result_count=(uint8_t)results;
                }
                if(start>(uint64_t)size-header || length>(uint64_t)size-header-start)valid=false;
                if(valid) {
                    signatures[current].code_offset=start;signatures[current].code_length=length;
                    plan.snapshot.functions=signatures;plan.snapshot.function_count=count;
                    plan.snapshot.code=bytes+header;plan.snapshot.code_size=(uint32_t)((size_t)size-header);
                    valid=rae_instructions_using(&plan,maps?rme_recipe:rae_old_recipe);
                }
            }
        }
        if(!valid)puts("0");
        else {
            NvmRecordArrayExecutionFunction fn=plan.functions[current];
            printf("1:%u:%u",plan.counts.globals,fn.instruction_count);
            for(uint32_t i=fn.instruction_start;fn.instruction_count<=1024 && i<fn.instruction_start+fn.instruction_count;i++) {
                NvmRecordArrayExecutionInstruction *r=&plan.instructions[i];
                printf(":%u,%u,%u,%u,%u,%u,%u,%u,%u",r->pc,r->next_pc,r->opcode,r->recipe,r->obligations,r->pops,r->pushes,r->callee,r->successor_count);
                for(uint8_t j=0;j<r->successor_count;j++)printf(",%u",r->successors[j]);
                printf(",%u",r->operand_count);
                for(uint8_t j=0;j<r->operand_count;j++)printf(",%u,%"PRIu32",%"PRIu32,r->operand_types[j],(uint32_t)r->operand_bits[j],(uint32_t)(r->operand_bits[j]>>32));
            }
            putchar('\n');
        }
        free(plan.instructions);free(bytes);
    }
    return 0;
}
