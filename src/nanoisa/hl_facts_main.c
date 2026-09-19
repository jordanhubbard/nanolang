/* I expose verified scalar reconstruction facts, not executable authority. */
#include "nanoisa.h"
#include "local_bindings.h"
#include "verifier.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
static void name(const uint8_t *s,uint32_t n) {
    putchar('"');
    for(uint32_t i=0;s && i<n && i<48;i++) {
        unsigned c=s[i];putchar((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'?(int)c:'_');
    }
    putchar('"');
}
static bool scalar(uint8_t tag){return tag==TAG_INT || tag==TAG_U8 || tag==TAG_BOOL || tag==TAG_FLOAT;}
int main(int argc,char **argv) {
    if(argc!=2){fputs("I require one NanoISA module.\n",stderr);return 2;}
    NanoisaErr error;NvmModule *m=nanoisa_load_file(argv[1],&error);
    if(!m){fprintf(stderr,"I cannot load this module: %s\n",error.message);return 1;}
    NvmVerifyResult verified=nvm_verify(m);
    if(!verified.ok || !(m->header.flags & NVM_FLAG_HAS_MAIN) || !m->function_count || m->function_count>32 ||
       m->import_count || m->module_ref_count || m->ownership_size || m->passive_size ||
       m->layout_size || m->struct_count || m->union_count || m->enum_count ||
       (m->header.flags & NVM_FLAG_NEEDS_EXTERN)) {
        fputs("I require a verified closed scalar module without retained runtime contracts.\n",stderr);
        nvm_module_free(m);return 1;
    }
    VmDecodedModule decoded;char detail[VM_DECODE_ERROR_SIZE];
    if(!vm_decode_module(m,&decoded,detail)){fprintf(stderr,"I cannot decode this module: %s\n",detail);nvm_module_free(m);return 1;}
    uint32_t total=0;
    for(uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];
        total+=decoded.functions[f].instruction_count;
        if(total>4096 || fn->local_count>128 || fn->upvalue_count || fn->result_count!=1 ||
           !scalar(fn->result_tag) || !nvm_get_string(m,fn->name_idx) || !strcmp(nvm_get_string(m,fn->name_idx),"__init__")) goto refused;
        if(fn->arity && (!m->function_param_types || !m->function_param_types[f]))goto refused;
        for(uint16_t p=0;p<fn->arity;p++)if(!scalar(m->function_param_types[f][p]))goto refused;
    }
    if(m->header.entry_point>=m->function_count || m->functions[m->header.entry_point].arity ||
       m->functions[m->header.entry_point].result_tag!=TAG_INT)goto refused;
    printf("{\"entry\":%u,\"functions\":[",m->header.entry_point);
    for(uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];const VmDecodedFunction *d=&decoded.functions[f];
        if(f)putchar(',');
        printf("{\"name\":");name((const uint8_t *)nvm_get_string(m,fn->name_idx),nvm_get_string_len(m,fn->name_idx));
        printf(",\"locals\":%u,\"result\":%u,\"params\":[",fn->local_count,fn->result_tag);
        for(uint16_t p=0;p<fn->arity;p++){if(p)putchar(',');printf("%u",m->function_param_types[f][p]);}
        printf("],\"names\":[");
        for(uint16_t slot=0;slot<fn->local_count;slot++) {
            if(slot)putchar(',');
            NvmLocalBinding b;bool found=false;
            for(uint32_t i=0;i<d->instruction_count;i++) {
                if(nvm_local_name_at(m,f,slot,d->instructions[i].byte_offset,&b)==NVM_LOCAL_NAMES_VALID){found=true;break;}
            }
            name(found?b.name:NULL,found?b.name_size:0);
        }
        printf("],\"size\":%u,\"code\":[",d->code_size);
        for(uint32_t i=0;i<d->instruction_count;i++) {
            const VmDecodedInstruction *di=&d->instructions[i];const DecodedInstruction *ins=&di->instruction;
            int64_t arg=0;
            if(ins->operand_count)switch(ins->operand_types[0]) {
                case OPERAND_U8:arg=ins->operands[0].u8;break;
                case OPERAND_U16:arg=ins->operands[0].u16;break;
                case OPERAND_U32:arg=ins->operands[0].u32;break;
                case OPERAND_I32:arg=ins->operands[0].i32;break;
                case OPERAND_I64:arg=ins->operands[0].i64;break;
                default:break;
            }
            if(i)putchar(',');
            printf("{\"pc\":%u,\"op\":\"%s\",\"arg\":%" PRId64 ",\"target\":%u",di->byte_offset,
                isa_get_info(ins->opcode)->name,arg,di->resolved_target==UINT32_MAX?UINT32_MAX:di->resolved_target-fn->code_offset);
            if(ins->operand_count && ins->operand_types[0]==OPERAND_F64) {
                uint64_t bits;
                _Static_assert(sizeof(bits)==sizeof(ins->operands[0].f64), "I require binary64 operand storage.");
                memcpy(&bits,&ins->operands[0].f64,sizeof(bits));
                printf(",\"f64_bits\":\"%016" PRIx64 "\"",bits);
            }
            putchar('}');
        }
        printf("]}");
    }
    puts("]}");vm_decoded_module_free(&decoded);nvm_module_free(m);return ferror(stdout)?1:0;
refused:
    fputs("I require bounded explicit int/u8/bool/float function signatures and no initializer.\n",stderr);
    vm_decoded_module_free(&decoded);nvm_module_free(m);return 1;
}
