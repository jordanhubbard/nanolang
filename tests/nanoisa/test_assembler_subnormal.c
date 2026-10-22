/* I preserve correctly rounded nonzero subnormal assembly operands. */
#include "assembler.h"
#include "isa.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void accepted(const char *token, uint64_t expected) {
    char source[1024];
    snprintf(source,sizeof source,".function main 0 0 0 float 1\n PUSH_F64 %s\n RET\n.end\n",token);
    AsmResult result;NvmModule *m=asm_assemble(source,&result);
    if(!m){fprintf(stderr,"I require accepted %s: %s\n",token,result.message);exit(1);}
    const uint8_t *code=m->code+m->functions[0].code_offset;
    uint64_t actual=0;for(unsigned i=0;i<8;i++)actual|=(uint64_t)code[1+i]<<(8*i);
    if(code[0]!=OP_PUSH_F64 || actual!=expected){fprintf(stderr,"I require bits %016" PRIx64 " for %s, got %016" PRIx64 "\n",expected,token,actual);exit(1);}
    nvm_module_free(m);
}
static void refused(const char *token) {
    char source[1024];snprintf(source,sizeof source,".function main 0 0 0 float 1\n PUSH_F64 %s\n RET\n.end\n",token);
    AsmResult result;NvmModule *m=asm_assemble(source,&result);
    if(m || result.error!=ASM_ERR_BAD_OPERAND){fprintf(stderr,"I require operand refusal for %s\n",token);exit(1);}
}
int main(void) {
    accepted("5e-324",UINT64_C(1));accepted("-5e-324",UINT64_C(0x8000000000000001));
    accepted("2.225073858507201e-308",UINT64_C(0x000fffffffffffff));
    accepted("-2.225073858507201e-308",UINT64_C(0x800fffffffffffff));
    accepted("2.2250738585072014e-308",UINT64_C(0x0010000000000000));
    accepted("1.7976931348623157e308",UINT64_C(0x7fefffffffffffff));
    accepted("0.0",0);accepted("-0.0",UINT64_C(0x8000000000000000));
    accepted("bits:7ff8000000000042",UINT64_C(0x7ff8000000000042));
    refused("1e9999");refused("-1e9999");refused("1e-9999");refused("-1e-9999");
    refused("not-a-float");refused("bits:0001");
    const char *integers[] = {"9223372036854775808", "-9223372036854775809"};
    for(unsigned i=0;i<2;i++) {
        char source[256];snprintf(source,sizeof source,".function main 0 0 0 int 1\n PUSH_I64 %s\n RET\n.end\n",integers[i]);
        AsmResult result;NvmModule *m=asm_assemble(source,&result);
        if(m || result.error!=ASM_ERR_BAD_OPERAND){fputs("I preserve integer overflow refusal.\n",stderr);return 1;}
    }
    puts("I preserve nine exact binary64 operands, six FLOAT refusals and two integer overflow refusals.");return 0;
}
