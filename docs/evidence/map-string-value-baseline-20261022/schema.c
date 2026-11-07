/* Nonexecuted declaration validation of the exact independently annotated text. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "nanoisa/assembler.h"
#include "nanoisa/ownership_map_declarations.h"
#include "nanoisa/record_map_admission.h"
#include "nanoisa/verifier.h"
#define CHECK(x) do {if(!(x)){fprintf(stderr,"schema check failed line %u: %s\n",__LINE__,#x);return 1;}}while(0)
int main(int argc,char **argv) {
 CHECK(argc==2);FILE *f=fopen(argv[1],"rb");CHECK(f);CHECK(fseek(f,0,SEEK_END)==0);long n=ftell(f);CHECK(n>=0);rewind(f);
 char *text=malloc((size_t)n+1);CHECK(text);CHECK(fread(text,1,(size_t)n,f)==(size_t)n);CHECK(fclose(f)==0);text[n]=0;
 AsmResult error;NvmModule *m=asm_assemble_unverified(text,&error);free(text);CHECK(m);CHECK(error.error==ASM_OK);
 NvmMapDeclarationPlan *p=NULL;NvmMapDeclarationResult result=nvm_prepare_map_declarations(m,&p);
 CHECK(result.status==NVM_MAP_DESCRIBED);CHECK(p);
 NvmMapDeclarationCounts counts;CHECK(nvm_map_declarations_counts(p,&counts));CHECK(counts.scalars==1&&counts.pairs==1&&counts.bindings==1);
 uint8_t tag=0;CHECK(nvm_map_declarations_scalar(p,0,&tag));CHECK(tag==TAG_STRING);
 NvmMapDeclarationPair pair;CHECK(nvm_map_declarations_pair(p,0,&pair));CHECK(pair.key_type==0&&pair.value_type==0);
 NvmMapDeclarationBinding binding;CHECK(nvm_map_declarations_binding(p,0,&binding));CHECK(binding.layout==0&&binding.field==0&&binding.pair==0);
 CHECK(nvm_map_declarations_field(p,0,0,&pair));CHECK(pair.key_type==0&&pair.value_type==0);
 CHECK(nvm_record_map_route(m)==NVM_RECORD_MAP_PRESENT);
 NvmRecordMapExecutionPlan *plan=(void *)(uintptr_t)1;NvmArrayEligibilityResult selected=nvm_record_map_admit(m,&plan);
 CHECK(selected.status==NVM_ARRAY_UNRESOLVED);CHECK(plan==(void *)(uintptr_t)1);
 CHECK(strstr(selected.message,"STRING-to-INT")!=NULL);NvmVerifyResult verified=nvm_verify(m);CHECK(!verified.ok);
 printf("PASS exact single STRING scalar/pair(0,0)/Holder.values declaration; selected UNRESOLVED: %s; verifier refuses: %s\n",selected.message,verified.error_msg);
 nvm_map_declarations_free(p);nvm_module_free(m);return 0;
}
