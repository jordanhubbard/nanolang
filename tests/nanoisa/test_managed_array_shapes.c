/* I observe private analysis results without changing verifier admission. */
#include "../../src/nanoisa/managed_array_shapes.h"
#include "../../src/nanoisa/verifier.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv) {
    if(argc<2 || argc>3)return 2;
    NvmArrayEligibilityReport absent={0},*absent_out=&absent;
    NvmArrayEligibilityResult missing=nvm_analyze_managed_arrays(NULL,&absent_out);
    if(missing.status!=NVM_ARRAY_INVALID || absent_out!=&absent)return 7;
    NanoisaErr error;NvmModule *module=nanoisa_assemble_file(argv[1],&error);
    if(!module){fprintf(stderr,"%s\n",error.message);return 2;}
    char *before=nanoisa_print(module);if(!before)return 2;
    NvmVerifyResult prior=nvm_verify_profile(module,NVM_PROFILE_CLOSED_MANAGED_STRINGS);
#ifdef NMA_TESTING
    if(argc==3)nvm_array_analysis_fail_after(strtoull(argv[2],NULL,10));
#endif
    NvmArrayEligibilityReport sentinel={0},*report=&sentinel;
    NvmArrayEligibilityResult result=nvm_analyze_managed_arrays(module,&report);
    if(result.status!=NVM_ARRAY_ELIGIBLE && report!=&sentinel)return 3;
    if(result.status==NVM_ARRAY_ELIGIBLE && report==&sentinel)return 4;
    char *after=nanoisa_print(module);if(!after || strcmp(before,after))return 5;
    NvmVerifyResult current=nvm_verify_profile(module,NVM_PROFILE_CLOSED_MANAGED_STRINGS);
    if(current.ok!=prior.ok || strcmp(current.error_msg,prior.error_msg))return 6;
    printf("%d %u %u %u %u %u\n",result.status,result.function,result.pc,
           report->origin_count,report->checked_writes,report->runtime_tag_checks);
    puts(result.message);
    if(result.status==NVM_ARRAY_ELIGIBLE) {
        for(uint32_t i=0;i<report->origin_count;i++) {
            const NvmArrayOrigin *o=&report->origins[i];
            printf("%u %u %u %u %u\n",o->function,o->pc,o->declared_tag,o->packed,o->child_tags);
        }
        nvm_array_eligibility_free(report);
    }
    free(before);free(after);nvm_module_free(module);return 0;
}
