/* I compare graph facts with unchanged leaf/profile decisions and module bytes. */
#include "../../src/nanoisa/managed_array_shapes.h"
#include "../../src/nanoisa/verifier.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv) {
    if(argc<2 || argc>3)return 2;
    NvmArrayGraphEligibilityReport sentinel={0},*report=&sentinel;
    if(nvm_analyze_managed_array_graphs(NULL,&report).status!=NVM_ARRAY_INVALID || report!=&sentinel)return 3;
    NanoisaErr error;NvmModule *module=nanoisa_assemble_file(argv[1],&error);
    if(!module){fprintf(stderr,"%s\n",error.message);return 4;}
    char *before=nanoisa_print(module);
    NvmArrayEligibilityReport *leaf_before=NULL,*leaf_after=NULL;
    NvmArrayEligibilityResult prior=nvm_analyze_managed_arrays(module,&leaf_before);
    NvmVerifyResult profile_before=nvm_verify_profile(module,NVM_PROFILE_CLOSED_MANAGED_STRINGS);
#ifdef NMA_TESTING
    if(argc==3)nvm_array_analysis_fail_after(strtoull(argv[2],NULL,10));
#endif
    NvmArrayEligibilityResult result=nvm_analyze_managed_array_graphs(module,&report);
    if(result.status!=NVM_ARRAY_ELIGIBLE && report!=&sentinel)return 5;
    if(result.status==NVM_ARRAY_ELIGIBLE && report==&sentinel)return 6;
#ifdef NMA_TESTING
    nvm_array_analysis_fail_after(UINT64_MAX);
#endif
    NvmArrayEligibilityResult after=nvm_analyze_managed_arrays(module,&leaf_after);
    NvmVerifyResult profile_after=nvm_verify_profile(module,NVM_PROFILE_CLOSED_MANAGED_STRINGS);
    if(prior.status!=after.status || strcmp(prior.message,after.message) ||
       profile_before.ok!=profile_after.ok || strcmp(profile_before.error_msg,profile_after.error_msg))return 7;
    if(leaf_before && (!leaf_after || memcmp(leaf_before,leaf_after,sizeof *leaf_before)))return 8;
    int selected=99;
#ifdef NMA_TESTING
    nvm_array_analysis_fail_after(0);
    if(nvm_select_managed_array_mode(module,&selected).status!=NVM_ARRAY_MEMORY || selected!=99)return 11;
    nvm_array_analysis_fail_after(UINT64_MAX);
#endif
    NvmArrayEligibilityResult selection=nvm_select_managed_array_mode(module,&selected);
    if(profile_before.ok) {
        if(selection.status!=NVM_ARRAY_ELIGIBLE || selected!=(prior.status!=NVM_ARRAY_ELIGIBLE))return 12;
    } else if(selection.status==NVM_ARRAY_ELIGIBLE || selected!=99)return 13;
    if(nvm_select_managed_array_mode(module,NULL).status!=NVM_ARRAY_INVALID)return 14;
    char *printed=nanoisa_print(module);
    if(!before || !printed || strcmp(before,printed))return 9;
    printf("%d %d %d %u %u\n",result.status,prior.status,profile_before.ok,
        report->arrays.origin_count,report->arrays.checked_writes);
    if(result.status==NVM_ARRAY_ELIGIBLE) {
        NvmArrayGraphEligibilityReport *again=NULL;
        if(nvm_analyze_managed_array_graphs(module,&again).status!=NVM_ARRAY_ELIGIBLE ||
           memcmp(report,again,sizeof *report))return 10;
        for(uint32_t i=0;i<report->arrays.origin_count;i++) {
            NvmArrayOrigin *o=&report->arrays.origins[i];
            printf("%u %u %u %u %u %llu %u\n",o->function,o->pc,o->declared_tag,o->packed,o->child_tags,
                (unsigned long long)report->child_origins[i],report->child_unknown[i]);
        }
        nvm_array_graph_eligibility_free(again);nvm_array_graph_eligibility_free(report);
    }
    nvm_array_eligibility_free(leaf_before);nvm_array_eligibility_free(leaf_after);
    free(before);free(printed);nvm_module_free(module);return 0;
}
