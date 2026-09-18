/* I test shared selection, owned publication and unchanged old profile boundaries. */
#include "../../src/nanoisa/managed_record_shapes.h"
#include "../../src/nanoisa/verifier.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    NvmManagedHeapPlan sentinel={0},*plan=&sentinel;
    if(nvm_select_managed_heap(NULL,0,&plan).status!=NVM_ARRAY_INVALID || plan!=&sentinel)return 3;
    NanoisaErr error;NvmModule *m=nanoisa_assemble_file(argv[1],&error);
    if(!m){fprintf(stderr,"%s\n",error.message);return 4;}
    char *before=nanoisa_print(m);if(!before)return 5;
    if(nvm_verify_profile(m,NVM_PROFILE_CLOSED_SCALAR).ok ||
       nvm_verify_profile(m,NVM_PROFILE_CLOSED_LITERAL_STRINGS).ok ||
       !nvm_verify_profile(m,NVM_PROFILE_CLOSED_MANAGED_STRINGS).ok)return 6;
    int old=71;
    if(nvm_select_managed_array_mode(m,&old).status!=NVM_ARRAY_UNRESOLVED || old!=71)return 7;
    unsigned failures=0;
    for(unsigned budget=0;budget<256;budget++) {
        nvm_array_analysis_fail_after(budget);plan=&sentinel;
        NvmArrayEligibilityResult r=nvm_select_managed_heap(m,0,&plan);
        nvm_array_analysis_fail_after(UINT64_MAX);
        if(r.status==NVM_ARRAY_ELIGIBLE) {
            if(plan==&sentinel || plan->mode!=NVM_MANAGED_RECORD || !plan->fields ||
               !plan->records || plan->records->authority!=NVM_RECORD_AUTHORITY_ORDINARY)return 8;
            nvm_managed_heap_plan_free(plan);break;
        }
        if(r.status!=NVM_ARRAY_MEMORY || plan!=&sentinel)return 9;
        failures++;
    }
    if(!failures || failures==256)return 10;
    char *after=nanoisa_print(m);
    if(!after || strcmp(before,after))return 11;
    /* Valid absent authority is unresolved and leaves the output untouched. */
    free(m->ownership_data);m->ownership_data=NULL;m->ownership_size=0;plan=&sentinel;
    if(nvm_select_managed_heap(m,0,&plan).status!=NVM_ARRAY_UNRESOLVED || plan!=&sentinel)return 12;
    free(before);free(after);nvm_module_free(m);return 0;
}
