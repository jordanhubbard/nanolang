/* I inspect non-admitting field facts and failure atomicity on ordinary modules. */
#include "../../src/nanoisa/managed_record_shapes.h"
#include "../../src/nanoisa/verifier.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv) {
    if(argc<2 || argc>3)return 2;
    NvmRecordEligibilityReport sentinel={0},*report=&sentinel;
    if(nvm_analyze_managed_records(NULL,&report).status!=NVM_ARRAY_INVALID || report!=&sentinel)return 3;
    NanoisaErr error;NvmModule *m=nanoisa_assemble_file(argv[1],&error);
    if(!m){fprintf(stderr,"%s\n",error.message);return 4;}
    char *before=nanoisa_print(m);if(!before)return 5;
    NvmVerifyResult prior=nvm_verify_profile(m,NVM_PROFILE_CLOSED_MANAGED_STRINGS);
    int selected=77;NvmArrayEligibilityResult old=nvm_select_managed_array_mode(m,&selected);
    int selected_before=selected;
#ifdef NMA_TESTING
    if(argc==3)nvm_array_analysis_fail_after(strtoull(argv[2],NULL,10));
#endif
    NvmArrayEligibilityResult result=nvm_analyze_managed_records(m,&report);
#ifdef NMA_TESTING
    nvm_array_analysis_fail_after(UINT64_MAX);
#endif
    if(result.status!=NVM_ARRAY_ELIGIBLE && report!=&sentinel)return 6;
    if(result.status==NVM_ARRAY_ELIGIBLE && report==&sentinel)return 7;
    char *after=nanoisa_print(m);if(!after || strcmp(before,after))return 8;
    NvmVerifyResult current=nvm_verify_profile(m,NVM_PROFILE_CLOSED_MANAGED_STRINGS);
    selected=77;NvmArrayEligibilityResult now=nvm_select_managed_array_mode(m,&selected);
    if(prior.ok!=current.ok || strcmp(prior.error_msg,current.error_msg) ||
       old.status!=now.status || selected!=selected_before || strcmp(old.message,now.message))return 9;
    free(before);free(after);nvm_module_free(m); /* The report owns all facts. */
    printf("%d %u %u %u %u %u %u\n",result.status,report->origin_count,report->field_value_count,
           report->checked_field_writes,report->checked_array_writes,report->runtime_tag_checks,old.status);
    puts(result.message);
    if(result.status==NVM_ARRAY_ELIGIBLE) {
        for(uint32_t i=0;i<report->origin_count;i++) {
            NvmRecordHeapOrigin *o=&report->origins[i];
            printf("O %u %u %u %u %u %u %u %u %u %llu\n",o->kind,o->function,o->pc,
                   o->record_ordinal,o->layout_index,o->field_start,o->field_count,
                   o->declared_tag,o->children.tags,(unsigned long long)o->children.origins);
        }
        for(uint32_t i=0;i<report->field_value_count;i++) {
            NvmRecordValueOrigins *v=&report->fields[i];
            printf("F %u %u %llu\n",v->tags,v->unknown,(unsigned long long)v->origins);
        }
        nvm_record_eligibility_free(report);
    }
    return 0;
}
