#include "record_map_publication.h"
NvmArrayEligibilityResult nvm_record_map_publication(const NvmModule *source) {
    NvmArrayEligibilityResult result={NVM_ARRAY_UNRESOLVED,0,0,"I require established map publication eligibility."};
    if(!source){result.status=NVM_ARRAY_INVALID;return result;}
    NvmRecordMapRoute route=nvm_record_map_route(source);
    if(route!=NVM_RECORD_MAP_PRESENT) {
        if(route==NVM_RECORD_MAP_INVALID)result.status=NVM_ARRAY_INVALID;
        if(route==NVM_RECORD_MAP_LIMIT)result.status=NVM_ARRAY_LIMIT;
        return result;
    }
    NvmRecordMapExecutionPlan *plan=NULL;NvmRecordMapImage *image=NULL;
    result=nvm_record_map_admit(source,&plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE)return result;
    result=nvm_record_map_image_copy(plan,&image);
    nvm_record_map_execution_free(plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE)return result;
    NvmRecordMapTargetResult target=nvm_record_map_native_target(image,NVM_RECORD_ARRAY_EXECUTION_STEPS);
    nvm_record_map_image_free(image);
    return target.eligibility;
}
