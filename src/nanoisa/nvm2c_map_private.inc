/* Public and private native consumers share the exact copied-image target. */
#include "record_map_native_target_internal.h"
static int nvm2c_map_target_error(char *error,size_t size,const char *message) {
    if(error && size)snprintf(error,size,"%s",message);
    return 0;
}
static int nvm2c_map_target(const NvmRecordMapImage *image,char *error,size_t size,uint8_t *value_out) {
    NvmRecordMapTargetResult target=nvm_record_map_native_target_pair(image,UINT64_MAX,value_out);
    if(target.eligibility.status==NVM_ARRAY_ELIGIBLE)return 1;
    if(target.eligibility.message[0])
        nvm2c_map_target_error(error,size,target.eligibility.message);
    return 0;
}
/* Internal seam for source-destruction controls, never an exported trust API. */
static char *nvm2c_map_image_emit(const NvmRecordMapImage *image,char *error,size_t size) {
    uint8_t value_tag=TAG_VOID;
    if(!nvm2c_map_target(image,error,size,&value_tag)) {
        if(error && size && !error[0])snprintf(error,size,"I require matching private native map recipes and target bounds.");
        return NULL;
    }
    return nvm2c_emit_core(nvm_record_map_image_borrow(image),error,size,nvm_record_map_image_plan(image),value_tag);
}
static char *nvm2c_emit_record_map_source(const NvmModule *source,char *error,size_t size,bool public_root) {
    if(error && size)error[0]=0;
    NvmRecordMapExecutionPlan *plan=NULL;NvmRecordMapImage *image=NULL;
    NvmArrayEligibilityResult result=public_root?nvm_record_map_admit(source,&plan):
        nvm_prepare_record_map_execution(source,&plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE) {
        nvm2c_map_target_error(error,size,result.message);return NULL;
    }
    result=nvm_record_map_image_copy(plan,&image);
    nvm_record_map_execution_free(plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE) {
        nvm2c_map_target_error(error,size,result.message);return NULL;
    }
    char *output=nvm2c_map_image_emit(image,error,size);
    nvm_record_map_image_free(image);
    return output;
}

#ifdef NANO_RECORD_MAP_PRIVATE_NATIVE
char *nvm2c_emit_record_map_private(const NvmModule *source,char *error,size_t size) {
    return nvm2c_emit_record_map_source(source,error,size,false);
}
#endif
