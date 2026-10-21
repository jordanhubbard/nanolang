#include "record_map_admission.h"
NvmArrayEligibilityResult nvm_record_map_admit(const NvmModule *m,NvmRecordMapExecutionPlan **out) {
    NvmArrayEligibilityResult r={NVM_ARRAY_UNRESOLVED,0,0,"I require complete ordinary map-field authority."};
    if(!out){r.status=NVM_ARRAY_INVALID;return r;}
    NvmRecordMapRoute route=nvm_record_map_route(m);
    if(route!=NVM_RECORD_MAP_PRESENT) {
        if(route==NVM_RECORD_MAP_INVALID)r.status=NVM_ARRAY_INVALID;
        if(route==NVM_RECORD_MAP_LIMIT)r.status=NVM_ARRAY_LIMIT;
        return r;
    }
    NvmRecordMapExecutionPlan *p=NULL;r=nvm_prepare_record_map_execution(m,&p);
    if(r.status!=NVM_ARRAY_ELIGIBLE)return r;
    NvmRecordMapExecutionCounts c;NvmRecordArrayExecutionFunction f;
    bool valid=nvm_record_map_execution_counts(p,&c) && c.carrier.functions<=256 &&
        nvm_record_map_execution_function(p,c.carrier.entry,&f) && !f.signature.arity &&
        !f.signature.upvalue_count && f.signature.result_count==1 && f.signature.result_tag==TAG_INT;
    if(valid && c.carrier.initializer!=UINT32_MAX)
        valid=nvm_record_map_execution_function(p,c.carrier.initializer,&f) && !f.signature.arity &&
            !f.signature.upvalue_count && f.signature.result_count<=1 &&
            ((!f.signature.result_count && f.signature.result_tag==TAG_VOID) ||
             (f.signature.result_count==1 && f.signature.result_tag==TAG_INT));
    if(!valid) {
        nvm_record_map_execution_free(p);r.status=NVM_ARRAY_UNRESOLVED;
        snprintf(r.message,sizeof r.message,"I require a zero-argument INT map entry and VOID/INT initializer.");return r;
    }
    *out=p;return r;
}
