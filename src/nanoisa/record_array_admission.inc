#include "record_array_admission.h"
#include "service_bindings_module.h"
/* I leave malformed noncandidates to the unchanged refusing validator. */
bool nvm_record_array_candidate(const NvmModule *m) {
    if(!m || !m->ownership_data || !m->layout_data)return false;
    NvmV2Cursor c;uint32_t version,count,layouts;const uint8_t *flags;
    nvm_v2_cursor_init(&c,m->ownership_data,m->ownership_size);
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK || version!=NVM_OWNERSHIP_EXTENSION_VERSION ||
       nvm_v2_u32(&c,&count)!=NVM_V2_OK || count>NVM_RECORD_PLAN_MAX_LAYOUTS ||
       nvm_v2_take(&c,count,&flags)!=NVM_V2_OK)return false;
    for(uint32_t i=0;i<count;i++)if(flags[i]&NVM_LAYOUT_RESOURCE)return false;
    nvm_v2_cursor_init(&c,m->layout_data,m->layout_size);
    if(nvm_v2_u32(&c,&layouts)!=NVM_V2_OK || layouts!=count)return false;
    uint32_t total=0;
    for(uint32_t i=0;i<count;i++) {
        uint8_t kind,reserved;uint16_t fields;uint32_t name;const uint8_t *field;
        if(nvm_v2_u8(&c,&kind)!=NVM_V2_OK || nvm_v2_u8(&c,&reserved)!=NVM_V2_OK ||
           nvm_v2_u16(&c,&fields)!=NVM_V2_OK || nvm_v2_u32(&c,&name)!=NVM_V2_OK ||
           fields>NVM_RECORD_PLAN_MAX_FIELDS-total)return false;
        total+=fields;
        for(uint16_t j=0;j<fields;j++) {
            if(nvm_v2_take(&c,12,&field)!=NVM_V2_OK)return false;
            if(field[0]==TAG_ARRAY)return !nvm_service_execution_pending(m);
        }
    }
    return false;
}
NvmArrayEligibilityResult nvm_record_array_admit(const NvmModule *m,NvmRecordArrayExecutionPlan **out) {
    NvmArrayEligibilityResult result={NVM_ARRAY_UNRESOLVED,0,0,"I require a complete ordinary array-field candidate."};
    if(!out){result.status=NVM_ARRAY_INVALID;return result;}
    if(!nvm_record_array_candidate(m))return result;
    NvmRecordArrayExecutionPlan *plan=NULL;
    result=nvm_prepare_record_array_execution(m,&plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE)return result;
    NvmRecordArrayExecutionCounts counts;NvmRecordArrayExecutionFunction entry;
    if(!nvm_record_array_execution_counts(plan,&counts) ||
       counts.functions>NVM_RECORD_ARRAY_ADMISSION_FUNCTIONS ||
       !nvm_record_array_execution_function(plan,counts.entry,&entry) ||
       entry.signature.arity || entry.signature.result_count!=1 || entry.signature.result_tag!=TAG_INT) {
        nvm_record_array_execution_free(plan);result.status=NVM_ARRAY_UNRESOLVED;
        snprintf(result.message,sizeof result.message,"I require a standalone integer-result record-array entry.");return result;
    }
    for(uint32_t i=0;i<counts.instructions;i++) {
        NvmRecordArrayExecutionInstruction instruction;
        if(!nvm_record_array_execution_instruction(plan,i,&instruction) ||
           instruction.opcode==OP_STRUCT_LITERAL || instruction.opcode==OP_STRUCT_GET || instruction.opcode==OP_STRUCT_SET || instruction.opcode==OP_AGG_SET) {
            nvm_record_array_execution_free(plan);result.status=NVM_ARRAY_UNRESOLVED;
            snprintf(result.message,sizeof result.message,"I require aggregate record opcodes shared by both public runtimes.");return result;
        }
    }
    for(uint32_t i=0;i<counts.records;i++) {
        NvmRecordArrayExecutionDescriptor descriptor;
        if(!nvm_record_array_execution_descriptor(plan,i,&descriptor) || descriptor.ordinal!=descriptor.layout) {
            nvm_record_array_execution_free(plan);result.status=NVM_ARRAY_UNRESOLVED;
            snprintf(result.message,sizeof result.message,"I require an identity record-layout projection for public execution.");return result;
        }
    }
    if(counts.initializer!=UINT32_MAX) {
        NvmRecordArrayExecutionFunction initializer;
        if(!nvm_record_array_execution_function(plan,counts.initializer,&initializer) ||
           initializer.signature.arity || (initializer.signature.result_count && initializer.signature.result_tag!=TAG_INT)) {
            nvm_record_array_execution_free(plan);result.status=NVM_ARRAY_UNRESOLVED;
            snprintf(result.message,sizeof result.message,"I require a scalar initializer for record-array execution.");return result;
        }
    }
    *out=plan;return result;
}
