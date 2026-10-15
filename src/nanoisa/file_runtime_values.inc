static bool fr_error_payload(NlFileResult r,int64_t out[NVM_FILE_RUNTIME_FIELDS]) {
    if(r.bytes>(uint64_t)INT64_MAX)return false;
    out[0]=(int64_t)r.status;out[1]=r.host_errno;out[2]=r.cleanup_errno;out[3]=(int64_t)r.bytes;
    out[4]=r.eof;out[5]=r.consumed;out[6]=r.cleanup_failed;return true;
}
static bool fr_scalar_result(NvmFileRuntime *c,uint32_t ordinal,NlFileScalarResult actual,NvmFileRuntimeView *out) {
    if(!ordinal || ordinal>=NVM_SERVICE_BINDING_COUNT || actual.kind!=(NlFileScalarKind)(ordinal-1) ||
       actual.ok!=(actual.detail.status==NL_FILE_OK))return false;
    NvmFileRuntimeView result=fr_view(c->types[ordinal+3]);
    result.arm=actual.ok?NVM_FILE_FLOW_ARM_OK:NVM_FILE_FLOW_ARM_ERROR;
    if(!actual.ok) {
        result.fields=NVM_FILE_RUNTIME_FIELDS;
        if(!fr_error_payload(actual.detail,result.values))return false;
    } else if(ordinal==1) {
        if(actual.value<0 || actual.value>1)return false;
        result.fields=1;result.values[0]=actual.value;
    } else if(ordinal==3) {
        if(actual.value<0 || actual.value>255 || (actual.eof && actual.value))return false;
        result.fields=2;result.values[0]=actual.value;result.values[1]=actual.eof;
    } else if(actual.value || actual.eof)return false;
    *out=result;return true;
}
static NlFileValueStatus fr_acquire(NlFileValues *values,const NvmFileRuntimeView *input,NlFileValue *out) {
    (void)input;return nl_file_values_temp(values,out);
}
static bool fr_scalar_payload(const NvmFileRuntimeView *p) {
    if(p->arm==NVM_FILE_FLOW_ARM_ERROR) {
        if(p->fields!=NVM_FILE_RUNTIME_FIELDS)return false;
        for(uint8_t i=4;i<7;i++)if(p->values[i]!=0 && p->values[i]!=1)return false;
        return true;
    }
    uint32_t ordinal=p->type.catalog_ordinal;
    if(ordinal==4)return p->fields==1;
    if(ordinal==6)return p->fields==2 && (p->values[1]==0 || p->values[1]==1);
    return (ordinal==5 || ordinal==7) && !p->fields;
}
