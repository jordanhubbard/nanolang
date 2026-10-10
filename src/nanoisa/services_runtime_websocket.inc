/* I validate passive bytes against the selected immutable member declarations. */
static bool fr_mixed_ws_fields(const NvmServicesRuntime *c,NvmServicesFlowDeclaration type,
                        const int64_t *values,uint8_t fields) {
    if(type.tag==TAG_VOID)return !fields;
    if(type.tag==TAG_INT)return fields==1;
    if(type.tag==TAG_BOOL)return fields==1 && (values[0]==0 || values[0]==1);
    if(type.tag==TAG_STRING)return fields==1 && fr_string_identity(c,values[0]);
    if(type.category!=NVM_SERVICES_CATEGORY_RECORD || !RUNTIME_TYPE_VALID(c,type.catalog_ordinal) ||
       !fr_same(type,c->types[type.catalog_ordinal]))return false;
    const NlServicePlanType *record=RUNTIME_CATALOG_TYPE(c,type.catalog_ordinal);
    if(!record || fields!=record->member_count || fields>NVM_SERVICES_RUNTIME_FIELDS)return false;
    for(uint8_t i=0;i<fields;i++) {
        NvmServicesFlowDeclaration member;
        if(!fr_member(c,type.catalog_ordinal,record->members[i].type_id,&member) ||
           (member.tag!=TAG_INT && member.tag!=TAG_BOOL && member.tag!=TAG_STRING) ||
           !fr_mixed_ws_fields(c,member,values+i,1))return false;
    }
    return true;
}
static bool fr_mixed_ws_payload(const NvmServicesRuntime *c,const NvmServicesRuntimeView *p) {
    if(!p->initialized || p->owning || p->formal || p->type.mode)return false;
    if(p->type.category!=NVM_SERVICES_CATEGORY_SCALAR_RESULT)
        return p->arm==NVM_SERVICES_FLOW_ARM_NONE && fr_mixed_ws_fields(c,p->type,p->values,p->fields);
    const NlServicePlanType *type=RUNTIME_CATALOG_TYPE(c,p->type.catalog_ordinal);
    if(!type || !fr_same(p->type,c->types[p->type.catalog_ordinal]) ||
       (p->arm!=NVM_SERVICES_FLOW_ARM_OK && p->arm!=NVM_SERVICES_FLOW_ARM_ERROR))return false;
    NvmServicesFlowDeclaration member;
    return fr_member(c,p->type.catalog_ordinal,type->members[p->arm==NVM_SERVICES_FLOW_ARM_ERROR].type_id,&member) &&
        fr_mixed_ws_fields(c,member,p->values,p->fields);
}
NvmServicesRuntimeStatus nvm_services_runtime_service_args(NvmServicesRuntime *c,uint32_t import,
    uint32_t reference,const uint32_t *inputs,size_t count,uint32_t output) {
    NvmServicesRuntimeStatus status=fr_effect(c);if(status!=NVM_SERVICES_RUNTIME_OK)return status;
    uint32_t ordinal;
    if(!fr_empty(c,output) || !fr_import(c,import,&ordinal) || ordinal>=RUNTIME_IMPORTS(c))
        return fr_error(c,NVM_SERVICES_RUNTIME_INVALID,NL_SERVICES_VALUE_ARGUMENT);
    uint32_t instance=ordinal/5,local=ordinal%5,base=instance*9;
    if(c->catalogs[instance]!=NL_SERVICES_WEBSOCKET) {
        const NlServicePlanMethod *legacy=c->catalogs[instance]==NL_SERVICES_TCP?
            nl_socket_catalog_method(local):nl_file_catalog_method(local);
        if(!legacy || count!=legacy->param_count-1-(legacy->input_mode==NL_SERVICE_INPUT_EXCLUSIVE?1:0) ||
           count>1 || (count && !inputs))return fr_error(c,NVM_SERVICES_RUNTIME_INVALID,NL_SERVICES_VALUE_ARGUMENT);
        return nvm_services_runtime_service(c,import,reference,count?inputs[0]:NVM_SERVICES_RUNTIME_NO_SLOT,output);
    }
    if(c->kind!=FR_ACYCLIC) {
        const FileRuntimeFrame *frame=&c->frames[c->frame_count-1];
        NvmServicesCodeInstruction instruction;NvmServicesBodyInstruction fact;
        if(!fr_instruction(c,frame->function,(uint16_t)frame->instruction,frame->variant,&instruction,&fact) ||
           instruction.decoded.opcode!=OP_FILE_SERVICE || instruction.decoded.operands[0].u32!=import ||
           instruction.catalog_ordinal!=ordinal)
            return fr_error(c,NVM_SERVICES_RUNTIME_TYPE,NL_SERVICES_VALUE_TYPE);
    }
    const NlServicePlanMethod *method=nl_websocket_catalog_method(local);
    if(!method)return fr_error(c,NVM_SERVICES_RUNTIME_INVALID,NL_SERVICES_VALUE_ARGUMENT);
    bool borrowed=method->input_mode==NL_SERVICE_INPUT_EXCLUSIVE;
    bool consumed=method->input_mode==NL_SERVICE_INPUT_CONSUME;
    size_t offset=borrowed?1:0;
    if(count!=method->param_count-1-offset || (count && !inputs))
        return fr_error(c,NVM_SERVICES_RUNTIME_INVALID,NL_SERVICES_VALUE_ARGUMENT);
    if(borrowed) {
        if(!fr_reference(c,reference) || c->references[reference].shared)
            return fr_error(c,NVM_SERVICES_RUNTIME_BORROWED,NL_SERVICES_VALUE_BORROWED);
        if(!fr_same(c->values[c->references[reference].owner_root].view.type,c->types[base]))
            return fr_error(c,NVM_SERVICES_RUNTIME_TYPE,NL_SERVICES_VALUE_TYPE);
    } else if(reference!=NVM_SERVICES_RUNTIME_NO_SLOT)
        return fr_error(c,NVM_SERVICES_RUNTIME_BORROWED,NL_SERVICES_VALUE_BORROWED);
    for(size_t i=0;i<count;i++) {
        for(size_t j=0;j<i;j++)if(inputs[i]==inputs[j])return fr_error(c,NVM_SERVICES_RUNTIME_INVALID,NL_SERVICES_VALUE_ARGUMENT);
        NvmServicesFlowDeclaration expected;
        if(!fr_live(c,inputs[i]) || !fr_member(c,base,method->params[i+offset].type_id,&expected))
            return fr_error(c,NVM_SERVICES_RUNTIME_TYPE,NL_SERVICES_VALUE_TYPE);
        const NvmServicesRuntimeView *value=&c->values[inputs[i]].view;
        bool owner=consumed && !i;
        if(!fr_same(value->type,expected) || value->formal || value->owning!=owner ||
           (!owner && !fr_mixed_ws_payload(c,value)))
            return fr_error(c,NVM_SERVICES_RUNTIME_TYPE,NL_SERVICES_VALUE_TYPE);
    }
    int64_t timeout=c->values[inputs[count-1]].view.values[0];
    NlServicesValueStatus core;NlWsTransportResult actual={0};NlWsMessage message={0};
    c->busy=true;
    switch(local) {
    case 0: {
        const ServiceRuntimeString *url=fr_string(c,inputs[0]);
        core=nl_services_values_websocket_connect(c->files,instance,url->bytes,url->length,timeout,&c->values[output].owner);break;
    }
    case 1: {
        const NvmServicesRuntimeView *value=&c->values[inputs[0]].view;
        const ServiceRuntimeString *data=fr_string_identity(c,value->values[1]);
        core=nl_services_value_websocket_send(c->files,instance,&c->references[reference].borrow,value->values[0]!=0,
            data->bytes,data->length,timeout,&actual);break;
    }
    case 2:core=nl_services_value_websocket_receive(c->files,instance,&c->references[reference].borrow,timeout,&message,&actual);break;
    default:core=nl_services_value_websocket_close(c->files,instance,&c->values[inputs[0]].owner,timeout,&actual);break;
    }
    c->busy=false;
    if(core!=NL_SERVICES_VALUE_OK){nl_ws_message_free(&message);return fr_core(c,core);}
    for(size_t i=0;i<count;i++)fr_clear(c,inputs[i]);
    if(!local){c->values[output].view=fr_view(c->types[base+3]);return NVM_SERVICES_RUNTIME_OK;}
    NvmServicesRuntimeView result=fr_view(c->types[base+local+3]);
    result.arm=actual.status==NL_WS_TRANSPORT_OK?NVM_SERVICES_FLOW_ARM_OK:NVM_SERVICES_FLOW_ARM_ERROR;
    if(result.arm==NVM_SERVICES_FLOW_ARM_ERROR) {
        result.fields=9;NlServicesOpenView error={.catalog=NL_SERVICES_WEBSOCKET};
        error.error.websocket=actual;fr_error_payload(error,result.values);
    } else if(local==1) {
        if(actual.bytes>INT64_MAX)return fr_error(c,NVM_SERVICES_RUNTIME_LIMIT,NL_SERVICES_VALUE_LIMIT);
        result.fields=1;result.values[0]=(int64_t)actual.bytes;
    } else if(local==2) {
        status=nvm_services_runtime_string(c,output,message.bytes,message.length);
        bool binary=message.binary;nl_ws_message_free(&message);
        if(status!=NVM_SERVICES_RUNTIME_OK)return status;
        result.fields=2;result.values[0]=binary;result.values[1]=c->values[output].view.values[0];
    }
    nl_ws_message_free(&message);
    c->values[output].view=result;return NVM_SERVICES_RUNTIME_OK;
}

NvmServicesRuntimeStatus nvm_services_runtime_websocket_policy(NvmServicesRuntime *c,uint32_t instance,const NlWsTransportPolicy *policy) {
    if(!c)return NVM_SERVICES_RUNTIME_INVALID;
    if(c->busy)return NVM_SERVICES_RUNTIME_BUSY;
    if(c->phase!=FR_READY)return NVM_SERVICES_RUNTIME_STATE;
    if(instance>=c->instances || c->catalogs[instance]!=NL_SERVICES_WEBSOCKET)return NVM_SERVICES_RUNTIME_TYPE;
    if(c->policy_set[instance])return NVM_SERVICES_RUNTIME_STATE;
    if(!policy || policy->max_timeout_ms>60000 || (policy->allow_lookup && !policy->resolver_helper))return NVM_SERVICES_RUNTIME_INVALID;
    size_t length=policy->resolver_helper?strnlen(policy->resolver_helper,4096):0;
    if(length>=4096 || (policy->resolver_helper && (!length || policy->resolver_helper[0]!='/')))return NVM_SERVICES_RUNTIME_INVALID;
    c->configs[instance].websocket=*policy;
    if(policy->resolver_helper) {
        memcpy(c->resolver_helpers[instance],policy->resolver_helper,length+1);
        c->configs[instance].websocket.resolver_helper=c->resolver_helpers[instance];
    }
    c->policy_set[instance]=true;return NVM_SERVICES_RUNTIME_OK;
}
