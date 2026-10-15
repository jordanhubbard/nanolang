/* I validate passive bytes against the selected immutable member declarations. */
static bool fr_ws_fields(const NvmWebSocketRuntime *c,NvmWebSocketFlowDeclaration type,
                        const int64_t *values,uint8_t fields) {
    if(type.tag==TAG_VOID)return !fields;
    if(type.tag==TAG_INT)return fields==1;
    if(type.tag==TAG_BOOL)return fields==1 && (values[0]==0 || values[0]==1);
    if(type.tag==TAG_STRING)return fields==1 && fr_string_identity(c,values[0]);
    if(type.category!=NVM_WEBSOCKET_CATEGORY_RECORD || !RUNTIME_TYPE_VALID(c,type.catalog_ordinal) ||
       !fr_same(type,c->types[type.catalog_ordinal]))return false;
    const NlServicePlanType *record=nl_websocket_catalog_type(type.catalog_ordinal);
    if(!record || fields!=record->member_count || fields>NVM_WEBSOCKET_RUNTIME_FIELDS)return false;
    for(uint8_t i=0;i<fields;i++) {
        NvmWebSocketFlowDeclaration member;
        if(!fr_member(c,type.catalog_ordinal,record->members[i].type_id,&member) ||
           (member.tag!=TAG_INT && member.tag!=TAG_BOOL && member.tag!=TAG_STRING) ||
           !fr_ws_fields(c,member,values+i,1))return false;
    }
    return true;
}
static bool fr_ws_payload(const NvmWebSocketRuntime *c,const NvmWebSocketRuntimeView *p) {
    if(!p->initialized || p->owning || p->formal || p->type.mode)return false;
    if(p->type.category!=NVM_WEBSOCKET_CATEGORY_SCALAR_RESULT)
        return p->arm==NVM_WEBSOCKET_FLOW_ARM_NONE && fr_ws_fields(c,p->type,p->values,p->fields);
    const NlServicePlanType *type=nl_websocket_catalog_type(p->type.catalog_ordinal);
    if(!type || !fr_same(p->type,c->types[p->type.catalog_ordinal]) ||
       (p->arm!=NVM_WEBSOCKET_FLOW_ARM_OK && p->arm!=NVM_WEBSOCKET_FLOW_ARM_ERROR))return false;
    NvmWebSocketFlowDeclaration member;
    return fr_member(c,p->type.catalog_ordinal,type->members[p->arm==NVM_WEBSOCKET_FLOW_ARM_ERROR].type_id,&member) &&
        fr_ws_fields(c,member,p->values,p->fields);
}
NvmWebSocketRuntimeStatus nvm_websocket_runtime_service(NvmWebSocketRuntime *c,uint32_t import,
    uint32_t reference,const uint32_t *inputs,size_t count,uint32_t output) {
    NvmWebSocketRuntimeStatus status=fr_effect(c);if(status!=NVM_WEBSOCKET_RUNTIME_OK)return status;
    uint32_t ordinal;
    if(!fr_empty(c,output) || !fr_import(c,import,&ordinal) || ordinal>=NVM_WEBSOCKET_NOMINAL_METHODS)
        return fr_error(c,NVM_WEBSOCKET_RUNTIME_INVALID,NL_WS_VALUE_ARGUMENT);
    if(c->kind!=FR_ACYCLIC) {
        const FileRuntimeFrame *frame=&c->frames[c->frame_count-1];
        NvmWebSocketCodeInstruction instruction;NvmWebSocketBodyInstruction fact;
        if(!fr_instruction(c,frame->function,(uint16_t)frame->instruction,frame->variant,&instruction,&fact) ||
           instruction.decoded.opcode!=OP_FILE_SERVICE || instruction.decoded.operands[0].u32!=import ||
           instruction.catalog_ordinal!=ordinal)
            return fr_error(c,NVM_WEBSOCKET_RUNTIME_TYPE,NL_WS_VALUE_TYPE);
    }
    const NlServicePlanMethod *method=nl_websocket_catalog_method(ordinal);
    bool borrowed=method->input_mode==NL_SERVICE_INPUT_EXCLUSIVE;
    bool consumed=method->input_mode==NL_SERVICE_INPUT_CONSUME;
    size_t offset=borrowed?1:0;
    if(count!=method->param_count-1-offset || (count && !inputs))
        return fr_error(c,NVM_WEBSOCKET_RUNTIME_INVALID,NL_WS_VALUE_ARGUMENT);
    if(borrowed) {
        if(!fr_reference(c,reference) || c->references[reference].shared)
            return fr_error(c,NVM_WEBSOCKET_RUNTIME_BORROWED,NL_WS_VALUE_BORROWED);
        if(!fr_same(c->values[c->references[reference].owner_root].view.type,c->types[0]))
            return fr_error(c,NVM_WEBSOCKET_RUNTIME_TYPE,NL_WS_VALUE_TYPE);
    } else if(reference!=NVM_WEBSOCKET_RUNTIME_NO_SLOT)
        return fr_error(c,NVM_WEBSOCKET_RUNTIME_BORROWED,NL_WS_VALUE_BORROWED);
    for(size_t i=0;i<count;i++) {
        for(size_t j=0;j<i;j++)if(inputs[i]==inputs[j])return fr_error(c,NVM_WEBSOCKET_RUNTIME_INVALID,NL_WS_VALUE_ARGUMENT);
        NvmWebSocketFlowDeclaration expected;
        if(!fr_live(c,inputs[i]) || !fr_member(c,0,method->params[i+offset].type_id,&expected))
            return fr_error(c,NVM_WEBSOCKET_RUNTIME_TYPE,NL_WS_VALUE_TYPE);
        const NvmWebSocketRuntimeView *value=&c->values[inputs[i]].view;
        bool owner=consumed && !i;
        if(!fr_same(value->type,expected) || value->formal || value->owning!=owner ||
           (!owner && !fr_ws_payload(c,value)))
            return fr_error(c,NVM_WEBSOCKET_RUNTIME_TYPE,NL_WS_VALUE_TYPE);
    }
    int64_t timeout=c->values[inputs[count-1]].view.values[0];
    NlWsValueStatus core;NlWsTransportResult actual={0};NlWsMessage message={0};
    c->busy=true;
    switch(ordinal) {
    case 0: {
        const ServiceRuntimeString *url=fr_string(c,inputs[0]);
        core=nl_ws_values_connect(c->files,url->bytes,url->length,timeout,&c->values[output].owner);break;
    }
    case 1: {
        const NvmWebSocketRuntimeView *value=&c->values[inputs[0]].view;
        const ServiceRuntimeString *data=fr_string_identity(c,value->values[1]);
        core=nl_ws_value_send(c->files,&c->references[reference].borrow,value->values[0]!=0,
            data->bytes,data->length,timeout,&actual);break;
    }
    case 2:core=nl_ws_value_receive(c->files,&c->references[reference].borrow,timeout,&message,&actual);break;
    default:core=nl_ws_value_close(c->files,&c->values[inputs[0]].owner,timeout,&actual);break;
    }
    c->busy=false;
    if(core!=NL_WS_VALUE_OK){nl_ws_message_free(&message);return fr_core(c,core);}
    for(size_t i=0;i<count;i++)fr_clear(c,inputs[i]);
    if(!ordinal){c->values[output].view=fr_view(c->types[3]);return NVM_WEBSOCKET_RUNTIME_OK;}
    NvmWebSocketRuntimeView result=fr_view(c->types[ordinal+3]);
    result.arm=actual.status==NL_WS_TRANSPORT_OK?NVM_WEBSOCKET_FLOW_ARM_OK:NVM_WEBSOCKET_FLOW_ARM_ERROR;
    if(result.arm==NVM_WEBSOCKET_FLOW_ARM_ERROR) {
        result.fields=NVM_WEBSOCKET_RUNTIME_FIELDS;fr_error_payload(actual,result.values);
    } else if(ordinal==1) {
        if(actual.bytes>INT64_MAX)return fr_error(c,NVM_WEBSOCKET_RUNTIME_LIMIT,NL_WS_VALUE_LIMIT);
        result.fields=1;result.values[0]=(int64_t)actual.bytes;
    } else if(ordinal==2) {
        status=nvm_websocket_runtime_string(c,output,message.bytes,message.length);
        bool binary=message.binary;nl_ws_message_free(&message);
        if(status!=NVM_WEBSOCKET_RUNTIME_OK)return status;
        result.fields=2;result.values[0]=binary;result.values[1]=c->values[output].view.values[0];
    }
    nl_ws_message_free(&message);
    c->values[output].view=result;return NVM_WEBSOCKET_RUNTIME_OK;
}
