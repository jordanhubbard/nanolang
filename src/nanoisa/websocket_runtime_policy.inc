NvmWebSocketRuntimeStatus nvm_websocket_runtime_policy(NvmWebSocketRuntime *c,const NlWsTransportPolicy *policy) {
    if(!c)return NVM_WEBSOCKET_RUNTIME_INVALID;
    if(c->busy)return NVM_WEBSOCKET_RUNTIME_BUSY;
    if(c->phase!=FR_READY || c->policy_set)return NVM_WEBSOCKET_RUNTIME_STATE;
    if(!policy || policy->max_timeout_ms>60000)return NVM_WEBSOCKET_RUNTIME_INVALID;
    size_t length=policy->resolver_helper?strnlen(policy->resolver_helper,sizeof c->resolver_helper):0;
    if(length>=sizeof c->resolver_helper || (policy->resolver_helper && (!length || policy->resolver_helper[0]!='/')))
        return NVM_WEBSOCKET_RUNTIME_INVALID;
    c->policy=*policy;
    if(policy->resolver_helper) {
        memcpy(c->resolver_helper,policy->resolver_helper,length+1);c->policy.resolver_helper=c->resolver_helper;
    }
    c->policy_set=true;return NVM_WEBSOCKET_RUNTIME_OK;
}
