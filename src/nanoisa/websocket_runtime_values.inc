/* I retain the exact nine-field error contract. String identities stay local. */
static bool fr_error_payload(NlWsTransportResult r,int64_t out[NVM_WEBSOCKET_RUNTIME_FIELDS]) {
    out[0]=r.status;out[1]=r.host_errno;out[2]=r.resolver_error;out[3]=r.supervisor_status;
    out[4]=r.close_code;out[5]=r.cleanup_errno;out[6]=r.cleanup_failed;
    out[7]=r.closure_unknown;out[8]=r.terminal;return true;
}
static bool fr_ws_payload(const NvmWebSocketRuntime *,const NvmWebSocketRuntimeView *);
