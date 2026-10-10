/* I check copied policy, revocation and serialization through installed headers. */
#include <nanolang/websocket/nanoisa/websocket_indirect_native_public.h>
#include <assert.h>
#include <pthread.h>
#include <string.h>

/* My instrumented provider must never attempt host I/O in these grant tests. */
int websocket_dispatch_socket(int domain,int type,int protocol) {
    (void)domain;(void)type;(void)protocol;assert(0);return -1;
}
int websocket_dispatch_close(int fd) {(void)fd;assert(0);return -1;}

static void *contender(void *unused) {
    (void)unused;
    void *poison=(void *)(uintptr_t)1;
    assert(nvm_websocket_host_grant_create(poison,poison)==NVM_WEBSOCKET_HOST_BUSY);
    assert(nvm_websocket_host_grant_revoke(poison)==NVM_WEBSOCKET_HOST_BUSY);
    assert(nvm_websocket_host_grant_destroy(poison)==NVM_WEBSOCKET_HOST_BUSY);
    assert(nvm_websocket_execute_indirect_bytes(poison,poison,1,poison,poison).runtime.status==NVM_WEBSOCKET_RUNTIME_BUSY);
    assert(nvm2c_emit_websocket_indirect_bytes(poison,1,poison,poison,poison,1)==NVM_WEBSOCKET_RUNTIME_BUSY);
    return NULL;
}
int main(void) {
    NvmWebSocketHostGrant *grant=NULL,*sentinel=(void *)(uintptr_t)1;
    char helper[]="/missing/resolver";
    NvmWebSocketHostPolicy policy={NVM_WEBSOCKET_HOST_POLICY_REVISION,true,true,2000,helper};
    assert(nvm_websocket_host_grant_create(&policy,&grant)==NVM_WEBSOCKET_HOST_OK);
    helper[1]='X';policy.allow_connections=false;policy.allow_lookup=false;policy.max_timeout_ms=1;
    assert(nvm_websocket_host_enter(grant,1,3)==NVM_WEBSOCKET_HOST_OK);
    NlWsTransportPolicy copied;
    assert(nvm_websocket_host_policy(grant,&copied)==NVM_WEBSOCKET_HOST_OK);
    assert(copied.allow_network && copied.allow_lookup && copied.max_timeout_ms==2000);
    assert(!strcmp(copied.resolver_helper,"/missing/resolver"));
    contender(NULL);
    pthread_t thread;assert(!pthread_create(&thread,NULL,contender,NULL));assert(!pthread_join(thread,NULL));
    nvm_websocket_host_leave();
    assert(nvm_websocket_host_enter(grant,2,3)==NVM_WEBSOCKET_HOST_UNRESOLVED);
    assert(nvm_websocket_host_enter(grant,1,2)==NVM_WEBSOCKET_HOST_UNRESOLVED);
    policy.revision=2;
    assert(nvm_websocket_host_grant_create(&policy,&sentinel)==NVM_WEBSOCKET_HOST_INVALID);
    policy.revision=1;policy.max_timeout_ms=60001;
    assert(nvm_websocket_host_grant_create(&policy,&sentinel)==NVM_WEBSOCKET_HOST_INVALID);
    policy.max_timeout_ms=2000;policy.allow_lookup=true;policy.resolver_helper=NULL;
    assert(nvm_websocket_host_grant_create(&policy,&sentinel)==NVM_WEBSOCKET_HOST_INVALID);
    policy.resolver_helper="relative/path";
    assert(nvm_websocket_host_grant_create(&policy,&sentinel)==NVM_WEBSOCKET_HOST_INVALID);
    assert(sentinel==(void *)(uintptr_t)1);
    NvmWebSocketIndirectOptions options={1,1000};NvmWebSocketScalar out={TAG_INT,12345};
    assert(nvm_websocket_execute_indirect_bytes(grant,NULL,0,&options,&out).runtime.status!=NVM_WEBSOCKET_RUNTIME_OK);
    assert(out.tag==TAG_INT && out.value==12345);
    assert(nvm_websocket_host_grant_revoke(grant)==NVM_WEBSOCKET_HOST_OK);
    assert(nvm_websocket_host_grant_revoke(grant)==NVM_WEBSOCKET_HOST_OK);
    assert(nvm_websocket_execute_indirect_bytes(grant,NULL,0,&options,&out).runtime.status==NVM_WEBSOCKET_RUNTIME_STATE);
    assert(out.tag==TAG_INT && out.value==12345);
    assert(nvm_websocket_host_grant_destroy(&grant)==NVM_WEBSOCKET_HOST_OK && !grant);
    assert(nvm_websocket_host_grant_destroy(&grant)==NVM_WEBSOCKET_HOST_OK);
    policy.allow_lookup=false;policy.resolver_helper=NULL;policy.max_timeout_ms=0;
    assert(nvm_websocket_host_grant_create(&policy,&grant)==NVM_WEBSOCKET_HOST_OK);
    assert(nvm_websocket_host_grant_destroy(&grant)==NVM_WEBSOCKET_HOST_OK);
    return 0;
}
