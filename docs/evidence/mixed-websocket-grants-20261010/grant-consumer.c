#include <nanolang/services/nanoisa/services_host_grant.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1; } } while(0)
int main(void) {
    char path[]="/resolver/copied";
    NvmServicesHostConfig configs[4]={
        {.revision=NVM_SERVICES_HOST_POLICY_REVISION,.catalog=NVM_SERVICES_HOST_FILE,.allowed=true},
        {.revision=NVM_SERVICES_HOST_POLICY_REVISION,.catalog=NVM_SERVICES_HOST_TCP,.allowed=true},
        {.revision=NVM_SERVICES_HOST_POLICY_REVISION,.catalog=NVM_SERVICES_HOST_WEBSOCKET,.allowed=true,
         .allow_lookup=true,.max_timeout_ms=17,.resolver_helper=path},
        {.revision=NVM_SERVICES_HOST_POLICY_REVISION,.catalog=NVM_SERVICES_HOST_WEBSOCKET,.allowed=true,
         .max_timeout_ms=29}};
    NvmServicesHostGrant *g=NULL;
    CHECK(nvm_services_host_grant_create_config(configs,4,&g)==NVM_SERVICES_HOST_OK && g);
    memset(configs,0,sizeof configs);memset(path,0,sizeof path);
    CHECK(nvm_services_host_grant_revoke_instance(g,2)==NVM_SERVICES_HOST_OK);
    CHECK(nvm_services_host_grant_revoke_instance(g,4)==NVM_SERVICES_HOST_INVALID);
    CHECK(nvm_services_host_grant_revoke(g)==NVM_SERVICES_HOST_OK);
    CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK && !g);
    NvmServicesHostPolicy legacy={NVM_SERVICES_HOST_WEBSOCKET,true};
    CHECK(nvm_services_host_grant_create(&legacy,1,&g)==NVM_SERVICES_HOST_INVALID && !g);
    puts("PASS relocated C99 mixed WebSocket grant consumer");return 0;
}
