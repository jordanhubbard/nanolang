#include "../src/nsi_socket_resolver.h"
#include <stdio.h>
#include <string.h>
int main(void) {
    char path[4096]="unchanged";
    if(!nl_socket_resolver_path(path,sizeof path))return strcmp(path,"unchanged")?2:1;
    char short_path[2]="x";
    if(nl_socket_resolver_path(short_path,sizeof short_path) || strcmp(short_path,"x"))return 3;
    NlSocketService *service=NULL;
    if(nl_socket_service_create(&service).status!=NL_SOCKET_OK)return 4;
    NlSocketResolution addresses;
    NlSocketLookupResult result=nl_socket_resolve_tcp_supervised(service,"localhost",9,80,true,path,3000,&addresses);
    if(nl_socket_service_destroy(service).status!=NL_SOCKET_OK)return 5;
    if(result.supervision!=NL_LOOKUP_COMPLETE || result.resolver.status!=NL_SOCKET_OK || !addresses.count)return 6;
    puts(path);return 0;
}
