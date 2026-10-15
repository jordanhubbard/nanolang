#define _POSIX_C_SOURCE 200809L
#include "nsi_socket_resolver_wire.h"
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc,char **argv) {
    if(argc!=3)return 2;
    char *end;errno=0;unsigned long port=strtoul(argv[2],&end,10);
    if(errno || !*argv[2] || *end || !port || port>65535)return 2;
    NlSocketService *service=NULL;
    NlSocketResult created=nl_socket_service_create(&service);
    if(created.status!=NL_SOCKET_OK)return 3;
    NlSocketResolution value={0};
    NlSocketResolveResult result=nl_socket_resolve_tcp(service,argv[1],strlen(argv[1]),(uint16_t)port,true,&value);
    if(nl_socket_service_destroy(service).status!=NL_SOCKET_OK)return 4;
    unsigned char wire[NL_RESOLVE_WIRE_SIZE];nl_resolve_encode(wire,result,&value);
    size_t sent=0;
    while(sent<sizeof wire) {
        ssize_t n=write(STDOUT_FILENO,wire+sent,sizeof wire-sent);
        if(n<0 && errno==EINTR)continue;
        if(n<=0)return 5;
        sent+=(size_t)n;
    }
    return 0;
}
