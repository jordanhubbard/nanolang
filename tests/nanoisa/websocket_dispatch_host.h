#ifndef TEST_WEBSOCKET_DISPATCH_HOST_H
#define TEST_WEBSOCKET_DISPATCH_HOST_H
#include <errno.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
static unsigned opens,closes;
static bool fail_close;
int websocket_dispatch_socket(int domain,int type,int protocol){int fd=socket(domain,type,protocol);if(fd>=0)opens++;return fd;}
int websocket_dispatch_close(int fd){int status=close(fd);closes++;if(!status && fail_close){fail_close=false;errno=EIO;return -1;}return status;}
static void report(NvmWebSocketIndirectExecutionReport r,NvmWebSocketRuntimeView out) {
    printf("{\"status\":%u,\"value\":%lld,\"fields\":%u,\"fuel\":%u,\"steps\":%llu,\"opens\":%u,\"closes\":%u,\"cleanup\":%llu,\"acquired\":%u}\n",
        r.runtime.status,(long long)out.values[0],out.fields,r.fuel_exhausted,
        (unsigned long long)r.instructions_started,opens,closes,
        (unsigned long long)r.runtime.cleanup.cleanup_failures,r.runtime.acquired);
}
#endif
