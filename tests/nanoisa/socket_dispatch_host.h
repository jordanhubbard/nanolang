#ifndef NANO_SOCKET_DISPATCH_TEST_HOST_H
#define NANO_SOCKET_DISPATCH_TEST_HOST_H
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
static unsigned dispatch_opens,dispatch_closes;
static bool dispatch_close_fault;
#ifndef SOCKET_DISPATCH_ON_OPEN
#define SOCKET_DISPATCH_ON_OPEN() ((void)0)
#endif
int dispatch_socket(int domain,int type,int protocol){SOCKET_DISPATCH_ON_OPEN();int fd=socket(domain,type,protocol);if(fd>=0)dispatch_opens++;return fd;}
int dispatch_close(int fd){dispatch_closes++;int result=close(fd);if(!result && dispatch_close_fault){dispatch_close_fault=false;errno=EIO;return -1;}return result;}
static void dispatch_report(NvmSocketIndirectExecutionReport r,NvmSocketRuntimeView out){
 printf("{\"status\":%u,\"acquired\":%u,\"steps\":%llu,\"fuel\":%u,\"cleanup\":%llu,\"unknown\":%u,\"value\":%lld,\"fields\":%u,\"opens\":%u,\"closes\":%u}\n",
 (unsigned)r.runtime.status,r.runtime.acquired,(unsigned long long)r.instructions_started,r.fuel_exhausted,
 (unsigned long long)r.runtime.cleanup.cleanup_failures,r.runtime.cleanup.first_cleanup.closure_unknown,
 (long long)out.values[0],out.fields,dispatch_opens,dispatch_closes);
}
#endif
