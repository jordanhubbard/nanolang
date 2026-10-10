#define _POSIX_C_SOURCE 200809L
#define send baseline_send
#define recv baseline_recv
#include "/Users/jordanh/Src/nanolang/modules/websocket/websocket_helpers.c"
#undef send
#undef recv
#include <stdio.h>
static const char *reply="HTTP/1.1 403 Forbidden\r\nX-Trace: 101\r\n\r\n";
static size_t pos;
ssize_t baseline_send(int fd,const void *bytes,size_t n,int flags) {
    (void)fd;(void)bytes;(void)flags;return (ssize_t)n;
}
ssize_t baseline_recv(int fd,void *bytes,size_t n,int flags) {
    (void)fd;(void)flags;
    if (!reply[pos]) return 0;
    if(n){*(char *)bytes=reply[pos++];return 1;}return 0;
}
int main(int argc,char **argv) {
    if(argc>1 && !strcmp(argv[1],"long-frame")) {
        char *s=malloc(65537);memset(s,'x',65536);s[65536]=0;
        int r=ws_send_text(1,s);free(s);return r;
    }
    char host[256]={0},port[16]={0},path[1024]={0},encoded[32];
    unsigned char nonce[16]={0};base64_encode(nonce,16,encoded);
    printf("zero nonce base64: %s\n",encoded);
    printf("wss parse status: %d\n",parse_ws_url("wss://localhost/private",host,sizeof host,port,sizeof port,path,sizeof path));
    printf("wss selected host/port: %s %s\n",host,port);
    printf("403 with 101 header handshake status: %d\n",ws_handshake(1,"localhost","/"));
    return 0;
}
