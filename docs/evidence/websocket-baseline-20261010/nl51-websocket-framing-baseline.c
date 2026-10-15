#define _POSIX_C_SOURCE 200809L
#define send baseline_send
#define recv baseline_recv
#define close baseline_close
#include "/Users/jordanh/Src/nanolang/modules/websocket/websocket_helpers.c"
#undef send
#undef recv
#undef close
static const unsigned char *input;
static size_t input_size, input_pos;
static unsigned char sent[128];
static size_t sent_size;
ssize_t baseline_send(int fd,const void *bytes,size_t n,int flags) {
    (void)fd;(void)flags;
    if(n<=sizeof sent-sent_size){memcpy(sent+sent_size,bytes,n);sent_size+=n;}
    return (ssize_t)n;
}
ssize_t baseline_recv(int fd,void *bytes,size_t n,int flags) {
    (void)fd;(void)flags;
    if(n>input_size-input_pos)n=input_size-input_pos;
    memcpy(bytes,input+input_pos,n);input_pos+=n;return (ssize_t)n;
}
int baseline_close(int fd){(void)fd;return 0;}
static void hex(const unsigned char *p,size_t n){for(size_t i=0;i<n;i++)printf("%02x",p[i]);puts("");}
static void frame(const char *label,const unsigned char *p,size_t n) {
    input=p;input_size=n;input_pos=0;sent_size=0;
    char *data=NULL;size_t count=0;
    int r=ws_recv_frame(1,&data,&count,-1);
    printf("%s: status=%d payload=",label,r);hex((const unsigned char *)data,count);
    printf("sent=");hex(sent,sent_size);free(data);
}
int main(int argc,char **argv) {
    if(argc>1 && !strcmp(argv[1],"forged"))return (int)nl_ws_is_connected(1);
    if(argc>1 && !strcmp(argv[1],"stale")) {
        WsCtx *c=ws_alloc();if(!c)return 2;c->connected=1;
        int64_t h=(int64_t)(uintptr_t)c;nl_ws_close(h);return (int)nl_ws_is_connected(h);
    }
    const unsigned char ping[]={0x89,3,'a','b','c'};
    const unsigned char fragment[]={0x01,3,'a','b','c'};
    const unsigned char reserved[]={0xc1,3,'a','b','c'};
    const unsigned char utf8[]={0x81,2,0xc0,0xaf};
    const unsigned char masked[]={0x81,0x83,1,2,3,4,'a'^1,'b'^2,'c'^3};
    frame("ping",ping,sizeof ping);frame("unfinished text",fragment,sizeof fragment);
    frame("reserved bit",reserved,sizeof reserved);frame("invalid UTF-8",utf8,sizeof utf8);
    frame("masked server",masked,sizeof masked);
    sent_size=0;WsCtx *c=ws_alloc();if(!c)return 2;c->fd=1;c->connected=1;
    nl_ws_close((int64_t)(uintptr_t)c);printf("close bytes=");hex(sent,sent_size);
    sent_size=0;ws_send_text(1,"abc");printf("first text frame=");hex(sent,sent_size);
    sent_size=0;ws_send_text(1,"abc");printf("second text frame=");hex(sent,sent_size);
    return 0;
}
