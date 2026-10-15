#include "../modules/websocket/websocket_helpers.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(int argc,char **argv) {
    CHECK(argc==3);
    CHECK(!nl_ws_is_connected(INT64_MAX) && !nl_ws_is_connected(1));
    CHECK(nl_ws_send(INT64_MAX,"x")==-1 && nl_ws_close(INT64_MAX)==-1);
    CHECK(!strcmp(nl_ws_receive_timeout(INT64_MAX,0),""));
    CHECK(strstr(nl_ws_last_error(INT64_MAX),"live"));
    CHECK(!nl_ws_connect("wss://localhost/private"));
    CHECK(!nl_ws_connect("ws://localhost:0/"));
    CHECK(!nl_ws_connect("ws://localhost/path\r\nX: injected"));
    CHECK(!nl_ws_connect("ws://user@localhost/"));
    CHECK(!nl_ws_connect("ws://localhost/#fragment"));
    int64_t h=nl_ws_connect(argv[1]);
    if(!strcmp(argv[2],"refuse")){CHECK(!h);puts("PASS refused upgrade");return 0;}
    CHECK(h>0 && nl_ws_is_connected(h));
    if(!strcmp(argv[2],"normal")) {
        char *long_text=malloc(70001);CHECK(long_text);memset(long_text,'x',70000);long_text[70000]=0;
        CHECK(nl_ws_send(h,long_text)==0 && nl_ws_send(h,long_text)==0);free(long_text);
        CHECK(!strcmp(nl_ws_receive_timeout(h,2000),"hello\xe2\x82\xac"));
        CHECK(!strcmp(nl_ws_receive_timeout(h,2000),"") && !nl_ws_is_connected(h));
    } else if(!strcmp(argv[2],"partial")) {
        CHECK(!strcmp(nl_ws_receive_timeout(h,30),"") && nl_ws_is_connected(h));
        CHECK(!strcmp(nl_ws_receive_timeout(h,2000),"hello"));
    } else if(!strcmp(argv[2],"invalid")) {
        CHECK(!strcmp(nl_ws_receive_timeout(h,2000),"") && !nl_ws_is_connected(h));
        CHECK(strlen(nl_ws_last_error(h))>0);
    } else CHECK(!strcmp(argv[2],"close"));
    CHECK(nl_ws_close(h)==0);
    CHECK(!nl_ws_is_connected(h) && nl_ws_close(h)==-1 && nl_ws_send(h,"x")==-1);
    CHECK(!strcmp(nl_ws_receive(h),""));
    CHECK(strstr(nl_ws_last_error(h),"live"));
    puts("PASS owned WebSocket transport and stale/forged identities");
    return 0;
}
