#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../src/nsi_websocket_values.h"
int main(int argc,char **argv) {
    assert(argc==3);
    NlWsTransportPolicy policy={true,true,getenv("NANOLANG_RESOLVER"),2000};
    NlWsValues *s=NULL;assert(nl_ws_values_create(&policy,&s)==NL_WS_VALUE_OK);
    NlWsValue result={0},value={0};NlWsConnectView view={0};
    assert(nl_ws_values_connect(s,argv[1],strlen(argv[1]),2000,&result)==NL_WS_VALUE_OK);
    assert(nl_ws_connect_view(s,&result,&view)==NL_WS_VALUE_OK && view.ok);
    if(!strcmp(argv[2],"unhandled")) {
        NlWsValuesFinish finish=nl_ws_values_destroy(s,NL_WS_VALUE_STATE);
        assert(finish.execution==NL_WS_VALUE_STATE && !finish.cleanup_failures);return 0;
    }
    NlWsValue stale=result;
    assert(nl_ws_connect_take_ok(s,&result,&value)==NL_WS_VALUE_OK);
    assert(nl_ws_value_validate(s,&stale,true)==NL_WS_VALUE_STALE);
    NlWsValueBorrow borrow={0};
    assert(nl_ws_value_borrow(s,&value,&borrow)==NL_WS_VALUE_OK);
    if(!strcmp(argv[2],"borrowed")) {
        NlWsValuesFinish finish=nl_ws_values_destroy(s,NL_WS_VALUE_STATE);
        assert(finish.execution==NL_WS_VALUE_STATE && !finish.cleanup_failures);return 0;
    }
    NlWsTransportResult r={0};
    assert(nl_ws_value_close(s,&value,1000,&r)==NL_WS_VALUE_BORROWED);
    assert(nl_ws_value_send(s,&borrow,true,"a\0b",3,1000,&r)==NL_WS_VALUE_OK);
    assert(r.status==NL_WS_TRANSPORT_OK && r.bytes==3);
    NlWsMessage message={0};
    assert(nl_ws_value_receive(s,&borrow,1000,&message,&r)==NL_WS_VALUE_OK);
    assert(r.status==NL_WS_TRANSPORT_OK && message.binary && message.length==3 && !memcmp(message.bytes,"a\0b",3));
    NlWsValueBorrow old_borrow=borrow;
    assert(nl_ws_value_end_borrow(s,&borrow)==NL_WS_VALUE_OK);
    assert(nl_ws_value_borrow_validate(s,&old_borrow)==NL_WS_VALUE_STALE);
    NlWsValue moved={0};stale=value;
    assert(nl_ws_value_move(s,&value,&moved)==NL_WS_VALUE_OK && !value.invocation);
    assert(nl_ws_value_validate(s,&stale,false)==NL_WS_VALUE_STALE);
    assert(nl_ws_value_close(s,&moved,!strcmp(argv[2],"invalid-close")?-1:1000,&r)==NL_WS_VALUE_OK);
    assert(!moved.invocation && r.terminal);
    assert(r.status==(!strcmp(argv[2],"invalid-close")?NL_WS_TRANSPORT_LIMIT:NL_WS_TRANSPORT_OK));
    NlWsValuesFinish finish=nl_ws_values_destroy(s,NL_WS_VALUE_OK);
    assert(!finish.cleanup_failures);
    assert(message.binary && message.length==3 && !memcmp(message.bytes,"a\0b",3));
    nl_ws_message_free(&message);return 0;
}
