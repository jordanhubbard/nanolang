/* I exercise the actual transport through the mixed carrier. */
#include "../src/nsi_services_values.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv){
 assert(argc==3);bool denied=!strcmp(argv[2],"denied"),abandon=!strcmp(argv[2],"borrowed"),unhandled=!strcmp(argv[2],"unhandled");
 size_t minimum=0,transport=0;assert(nl_ws_values_minimum_storage(&minimum) && nl_ws_transport_storage_bound(&transport));
 NlServicesValueConfig configs[4]={{.catalog=NL_SERVICES_FILE},{.catalog=NL_SERVICES_TCP}};
 for(unsigned i=2;i<4;i++)configs[i]=(NlServicesValueConfig){NL_SERVICES_WEBSOCKET,{!denied,true,getenv("NANOLANG_RESOLVER"),2000},minimum+transport};
 NlServicesValues *c=NULL;assert(nl_services_values_create_config(configs,4,&c)==NL_SERVICES_VALUE_OK);memset(configs,0,sizeof configs);
 NlServicesValue result={0},file={0};NlServicesOpenView view;
 assert(nl_services_values_acquire(c,0,NULL,&result)==NL_SERVICES_VALUE_OK);
 assert(nl_services_value_take_ok(c,&result,&file)==NL_SERVICES_VALUE_OK);
 NlServicesBorrow file_borrow={0};assert(nl_services_value_borrow(c,&file,&file_borrow)==NL_SERVICES_VALUE_OK);
 NlServicesScalarResult scalar;assert(nl_services_value_call(c,0,1,&file_borrow,42,&scalar)==NL_SERVICES_VALUE_OK);
 assert(nl_services_value_call(c,0,2,&file_borrow,0,&scalar)==NL_SERVICES_VALUE_OK);
 assert(nl_services_value_call(c,0,3,&file_borrow,0,&scalar)==NL_SERVICES_VALUE_OK && scalar.result.file.value==42);
 NlSocketEndpoint invalid={0};assert(nl_services_values_acquire(c,1,&invalid,&result)==NL_SERVICES_VALUE_OK);
 assert(nl_services_value_take_error(c,&result,&view)==NL_SERVICES_VALUE_OK && view.error.tcp.status==NL_SOCKET_ARGUMENT);
 NlWsMessage retained={0};
 for(unsigned i=2;i<4;i++){
  assert(nl_services_values_websocket_connect(c,i,argv[1],strlen(argv[1]),1000,&result)==NL_SERVICES_VALUE_OK);
  assert(nl_services_value_view(c,&result,&view)==NL_SERVICES_VALUE_OK);
  if(denied){assert(!view.ok && view.error.websocket.status==NL_WS_TRANSPORT_RIGHTS);assert(nl_services_value_take_error(c,&result,&view)==NL_SERVICES_VALUE_OK);continue;}
  assert(view.ok);if(unhandled)break;
  NlServicesValue value={0};assert(nl_services_value_take_ok(c,&result,&value)==NL_SERVICES_VALUE_OK);
  NlServicesBorrow borrow={0};assert(nl_services_value_borrow(c,&value,&borrow)==NL_SERVICES_VALUE_OK);
  for(unsigned pass=0;pass<2;pass++){
   NlWsTransportResult r;assert(nl_services_value_websocket_send(c,i,&borrow,true,"a\0b",3,1000,&r)==NL_SERVICES_VALUE_OK && r.status==NL_WS_TRANSPORT_OK);
   NlWsMessage message={0};assert(nl_services_value_websocket_receive(c,i,&borrow,1000,&message,&r)==NL_SERVICES_VALUE_OK);
   assert(r.status==NL_WS_TRANSPORT_OK && message.binary && message.length==3 && !memcmp(message.bytes,"a\0b",3));
   nl_ws_message_free(&retained);retained=message;
  }
  if(abandon)break;
  assert(nl_services_value_end_borrow(c,&borrow)==NL_SERVICES_VALUE_OK);NlWsTransportResult r;
  bool invalid_close=!strcmp(argv[2],"invalid-close");
  assert(nl_services_value_websocket_close(c,i,&value,invalid_close?-1:1000,&r)==NL_SERVICES_VALUE_OK && !value.instance);
  assert(r.status==(invalid_close?NL_WS_TRANSPORT_LIMIT:NL_WS_TRANSPORT_OK));
 }
 NlServicesFinish finish;assert(nl_services_values_destroy(&c,NL_SERVICES_VALUE_OK,&finish)==NL_SERVICES_VALUE_OK && !c && !finish.cleanup_failures);
 if(retained.bytes){assert(retained.binary && retained.length==3 && !memcmp(retained.bytes,"a\0b",3));nl_ws_message_free(&retained);}
 return 0;
}
