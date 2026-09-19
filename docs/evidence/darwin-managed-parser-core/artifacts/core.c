#include "managed_strings.h"
#include "inputs.h"
#include "expected.h"
int run(void){NmsRuntime r;nms_init(&r,0,0);
for(unsigned i=0;i<sizeof inputs/sizeof inputs[0];i++){
 NmsHandle handle;uint64_t bits=0;
 if(nms_create(&r,inputs[i].text,inputs[i].length,&handle)!=NMS_OK)return 10000+i;
 if(nms_retain(&r,handle)!=NMS_OK)return 20000+i;
#ifdef NMS_TESTING
 nms_test_fail_after(&r,0);
#endif
 if(nms_parse_f64(&r,handle,&bits)!=NMS_OK||bits!=expected[i])return 30000+i;
 if(r.live_objects!=1||r.slots[handle&~NMS_DYNAMIC].references!=2)return 40000+i;
 if(nms_release(&r,handle)!=NMS_OK||nms_release(&r,handle)!=NMS_OK||r.live_objects)return 50000+i;
#ifdef NMS_TESTING
 nms_test_fail_after(&r,UINT64_MAX);
#endif
}return nms_dispose(&r);}
#ifndef __wasm32__
#include <stdio.h>
int main(void){int result=run();if(result)fprintf(stderr,"parser reference failure %d\n",result);return result?1:0;}
#endif
