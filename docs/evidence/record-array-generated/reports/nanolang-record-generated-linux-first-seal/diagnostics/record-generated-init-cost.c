#include "record_array_generated_private.h"
#include "record_array_alloc.h"
#include <stdio.h>
#include <time.h>
static void body(NrgInstance *p){nrg_return(p);}
static double now(void){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))return -1;return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){setvbuf(stdout,NULL,_IONBF,0);const NrgFunction f={0,0,0,0,0,NULL,body};const NrgProgram program={NRG_ABI,sizeof(NmsValue),offsetof(NmsValue,tag),NRG_FRAMES,1,0,UINT32_MAX,0,1,0,0,0,&f,NULL,NULL,NULL,NULL};for(unsigned i=0;i<2;i++){double begin=now();NrgInstance *p=NULL;if(nrg_create(&program,&p)!=NRG_OK)return 1;double created=now();nrg_destroy(p);double done=now();if(ra_live||ra_bytes)return 2;printf("iteration=%u create=%.9f destroy=%.9f allocation_calls=%zu\n",i,created-begin,done-created,ra_calls);}return 0;}
