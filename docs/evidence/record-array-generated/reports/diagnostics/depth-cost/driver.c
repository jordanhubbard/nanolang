#include "record_array_generated_private.h"
#include "record_array_alloc.h"
#include <stdio.h>
#include <time.h>
NrgStatus nrg_generated_create(NrgInstance **);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){setvbuf(stdout,NULL,_IONBF,0);for(unsigned i=0;i<2;i++){ra_calls=0;ra_peak=0;double a=now();NrgInstance *p=NULL;NrgStatus s=nrg_generated_create(&p);double b=now();if(s)return 1;s=nrg_run(p);double c=now();NrgStats stats;if(!nrg_stats(p,&stats)||s||stats.maximum_frames!=1024)return 2;nrg_destroy(p);double d=now();printf("iteration=%u create=%.9f run=%.9f destroy=%.9f calls=%zu peak=%zu live=%zu bytes=%zu\n",i,b-a,c-b,d-c,ra_calls,ra_peak,ra_live,ra_bytes);if(ra_live||ra_bytes)return 3;}return 0;}
