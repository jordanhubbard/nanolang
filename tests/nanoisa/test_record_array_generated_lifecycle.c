/* I check private adapter boundaries separately from generated program parity. */
#include "../../src/nanoisa/record_array_generated_private.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks,entries;
static bool fail_entry;
#define CHECK(x) do { checks++; if(!(x)) { fprintf(stderr,"lifecycle check %u at %d\n",checks,__LINE__); abort(); } } while(0)
static void body(NrgInstance *p) {
    entries++;
    CHECK(nrg_run(p)==NRG_BUSY);
    NrgStats s,before;memset(&s,0xa5,sizeof s);memcpy(&before,&s,sizeof s);
    CHECK(!nrg_stats(p,&s)&&!memcmp(&s,&before,sizeof s));
    nrg_destroy(p); /* Busy destruction must leave the acquired invocation alive. */
    CHECK(nrg_status(p)==NRG_OK);
    NmsValue value={42,1};CHECK(nrg_push_move(p,&value));
    CHECK(value.tag==0);
    if(fail_entry) { nrg_fail(p,NRG_ASSERT);nrg_fail(p,NRG_TYPE);return; }
    nrg_return(p);
}
static void *foreign(void *arg) {
    NrgInstance *p=arg;NrgStats s,before;
    memset(&s,0xa5,sizeof s);memcpy(&before,&s,sizeof s);
    CHECK(nrg_run(p)==NRG_TYPE);
    CHECK(!nrg_stats(p,&s)&&!memcmp(&s,&before,sizeof s));
    nrg_destroy(p);return NULL;
}
static void refuse(const NrgProgram *program) {
    NrgInstance *p=(void *)(uintptr_t)1;
    CHECK(nrg_create(program,&p)==NRG_STATE&&p==(void *)(uintptr_t)1);
}
int main(void) {
    const NrgFunction function={0,0,1,1,1,NULL,body};
    const NrgProgram program={NRG_ABI,sizeof(NmsValue),offsetof(NmsValue,tag),NRG_FRAMES,
        1,0,UINT32_MAX,0,1,0,0,0,&function,NULL,NULL,NULL,NULL};
    NrgProgram bad=program;
    bad.abi++;refuse(&bad);bad=program;bad.value_size++;refuse(&bad);
    bad=program;bad.value_tag_offset++;refuse(&bad);
    bad=program;bad.frame_limit--;refuse(&bad);refuse(NULL);
    CHECK(nrg_create(&program,NULL)==NRG_STATE);
    NrgInstance *p=NULL;CHECK(nrg_create(&program,&p)==NRG_OK);
    NrgStats s;CHECK(nrg_stats(p,&s)&&s.epoch==0&&!s.has_result&&!s.frames);
    pthread_t thread;CHECK(!pthread_create(&thread,NULL,foreign,p));CHECK(!pthread_join(thread,NULL));
    CHECK(entries==0&&nrg_stats(p,&s)&&s.epoch==0);
    CHECK(nrg_run(p)==NRG_OK&&entries==1);
    NrgObservation original,after;CHECK(nrg_observe(p,false,0,NULL,0,&original));
    CHECK(original.tag==1&&original.scalar_bits==42&&original.epoch==1);
    fail_entry=true;CHECK(nrg_run(p)==NRG_ASSERT&&entries==2);
    CHECK(nrg_stats(p,&s)&&s.epoch==2&&!s.frames&&s.has_result&&s.status==NRG_ASSERT);
    CHECK(nrg_observe(p,false,0,NULL,0,&after));
    CHECK(after.tag==original.tag&&after.scalar_bits==original.scalar_bits&&after.identity==original.identity);
    fail_entry=false;CHECK(nrg_run(p)==NRG_OK&&entries==3);
    CHECK(nrg_stats(p,&s)&&s.epoch==3&&!s.frames&&s.status==NRG_OK);
    nrg_destroy(p);
    printf("I passed %u private lifecycle checks: ABI, wrong thread, BUSY, acquired finish, retained result and recovery.\n",checks);
    return 0;
}
