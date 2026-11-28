/* I qualify scratch reuse without retaining payload roots across collection. */
#include "eval/carrier_runtime.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail collection scratch check %s at line %d.\n",#c,__LINE__);exit(1); } } while (0)
int g_argc;
char **g_argv;
static size_t attempts, calls, released_scratch;
static bool deny_growth, deny_collection, reenter;
static Environment *active;
static const NlCarrierRef *last_roots;
static size_t last_count;
static void *test_realloc(void *p,size_t bytes) {
    attempts++;
    if(deny_growth)return NULL;
    return realloc(p,bytes);
}
static void test_free(void *p) {
    if(p && p==last_roots)released_scratch++;
    free(p);
}
static bool test_collect(NlCarrierValues *values,const NlCarrierRef *roots,size_t count) {
    calls++;last_roots=roots;last_count=count;
    if(reenter) {
        CHECK(count>0);unsigned char first[sizeof *roots];memcpy(first,roots,sizeof first);
        CHECK(!env_carrier_collect(active));
        CHECK(memcmp(first,roots,sizeof first)==0);
    }
    return !deny_collection && nl_carrier_values_collect(values,roots,count);
}
#define realloc test_realloc
#define free test_free
#define nl_carrier_values_collect test_collect
#include "../src/eval/carrier_runtime.c"
#undef realloc
#undef free
#undef nl_carrier_values_collect
static void cleared(void) {
    const unsigned char *p=(const unsigned char *)last_roots;
    for(size_t i=0;i<last_count*sizeof *last_roots;i++)CHECK(p[i]==0);
}
int main(void) {
    active=create_environment();CHECK(active);CHECK(env_carrier_enable(active));
    CHECK(env_carrier_enable_collection(active,NULL,0));CHECK(env_carrier_collect(active));
    CHECK(last_count==0);
    Value owned=create_struct("ScratchOwned",NULL,NULL,0);env_carrier_track(active,owned,false);
    NlCarrierRef hosts[80];for(size_t i=0;i<80;i++)hosts[i]=nl_carrier_value(owned);
    CHECK(env_carrier_enable_collection(active,hosts,1));
    size_t before=calls;deny_growth=true;CHECK(!env_carrier_collect(active));
    CHECK(calls==before && env_carrier_owns(active,owned));
    deny_growth=false;CHECK(env_carrier_collect(active));cleared();
    CHECK(env_carrier_owns(active,owned));
    size_t bounded=attempts;
    reenter=true;
    for(size_t i=0;i<128;i++){CHECK(env_carrier_collect(active));cleared();CHECK(env_carrier_owns(active,owned));}
    reenter=false;CHECK(attempts==bounded);
    deny_collection=true;CHECK(!env_carrier_collect(active));cleared();CHECK(env_carrier_owns(active,owned));deny_collection=false;
    CHECK(env_carrier_enable_collection(active,hosts,80));
    deny_growth=true;before=calls;CHECK(!env_carrier_collect(active));CHECK(calls==before && env_carrier_owns(active,owned));cleared();
    CHECK(env_carrier_enable_collection(active,hosts,1));CHECK(env_carrier_collect(active));cleared();
    deny_growth=false;CHECK(env_carrier_enable_collection(active,hosts,80));CHECK(env_carrier_collect(active));cleared();
    CHECK(last_count==80 && env_carrier_owns(active,owned));
    CHECK(env_carrier_enable_collection(active,NULL,0));CHECK(env_carrier_collect(active));cleared();CHECK(env_carrier_stats(active).live==0);
    Value text=create_string("caller survives");char *names[]={"text"};Value fields[]={text};
    StructValue caller={0};caller.struct_name="BorrowedScratch";caller.field_count=1;caller.field_names=names;caller.field_values=fields;
    Value borrowed={0};borrowed.type=VAL_STRUCT;borrowed.as.struct_val=&caller;hosts[0]=nl_carrier_value(borrowed);
    CHECK(env_carrier_enable_collection(active,hosts,1));CHECK(env_carrier_collect(active));cleared();
    CHECK(env_carrier_stats(active).registered==1 && strcmp(text.as.string_val,"caller survives")==0);
    CHECK(gc_get_header(text.as.string_val)->ref_count==1);
    free_environment(active);CHECK(released_scratch==1);
    CHECK(strcmp(text.as.string_val,"caller survives")==0);gc_release(text.as.string_val);
    CHECK(gc_get_stats().current_usage==0 && gc_get_stats().num_objects==0);
    puts("I preserve failed growth, failed collection, reentrant refusal, bounded reuse, changing roots and borrowed caller lifetime; scratch retires once.");
    return 0;
}
