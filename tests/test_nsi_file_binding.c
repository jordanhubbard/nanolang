/* I observe pure binding bytes and all selected provider allocations.
 * Forward source text is not parsed, lowered, or executed here. */
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#define BINDING_TEST_IMPLEMENTATION
#include "file_binding_hooks.h"
#include "nsi_file_binding.h"
#include "nsi_internal.h"
#include "nsi_file_plan.h"
#include "binding_cases.h"
static size_t checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"I failed line %d: %s\n",__LINE__,#x); abort(); } } while (0)

enum { ALLOC_MALLOC, ALLOC_CALLOC, ALLOC_REALLOC, ALLOC_STRDUP };
typedef struct { void *pointer; size_t bytes; } Allocation;
static Allocation live_allocations[32768];
static unsigned trace[32768];
static size_t live, live_bytes, peak_bytes, calls, fail_at=SIZE_MAX, fail_hits;
static size_t kinds[4];
static bool transient, tracing;
static bool inject(unsigned kind) {
    if (tracing) { CHECK(calls < sizeof trace / sizeof trace[0]); trace[calls]=kind; }
    kinds[kind]++;
    size_t index=calls++;
    bool fail=transient ? index==fail_at : index>=fail_at;
    if (fail) fail_hits++;
    return fail;
}
static void account(void *p,size_t n) {
    if (!p) return;
    CHECK(live < sizeof live_allocations / sizeof live_allocations[0]);
    CHECK(n<=SIZE_MAX-live_bytes);
    live_allocations[live++]=(Allocation){p,n};live_bytes+=n;
    if(live_bytes>peak_bytes)peak_bytes=live_bytes;
}
static size_t locate(void *p) {
    for(size_t i=0;i<live;i++)if(live_allocations[i].pointer==p)return i;
    CHECK(false);return 0;
}
void *binding_test_malloc(size_t n) {
    if(inject(ALLOC_MALLOC))return NULL;
    void *p=malloc(n?n:1);account(p,n);return p;
}
void *binding_test_calloc(size_t n,size_t width) {
    CHECK(!n||width<=SIZE_MAX/n);
    if(inject(ALLOC_CALLOC))return NULL;
    void *p=calloc(n?n:1,width?width:1);account(p,n*width);return p;
}
void binding_test_free(void *p) {
    if(!p)return;
    size_t i=locate(p);live_bytes-=live_allocations[i].bytes;
    live_allocations[i]=live_allocations[--live];free(p);
}
void *binding_test_realloc(void *p,size_t n) {
    if(!p){if(inject(ALLOC_REALLOC))return NULL;void *q=malloc(n?n:1);account(q,n);return q;}
    size_t i=locate(p);
    if(inject(ALLOC_REALLOC))return NULL;
    if(!n){binding_test_free(p);return NULL;}
    void *q=realloc(p,n);
    if(q){live_bytes-=live_allocations[i].bytes;live_allocations[i]=(Allocation){q,n};live_bytes+=n;if(live_bytes>peak_bytes)peak_bytes=live_bytes;}
    return q;
}
char *binding_test_strdup(const char *s) {
    if(inject(ALLOC_STRDUP))return NULL;
    size_t n=strlen(s)+1;char *p=malloc(n);if(p)memcpy(p,s,n);account(p,n);return p;
}
static void reset_fault(size_t index,bool once) {
    CHECK(live==0&&live_bytes==0);calls=0;fail_hits=0;fail_at=index;transient=once;peak_bytes=0;tracing=false;
}
static unsigned char *read_bytes(const char *path,size_t *size) {
    FILE *f=fopen(path,"rb");CHECK(f!=NULL);CHECK(!fseek(f,0,SEEK_END));long n=ftell(f);CHECK(n>=0);CHECK(!fseek(f,0,SEEK_SET));
    unsigned char *p=malloc((size_t)n+1);CHECK(p!=NULL);CHECK(fread(p,1,(size_t)n,f)==(size_t)n);CHECK(!fclose(f));p[n]=0;*size=(size_t)n;return p;
}
static unsigned char *case_bytes(const char *base,const char *name,size_t *size) {
    char path[4096];int n=snprintf(path,sizeof path,"%s/%s",base,name);CHECK(n>0&&(size_t)n<sizeof path);return read_bytes(path,size);
}
static unsigned char *gold_json,*gold_source;
static size_t gold_json_size,gold_source_size,bound;
static void check_plan(NlFileBindingPlan *p) {
    size_t n=0;const unsigned char *j=nl_file_binding_interface_bytes(p,&n);
    CHECK(j!=NULL&&n==gold_json_size&&!memcmp(j,gold_json,n)&&j[n]==0);
    const unsigned char *s=nl_file_binding_source_bytes(p,&n);
    CHECK(s!=NULL&&n==gold_source_size&&!memcmp(s,gold_source,n)&&s[n]==0);
    CHECK(nl_file_binding_peak_bound(p)==bound);
    CHECK(nl_file_binding_storage_size(p)>=gold_json_size+gold_source_size+2);
#ifdef BINDING_INSTRUMENT
    CHECK(live==1&&live_bytes==nl_file_binding_storage_size(p));
    CHECK(peak_bytes<=bound);
#endif
}
static NlFileBindingStatus prepare_case(const unsigned char *bytes,size_t size,uint32_t mask,bool preallocation) {
    NlFileBindingPlan *sentinel=(NlFileBindingPlan *)(uintptr_t)1,*p=sentinel;
    NlFileBindingStatus status=nl_file_binding_prepare(bytes,size,&p);
    CHECK((unsigned)status<32&&(mask&(1u<<(unsigned)status)));
    if(status==NL_FILE_BINDING_OK){CHECK(p!=sentinel);check_plan(p);nl_file_binding_free(p);}
    else CHECK(p==sentinel);
#ifdef BINDING_INSTRUMENT
    CHECK(live==0&&live_bytes==0&&peak_bytes<=bound);
    if(preallocation)CHECK(calls==0);
#else
    (void)preallocation;
#endif
    return status;
}
static void retain_output(const char *base,const char *name,const unsigned char *data,size_t size) {
    char path[4096];int n=snprintf(path,sizeof path,"%s/%s",base,name);CHECK(n>0&&(size_t)n<sizeof path);
    FILE *f=fopen(path,"wb");CHECK(f!=NULL);CHECK(fwrite(data,1,size,f)==size);CHECK(!fclose(f));
}
static void basic(const char *base,const unsigned char *bytes,size_t size) {
    CHECK(!nl_file_binding_allocation_bound(NULL));
    CHECK(nl_file_binding_allocation_bound(&bound));CHECK(bound<=NL_FILE_BINDING_MAX_ALLOCATION);
    CHECK(nl_file_binding_prepare(bytes,size,NULL)==NL_FILE_BINDING_INVALID);
    CHECK(nl_file_binding_storage_size(NULL)==0&&nl_file_binding_peak_bound(NULL)==0);
    size_t sentinel_size=73;
    CHECK(nl_file_binding_interface_bytes(NULL,&sentinel_size)==NULL&&sentinel_size==73);
    CHECK(nl_file_binding_source_bytes(NULL,&sentinel_size)==NULL&&sentinel_size==73);
    nl_file_binding_free(NULL);
    reset_fault(SIZE_MAX,false);prepare_case(NULL,1,1u<<NL_FILE_BINDING_INVALID,true);
    reset_fault(SIZE_MAX,false);prepare_case(bytes,0,1u<<NL_FILE_BINDING_INVALID,true);
    NlFileBindingPlan *p=NULL;reset_fault(SIZE_MAX,false);
    unsigned char *copy=malloc(size);CHECK(copy!=NULL);memcpy(copy,bytes,size);
    CHECK(nl_file_binding_prepare(copy,size,&p)==NL_FILE_BINDING_OK);
    memset(copy,0xa5,size);free(copy);check_plan(p);
    CHECK(nl_file_binding_interface_bytes(p,NULL)==NULL);CHECK(nl_file_binding_source_bytes(p,NULL)==NULL);
    size_t n;const unsigned char *canonical=nl_file_binding_interface_bytes(p,&n);
#ifdef BINDING_INSTRUMENT
    retain_output(base,"actual-instrumented-interface.json",canonical,n);
    size_t source_count;const unsigned char *source=nl_file_binding_source_bytes(p,&source_count);
    retain_output(base,"actual-instrumented-binding.nano.txt",source,source_count);
#else
    retain_output(base,"actual-linked-interface.json",canonical,n);
    size_t source_count;const unsigned char *source=nl_file_binding_source_bytes(p,&source_count);
    retain_output(base,"actual-linked-binding.nano.txt",source,source_count);
#endif
    unsigned char *snapshot=malloc(n);CHECK(snapshot!=NULL);memcpy(snapshot,canonical,n);
    nl_file_binding_free(p);p=NULL;reset_fault(SIZE_MAX,false);
    CHECK(nl_file_binding_prepare(snapshot,n,&p)==NL_FILE_BINDING_OK);check_plan(p);free(snapshot);nl_file_binding_free(p);
    CHECK(live==0&&live_bytes==0);
}
#ifdef BINDING_INSTRUMENT
static void faults(const unsigned char *bytes,size_t size,const char *label,NlFileBindingStatus baseline_status) {
    reset_fault(SIZE_MAX,false);tracing=true;
    uint32_t baseline=1u<<(unsigned)baseline_status;
    prepare_case(bytes,size,baseline,false);size_t count=calls;
    CHECK(count>4);CHECK(kinds[ALLOC_MALLOC]>0&&kinds[ALLOC_CALLOC]>0&&kinds[ALLOC_STRDUP]>0);
    tracing=false;
    size_t failures=0,recovered=0;
    for(unsigned once=0;once<2;once++)for(size_t i=0;i<count;i++){
        reset_fault(i,once!=0);
        NlFileBindingStatus status=prepare_case(bytes,size,baseline|(1u<<NL_FILE_BINDING_MEMORY)|(1u<<NL_FILE_BINDING_UNRESOLVED),false);
        CHECK(fail_hits>0);
        if(status==baseline_status)recovered++;else failures++;
        printf("FAULT %s %s %zu status=%u hits=%zu attempts=%zu peak=%zu\n",label,once?"transient":"prefix",i,(unsigned)status,fail_hits,calls,peak_bytes);
        reset_fault(SIZE_MAX,false);prepare_case(bytes,size,baseline,false);
    }
    printf("FAULT_SUMMARY %s prefixes=%zu failures=%zu baseline_recoveries=%zu\n",label,2*count,failures,recovered);
}
static void named_failures(const unsigned char *bytes,size_t size) {
    reset_fault(SIZE_MAX,false);
    const char *end=NULL;cJSON *tree=cJSON_ParseWithLengthOpts((const char *)bytes,size+1,&end,1);
    CHECK(tree!=NULL&&end==(const char *)bytes+size);
    size_t before=live,before_bytes=live_bytes;
    calls=0;tracing=true;NlNsi *n=nl_nsi_decode_object(tree);CHECK(n!=NULL);nl_nsi_free(n);tracing=false;
    CHECK(trace[0]==ALLOC_CALLOC&&trace[1]==ALLOC_STRDUP&&trace[2]==ALLOC_STRDUP);
    CHECK(live==before&&live_bytes==before_bytes);
    for(size_t index=1;index<=2;index++){
        calls=0;fail_hits=0;fail_at=index;transient=true;
        CHECK(nl_nsi_decode_object(tree)==NULL);CHECK(fail_hits==1&&live==before&&live_bytes==before_bytes);
        fail_at=SIZE_MAX;calls=0;n=nl_nsi_decode_object(tree);CHECK(n!=NULL);
        CHECK(!strcmp(n->iface.id,"nsi:nanolang/filesystem")&&!strcmp(n->iface.name,"filesystem"));nl_nsi_free(n);
        CHECK(live==before&&live_bytes==before_bytes);printf("NAMED_FAILURE index=%zu retained_tree=%zu\n",index,before);
    }
    cJSON_Delete(tree);CHECK(live==0&&live_bytes==0);reset_fault(SIZE_MAX,false);
}
#endif
int main(int argc,char **argv) {
    CHECK(argc==4);const char *base=argv[1];
    gold_json=read_bytes(argv[2],&gold_json_size);gold_source=read_bytes(argv[3],&gold_source_size);
    size_t size;unsigned char *bytes=case_bytes(base,"valid.json",&size);
    basic(base,bytes,size);
    for(size_t i=0;i<sizeof binding_cases/sizeof binding_cases[0];i++){
        const BindingCase *c=&binding_cases[i];size_t n;unsigned char *input=case_bytes(base,c->filename,&n);
        reset_fault(SIZE_MAX,false);NlFileBindingStatus status=prepare_case(input,n,c->statuses,c->preallocation);
        printf("CASE %s status=%u bytes=%zu allocations=%zu\n",c->name,(unsigned)status,n,calls);free(input);
    }
#ifdef BINDING_INSTRUMENT
    named_failures(bytes,size);faults(bytes,size,"catalog",NL_FILE_BINDING_OK);
    for(size_t i=0;i<3;i++){
        const char *names[]={"alloc-array.json","alloc-callback.json","alloc-async.json"};
        size_t n;unsigned char *input=case_bytes(base,names[i],&n);
        faults(input,n,names[i],NL_FILE_BINDING_INVALID);free(input);
    }
#endif
    free(bytes);free(gold_json);free(gold_source);CHECK(live==0&&live_bytes==0);
    printf("PASS strict File binding %zu checks cases=%zu allocations=%zu/%zu/%zu/%zu bound=%zu mode=%s\n",checks,sizeof binding_cases/sizeof binding_cases[0],kinds[0],kinds[1],kinds[2],kinds[3],bound,
#ifdef BINDING_INSTRUMENT
           "instrumented"
#else
           "linked"
#endif
    );return 0;
}
