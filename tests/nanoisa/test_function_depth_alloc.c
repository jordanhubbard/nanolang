/* I retire every verifier allocation and continue after a transient refusal. */
#define main retained_depth_controls_main
#include "test_function_depth_batch.c"
#undef main
static void *live_objects[64];
static size_t live_count,attempts,fail_at,hits;
static void remember(void *p) {if(p){assert(live_count<64);live_objects[live_count++]=p;}}
void *depth_test_malloc(size_t size) {++attempts;if(attempts==fail_at){++hits;return NULL;}void *p=malloc(size);remember(p);return p;}
void *depth_test_calloc(size_t count,size_t size) {++attempts;if(attempts==fail_at){++hits;return NULL;}void *p=calloc(count,size);remember(p);return p;}
void *depth_test_realloc(void *p,size_t size) {
    ++attempts;if(attempts==fail_at){++hits;return NULL;}
    size_t i=0;while(i<live_count && live_objects[i]!=p)++i;
    assert(!p || i<live_count);bool had=p!=NULL;void *next=realloc(p,size);
    if(next){if(had)live_objects[i]=next;else remember(next);}return next;
}
void depth_test_free(void *p) {
    if(!p)return;
    size_t i=0;while(i<live_count && live_objects[i]!=p)++i;
    assert(i<live_count);live_objects[i]=live_objects[--live_count];free(p);
}
static NvmVerifyResult query(NvmModule *m,uint16_t *out,size_t failure) {
    attempts=hits=0;fail_at=failure;memset(out,0xa5,3*sizeof *out);
    NvmVerifyResult r=nvm_infer_function_max_stacks(m,out,3,sizeof *out,0);
    fail_at=0;assert(!live_count);return r;
}
int main(void) {
    NvmModule *m=ordinary(3);uint16_t output[3];
    assert(query(m,output,0).ok);size_t count=attempts;assert(count);
    for(unsigned i=0;i<3;++i)assert(output[i]==i+1);
    for(size_t failure=1;failure<=count;++failure) {
        assert(!query(m,output,failure).ok && hits==1);
        unsigned refused=0;
        for(unsigned i=0;i<3;++i){assert(!output[i] || output[i]==i+1);refused+=!output[i];}
        assert(refused==1);
        assert(query(m,output,0).ok);
        for(unsigned i=0;i<3;++i)assert(output[i]==i+1);
    }
    nvm_module_free(m);printf("I passed %zu transient verifier allocation positions and recovery.\n",count);return 0;
}
