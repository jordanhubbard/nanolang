#include "portable_read_module.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern uint64_t nano_try_entry(void);
extern uint64_t nms_module_live_objects(void), nms_module_live_bytes(void);
extern uint32_t nano_dispose(void);
#ifdef TEST_PORTABLE_BYTES
#define npr_module_bind npr_module_bind_bytes
#endif
static long budget=-1;
static unsigned requests, live, callbacks;
typedef union { max_align_t alignment; size_t size; } Header;
static void *allocate(size_t size) {
    requests++;
    if(budget==0)return NULL;
    if(budget>0)budget--;
    assert(size<=SIZE_MAX-sizeof(Header));
    Header *header=malloc(sizeof(*header)+size);
    assert(header);
    header->size=size;
    live++;
    return header+1;
}
static void release(void *pointer) {
    if(!pointer)return;
    assert(live);
    live--;
    free((Header *)pointer-1);
}
void *nano_core_malloc(size_t size) { return allocate(size); }
void nano_core_free(void *pointer) { release(pointer); }
void *nano_scratch_malloc(size_t size) { return allocate(size); }
void nano_scratch_free(void *pointer) { release(pointer); }
static int32_t read_text(void *context,const uint8_t *path,uint32_t length,
                         uint8_t *out,uint32_t capacity,uint32_t *written) {
    (void)context;
    assert(length && path && capacity>=6);
    callbacks++;
    memcpy(out,"copied",6);
    *written=6;
    return NPR_OK;
}
int main(int argc,char **argv) {
    assert(argc==2);
    long requested=strtol(argv[1],NULL,10);
    NprHostBinding binding={read_text,&callbacks};
    assert(npr_module_bind(&binding)==NPR_OK);
    budget=requested;
    uint64_t result=nano_try_entry();
    unsigned count=requests, effects=callbacks;
    if(requested<0)assert(result==6);
    else {
        assert((result>>32)==3 || (result>>32)==6);
        if(npr_module_host_status()==NPR_MEMORY)assert(!effects);
    }
    assert(nms_module_live_objects()==0 && nms_module_live_bytes()==0);
    budget=-1;
    assert(nano_try_entry()==6 && npr_module_host_status()==NPR_OK);
    assert(nms_module_live_objects()==0 && nms_module_live_bytes()==0);
    assert(npr_module_bind(NULL)==NPR_OK);
    assert(nano_dispose()==0 && live==0);
    printf("requests=%u effects=%u\n",count,effects);
    return 0;
}
