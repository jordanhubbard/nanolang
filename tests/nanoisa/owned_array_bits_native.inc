#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static size_t live;
static FILE *output;
static void *bits_malloc(size_t n){void *p=malloc(n);if(p)live++;return p;}
static void *bits_calloc(size_t n,size_t s){void *p=calloc(n,s);if(p)live++;return p;}
static void bits_free(void *p){if(p){assert(live);live--;free(p);}}
#define malloc bits_malloc
#define calloc bits_calloc
#define free bits_free
#undef stdout
#define stdout output
#define NVM2C_NO_MAIN
#include "GENERATED"
#undef malloc
#undef calloc
#undef free
static const uint64_t patterns[]={UINT64_C(0),UINT64_C(0x8000000000000000),UINT64_C(0x7ff8000000000042),UINT64_C(0xfff8000000000123)};
static void observe(NmsRuntime *r,NmsHandle array,unsigned count,int mutated){
    uint32_t length=0;assert(nms_value_array_length(r,array,&length)==NMS_OK && length==count);
    for(unsigned i=0;i<count;i++){NmsValue v={0};assert(nms_value_array_get(r,array,i,&v)==NMS_OK);uint64_t expected=mutated?(i==0?patterns[3]:i<4?patterns[i]:patterns[(i-4)%4]):patterns[i];assert(v.tag==3 && v.payload==expected);assert(nms_value_release(r,v)==NMS_OK);}
}
/* I label this separately from the public entry below: these are actual emitted
 * helpers in the fully admitted module, with their existing private signature. */
static void carrier_probe(void){
    NmsRuntime managed;nms_init(&managed,NULL,0);assert(nms_bind_records(&managed,NULL,0)==NMS_OK);assert(nms_begin(&managed)==NMS_OK && managed.active);uint64_t generation=0;
    nown_value owner={0},argument={0},relayed={0},result={0};
    assert(nown_function_1(&argument,&generation,++generation,&owner,&managed)==0);
    assert(!argument.tag && owner.tag==8 && owner.category==1 && owner.record && owner.record->layout==1 && owner.record->count==3);
    nown_value alias=owner.record->fields[1];assert(alias.tag==7 && alias.category==2 && alias.managed==&managed);assert(nown_retain(alias));observe(&managed,alias.handle,4,0);
    assert(nown_function_2(&owner,&generation,++generation,&relayed,&managed)==0 && !owner.tag);
    assert(relayed.record && relayed.record->layout==1 && relayed.record->fields[1].handle==alias.handle);observe(&managed,alias.handle,4,0);
    assert(nown_function_3(&relayed,&generation,++generation,&result,&managed)==0 && !relayed.tag && result.tag==1 && result.scalar==0);observe(&managed,alias.handle,9,1);nown_release(result);
    const uint64_t indices[]={UINT64_MAX,UINT64_C(0x8000000000000000),INT64_MAX};
    for(unsigned i=0;i<3;i++){NmsValue v={99,99};assert(nms_value_array_get(&managed,alias.handle,indices[i],&v)==NMS_OK && !v.tag && !v.payload);assert(nms_value_array_set(&managed,alias.handle,indices[i],(NmsValue){patterns[0],3})==NMS_STATE);observe(&managed,alias.handle,9,1);}
    nown_release(alias);assert(!managed.live_objects && !managed.live_bytes);assert(nms_finish(&managed,NMS_OK,0)==0 && !managed.active);assert(nms_dispose(&managed)==NMS_OK);assert(!live);
    fprintf(stderr,"I retain native internal carrier bit observations separately from public entry.\n");
}
static void public_run(void){
    output=tmpfile();assert(output);int64_t result=-91;int status=nvm_owned_entry(&result);assert(!live);assert(!fflush(output));rewind(output);char text[32]={0};size_t n=fread(text,1,sizeof text-1,output);assert(!ferror(output) && n<sizeof text-1);assert(!fclose(output));output=NULL;
    unsigned group=CASE?(CASE-1)/3:0;int wanted=CASE && group?3:0;fprintf(stderr,"native public bit-boundary case=%d status=%d result=%lld\n",CASE,status,(long long)result);assert(status==wanted && result==(wanted?-91:0));assert(!strcmp(text,wanted?"7\n":"7\n9\n"));
}
int main(void){public_run();if(!CASE)carrier_probe();public_run();return 0;}
