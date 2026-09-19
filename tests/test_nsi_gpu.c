/* I run each selected case in a fresh process. Hooks wrap actual CUDA calls;
 * injected before-call failures do not pretend that release happened. */
#define _DEFAULT_SOURCE 1
#include <dlfcn.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void *test_dlopen(const char *,int);
static void *test_dlsym(void *,const char *);
static int test_dlclose(void *);
static void *test_malloc(size_t);
static void *test_calloc(size_t,size_t);
#define dlopen test_dlopen
#define dlsym test_dlsym
#define dlclose test_dlclose
#define malloc test_malloc
#define calloc test_calloc
#include "../src/nsi_gpu.c"
#include "../src/nsi_cap.c"
#undef dlopen
#undef dlsym
#undef dlclose
#undef malloc
#undef calloc

enum { INIT,VERSION,COUNT,DEVICE,NAME,UUID,CREATE,PUSH,POP,CURRENT,SYNC,DESTROY,ALLOC,RELEASE,WRITE,READ,OPS };
static const char *op_names[]={"init","version","count","device","name","uuid","create","push","pop","current","sync","destroy","alloc","release","write","read"};
static unsigned calls[OPS],checks,real_creates,real_destroys,real_allocs,real_frees,recovery_destroys;
static unsigned loader_calls,symbol_calls,loader_closes;
static int fail_loader,fail_symbol,fail_loader_close;
static long allocation_budget=-1;
static GpuApi real_api;
static GpuContext acquired[32];static bool context_live[32];
typedef struct { int op;unsigned call;bool after;int error; } Fault;
static Fault faults[4];static unsigned fault_count;
#define CHECK(x) do { checks++;if(!(x)){fprintf(stderr,"FAIL line%d case=%s: %s\n",__LINE__,case_name,#x);exit(1);} } while(0)
static const char *case_name;
static void fault(int op,unsigned nth,bool after,int error){faults[fault_count++]=(Fault){op,calls[op]+nth,after,error};}
static int before(int op){calls[op]++;for(unsigned i=0;i<fault_count;i++)if(faults[i].op==op && faults[i].call==calls[op] && !faults[i].after)return faults[i].error;return 0;}
static int after(int op,int rc){for(unsigned i=0;i<fault_count;i++)if(faults[i].op==op && faults[i].call==calls[op] && faults[i].after)return faults[i].error;return rc;}
#define WRAP(op,expression) do { int injected=before(op);if(injected)return injected;int result=(expression);return after(op,result); } while(0)
static int w_init(unsigned int x){WRAP(INIT,real_api.init(x));}
static int w_version(int *x){WRAP(VERSION,real_api.version(x));}
static int w_count(int *x){WRAP(COUNT,real_api.count(x));}
static int w_device(GpuDevice *x,int y){WRAP(DEVICE,real_api.device(x,y));}
static int w_name(char *x,int y,GpuDevice z){WRAP(NAME,real_api.name(x,y,z));}
static int w_uuid(GpuUuid *x,GpuDevice y){WRAP(UUID,real_api.uuid(x,y));}
static int w_create(GpuContext *x,unsigned int y,GpuDevice z){
    int injected=before(CREATE);if(injected)return injected;int rc=real_api.create(x,y,z);
    if(!rc){CHECK(real_creates<32);acquired[real_creates]=*x;context_live[real_creates++]=true;}return after(CREATE,rc);
}
static int w_push(GpuContext x){WRAP(PUSH,real_api.push(x));}
static int w_pop(GpuContext *x){WRAP(POP,real_api.pop(x));}
static int w_current(GpuContext *x){WRAP(CURRENT,real_api.current(x));}
static int w_sync(void){WRAP(SYNC,real_api.sync());}
static int w_destroy(GpuContext x){
    int injected=before(DESTROY);if(injected)return injected;int rc=real_api.destroy(x);
    if(!rc){real_destroys++;for(unsigned i=0;i<real_creates;i++)if(acquired[i]==x && context_live[i]){context_live[i]=false;break;}}return after(DESTROY,rc);
}
static int w_alloc(GpuPointer *x,size_t n){int injected=before(ALLOC);if(injected)return injected;int rc=real_api.alloc(x,n);if(!rc)real_allocs++;return after(ALLOC,rc);}
static int w_release(GpuPointer x){int injected=before(RELEASE);if(injected)return injected;int rc=real_api.release(x);if(!rc)real_frees++;return after(RELEASE,rc);}
static int w_write(GpuPointer x,const void *y,size_t n){WRAP(WRITE,real_api.write(x,y,n));}
static int w_read(void *x,GpuPointer y,size_t n){WRAP(READ,real_api.read(x,y,n));}
static GpuApi wrapped={w_init,w_version,w_count,w_device,w_name,w_uuid,w_create,w_push,w_pop,w_current,w_sync,w_destroy,w_alloc,w_release,w_write,w_read};
static void *test_dlopen(const char *s,int flags){loader_calls++;return fail_loader?NULL:dlopen(s,flags);}
static void *test_dlsym(void *handle,const char *name){
    symbol_calls++;if(fail_symbol==(int)symbol_calls)return NULL;void *real=dlsym(handle,name);if(!real)return NULL;
#define SYMBOL(field,symbol) if(!strcmp(name,symbol)){void *p=NULL;memcpy(&real_api.field,&real,sizeof real);memcpy(&p,&wrapped.field,sizeof p);return p;}
    SYMBOL(init,"cuInit") SYMBOL(version,"cuDriverGetVersion") SYMBOL(count,"cuDeviceGetCount") SYMBOL(device,"cuDeviceGet")
    SYMBOL(name,"cuDeviceGetName") SYMBOL(uuid,"cuDeviceGetUuid_v2") SYMBOL(create,"cuCtxCreate_v2") SYMBOL(push,"cuCtxPushCurrent_v2")
    SYMBOL(pop,"cuCtxPopCurrent_v2") SYMBOL(current,"cuCtxGetCurrent") SYMBOL(sync,"cuCtxSynchronize") SYMBOL(destroy,"cuCtxDestroy_v2")
    SYMBOL(alloc,"cuMemAlloc_v2") SYMBOL(release,"cuMemFree_v2") SYMBOL(write,"cuMemcpyHtoD_v2") SYMBOL(read,"cuMemcpyDtoH_v2")
#undef SYMBOL
    return real;
}
static int test_dlclose(void *h){loader_closes++;if(fail_loader_close)return -1;return dlclose(h);}
static void *test_malloc(size_t n){if(!allocation_budget)return NULL;if(allocation_budget>0)allocation_budget--;return malloc(n);}
static void *test_calloc(size_t n,size_t z){if(!allocation_budget)return NULL;if(allocation_budget>0)allocation_budget--;return calloc(n,z);}
static NlGpuService *open_service(void){NlGpuService *s=NULL;NlGpuResult r=nl_gpu_service_create(0,&s);if(r.status!=NL_GPU_OK)fprintf(stderr,"actual GPU create status=%d driver=%d cleanup=%d\n",r.status,r.driver_error,r.cleanup_error);CHECK(r.status==NL_GPU_OK && s);return s;}
static NlGpuToken allocate(NlGpuService *s){NlGpuToken t;CHECK(nl_gpu_buffer_allocate(s,257,GPU_RIGHTS,&t).status==NL_GPU_OK);return t;}
static NlGpuAllocation row(NlGpuService *s){NlGpuDiagnostics d=nl_gpu_service_diagnostics();for(unsigned i=0;i<d.records;i++)if(d.lifetimes[i].record_id==s->lifetime->identity)return d.lifetimes[i].allocations_by_slot[0];CHECK(false);return (NlGpuAllocation){0};}
static void cleanup_external(void){
    /* I directly recover only contexts my hook observed as still live. This is
     * fixture cleanup, never an adapter retry or a corrected adapter outcome. */
    for(unsigned i=0;i<real_creates;i++)if(context_live[i]){CHECK(real_api.destroy(acquired[i])==0);context_live[i]=false;recovery_destroys++;}
}
static void normal(void){
    NlGpuService *s=open_service(),*t=open_service();NlGpuDevice d;CHECK(nl_gpu_service_device(s,&d).status==NL_GPU_OK && d.cuda_gpu);
    printf("actual backend=CUDA class=GPU driver=%d name=%s uuid=",d.driver_version,d.name);for(unsigned i=0;i<16;i++)printf("%02x",d.uuid[i]);puts("");
    void *foreign_library=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);CHECK(foreign_library);
    GpuDevice dev;CHECK(real_api.device(&dev,0)==0);GpuContext foreign=NULL,prior=NULL,now=NULL;CHECK(real_api.current(&prior)==0);
    CHECK(real_api.create(&foreign,0,dev)==0);NlGpuToken a=allocate(s),b=allocate(s),original=a;
    CHECK(real_api.current(&now)==0 && now==foreign);
    unsigned char input[257],out[257];for(unsigned i=0;i<257;i++)input[i]=(unsigned char)(i*37);
    CHECK(nl_gpu_buffer_write(s,&a,0,input,257).status==NL_GPU_OK);
    CHECK(nl_gpu_buffer_write(s,&b,0,input,257).status==NL_GPU_OK);
    memset(out,0,257);CHECK(nl_gpu_buffer_read(s,&a,0,out,257).status==NL_GPU_OK && !memcmp(input,out,257));
    CHECK(nl_gpu_buffer_read(s,&a,13,out,29).status==NL_GPU_OK && !memcmp(input+13,out,29));
    unsigned saved=calls[READ];CHECK(nl_gpu_buffer_read(s,&a,257,NULL,0).status==NL_GPU_OK && calls[READ]==saved);
    memset(out,0xaa,257);CHECK(nl_gpu_buffer_read(s,&a,257,out,1).status==NL_GPU_ARGUMENT && out[0]==0xaa);
    CHECK(nl_gpu_buffer_read(t,&a,0,out,1).status==NL_GPU_TOKEN && calls[READ]==saved);
    CHECK(nl_gpu_buffer_read(s,&a,0,(unsigned char *)&a,1).status==NL_GPU_ARGUMENT);
    CHECK(nl_gpu_buffer_read(s,&a,0,((unsigned char *)&a)+sizeof a-1,1).status==NL_GPU_ARGUMENT);
    CHECK(!memcmp(&a,&original,sizeof a));
    CHECK(nl_gpu_buffer_transfer(s,&a,&a).status==NL_GPU_OK);
    CHECK(nl_gpu_buffer_close(s,&original).status==NL_GPU_TOKEN);
    CHECK(nl_gpu_buffer_close(s,&a).status==NL_GPU_OK);CHECK(nl_gpu_buffer_close(s,&a).status==NL_GPU_TOKEN);
    CHECK(nl_gpu_buffer_read(s,&b,0,out,257).status==NL_GPU_OK && !memcmp(out,input,257));
    CHECK(nl_gpu_buffer_close(s,&b).status==NL_GPU_OK);
    NlGpuToken ro;CHECK(nl_gpu_buffer_allocate(s,1,NL_CAP_READ,&ro).status==NL_GPU_OK);
    saved=calls[WRITE];CHECK(nl_gpu_buffer_write(s,&ro,0,input,1).status==NL_GPU_RIGHTS && calls[WRITE]==saved);
    CHECK(nl_gpu_buffer_transfer(s,&ro,&a).status==NL_GPU_RIGHTS);CHECK(nl_gpu_buffer_close(s,&ro).status==NL_GPU_OK);
    for(unsigned i=0;i<96;i++){a=allocate(s);CHECK(nl_gpu_buffer_close(s,&a).status==NL_GPU_OK);}
    NlGpuToken full[64];for(unsigned i=0;i<64;i++)full[i]=allocate(s);
    a=full[0];CHECK(nl_gpu_buffer_transfer(s,&full[0],&full[0]).status==NL_GPU_CAPACITY && !memcmp(&a,&full[0],sizeof a));
    CHECK(nl_gpu_buffer_allocate(s,1,GPU_RIGHTS,&a).status==NL_GPU_CAPACITY);
    CHECK(nl_gpu_buffer_close(s,&full[1]).status==NL_GPU_OK);
    CHECK(nl_gpu_buffer_transfer(s,&full[0],&full[0]).status==NL_GPU_OK);
    CHECK(real_api.current(&now)==0 && now==foreign);
    CHECK(nl_gpu_service_destroy(t).status==NL_GPU_OK);CHECK(nl_gpu_service_destroy(s).status==NL_GPU_OK);
    CHECK(real_api.current(&now)==0 && now==foreign);CHECK(real_api.destroy(foreign)==0);
    CHECK(real_api.current(&now)==0 && now==prior);CHECK(dlclose(foreign_library)==0);CHECK(nl_gpu_service_diagnostics().records==0);
}
static void admission_fault(const char *arg){
    NlGpuService *sentinel=(NlGpuService *)(uintptr_t)1,*s=sentinel;
    if(!strcmp(arg,"loader"))fail_loader=1;
    else if(!strncmp(arg,"symbol",6))fail_symbol=atoi(arg+6);
    else if(!strncmp(arg,"host",4))allocation_budget=atoi(arg+4);
    else {int op=-1;for(int i=0;i<OPS;i++)if(!strcmp(arg,op_names[i]))op=i;CHECK(op>=0);fault(op,1,false,701);}
    NlGpuResult r=nl_gpu_service_create(0,&s);CHECK(r.status!=NL_GPU_OK && s==sentinel);
    CHECK(!nl_gpu_service_diagnostics().records);CHECK(!real_creates);
}
static void create_terminal(const char *arg){
    if(!strcmp(arg,"post"))fault(CREATE,1,true,701);
    else if(!strcmp(arg,"popbefore"))fault(POP,1,false,701);
    else if(!strcmp(arg,"popafter"))fault(POP,1,true,701);
    else {CHECK(!strcmp(arg,"query"));fault(CURRENT,2,false,701);}
    NlGpuService *s=(NlGpuService *)(uintptr_t)1;NlGpuResult r=nl_gpu_service_create(0,&s);
    CHECK(r.status!=NL_GPU_OK && s==(NlGpuService *)(uintptr_t)1 && r.context_restore_unknown);
    NlGpuDiagnostics d=nl_gpu_service_diagnostics();CHECK(d.records==1 && d.quarantined==1 && d.acquisition_latched);
    unsigned init_calls=calls[INIT];CHECK(nl_gpu_service_create(0,&s).status==NL_GPU_TERMINAL && calls[INIT]==init_calls);
}
static void operation_fault(const char *arg){
    NlGpuService *s=open_service();NlGpuToken t;NlGpuResult r;unsigned char bytes[257],out[257];memset(bytes,0x37,257);memset(out,0xaa,257);
    if(!strncmp(arg,"pop-",4)) {
        NlGpuToken sentinel;memset(&sentinel,0xa5,sizeof sentinel);t=sentinel;
        fault(POP,1,!strcmp(arg,"pop-after"),707);
        r=nl_gpu_buffer_allocate(s,257,GPU_RIGHTS,&t);CHECK(r.status==NL_GPU_DRIVER && r.context_restore_unknown && !memcmp(&t,&sentinel,sizeof t));
        NlGpuAllocation x=row(s);CHECK(x.disposition==NL_GPU_RELEASE_SKIPPED && !x.release_attempted && !calls[RELEASE]);
        r=nl_gpu_service_dispose(s);x=row(s);CHECK(r.context_destroyed && x.context_reclaimed && !calls[RELEASE]);
    }else if(!strcmp(arg,"loader-close")) {
        fail_loader_close=1;r=nl_gpu_service_destroy(s);s=NULL;
        CHECK(r.context_destroyed && r.library_retained && nl_gpu_service_diagnostics().quarantined==1);
    }else if(!strncmp(arg,"alloc",5) || !strcmp(arg,"rollback")){
        NlGpuToken sentinel;memset(&sentinel,0xa5,sizeof sentinel);t=sentinel;
        bool post=strcmp(arg,"allocbefore")!=0;fault(ALLOC,1,post,701);
        if(!strcmp(arg,"rollback"))fault(CURRENT,3,false,702);
        r=nl_gpu_buffer_allocate(s,257,GPU_RIGHTS,&t);CHECK(r.status==NL_GPU_DRIVER && r.driver_error==701 && !memcmp(&t,&sentinel,sizeof t));
        if(!strcmp(arg,"rollback")){
            NlGpuAllocation x=row(s);CHECK(r.cleanup_error==702 && x.disposition==NL_GPU_RELEASE_SKIPPED && !x.release_attempted && x.skipped_error==702);
            CHECK(calls[RELEASE]==0);r=nl_gpu_service_dispose(s);x=row(s);
            CHECK(r.context_destroyed && x.context_reclaimed && x.disposition==NL_GPU_RELEASE_SKIPPED && calls[RELEASE]==0);
        }else if(post){NlGpuAllocation x=row(s);CHECK(x.disposition==NL_GPU_RELEASE_FREED && x.release_attempted && x.release_error==0 && calls[RELEASE]==1);}
    }
    else {
        t=allocate(s);CHECK(nl_gpu_buffer_write(s,&t,0,bytes,257).status==NL_GPU_OK);
        if(!strncmp(arg,"free",4)){
            bool post=!strcmp(arg,"freeafter");fault(RELEASE,1,post,703);r=nl_gpu_buffer_close(s,&t);
            NlGpuAllocation x=row(s);CHECK(r.consumed && r.release_unknown && x.disposition==NL_GPU_RELEASE_FAILED && x.release_attempted && x.release_error==703);
            CHECK(real_frees==(post?1u:0u));unsigned released=calls[RELEASE];r=nl_gpu_service_dispose(s);x=row(s);
            CHECK(r.context_destroyed && x.context_reclaimed && x.disposition==NL_GPU_RELEASE_FAILED && calls[RELEASE]==released);
        }else if(!strcmp(arg,"closepush")){
            fault(PUSH,1,false,704);r=nl_gpu_buffer_close(s,&t);NlGpuAllocation x=row(s);
            CHECK(r.consumed && x.disposition==NL_GPU_RELEASE_SKIPPED && !x.release_attempted && x.skipped_error==704 && !calls[RELEASE]);
        }else if(!strncmp(arg,"destroy",7)){
            fault(DESTROY,1,!strcmp(arg,"destroyafter"),705);r=nl_gpu_service_dispose(s);CHECK(r.context_unknown && r.context_destroy_attempts==1);
            unsigned destroyed=calls[DESTROY];r=nl_gpu_service_destroy(s);s=NULL;CHECK(r.context_unknown && calls[DESTROY]==destroyed);
            CHECK(nl_gpu_service_diagnostics().quarantined==1);
        }else if(!strcmp(arg,"staging")){
            allocation_budget=0;unsigned reads=calls[READ];r=nl_gpu_buffer_read(s,&t,0,out,257);CHECK(r.status==NL_GPU_MEMORY && calls[READ]==reads && out[0]==0xaa);allocation_budget=-1;
        }else if(!strcmp(arg,"generation")){
            s->caps->next_generation=UINT32_MAX;unsigned allocated=calls[ALLOC];NlGpuToken untouched=t;r=nl_gpu_buffer_allocate(s,1,GPU_RIGHTS,&untouched);
            CHECK(r.status==NL_GPU_LIMIT && calls[ALLOC]==allocated && !memcmp(&untouched,&t,sizeof t));
        }else {
            bool writing=!strncmp(arg,"write",5);bool sync=!strcmp(arg,"readsync") || !strcmp(arg,"writesync");
            int op=sync?SYNC:(writing?WRITE:READ);fault(op,1,strstr(arg,"after")!=NULL,706);
            r=writing?nl_gpu_buffer_write(s,&t,0,bytes,257):nl_gpu_buffer_read(s,&t,0,out,257);
            CHECK(r.status==NL_GPU_DRIVER && r.driver_error==706);
            if(writing)CHECK(nl_gpu_buffer_read(s,&t,0,out,1).status==NL_GPU_TERMINAL);
            else {for(unsigned i=0;i<257;i++)CHECK(out[i]==0xaa);}
        }
    }
    if(s)(void)nl_gpu_service_destroy(s);
}
int main(int argc,char **argv){
    CHECK(argc==2);case_name=argv[1];printf("CASE %s\n",case_name);fflush(stdout);
    if(!strcmp(case_name,"normal"))normal();
    else if(!strcmp(case_name,"contexts")) {
        NlGpuService *services[8];for(unsigned i=0;i<8;i++)services[i]=open_service();
        unsigned loads=loader_calls;NlGpuService *out=(NlGpuService *)(uintptr_t)1;
        CHECK(nl_gpu_service_create(0,&out).status==NL_GPU_CAPACITY && out==(NlGpuService *)(uintptr_t)1 && loader_calls==loads);
        for(unsigned i=0;i<8;i++)CHECK(nl_gpu_service_destroy(services[i]).status==NL_GPU_OK);
        CHECK(nl_gpu_service_diagnostics().records==0);
    }
    else if(!strncmp(case_name,"admit-",6))admission_fault(case_name+6);
    else if(!strncmp(case_name,"create-",7))create_terminal(case_name+7);
    else operation_fault(case_name);
    cleanup_external();NlGpuDiagnostics d=nl_gpu_service_diagnostics();
    printf("PASS checks=%u real_contexts=%u real_destroyed=%u real_allocs=%u real_frees=%u fixture_recovery_contexts=%u records=%u quarantine=%u latch=%u\n",checks,real_creates,real_destroys,real_allocs,real_frees,recovery_destroys,d.records,d.quarantined,d.acquisition_latched);
    for(unsigned i=0;i<d.records;i++)printf("record=%llu unknown_context=%u restore=%u release=%u tracked=%u unresolved=%u\n",(unsigned long long)d.lifetimes[i].record_id,d.lifetimes[i].context_unknown,d.lifetimes[i].context_restore_unknown,d.lifetimes[i].release_unknown,d.lifetimes[i].tracked_allocations,d.lifetimes[i].unresolved_allocations);
    return 0;
}
