#include "nsi_gpu.h"
#include "nsi_cap_private.h"
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define GPU_TYPE "nsi:nanolang/gpu#Buffer"
#define GPU_SERVICE "nsi:nanolang/gpu/private-cuda-buffer"
#define GPU_RIGHTS (NL_CAP_READ | NL_CAP_WRITE | NL_CAP_TRANSFER)
#define GPU_HOST_CONTRACT (-1)
#define GPU_HOST_LOADER (-2)

/* I match the pinned CUDA13 Linux LP64 ABI, including explicit v2 symbols.
 * Other hosts compile the unavailable path and never call this ABI. */
typedef int GpuResult;
typedef int GpuDevice;
typedef unsigned long long GpuPointer;
typedef struct CUctx_st *GpuContext;
typedef struct { char bytes[16]; } GpuUuid;
_Static_assert(sizeof(GpuPointer)==8 && sizeof(GpuUuid)==16 && sizeof(int)==4,
               "I require the reviewed CUDA storage ABI");
typedef struct {
    GpuResult (*init)(unsigned int);
    GpuResult (*version)(int *);
    GpuResult (*count)(int *);
    GpuResult (*device)(GpuDevice *, int);
    GpuResult (*name)(char *, int, GpuDevice);
    GpuResult (*uuid)(GpuUuid *, GpuDevice);
    GpuResult (*create)(GpuContext *, unsigned int, GpuDevice);
    GpuResult (*push)(GpuContext);
    GpuResult (*pop)(GpuContext *);
    GpuResult (*current)(GpuContext *);
    GpuResult (*sync)(void);
    GpuResult (*destroy)(GpuContext);
    GpuResult (*alloc)(GpuPointer *, size_t);
    GpuResult (*release)(GpuPointer);
    GpuResult (*write)(GpuPointer, const void *, size_t);
    GpuResult (*read)(void *, GpuPointer, size_t);
} GpuApi;
typedef struct {
    bool occupied, live, contents_unknown, release_attempted, context_reclaimed;
    NlGpuReleaseDisposition disposition;
    int release_error, skipped_error;
    uint64_t allocation_id;
    GpuPointer pointer;
    size_t bytes;
    NlCap token;
} GpuBuffer;
typedef struct {
    bool reserved, quarantined, faulted, release_unknown, restore_unknown;
    bool context_unknown, destroy_attempted, context_destroyed;
    int first_error;
    uint64_t identity;
    void *library;
    GpuApi api;
    GpuContext context;
    uint64_t allocations, free_attempts, freed_count, skipped_count;
    GpuBuffer buffers[NL_GPU_BUFFER_LIMIT];
} GpuLifetime;
struct NlGpuService {
    GpuLifetime *lifetime;
    NlCapTable *caps;
    NlGpuDevice device;
    bool disposed;
};
typedef struct { GpuContext prior; bool active; } GpuBracket;
/* I keep unknown resources here after unpublished failure/wrapper destruction.
 * Serialization is a C API precondition, not an internal concurrency promise. */
static GpuLifetime gpu_lifetimes[NL_GPU_CONTEXT_LIMIT];
static uint64_t gpu_next_identity;
static bool gpu_acquisition_latched;

static NlGpuResult gpu_result(NlGpuStatus status) {
    NlGpuResult r={0};r.status=status;return r;
}
static void gpu_error(NlGpuResult *r,int code) {
    if(r->status==NL_GPU_OK){r->status=NL_GPU_DRIVER;r->driver_error=code;}
    else if(!r->cleanup_failed){r->cleanup_failed=true;r->cleanup_error=code;}
}
static void gpu_uncertain(GpuLifetime *life,int code,bool restoration) {
    gpu_acquisition_latched=true;life->faulted=true;
    if(!life->first_error)life->first_error=code?code:GPU_HOST_CONTRACT;
    if(restoration)life->restore_unknown=true;
    else life->release_unknown=true;
}
static NlGpuResult gpu_report(GpuLifetime *life,NlGpuResult r) {
    r.record_id=life->identity;r.release_unknown=life->release_unknown;
    r.context_restore_unknown=life->restore_unknown;
    r.context_unknown=life->context_unknown;
    r.context_destroyed=life->context_destroyed;
    r.library_retained=life->library!=NULL;
    if(r.status==NL_GPU_OK && (life->release_unknown || life->restore_unknown || life->context_unknown)) {
        r.status=NL_GPU_DRIVER;r.driver_error=life->first_error?life->first_error:GPU_HOST_CONTRACT;
    }
    return r;
}
static NlGpuStatus gpu_cap_status(int code) {
    switch(code){
    case NL_CAP_OK:return NL_GPU_OK;
    case NL_CAP_ERR_RIGHTS:case NL_CAP_ERR_TRANSFER:return NL_GPU_RIGHTS;
    case NL_CAP_ERR_FULL:return NL_GPU_CAPACITY;
    case NL_CAP_PRIVATE_ERR_GENERATION:return NL_GPU_LIMIT;
    default:return NL_GPU_TOKEN;
    }
}
static NlGpuStatus gpu_context(const NlGpuService *s) {
    if(!s)return NL_GPU_ARGUMENT;
    return s->disposed?NL_GPU_DISPOSED:NL_GPU_OK;
}
static bool gpu_same_cap(NlCap a,NlCap b) {
    return a.slot==b.slot && a.generation==b.generation && a.secret==b.secret;
}
static NlGpuStatus gpu_resolve(NlGpuService *s,const NlGpuToken *token,uint32_t rights,
                              GpuBuffer **out) {
    NlGpuStatus status=gpu_context(s);if(status!=NL_GPU_OK)return status;
    if(!token)return NL_GPU_ARGUMENT;
    GpuLifetime *life=s->lifetime;
    if(token->context_id!=life->identity || token->cap.slot>=NL_CAP_PRIVATE_SLOTS)return NL_GPU_TOKEN;
    int rc=nl_cap_check(s->caps,&token->cap,rights);if(rc!=NL_CAP_OK)return gpu_cap_status(rc);
    const char *type=nl_cap_type_id(s->caps,&token->cap),*service=nl_cap_service_id(s->caps,&token->cap);
    if(!type || !service || strcmp(type,GPU_TYPE) || strcmp(service,GPU_SERVICE))return NL_GPU_TOKEN;
    for(unsigned i=0;i<NL_GPU_BUFFER_LIMIT;i++) {
        GpuBuffer *entry=&life->buffers[i];
        if(entry->occupied && entry->live && gpu_same_cap(entry->token,token->cap)){*out=entry;return NL_GPU_OK;}
    }
    return NL_GPU_TOKEN;
}
static bool gpu_range(const void *p,size_t n) {
    return !n || (p && n-1<=UINTPTR_MAX-(uintptr_t)p);
}
static bool gpu_overlap(const void *a,size_t na,const void *b,size_t nb) {
    if(!na || !nb)return false;
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<=y?y-x<na:x-y<nb;
}
static bool gpu_load(GpuLifetime *life) {
#if defined(__linux__) && defined(__LP64__)
    _Static_assert(sizeof(void *)==8,"I require LP64 CUDA pointers");
    life->library=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    if(!life->library)return false;
#define GPU_LOAD(field,symbol) do { \
    void *address=dlsym(life->library,symbol); \
    _Static_assert(sizeof life->api.field==sizeof address,"I require POSIX function pointer storage"); \
    if(!address)return false; \
    memcpy(&life->api.field,&address,sizeof address); \
} while(0)
    GPU_LOAD(init,"cuInit");GPU_LOAD(version,"cuDriverGetVersion");
    GPU_LOAD(count,"cuDeviceGetCount");GPU_LOAD(device,"cuDeviceGet");
    GPU_LOAD(name,"cuDeviceGetName");GPU_LOAD(uuid,"cuDeviceGetUuid_v2");
    GPU_LOAD(create,"cuCtxCreate_v2");GPU_LOAD(push,"cuCtxPushCurrent_v2");
    GPU_LOAD(pop,"cuCtxPopCurrent_v2");GPU_LOAD(current,"cuCtxGetCurrent");
    GPU_LOAD(sync,"cuCtxSynchronize");GPU_LOAD(destroy,"cuCtxDestroy_v2");
    GPU_LOAD(alloc,"cuMemAlloc_v2");GPU_LOAD(release,"cuMemFree_v2");
    GPU_LOAD(write,"cuMemcpyHtoD_v2");GPU_LOAD(read,"cuMemcpyDtoH_v2");
#undef GPU_LOAD
    return true;
#else
    (void)life;return false;
#endif
}
static void gpu_end(GpuLifetime *life,GpuBracket *bracket,NlGpuResult *r) {
    if(!bracket->active)return;
    bracket->active=false;GpuContext popped=NULL,current=NULL;
    int rc=life->api.pop(&popped);
    if(rc || popped!=life->context) {
        gpu_error(r,rc?rc:GPU_HOST_CONTRACT);gpu_uncertain(life,rc,true);
    }
    rc=life->api.current(&current);
    if(rc || current!=bracket->prior) {
        gpu_error(r,rc?rc:GPU_HOST_CONTRACT);gpu_uncertain(life,rc,true);
    }
}
static bool gpu_begin(GpuLifetime *life,GpuBracket *bracket,NlGpuResult *r) {
    *bracket=(GpuBracket){0};
    int rc=life->api.current(&bracket->prior);
    if(rc){gpu_error(r,rc);return false;}
    rc=life->api.push(life->context);
    if(rc){gpu_error(r,rc);gpu_uncertain(life,rc,true);return false;}
    bracket->active=true;GpuContext current=NULL;
    rc=life->api.current(&current);
    if(rc || current!=life->context) {
        gpu_error(r,rc?rc:GPU_HOST_CONTRACT);gpu_uncertain(life,rc,true);
        gpu_end(life,bracket,r);return false;
    }
    return true;
}
static void gpu_skip_free(GpuLifetime *life,GpuBuffer *entry,int code) {
    if(!entry->occupied || entry->disposition!=NL_GPU_RELEASE_PENDING)return;
    entry->disposition=NL_GPU_RELEASE_SKIPPED;
    entry->skipped_error=code?code:GPU_HOST_CONTRACT;
    life->skipped_count++;gpu_uncertain(life,entry->skipped_error,false);
}
static void gpu_free_once(GpuLifetime *life,GpuBuffer *entry,NlGpuResult *r) {
    if(!entry->occupied || entry->disposition!=NL_GPU_RELEASE_PENDING)return;
    entry->release_attempted=true;r->free_attempts++;life->free_attempts++;
    int rc=life->api.release(entry->pointer);entry->release_error=rc;
    if(rc) {
        entry->disposition=NL_GPU_RELEASE_FAILED;
        gpu_error(r,rc);gpu_uncertain(life,rc,false);
    } else {
        entry->disposition=NL_GPU_RELEASE_FREED;
        r->freed_count++;life->freed_count++;
    }
}
/* I never use a failed free pointer again. Explicit context destruction is the
 * separately documented final reclamation operation, not a repeated free. */
static void gpu_destroy_context(GpuLifetime *life,NlGpuResult *r) {
    if(life->context && !life->destroy_attempted) {
        life->destroy_attempted=true;r->context_destroy_attempts++;
        GpuContext prior=NULL,after=NULL;int before=life->api.current(&prior);
        int rc=life->api.destroy(life->context);
        if(rc){gpu_error(r,rc);life->context_unknown=true;gpu_uncertain(life,rc,false);}
        else {
            life->context=NULL;life->context_destroyed=true;
            for(unsigned i=0;i<NL_GPU_BUFFER_LIMIT;i++) {
                GpuBuffer *entry=&life->buffers[i];
                if(entry->occupied && entry->disposition!=NL_GPU_RELEASE_FREED)
                    entry->context_reclaimed=true;
            }
        }
        int query=life->api.current(&after);
        /* Normal destruction sees the caller's context, not my detached one.
         * An already unknown restoration never becomes a success claim here. */
        if(before || query || (!life->restore_unknown && after!=prior)) {
            gpu_error(r,before?before:(query?query:GPU_HOST_CONTRACT));
            gpu_uncertain(life,before?before:query,true);
        }
    }
    if(!life->context && !life->context_unknown && !life->restore_unknown && life->library) {
        if(dlclose(life->library)!=0){gpu_error(r,GPU_HOST_LOADER);gpu_uncertain(life,GPU_HOST_LOADER,false);life->quarantined=true;}
        else life->library=NULL;
    }
}
static void gpu_release_record(GpuLifetime *life) {
    if(life->context || life->library || life->context_unknown || life->restore_unknown)life->quarantined=true;
    else *life=(GpuLifetime){0};
}
static void gpu_rollback_buffer(GpuLifetime *life,GpuBuffer *entry,NlGpuResult *r) {
    if(!entry->occupied || entry->disposition!=NL_GPU_RELEASE_PENDING)return;
    if(life->restore_unknown) {
        gpu_skip_free(life,entry,GPU_HOST_CONTRACT);return;
    }
    GpuBracket bracket;
    /* The allocation already owns the primary error. I separately retain the
     * cleanup bracket's cause before merging its first/secondary diagnostics. */
    NlGpuResult cleanup=gpu_result(NL_GPU_OK);
    if(gpu_begin(life,&bracket,&cleanup)){gpu_free_once(life,entry,r);gpu_end(life,&bracket,r);}
    else {
        gpu_error(r,cleanup.driver_error);
        if(cleanup.cleanup_failed)gpu_error(r,cleanup.cleanup_error);
        gpu_skip_free(life,entry,cleanup.driver_error);
    }
}

NlGpuResult nl_gpu_service_create(int ordinal,NlGpuService **out) {
    if(!out || ordinal<0)return gpu_result(NL_GPU_ARGUMENT);
    if(gpu_acquisition_latched)return gpu_result(NL_GPU_TERMINAL);
    if(gpu_next_identity==UINT64_MAX)return gpu_result(NL_GPU_LIMIT);
    GpuLifetime *life=NULL;
    for(unsigned i=0;i<NL_GPU_CONTEXT_LIMIT;i++)if(!gpu_lifetimes[i].reserved){life=&gpu_lifetimes[i];break;}
    if(!life)return gpu_result(NL_GPU_CAPACITY);
    *life=(GpuLifetime){.reserved=true,.identity=++gpu_next_identity};
    NlGpuResult r=gpu_result(NL_GPU_OK);r.record_id=life->identity;
    NlGpuService *s=calloc(1,sizeof *s);
    if(!s){r.status=NL_GPU_MEMORY;gpu_release_record(life);return r;}
    s->lifetime=life;s->caps=nl_cap_table_create();
    if(!s->caps){r.status=NL_GPU_MEMORY;goto fail;}
    if(!gpu_load(life)){r.status=NL_GPU_UNAVAILABLE;goto fail;}
    int rc=life->api.init(0),count=0;GpuDevice device=0;GpuUuid uuid={{0}};
    if(rc){gpu_error(&r,rc);goto fail;}
    rc=life->api.version(&s->device.driver_version);if(rc){gpu_error(&r,rc);goto fail;}
    rc=life->api.count(&count);if(rc){gpu_error(&r,rc);goto fail;}
    if(count<=ordinal){r.status=NL_GPU_UNAVAILABLE;goto fail;}
    rc=life->api.device(&device,ordinal);if(rc){gpu_error(&r,rc);goto fail;}
    rc=life->api.name(s->device.name,(int)sizeof s->device.name,device);if(rc){gpu_error(&r,rc);goto fail;}
    s->device.name[sizeof s->device.name-1]='\0';
    rc=life->api.uuid(&uuid,device);if(rc){gpu_error(&r,rc);goto fail;}
    memcpy(s->device.uuid,uuid.bytes,sizeof uuid.bytes);s->device.ordinal=ordinal;s->device.cuda_gpu=true;
    GpuBracket bracket={0};rc=life->api.current(&bracket.prior);if(rc){gpu_error(&r,rc);goto fail;}
    rc=life->api.create(&life->context,0,device);
    if(rc || !life->context) {
        gpu_error(&r,rc?rc:GPU_HOST_CONTRACT);
        GpuContext current=NULL;int query=life->api.current(&current);
        if(query || current!=bracket.prior)gpu_uncertain(life,query,true);
        if(!rc && !life->context){life->context_unknown=true;gpu_uncertain(life,GPU_HOST_CONTRACT,false);}
        goto fail;
    }
    bracket.active=true;GpuContext current=NULL;rc=life->api.current(&current);
    if(rc || current!=life->context){gpu_error(&r,rc?rc:GPU_HOST_CONTRACT);gpu_uncertain(life,rc,true);}
    gpu_end(life,&bracket,&r);
    if(r.status!=NL_GPU_OK)goto fail;
    *out=s;return gpu_report(life,r);
fail:
    gpu_destroy_context(life,&r);r=gpu_report(life,r);
    nl_cap_table_destroy(s->caps);free(s);gpu_release_record(life);return r;
}
NlGpuResult nl_gpu_service_device(const NlGpuService *s,NlGpuDevice *out) {
    NlGpuStatus status=gpu_context(s);if(status!=NL_GPU_OK)return gpu_result(status);
    if(!out)return gpu_result(NL_GPU_ARGUMENT);
    if(s->lifetime->faulted)return gpu_report(s->lifetime,gpu_result(NL_GPU_TERMINAL));
    *out=s->device;return gpu_report(s->lifetime,gpu_result(NL_GPU_OK));
}
NlGpuResult nl_gpu_buffer_allocate(NlGpuService *s,size_t bytes,uint32_t rights,NlGpuToken *out) {
    NlGpuStatus status=gpu_context(s);if(status!=NL_GPU_OK)return gpu_result(status);
    if(!out || !bytes || bytes>NL_GPU_BYTE_LIMIT || (rights&~GPU_RIGHTS))return gpu_result(NL_GPU_ARGUMENT);
    GpuLifetime *life=s->lifetime;
    if(gpu_acquisition_latched || life->faulted)return gpu_report(life,gpu_result(NL_GPU_TERMINAL));
    GpuBuffer *entry=NULL;
    for(unsigned i=0;i<NL_GPU_BUFFER_LIMIT;i++)if(!life->buffers[i].occupied || life->buffers[i].disposition==NL_GPU_RELEASE_FREED){entry=&life->buffers[i];break;}
    if(!entry)return gpu_result(NL_GPU_CAPACITY);
    if(life->allocations==UINT64_MAX)return gpu_result(NL_GPU_LIMIT);
    NlCap cap;int rc=nl_cap_private_mint(s->caps,GPU_TYPE,GPU_SERVICE,rights,&cap);
    if(rc!=NL_CAP_OK)return gpu_result(gpu_cap_status(rc));
    NlGpuResult r=gpu_result(NL_GPU_OK);GpuBracket bracket;
    if(gpu_begin(life,&bracket,&r)) {
        GpuPointer pointer=0;rc=life->api.alloc(&pointer,bytes);
        if(pointer)*entry=(GpuBuffer){.occupied=true,.pointer=pointer,.bytes=bytes,.token=cap,
            .allocation_id=++life->allocations,.disposition=NL_GPU_RELEASE_PENDING};
        if(rc || !pointer){gpu_error(&r,rc?rc:GPU_HOST_CONTRACT);if(!rc && !pointer)gpu_uncertain(life,GPU_HOST_CONTRACT,false);}
        gpu_end(life,&bracket,&r);
    }
    if(r.status!=NL_GPU_OK) {
        (void)nl_cap_private_consume(s->caps,&cap);gpu_rollback_buffer(life,entry,&r);return gpu_report(life,r);
    }
    entry->live=true;*out=(NlGpuToken){life->identity,cap};return gpu_report(life,r);
}
static NlGpuResult gpu_copy(NlGpuService *s,const NlGpuToken *token,size_t offset,
                           void *destination,const void *source,size_t count,bool reading) {
    const void *host=reading?destination:source;
    if(!token || !gpu_range(host,count) || !gpu_range(token,sizeof *token) ||
       (reading && gpu_overlap(host,count,token,sizeof *token)))return gpu_result(NL_GPU_ARGUMENT);
    GpuBuffer *entry=NULL;NlGpuStatus status=gpu_resolve(s,token,reading?NL_CAP_READ:NL_CAP_WRITE,&entry);
    if(status!=NL_GPU_OK)return gpu_result(status);
    GpuLifetime *life=s->lifetime;
    if(life->faulted || entry->contents_unknown)return gpu_report(life,gpu_result(NL_GPU_TERMINAL));
    if(offset>entry->bytes || count>entry->bytes-offset || offset>ULLONG_MAX-entry->pointer)
        return gpu_result(NL_GPU_ARGUMENT);
    NlGpuResult r=gpu_result(NL_GPU_OK);if(!count)return gpu_report(life,r);
    void *staging=reading?malloc(count):NULL;
    if(reading && !staging)return gpu_result(NL_GPU_MEMORY);
    GpuBracket bracket;
    if(gpu_begin(life,&bracket,&r)) {
        int rc=reading?life->api.read(staging,entry->pointer+offset,count):life->api.write(entry->pointer+offset,source,count);
        if(rc)gpu_error(&r,rc);
        else {rc=life->api.sync();if(rc)gpu_error(&r,rc);}
        if(rc && !reading)entry->contents_unknown=true;
        gpu_end(life,&bracket,&r);
    }
    if(r.status==NL_GPU_OK){if(reading)memcpy(destination,staging,count);r.bytes=count;}
    free(staging);return gpu_report(life,r);
}
NlGpuResult nl_gpu_buffer_write(NlGpuService *s,const NlGpuToken *token,size_t offset,
                                const void *bytes,size_t count) {
    return gpu_copy(s,token,offset,NULL,bytes,count,false);
}
NlGpuResult nl_gpu_buffer_read(NlGpuService *s,const NlGpuToken *token,size_t offset,
                               void *bytes,size_t count) {
    return gpu_copy(s,token,offset,bytes,NULL,count,true);
}
NlGpuResult nl_gpu_buffer_transfer(NlGpuService *s,const NlGpuToken *token,NlGpuToken *out) {
    if(!out)return gpu_result(NL_GPU_ARGUMENT);
    GpuBuffer *entry=NULL;NlGpuStatus status=gpu_resolve(s,token,NL_CAP_TRANSFER,&entry);
    if(status!=NL_GPU_OK)return gpu_result(status);
    if(s->lifetime->faulted)return gpu_report(s->lifetime,gpu_result(NL_GPU_TERMINAL));
    NlCap next;int rc=nl_cap_private_transfer(s->caps,&entry->token,&next);
    if(rc!=NL_CAP_OK)return gpu_result(gpu_cap_status(rc));
    entry->token=next;*out=(NlGpuToken){s->lifetime->identity,next};
    NlGpuResult r=gpu_result(NL_GPU_OK);r.consumed=true;return gpu_report(s->lifetime,r);
}
NlGpuResult nl_gpu_buffer_close(NlGpuService *s,const NlGpuToken *token) {
    GpuBuffer *entry=NULL;NlGpuStatus status=gpu_resolve(s,token,0,&entry);
    if(status!=NL_GPU_OK)return gpu_result(status);
    NlCap cap=entry->token;entry->live=false;int rc=nl_cap_private_consume(s->caps,&cap);
    if(rc!=NL_CAP_OK){entry->live=true;return gpu_result(gpu_cap_status(rc));}
    NlGpuResult r=gpu_result(NL_GPU_OK);r.consumed=true;GpuLifetime *life=s->lifetime;
    if(life->faulted){gpu_skip_free(life,entry,life->first_error);return gpu_report(life,r);}
    GpuBracket bracket;
    if(gpu_begin(life,&bracket,&r)){gpu_free_once(life,entry,&r);gpu_end(life,&bracket,&r);}
    else gpu_skip_free(life,entry,r.driver_error);
    return gpu_report(life,r);
}
NlGpuResult nl_gpu_service_dispose(NlGpuService *s) {
    if(!s)return gpu_result(NL_GPU_ARGUMENT);
    GpuLifetime *life=s->lifetime;NlGpuResult r=gpu_result(NL_GPU_OK);
    if(!s->disposed) {
        s->disposed=true;r.consumed=true;
        for(unsigned i=0;i<NL_GPU_BUFFER_LIMIT;i++) {
            GpuBuffer *entry=&life->buffers[i];if(!entry->live)continue;
            entry->live=false;int rc=nl_cap_private_consume(s->caps,&entry->token);
            if(rc!=NL_CAP_OK)gpu_error(&r,GPU_HOST_CONTRACT);
        }
        GpuBracket bracket;
        if(!life->restore_unknown && life->context && gpu_begin(life,&bracket,&r)) {
            for(unsigned i=0;i<NL_GPU_BUFFER_LIMIT;i++)if(life->buffers[i].occupied)
                gpu_free_once(life,&life->buffers[i],&r);
            gpu_end(life,&bracket,&r);
        } else {
            for(unsigned i=0;i<NL_GPU_BUFFER_LIMIT;i++)if(life->buffers[i].occupied)
                gpu_skip_free(life,&life->buffers[i],r.driver_error?r.driver_error:life->first_error);
        }
        gpu_destroy_context(life,&r);
    }
    return gpu_report(life,r);
}
NlGpuResult nl_gpu_service_destroy(NlGpuService *s) {
    if(!s)return gpu_result(NL_GPU_ARGUMENT);
    NlGpuResult r=nl_gpu_service_dispose(s);GpuLifetime *life=s->lifetime;
    nl_cap_table_destroy(s->caps);free(s);gpu_release_record(life);return r;
}
NlGpuDiagnostics nl_gpu_service_diagnostics(void) {
    NlGpuDiagnostics out={0};out.acquisition_latched=gpu_acquisition_latched;
    for(unsigned i=0;i<NL_GPU_CONTEXT_LIMIT;i++) {
        const GpuLifetime *life=&gpu_lifetimes[i];if(!life->reserved)continue;
        NlGpuLifetime *row=&out.lifetimes[out.records++];row->record_id=life->identity;
        row->quarantined=life->quarantined;out.quarantined+=life->quarantined;
        row->context_retained=life->context!=NULL;row->library_retained=life->library!=NULL;
        row->release_unknown=life->release_unknown;row->context_restore_unknown=life->restore_unknown;
        row->context_unknown=life->context_unknown;row->first_error=life->first_error;
        row->allocations=life->allocations;row->free_attempts=life->free_attempts;
        row->freed_count=life->freed_count;row->skipped_count=life->skipped_count;
        for(unsigned n=0;n<NL_GPU_BUFFER_LIMIT;n++) {
            const GpuBuffer *entry=&life->buffers[n];if(!entry->occupied)continue;
            row->tracked_allocations++;
            row->unresolved_allocations+=entry->disposition!=NL_GPU_RELEASE_FREED && !entry->context_reclaimed;
            row->allocations_by_slot[n]=(NlGpuAllocation){
                .allocation_id=entry->allocation_id,.bytes=entry->bytes,.disposition=entry->disposition,
                .release_error=entry->release_error,.skipped_error=entry->skipped_error,
                .live_owner=entry->live,.release_attempted=entry->release_attempted,
                .context_reclaimed=entry->context_reclaimed};
        }
    }
    return out;
}
