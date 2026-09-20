#include "nsi_file_values.h"
#include "nsi_file_values_internal.h"
#include "nsi_cap_private.h"
#if NL_FILE_VALUE_SLOTS != NL_CAP_PRIVATE_SLOTS
#error "I require matching private File and value slot bounds"
#endif
#include <stdlib.h>
#include <string.h>

typedef enum { FV_EMPTY, FV_FILE, FV_OPEN_OK, FV_OPEN_ERROR } FvKind;
typedef struct {
    uint64_t generation, borrow_epoch;
    FvKind kind;
    bool borrowed;
    NlFileToken token;
    NlFileResult error;
} FvSlot;
struct NlFileValues {
    uint64_t identity;
    NlFileService *service;
    bool finished;
    NlFileValuesFinish report;
    FvSlot slots[NL_FILE_VALUE_SLOTS];
};
bool nl_file_values_storage_bound(size_t *out) {
    size_t service;
    if (!out || !nl_file_service_storage_bound(&service) ||
        service > SIZE_MAX - sizeof(NlFileValues)) return false;
    *out = sizeof(NlFileValues) + service;
    return true;
}

/* External serialization includes this counter and the adapter's counter. */
static uint64_t fv_identity;
static NlFileResult fv_result(NlFileStatus status) {
    NlFileResult r={0};r.status=status;return r;
}
static bool fv_empty(NlFileValue v) {
    return !v.invocation && !v.generation && !v.slot;
}
static bool fv_disjoint(const void *a,const void *b,size_t size) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<y ? y-x>=size : x-y>=size;
}
static NlFileValueStatus fv_context(NlFileValues *s) {
    return !s?NL_FILE_VALUE_ARGUMENT:s->finished?NL_FILE_VALUE_DISPOSED:NL_FILE_VALUE_OK;
}
static NlFileValueStatus fv_resolve(NlFileValues *s,const NlFileValue *v,FvSlot **out) {
    NlFileValueStatus status=fv_context(s);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(!v)return NL_FILE_VALUE_ARGUMENT;
    if(v->invocation!=s->identity || v->slot>=NL_FILE_VALUE_SLOTS || !v->generation)
        return NL_FILE_VALUE_STALE;
    FvSlot *slot=&s->slots[v->slot];
    if(slot->kind==FV_EMPTY || slot->generation!=v->generation)return NL_FILE_VALUE_STALE;
    *out=slot;return NL_FILE_VALUE_OK;
}
static NlFileValue fv_handle(NlFileValues *s,uint32_t i) {
    return (NlFileValue){s->identity,s->slots[i].generation,i};
}
static void fv_clear(FvSlot *slot) {
    uint64_t generation=slot->generation,epoch=slot->borrow_epoch;
    memset(slot,0,sizeof *slot);slot->generation=generation;slot->borrow_epoch=epoch;
}
static void fv_cleanup(NlFileValues *s,NlFileResult r) {
    if(r.status==NL_FILE_OK && !r.cleanup_failed)return;
    if(!s->report.cleanup_failures)s->report.first_cleanup=r;
    else if(s->report.cleanup_failures==1)s->report.next_cleanup=r;
    /* Saturation preserves failure rather than wrapping to clean completion. */
    if(s->report.cleanup_failures<UINT64_MAX)s->report.cleanup_failures++;
}
static NlFileValueStatus fv_close_slot(NlFileValues *s,FvSlot *slot,NlFileResult *out) {
    NlFileResult r=nl_file_consume_close(s->service,&slot->token);
    fv_cleanup(s,r);
    if(!r.consumed)return NL_FILE_VALUE_STATE;
    fv_clear(slot);*out=r;return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_values_create(NlFileValues **out) {
    if(!out || *out)return NL_FILE_VALUE_ARGUMENT;
    if(fv_identity==UINT64_MAX)return NL_FILE_VALUE_LIMIT;
    NlFileValues *s=calloc(1,sizeof *s);
    if(!s)return NL_FILE_VALUE_MEMORY;
    NlFileResult r=nl_file_service_create(&s->service);
    if(r.status!=NL_FILE_OK) {
        free(s);return r.status==NL_FILE_LIMIT?NL_FILE_VALUE_LIMIT:
                       r.status==NL_FILE_MEMORY?NL_FILE_VALUE_MEMORY:NL_FILE_VALUE_STATE;
    }
    s->identity=++fv_identity;*out=s;return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_values_temp(NlFileValues *s,NlFileValue *out) {
    NlFileValueStatus status=fv_context(s);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(!out || !fv_empty(*out))return NL_FILE_VALUE_ARGUMENT;
    uint32_t i=0;
    while(i<NL_FILE_VALUE_SLOTS && (s->slots[i].kind!=FV_EMPTY ||
          s->slots[i].generation==UINT64_MAX || s->slots[i].borrow_epoch==UINT64_MAX))i++;
    if(i==NL_FILE_VALUE_SLOTS)return NL_FILE_VALUE_LIMIT;
    /* All owner/Result storage is prepared before acquisition. Publication below
     * cannot allocate. Adapter mint failure performs its own checked rollback. */
    NlFileToken token={0};
    NlFileResult r=nl_file_acquire_temp(s->service,NL_CAP_READ|NL_CAP_WRITE|NL_CAP_TRANSFER,&token);
    FvSlot *slot=&s->slots[i];slot->generation++;
    if(r.status==NL_FILE_OK){slot->kind=FV_OPEN_OK;slot->token=token;}
    else {slot->kind=FV_OPEN_ERROR;slot->error=r;if(r.cleanup_failed)fv_cleanup(s,r);}
    *out=fv_handle(s,i);return NL_FILE_VALUE_OK;
}
static NlFileValueStatus fv_transfer(NlFileValues *s,NlFileValue *v,NlFileValue *out,bool take_ok) {
    if(!v || !out || !fv_disjoint(v,out,sizeof *v) || !fv_empty(*out))return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;NlFileValueStatus status=fv_resolve(s,v,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(take_ok && slot->kind!=FV_OPEN_OK)return NL_FILE_VALUE_TYPE;
    if(slot->borrowed)return NL_FILE_VALUE_BORROWED;
    if(slot->generation==UINT64_MAX)return NL_FILE_VALUE_LIMIT;
    uint32_t i=v->slot;slot->generation++;
    if(take_ok)slot->kind=FV_FILE;
    *v=(NlFileValue){0};*out=fv_handle(s,i);return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_value_move(NlFileValues *s,NlFileValue *v,NlFileValue *out) {
    return fv_transfer(s,v,out,false);
}
NlFileValueStatus nl_file_open_view(NlFileValues *s,const NlFileValue *v,NlFileOpenView *out) {
    if(!out)return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;NlFileValueStatus status=fv_resolve(s,v,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(slot->kind!=FV_OPEN_OK && slot->kind!=FV_OPEN_ERROR)return NL_FILE_VALUE_TYPE;
    NlFileOpenView view={0};view.ok=slot->kind==FV_OPEN_OK;
    if(!view.ok)view.error=slot->error;
    *out=view;return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_open_take_ok(NlFileValues *s,NlFileValue *v,NlFileValue *out) {
    return fv_transfer(s,v,out,true);
}
NlFileValueStatus nl_file_open_take_error(NlFileValues *s,NlFileValue *v,NlFileResult *out) {
    if(!out)return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;NlFileValueStatus status=fv_resolve(s,v,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(slot->kind!=FV_OPEN_ERROR)return NL_FILE_VALUE_TYPE;
    NlFileResult r=slot->error;fv_clear(slot);*v=(NlFileValue){0};*out=r;return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_value_borrow(NlFileValues *s,const NlFileValue *v,NlFileValueBorrow *out) {
    if(!out || !fv_empty(out->value) || out->epoch)return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;NlFileValueStatus status=fv_resolve(s,v,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(slot->kind!=FV_FILE)return NL_FILE_VALUE_TYPE;
    if(slot->borrowed)return NL_FILE_VALUE_BORROWED;
    if(slot->borrow_epoch==UINT64_MAX)return NL_FILE_VALUE_LIMIT;
    slot->borrow_epoch++;slot->borrowed=true;
    *out=(NlFileValueBorrow){*v,slot->borrow_epoch};return NL_FILE_VALUE_OK;
}
static NlFileValueStatus fv_borrow(NlFileValues *s,const NlFileValueBorrow *b,FvSlot **out) {
    if(!b)return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;NlFileValueStatus status=fv_resolve(s,&b->value,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(slot->kind!=FV_FILE || !slot->borrowed || !b->epoch || b->epoch!=slot->borrow_epoch)
        return NL_FILE_VALUE_STALE;
    *out=slot;return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_values_live_slots(NlFileValues *s,uint64_t *owners,uint64_t *borrowed) {
    NlFileValueStatus status=fv_context(s);if(status!=NL_FILE_VALUE_OK)return status;
    if(!owners || !borrowed || !fv_disjoint(owners,borrowed,sizeof *owners))return NL_FILE_VALUE_ARGUMENT;
    uint64_t live=0,held=0;
    for(uint32_t i=0;i<NL_FILE_VALUE_SLOTS;i++) {
        if(s->slots[i].kind!=FV_EMPTY)live|=UINT64_C(1)<<i;
        if(s->slots[i].borrowed)held|=UINT64_C(1)<<i;
    }
    *owners=live;*borrowed=held;return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_value_validate(NlFileValues *s,const NlFileValue *v,bool open_result) {
    FvSlot *slot;NlFileValueStatus status=fv_resolve(s,v,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    return (open_result?(slot->kind==FV_OPEN_OK || slot->kind==FV_OPEN_ERROR):
                        slot->kind==FV_FILE)?NL_FILE_VALUE_OK:NL_FILE_VALUE_TYPE;
}
NlFileValueStatus nl_file_value_borrow_validate(NlFileValues *s,const NlFileValueBorrow *b) {
    if(!b)return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;return fv_borrow(s,b,&slot);
}
NlFileValueStatus nl_file_value_end_borrow(NlFileValues *s,NlFileValueBorrow *b) {
    FvSlot *slot;NlFileValueStatus status=fv_borrow(s,b,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    slot->borrowed=false;*b=(NlFileValueBorrow){0};return NL_FILE_VALUE_OK;
}
static NlFileScalarResult fv_scalar(NlFileScalarKind kind,NlFileResult r) {
    NlFileScalarResult out={0};out.kind=kind;out.ok=r.status==NL_FILE_OK;out.detail=r;return out;
}
NlFileValueStatus nl_file_value_write_byte(NlFileValues *s,const NlFileValueBorrow *b,int64_t byte,NlFileScalarResult *out) {
    if(!out)return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;NlFileValueStatus status=fv_borrow(s,b,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    NlFileResult r;
    if(byte<0 || byte>255)r=fv_result(NL_FILE_ARGUMENT);
    else {unsigned char value=(unsigned char)byte;r=nl_file_write(s->service,&slot->token,&value,1);}
    NlFileScalarResult result=fv_scalar(NL_FILE_VALUE_WRITE,r);
    if(result.ok)result.value=(int64_t)r.bytes;
    *out=result;return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_value_rewind(NlFileValues *s,const NlFileValueBorrow *b,NlFileScalarResult *out) {
    if(!out)return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;NlFileValueStatus status=fv_borrow(s,b,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    *out=fv_scalar(NL_FILE_VALUE_POSITION,nl_file_rewind(s->service,&slot->token));return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_value_read_byte(NlFileValues *s,const NlFileValueBorrow *b,NlFileScalarResult *out) {
    if(!out)return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;NlFileValueStatus status=fv_borrow(s,b,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    unsigned char byte=0;NlFileResult r=nl_file_read(s->service,&slot->token,&byte,1);
    NlFileScalarResult result=fv_scalar(NL_FILE_VALUE_READ,r);
    if(result.ok){result.value=r.bytes?byte:0;result.eof=r.eof;}
    *out=result;return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_value_close(NlFileValues *s,NlFileValue *v,NlFileScalarResult *out) {
    if(!out)return NL_FILE_VALUE_ARGUMENT;
    FvSlot *slot;NlFileValueStatus status=fv_resolve(s,v,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(slot->kind!=FV_FILE)return NL_FILE_VALUE_TYPE;
    if(slot->borrowed)return NL_FILE_VALUE_BORROWED;
    NlFileResult r;status=fv_close_slot(s,slot,&r);
    if(status!=NL_FILE_VALUE_OK)return status;
    *v=(NlFileValue){0};*out=fv_scalar(NL_FILE_VALUE_CLOSE,r);return NL_FILE_VALUE_OK;
}
NlFileValueStatus nl_file_value_drop(NlFileValues *s,NlFileValue *v) {
    NlFileValueStatus status=fv_context(s);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(!v)return NL_FILE_VALUE_ARGUMENT;
    if(fv_empty(*v))return NL_FILE_VALUE_OK;
    FvSlot *slot;status=fv_resolve(s,v,&slot);
    if(status!=NL_FILE_VALUE_OK)return status;
    if(slot->borrowed)return NL_FILE_VALUE_BORROWED;
    if(slot->kind==FV_OPEN_ERROR)fv_clear(slot);
    else {NlFileResult r;status=fv_close_slot(s,slot,&r);if(status!=NL_FILE_VALUE_OK)return status;}
    *v=(NlFileValue){0};return NL_FILE_VALUE_OK;
}
bool nl_file_values_report(const NlFileValues *s, NlFileValuesFinish *out) {
    if(!s || !out)return false;
    *out=s->report;return true;
}
NlFileValuesFinish nl_file_values_finish(NlFileValues *s,NlFileValueStatus error) {
    if(!s)return (NlFileValuesFinish){.execution=NL_FILE_VALUE_ARGUMENT};
    if(s->finished)return s->report;
    if((unsigned)error>=NL_FILE_VALUE_STATUS_COUNT)
        return (NlFileValuesFinish){.execution=NL_FILE_VALUE_ARGUMENT};
    s->report.execution=error;
    for(uint32_t i=0;i<NL_FILE_VALUE_SLOTS;i++) {
        FvSlot *slot=&s->slots[i];
        if(slot->kind==FV_FILE || slot->kind==FV_OPEN_OK) {
            NlFileResult r;NlFileValueStatus status=fv_close_slot(s,slot,&r);
            if(status!=NL_FILE_VALUE_OK && s->report.execution==NL_FILE_VALUE_OK)s->report.execution=status;
        }
        fv_clear(slot);
    }
    fv_cleanup(s,nl_file_service_dispose(s->service));
    fv_cleanup(s,nl_file_service_destroy(s->service));s->service=NULL;
    s->finished=true;return s->report;
}
NlFileValuesFinish nl_file_values_destroy(NlFileValues *s,NlFileValueStatus error) {
    NlFileValuesFinish result=nl_file_values_finish(s,error);
    /* An invalid status must not free a context whose cleanup was refused. */
    if(s && s->finished)free(s);
    return result;
}
