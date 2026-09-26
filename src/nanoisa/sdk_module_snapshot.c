#include "sdk_module_snapshot.h"
#include <stdlib.h>
#include <string.h>
struct NvmSdkModuleSnapshot {
    NvmV2Module module;
    NvmSdkSignatureSnapshot *signatures;
    uint8_t *payloads;
    size_t bytes;
};
typedef struct { size_t bytes,payloads,signature_bytes; uint32_t work,work_limit; } ModuleSnapshotPlan;
static bool module_add(size_t *used,size_t count,size_t width,size_t limit) {
    if(*used>limit || (width && count>(limit-*used)/width))return false;
    *used+=count*width;return true;
}
static bool module_work(ModuleSnapshotPlan *p,uint32_t count) {
    if(count>p->work_limit-p->work)return false;
    p->work+=count;return true;
}
static NvmSdkResult module_payload(ModuleSnapshotPlan *p,const void *data,size_t size,size_t limit) {
    if(size&&!data)return NVM_SDK_INVALID;
    if(!module_add(&p->bytes,size,1,limit)||!module_add(&p->payloads,size,1,limit)||
       !module_work(p,2))return NVM_SDK_LIMIT;
    return NVM_SDK_OK;
}
static NvmSdkResult module_plan(const NvmV2Module *m,size_t limit,uint32_t work_limit,ModuleSnapshotPlan *out) {
    if(!m||!out)return NVM_SDK_INVALID;
    ModuleSnapshotPlan p={.bytes=sizeof(NvmSdkModuleSnapshot),.work_limit=work_limit};
    if(p.bytes>limit)return NVM_SDK_LIMIT;
#define PLAN_TABLE(name) do { \
    if(m->name.count&&!m->name.items)return NVM_SDK_INVALID; \
    if(!module_add(&p.bytes,m->name.count,sizeof *m->name.items,limit)|| \
       !module_work(&p,m->name.count)||!module_work(&p,m->name.count))return NVM_SDK_LIMIT; \
} while(0)
    PLAN_TABLE(metadata);PLAN_TABLE(constants);PLAN_TABLE(layouts);PLAN_TABLE(functions);
    PLAN_TABLE(globals);PLAN_TABLE(imports);PLAN_TABLE(callbacks);PLAN_TABLE(links);PLAN_TABLE(debug);
#undef PLAN_TABLE
    uint32_t signature_work;
    uint32_t signature_limit=((p.work_limit-p.work)/3)*2;
    NvmSdkResult result=nvm_sdk_signature_snapshot_measure(m,limit-p.bytes,signature_limit,
        &p.signature_bytes,&signature_work);
    if(result!=NVM_SDK_OK)return result;
    /* Measurement validation plus prepare's validation/copy = three passes. */
    if(!module_add(&p.bytes,p.signature_bytes,1,limit)||
       !module_work(&p,signature_work)||!module_work(&p,signature_work/2))return NVM_SDK_LIMIT;
    for(uint32_t i=0;i<m->layouts.count;i++) {
        const NvmV2Layout *row=&m->layouts.items[i];
        if(row->field_count&&!row->fields)return NVM_SDK_INVALID;
        if(!module_add(&p.bytes,row->field_count,sizeof *row->fields,limit)||
           !module_work(&p,row->field_count)||!module_work(&p,row->field_count))return NVM_SDK_LIMIT;
    }
    for(uint32_t i=0;i<m->constants.count;i++) {
        const NvmV2Constant *row=&m->constants.items[i];
        result=module_payload(&p,row->payload,row->length,limit);
        if(result!=NVM_SDK_OK)return result;
    }
    if(m->code_size>SIZE_MAX)return NVM_SDK_LIMIT;
    result=module_payload(&p,m->code,(size_t)m->code_size,limit);
    if(result!=NVM_SDK_OK)return result;
#define PLAN_PAYLOAD(name) do { result=module_payload(&p,m->name##_data,m->name##_size,limit); \
    if(result!=NVM_SDK_OK)return result; } while(0)
    PLAN_PAYLOAD(capture);PLAN_PAYLOAD(ownership);PLAN_PAYLOAD(passive);PLAN_PAYLOAD(service);
#undef PLAN_PAYLOAD
    *out=p;return NVM_SDK_OK;
}
void nvm_sdk_module_snapshot_free(NvmSdkModuleSnapshot *p) {
    if(!p)return;
    NvmV2Module *m=&p->module;
    for(uint32_t i=0;i<m->layouts.count;i++)free(m->layouts.items[i].fields);
    free(m->layouts.items);free(m->constants.items);free(m->metadata.items);
    free(m->functions.items);free(m->globals.items);free(m->imports.items);
    free(m->callbacks.items);free(m->links.items);free(m->debug.items);
    nvm_sdk_signature_snapshot_free(p->signatures);
    free(p->payloads);free(p);
}
static const uint8_t *module_copy_payload(NvmSdkModuleSnapshot *p,size_t *at,
                                          const uint8_t *source,size_t size) {
    if(!size)return NULL;
    uint8_t *target=p->payloads+*at;
    memcpy(target,source,size);*at+=size;return target;
}
static NvmSdkResult module_prepare(const NvmV2Module *source,size_t limit,
    NvmPreparationBudget *budget,NvmSdkModuleSnapshot **out) {
    if(!out)return NVM_SDK_INVALID;
    if(limit>NVM_SDK_GENERATION_MAX_BYTES)limit=NVM_SDK_GENERATION_MAX_BYTES;
    NvmPreparationBudget remaining=budget?*budget:(NvmPreparationBudget){0,0};
    uint32_t work_limit=NVM_SDK_GENERATION_MAX_WORK;
    if(budget) {
        if(limit>remaining.bytes)limit=remaining.bytes;
        if(work_limit>remaining.steps)work_limit=remaining.steps;
    }
    ModuleSnapshotPlan plan;
    NvmSdkResult result=module_plan(source,limit,work_limit,&plan);
    if(result!=NVM_SDK_OK)return result;
    if(budget&&!nvm_preparation_charge(&remaining,plan.bytes,plan.work))return NVM_SDK_LIMIT;
    NvmSdkModuleSnapshot *p=calloc(1,sizeof *p);if(!p)return NVM_SDK_MEMORY;
    NvmV2Module *m=&p->module;
    p->bytes=plan.bytes;
    m->isa_version=source->isa_version;m->entry_point=source->entry_point;
    m->has_debug=source->has_debug;m->extra_features=source->extra_features;
#define COPY_TABLE(name) do { if(source->name.count) { \
    m->name.items=malloc((size_t)source->name.count*sizeof *m->name.items); \
    if(!m->name.items)goto memory; \
    memcpy(m->name.items,source->name.items,(size_t)source->name.count*sizeof *m->name.items); \
    m->name.count=source->name.count; } } while(0)
    COPY_TABLE(metadata);COPY_TABLE(constants);COPY_TABLE(functions);COPY_TABLE(globals);
    COPY_TABLE(imports);COPY_TABLE(callbacks);COPY_TABLE(links);COPY_TABLE(debug);
#undef COPY_TABLE
    if(source->layouts.count) {
        m->layouts.items=calloc(source->layouts.count,sizeof *m->layouts.items);
        if(!m->layouts.items)goto memory;
        m->layouts.count=source->layouts.count;
    }
    for(uint32_t i=0;i<source->layouts.count;i++) {
        const NvmV2Layout *s=&source->layouts.items[i];NvmV2Layout *d=&m->layouts.items[i];
        d->kind=s->kind;d->name_idx=s->name_idx;
        if(s->field_count) {
            d->fields=malloc((size_t)s->field_count*sizeof *d->fields);
            if(!d->fields)goto memory;
            memcpy(d->fields,s->fields,(size_t)s->field_count*sizeof *d->fields);
        }
        d->field_count=s->field_count;
    }
    result=nvm_sdk_signature_snapshot_prepare(source,plan.signature_bytes,&p->signatures);
    if(result!=NVM_SDK_OK){nvm_sdk_module_snapshot_free(p);return result;}
    m->signatures=*nvm_sdk_signature_snapshot_rows(p->signatures);
    if(plan.payloads){p->payloads=malloc(plan.payloads);if(!p->payloads)goto memory;}
    size_t at=0;
    for(uint32_t i=0;i<m->constants.count;i++)
        m->constants.items[i].payload=module_copy_payload(p,&at,source->constants.items[i].payload,
            source->constants.items[i].length);
    m->code_size=source->code_size;
    m->code=module_copy_payload(p,&at,source->code,(size_t)source->code_size);
#define COPY_PAYLOAD(name) do { m->name##_size=source->name##_size; \
    m->name##_data=module_copy_payload(p,&at,source->name##_data,source->name##_size); } while(0)
    COPY_PAYLOAD(capture);COPY_PAYLOAD(ownership);COPY_PAYLOAD(passive);COPY_PAYLOAD(service);
#undef COPY_PAYLOAD
    *out=p;if(budget)*budget=remaining;return NVM_SDK_OK;
memory:
    nvm_sdk_module_snapshot_free(p);return NVM_SDK_MEMORY;
}
NvmSdkResult nvm_sdk_module_snapshot_prepare(const NvmV2Module *source,size_t limit,
    NvmSdkModuleSnapshot **out) {
    return module_prepare(source,limit,NULL,out);
}
NvmSdkResult nvm_sdk_module_snapshot_prepare_budget(const NvmV2Module *source,
    NvmPreparationBudget *budget,NvmSdkModuleSnapshot **out) {
    if(!budget)return NVM_SDK_INVALID;
    if(!nvm_preparation_budget_valid(budget))return NVM_SDK_LIMIT;
    return module_prepare(source,NVM_SDK_GENERATION_MAX_BYTES,budget,out);
}
const NvmV2Module *nvm_sdk_module_snapshot_view(const NvmSdkModuleSnapshot *p) {
    return p?&p->module:NULL;
}
size_t nvm_sdk_module_snapshot_bytes(const NvmSdkModuleSnapshot *p) {return p?p->bytes:0;}
