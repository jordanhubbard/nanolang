#include "managed_record_map_boundary.h"
struct NvmRecordMapBoundaryPlan {
    NvmRecordMapExecutionPlan *execution;
    NvmRecordValueOrigins *globals;
    uint32_t count, reserved;
    uint64_t bytes, work;
};
_Static_assert(sizeof(NvmRecordMapBoundaryPlan)==40,"boundary owner ledger");
_Static_assert(sizeof(NvmRecordValueOrigins)==16,"boundary row ledger");
void nvm_record_map_boundary_free(NvmRecordMapBoundaryPlan *p) {
    if(!p)return;
    nvm_record_map_execution_free(p->execution);free(p->globals);free(p);
}
/* Positive-only whitebox reservation; ordinary constructor supplies zero. */
static NvmArrayEligibilityResult rmb_prepare_reserved(const NvmModule *m,NvmRecordMapBoundaryPlan **out,
    uint64_t extra_bytes,uint64_t extra_work) {
    NvmArrayEligibilityResult r={NVM_ARRAY_INVALID,0,0,"I require an original module and boundary output."};
    if(!m || !out)return r;
    if(extra_bytes>UINT64_MAX-sizeof(NvmRecordMapBoundaryPlan) ||
       extra_work>UINT64_MAX-sizeof(NvmRecordMapBoundaryPlan)) {r.status=NVM_ARRAY_LIMIT;return r;}
    /* rme checks its complete fixed reserve before allocating. Owner allocation
     * follows proof, but its full live overlap is reserved before proof begins. */
    NvmRecordMapExecutionPlan *execution=NULL;NvmMapGlobalExport globals={0};
    r=rme_prepare_boundary(m,&execution,extra_bytes+sizeof(NvmRecordMapBoundaryPlan),
                          extra_work+sizeof(NvmRecordMapBoundaryPlan),&globals);
    if(r.status!=NVM_ARRAY_ELIGIBLE)return r;
    NvmRecordMapBoundaryPlan *p=calloc(1,sizeof *p);
    if(!p){free(globals.rows);nvm_record_map_execution_free(execution);r.status=NVM_ARRAY_MEMORY;
        snprintf(r.message,sizeof r.message,"I could not allocate my copied boundary owner.");return r;}
    p->execution=execution;p->globals=globals.rows;p->count=globals.count;
    p->bytes=execution->carrier->bytes;p->work=execution->carrier->work;
    *out=p;return r;
}
NvmArrayEligibilityResult nvm_prepare_record_map_boundary(const NvmModule *m,NvmRecordMapBoundaryPlan **out) {
    return rmb_prepare_reserved(m,out,0,0);
}
bool nvm_record_map_boundary_counts(const NvmRecordMapBoundaryPlan *p,NvmRecordMapBoundaryCounts *out) {
    if(!p || !out)return false;
    NvmRecordMapBoundaryCounts row={0};row.globals=p->count;
    row.peak_bytes_reserved=p->bytes;row.work_reserved=p->work;*out=row;return true;
}
bool nvm_record_map_boundary_global(const NvmRecordMapBoundaryPlan *p,uint32_t i,NvmRecordValueOrigins *out) {
    if(!p || !out || i>=p->count)return false;
    *out=p->globals[i];return true;
}
const NvmRecordMapExecutionPlan *nvm_record_map_boundary_execution(const NvmRecordMapBoundaryPlan *p) {
    return p?p->execution:NULL;
}
