#include "../../src/nanoisa/managed_record_map_execution.h"
#include "../../src/nanoisa/record_map_structure_private.h"
static void execution_original(const NvmModule *);
#define MAP_ORIGINAL_CHECK execution_original
#define MAP_ORIGIN_MAIN old_map_origin_main
#include "test_record_map_origins.c"
static NvmRecordMapExecutionPlan *execution_prepare(const NvmModule *m) {
    NvmRecordMapExecutionPlan *p=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult result=nvm_prepare_record_map_execution(m,&p);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"I refused recipes: %s\n",result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);CHECK(p&&p!=(void *)(uintptr_t)1);return p;
}
static void execution_rows(const NvmModule *m,NvmRecordMapExecutionPlan *p) {
    NvmRecordMapExecutionCounts counts;CHECK(nvm_record_map_execution_counts(p,&counts));
    CHECK(counts.carrier.functions==m->function_count);unsigned maps=0,visited=0;
    for(uint32_t fi=0;fi<m->function_count;fi++) {
        NvmRecordArrayExecutionFunction f;CHECK(nvm_record_map_execution_function(p,fi,&f));
        CHECK(f.signature.code_length==m->functions[fi].code_length);CHECK(f.maximum_stack>0);
        uint32_t pc=0;
        for(uint32_t j=0;j<f.instruction_count;j++) {
            NvmRecordMapExecutionInstruction row;CHECK(nvm_record_map_execution_instruction(p,f.instruction_start+j,&row));
            DecodedInstruction ins;uint32_t size=isa_decode(m->code+f.signature.code_offset+pc,f.signature.code_length-pc,&ins);
            CHECK(size);CHECK(row.base.function==fi&&row.base.pc==pc&&row.base.next_pc==pc+size&&row.base.opcode==ins.opcode);
            if(nvm_record_map_opcode(ins.opcode)) {
                maps++;CHECK(row.base.recipe==NVM_RM_INVALID_BASE_RECIPE);CHECK(row.key_tags==MASK(TAG_STRING)&&row.value_tags==MASK(TAG_INT));
                CHECK(row.base.pushes==1);CHECK(row.base.obligations&NVM_RA_FIRST_ERROR_CLEANUP);
                switch(ins.opcode) {
                case OP_HM_NEW:CHECK(row.map_operation==NVM_RM_NEW&&row.base.pops==0&&row.result_tags==MASK(TAG_HASHMAP));break;
                case OP_HM_SET:CHECK(row.map_operation==NVM_RM_SET&&row.base.pops==3&&row.result_tags==MASK(TAG_HASHMAP));break;
                case OP_HM_GET:CHECK(row.map_operation==NVM_RM_GET&&row.base.pops==2&&row.result_tags==(MASK(TAG_INT)|MASK(TAG_VOID)));break;
                case OP_HM_HAS:CHECK(row.map_operation==NVM_RM_HAS&&row.base.pops==2&&row.result_tags==MASK(TAG_BOOL));break;
                case OP_HM_LEN:CHECK(row.map_operation==NVM_RM_LEN&&row.base.pops==1&&row.result_tags==MASK(TAG_INT));break;
                }
            } else CHECK(row.map_operation==NVM_RM_NONE&&!row.key_tags&&!row.value_tags&&!row.result_tags&&row.base.recipe!=NVM_RM_INVALID_BASE_RECIPE);
            pc+=size;visited++;
        }
        CHECK(pc==f.signature.code_length);
    }
    CHECK(visited==counts.carrier.instructions&&maps>0);
    NvmRecordMapExecutionInstruction bad,saved;memset(&bad,0x5a,sizeof bad);saved=bad;
    CHECK(!nvm_record_map_execution_instruction(p,counts.carrier.instructions,&bad));CHECK(!memcmp(&bad,&saved,sizeof bad));
    map_public_transport(m,true);
}
static void execution_original(const NvmModule *m) {
#ifdef MAP_EXECUTION_ORIGINAL_CHECK
    size_t retained_before=ra_bytes;
#endif
    NvmRecordMapExecutionPlan *p=execution_prepare(m);execution_rows(m,p);
#ifdef MAP_EXECUTION_ORIGINAL_CHECK
    MAP_EXECUTION_ORIGINAL_CHECK(m,p,retained_before);
#endif
    nvm_record_map_execution_free(p);
}
static void execution_copied(void) {
    Input c;map_basic(&c);NvmRecordMapExecutionPlan *p=execution_prepare(&c.m);execution_rows(&c.m,p);
    uint32_t size;CHECK(nvm_record_map_execution_size(p,NVM_RA_SNAPSHOT_CODE,0,&size));CHECK(size==c.m.code_size);
    uint8_t *code=malloc(size);CHECK(code);memcpy(code,c.code,size);memset(&c,0,sizeof c);
    uint8_t *copy=malloc(size);CHECK(copy);CHECK(nvm_record_map_execution_bytes(p,NVM_RA_SNAPSHOT_CODE,0,0,size,copy));CHECK(!memcmp(code,copy,size));free(code);free(copy);
    NvmRecordMapExecutionCounts counts;CHECK(nvm_record_map_execution_counts(p,&counts));
    NvmRecordArrayOriginCounts origins;CHECK(nvm_record_map_execution_origin_counts(p,&origins));CHECK(origins.origins==counts.maps.origins);
    uint8_t key=99,value=98;CHECK(nvm_record_map_execution_map_field(p,0,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
    CHECK(!nvm_record_map_execution_map_field(p,0,0,&key,&key)&&key==TAG_STRING);
    CHECK(nvm_record_map_execution_map_origin(p,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
    NvmRecordHeapOrigin origin;CHECK(nvm_record_map_execution_origin(p,0,&origin));
    nvm_record_map_execution_free(p);
}
static void execution_failures(void) {
    Input c;map_basic(&c);c.code[0]=OP_HM_DELETE;
    NvmRecordMapExecutionPlan *p=(void *)(uintptr_t)1;CHECK(nvm_prepare_record_map_execution(&c.m,&p).status!=NVM_ARRAY_ELIGIBLE);CHECK(p==(void *)(uintptr_t)1);
    CHECK(nvm_prepare_record_map_execution(NULL,&p).status==NVM_ARRAY_INVALID&&p==(void *)(uintptr_t)1);
#ifdef RA_WHITEBOX
    for(unsigned op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction row={0};
        CHECK(!!rme_recipe((uint8_t)op,&row)==!!(nvm_record_map_opcode((uint8_t)op)||nvm_record_array_opcode_supported((uint8_t)op)));
    }
    NvmRecordArrayExecutionPlan budget={0};budget.bytes=NVM_RECORD_ARRAY_EXECUTION_BYTES-1;budget.work=NVM_RECORD_ARRAY_EXECUTION_STEPS-1;
    CHECK(rae_charge(&budget,1,1));CHECK(!rae_charge(&budget,1,0)&&budget.failure==NVM_ARRAY_LIMIT);
    budget.failure=NVM_ARRAY_INVALID;CHECK(!rae_charge(&budget,0,1)&&budget.failure==NVM_ARRAY_LIMIT);
    map_basic(&c);uint16_t stacks[256],before[256];memset(stacks,0xa5,sizeof stacks);memcpy(before,stacks,sizeof stacks);
    NvmRecordMapFrozenContext *facts=(void *)(uintptr_t)1;
    c.code[0]=OP_HM_DELETE;CHECK(analyze(&c.m,NULL,NULL,NULL,NULL,stacks,0,&facts).status!=NVM_ARRAY_ELIGIBLE);
    CHECK(facts==(void *)(uintptr_t)1&&!memcmp(stacks,before,sizeof stacks));
    map_basic(&c);CHECK(analyze(&c.m,NULL,NULL,NULL,NULL,stacks,0,&facts).status==NVM_ARRAY_ELIGIBLE);CHECK(stacks[0]>0);nvm_record_map_frozen_free(facts);
#endif
}
static void execution_allocations(void) {
#ifdef RA_WHITEBOX
    Input c;map_basic(&c);CHECK(!ra_live&&!ra_bytes);ra_calls=ra_peak=0;
    NvmRecordMapExecutionPlan *p=execution_prepare(&c.m);size_t count=ra_calls;
    NvmRecordMapExecutionCounts costs;CHECK(nvm_record_map_execution_counts(p,&costs));CHECK(ra_peak<=costs.carrier.peak_bytes_reserved);
    nvm_record_map_execution_free(p);CHECK(!ra_live&&!ra_bytes);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<count;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;p=(void *)(uintptr_t)1;
        CHECK(nvm_prepare_record_map_execution(&c.m,&p).status==NVM_ARRAY_MEMORY);CHECK(p==(void *)(uintptr_t)1);CHECK(!ra_live&&!ra_bytes);
        ra_fail=SIZE_MAX;ra_persistent=0;p=execution_prepare(&c.m);nvm_record_map_execution_free(p);CHECK(!ra_live&&!ra_bytes);
    }
    for(unsigned kind=0;kind<2;kind++) {
        uint64_t extra_bytes=kind?0:NVM_RECORD_ARRAY_EXECUTION_BYTES-costs.carrier.peak_bytes_reserved;
        uint64_t extra_work=kind?NVM_RECORD_ARRAY_EXECUTION_STEPS-costs.carrier.work_reserved:0;
        CHECK(rme_prepare_reserved(&c.m,&p,extra_bytes,extra_work).status==NVM_ARRAY_ELIGIBLE);
        NvmRecordMapExecutionCounts exact;CHECK(nvm_record_map_execution_counts(p,&exact));
        CHECK(kind?exact.carrier.work_reserved==NVM_RECORD_ARRAY_EXECUTION_STEPS:exact.carrier.peak_bytes_reserved==NVM_RECORD_ARRAY_EXECUTION_BYTES);
        nvm_record_map_execution_free(p);p=(void *)(uintptr_t)1;
        CHECK(rme_prepare_reserved(&c.m,&p,extra_bytes+!kind,extra_work+kind).status==NVM_ARRAY_LIMIT);
        CHECK(p==(void *)(uintptr_t)1&&!ra_live&&!ra_bytes);
        p=execution_prepare(&c.m);nvm_record_map_execution_free(p);CHECK(!ra_live&&!ra_bytes);
    }
    for(unsigned kind=0;kind<2;kind++) {
        size_t calls=ra_calls;p=(void *)(uintptr_t)1;
        CHECK(rme_prepare_reserved(&c.m,&p,kind?0:UINT64_MAX,kind?UINT64_MAX:0).status==NVM_ARRAY_LIMIT);
        CHECK(p==(void *)(uintptr_t)1&&calls==ra_calls);
    }
    /* I test the private dual-output publication under every proof allocation failure. */
    uint16_t stacks[256],saved[256];memset(saved,0xa5,sizeof saved);
    NvmRecordMapFrozenContext *facts=NULL;ra_calls=0;
    CHECK(analyze(&c.m,NULL,NULL,NULL,NULL,stacks,0,&facts).status==NVM_ARRAY_ELIGIBLE);
    size_t proof_count=ra_calls;nvm_record_map_frozen_free(facts);CHECK(!ra_live&&!ra_bytes);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<proof_count;i++) {
        memcpy(stacks,saved,sizeof saved);facts=(void *)(uintptr_t)1;
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;
        CHECK(analyze(&c.m,NULL,NULL,NULL,NULL,stacks,0,&facts).status==NVM_ARRAY_MEMORY);
        CHECK(facts==(void *)(uintptr_t)1&&!memcmp(stacks,saved,sizeof stacks));CHECK(!ra_live&&!ra_bytes);
        ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(analyze(&c.m,NULL,NULL,NULL,NULL,stacks,0,&facts).status==NVM_ARRAY_ELIGIBLE);
        nvm_record_map_frozen_free(facts);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I retained %zu stack/plan proof failure prefixes.\n",proof_count);
    printf("I retained %zu recipe allocation prefixes; reserved %llu bytes/%llu work.\n",count,(unsigned long long)costs.carrier.peak_bytes_reserved,(unsigned long long)costs.carrier.work_reserved);
#endif
}
#ifndef MAP_EXECUTION_MAIN
#define MAP_EXECUTION_MAIN main
#endif
int MAP_EXECUTION_MAIN(int argc,char **argv) {
    old_map_origin_main(argc,argv);execution_copied();execution_failures();execution_allocations();
    printf("PASS %u copied map recipe checks; public boundaries checked\n",checks);return 0;
}
