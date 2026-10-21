#include "../../src/nanoisa/record_map_snapshot_private.h"
#include <stddef.h>
static void image_compare(const NvmModule *,const NvmRecordMapExecutionPlan *,const NvmRecordMapImage *);
static void image_original(const NvmModule *,const NvmRecordMapExecutionPlan *,size_t);
#define MAP_EXECUTION_ORIGINAL_CHECK image_original
#define MAP_EXECUTION_MAIN old_map_execution_main
#include "test_record_map_execution.c"
static void image_original(const NvmModule *m,const NvmRecordMapExecutionPlan *p,size_t before) {
    (void)before;
#ifdef RA_WHITEBOX
    size_t calls=ra_calls,live=ra_live,bytes=ra_bytes;uint64_t retained=UINT64_MAX;
    NvmMapCopyWork work={0,NVM_RECORD_ARRAY_EXECUTION_STEPS,false};
    CHECK(rme_plan_retained_cost(p,&work,&retained)==NVM_MAP_COST_OK);
    CHECK(retained==bytes-before);CHECK(ra_calls==calls&&ra_live==live&&ra_bytes==bytes);
    NvmDeclarationCounts dc;CHECK(nvm_record_map_execution_declaration_counts(p,&dc));
    CHECK(work.used==180u+12u*((uint64_t)dc.layouts+m->function_count+m->string_count));
#endif
    NvmRecordMapImage *image=(void *)(uintptr_t)1;
    CHECK(nvm_record_map_image_copy(p,&image).status==NVM_ARRAY_ELIGIBLE);
    image_compare(m,p,image);nvm_record_map_image_free(image);
}
static void image_cost_cases(void) {
#ifdef RA_WHITEBOX
    for(unsigned variant=0;variant<8;variant++) {
        Input c;map_basic(&c);char **extra_strings=NULL;uint32_t *extra_lengths=NULL;
        if(variant>=1)c.m.function_param_types=c.param_rows;
        if(variant>=2)c.param_rows[0]=c.params; /* present zero-arity allocation */
        if(variant>=3){c.strings[1]="";c.lengths[1]=0;}
        if(variant>=4) {
            c.n--;arg32(&c,OP_CALL,1);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;
            c.m.function_count=2;c.fn[1].name_idx=2;c.fn[1].arity=1;c.fn[1].local_count=1;
            c.fn[1].result_tag=TAG_INT;c.fn[1].result_count=1;c.fn[1].code_offset=(uint32_t)c.n;
            c.params[0]=TAG_INT;c.param_rows[1]=c.params;arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);
            c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);
        }
        if(variant==5) {
            size_t at=0;b32(c.layouts,&at,3);b8(c.layouts,&c.l,NVM_V2_LAYOUT_STRUCT);b8(c.layouts,&c.l,0);
            b16(c.layouts,&c.l,0);b32(c.layouts,&c.l,NO);c.m.layout_size=(uint32_t)c.l;c.m.struct_count=3;
            at=4;b32(c.ownership,&at,3);c.ownership[10]=NVM_LAYOUT_COMPLETE;
        }
        if(variant==6){init(&c,TAG_INT,false);map_finish(&c);}
        if(variant==7) {
            extra_strings=calloc(NVM_MAX_STRINGS,sizeof *extra_strings);extra_lengths=calloc(NVM_MAX_STRINGS,sizeof *extra_lengths);
            CHECK(extra_strings&&extra_lengths);
            for(unsigned i=0;i<NVM_MAX_STRINGS;i++){extra_strings[i]=i<4?c.strings[i]:"";extra_lengths[i]=i<4?c.lengths[i]:0;}
            c.m.strings=extra_strings;c.m.string_lengths=extra_lengths;c.m.string_count=NVM_MAX_STRINGS;
        }
        CHECK(!ra_live&&!ra_bytes);NvmRecordMapExecutionPlan *p=execution_prepare(&c.m);
        image_original(&c.m,p,0);
        NvmMapCopyWork work={0,NVM_RECORD_ARRAY_EXECUTION_STEPS,false};uint64_t bytes=99;
        CHECK(rme_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_OK);
        uint64_t needed=work.used,retained=bytes;size_t calls=ra_calls;
        work=(NvmMapCopyWork){0,needed,false};bytes=99;
        CHECK(rme_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_OK&&bytes==retained&&work.used==needed);
        work=(NvmMapCopyWork){0,needed-1,false};bytes=99;
        CHECK(rme_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_LIMIT&&bytes==99&&work.limited);
        CHECK(ra_calls==calls);
        work=(NvmMapCopyWork){0,NVM_RECORD_ARRAY_EXECUTION_STEPS,false};bytes=99;
        CHECK(rme_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_OK&&bytes==retained);
        /* Invalid owner bookkeeping is a defensive private control, not public input. */
        p->carrier->origins=p->facts->base;work=(NvmMapCopyWork){0,NVM_RECORD_ARRAY_EXECUTION_STEPS,false};bytes=99;
        CHECK(rme_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_INVALID&&bytes==99);p->carrier->origins=NULL;
        uint32_t count=p->carrier->snapshot.function_count;p->carrier->snapshot.function_count=257;
        work=(NvmMapCopyWork){0,NVM_RECORD_ARRAY_EXECUTION_STEPS,false};bytes=99;
        CHECK(rme_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_INVALID&&bytes==99);p->carrier->snapshot.function_count=count;
        /* Cross-check every redundant count/range/signature without dereferencing rows beyond bounds. */
#define COST_BAD(field) do {uint32_t saved_count=(field);(field)++;work=(NvmMapCopyWork){0,NVM_RECORD_ARRAY_EXECUTION_STEPS,false};bytes=99;CHECK(rme_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_INVALID&&bytes==99);(field)=saved_count;}while(0)
        COST_BAD(p->carrier->counts.metadata);COST_BAD(p->carrier->counts.debug);COST_BAD(p->carrier->counts.sections);
        COST_BAD(p->carrier->functions[0].instruction_start);COST_BAD(p->carrier->functions[0].instruction_count);
        COST_BAD(p->carrier->functions[0].signature.arity);COST_BAD(p->carrier->functions[0].signature.local_count);
#undef COST_BAD
        nvm_record_map_execution_free(p);CHECK(!ra_live&&!ra_bytes);free(extra_strings);free(extra_lengths);
    }
    printf("I matched retained payload and exact logical work without allocations.\n");
#endif
}
static void image_compare(const NvmModule *m,const NvmRecordMapExecutionPlan *p,const NvmRecordMapImage *image) {
    const NvmModule *copy=nvm_record_map_image_borrow(image);CHECK(copy&&copy!=m);
    CHECK(copy->code_size==m->code_size&&!memcmp(copy->code,m->code,m->code_size));
    CHECK(copy->ownership_size==m->ownership_size&&!memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
    CHECK(copy->layout_size==m->layout_size&&!memcmp(copy->layout_data,m->layout_data,m->layout_size));
    CHECK(copy->string_count==m->string_count&&copy->function_count==m->function_count);
    for(uint32_t i=0;i<m->string_count;i++)CHECK(copy->string_lengths[i]==m->string_lengths[i]&&!memcmp(copy->strings[i],m->strings[i],(size_t)m->string_lengths[i]+1));
    for(uint32_t i=0;i<m->function_count;i++) {
        CHECK(!memcmp(&copy->functions[i],&m->functions[i],sizeof m->functions[i]));
        CHECK(!!copy->function_param_types==!!m->function_param_types);
        if(m->function_param_types) {
            CHECK(!!copy->function_param_types[i]==!!m->function_param_types[i]);
            if(m->function_param_types[i])CHECK(!memcmp(copy->function_param_types[i],m->function_param_types[i],m->functions[i].arity));
        }
    }
    NvmRecordMapExecutionCounts a,b;CHECK(nvm_record_map_execution_counts(p,&a)&&nvm_record_map_image_counts(image,&b));
    CHECK(a.carrier.instructions==b.carrier.instructions&&a.carrier.functions==b.carrier.functions&&a.maps.origins==b.maps.origins);
    for(uint32_t i=0;i<a.carrier.instructions;i++) {
        NvmRecordMapExecutionInstruction x,y;
        CHECK(nvm_record_map_execution_instruction(p,i,&x)&&nvm_record_map_image_instruction(image,i,&y));CHECK(!memcmp(&x,&y,sizeof x));
    }
    map_public_transport(copy,true);
}
static void image_lifetimes(void) {
    Input c;map_basic(&c);NvmRecordMapExecutionPlan *source=execution_prepare(&c.m);NvmRecordMapImage *image=NULL;
    CHECK(nvm_record_map_image_copy(source,&image).status==NVM_ARRAY_ELIGIBLE);image_compare(&c.m,source,image);
    nvm_record_map_execution_free(source);memset(&c,0,sizeof c);
    const NvmModule *m=nvm_record_map_image_borrow(image);CHECK(m&&m->function_count==1);map_public_transport(m,true);
    uint8_t key=99,value=98;CHECK(nvm_record_map_image_field(image,0,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
    CHECK(nvm_record_map_image_origin(image,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
    CHECK(!nvm_record_map_image_field(image,0,0,&key,&key)&&key==TAG_STRING);
    NvmRecordMapImageCost cost;CHECK(nvm_record_map_image_cost(image,&cost)&&cost.source_retained_bytes>0);
#ifdef RA_WHITEBOX
    uint64_t retained;NvmMapCopyWork w={0,NVM_RECORD_ARRAY_EXECUTION_STEPS,false};
    CHECK(rme_plan_retained_cost(image->plan,&w,&retained)==NVM_MAP_COST_OK);
    CHECK(retained+sizeof *image==ra_bytes);
#endif
    nvm_record_map_image_free(image);
}
static void image_failure_controls(void) {
#ifdef RA_WHITEBOX
    Input c;map_basic(&c);NvmRecordMapExecutionPlan *source=execution_prepare(&c.m);
    size_t source_bytes=ra_bytes,source_live=ra_live;ra_calls=ra_peak=0;
    NvmRecordMapImage *image=NULL;CHECK(nvm_record_map_image_copy(source,&image).status==NVM_ARRAY_ELIGIBLE);
    size_t allocations=ra_calls;NvmRecordMapImageCost cost;CHECK(nvm_record_map_image_cost(image,&cost));
    CHECK(cost.source_retained_bytes==source_bytes&&ra_peak<=cost.preparation_bytes_reserved);
    nvm_record_map_image_free(image);CHECK(ra_bytes==source_bytes&&ra_live==source_live);
    NvmRecordMapImage *prior=NULL;CHECK(nvm_record_map_image_copy(source,&prior).status==NVM_ARRAY_ELIGIBLE);
    size_t held_bytes=ra_bytes,held_live=ra_live;
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<allocations;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;image=prior;
        CHECK(nvm_record_map_image_copy(source,&image).status==NVM_ARRAY_MEMORY&&image==prior);
        CHECK(ra_bytes==held_bytes&&ra_live==held_live);CHECK(nvm_record_map_image_borrow(prior)->function_count==1);
        NvmRecordMapExecutionCounts counts;CHECK(nvm_record_map_execution_counts(source,&counts)&&counts.carrier.functions==1);
        ra_fail=SIZE_MAX;ra_persistent=0;CHECK(nvm_record_map_image_copy(source,&image).status==NVM_ARRAY_ELIGIBLE);
        nvm_record_map_image_free(image);CHECK(ra_bytes==held_bytes&&ra_live==held_live);
    }
    nvm_record_map_image_free(prior);CHECK(ra_bytes==source_bytes&&ra_live==source_live);
    for(unsigned kind=0;kind<2;kind++) {
        uint64_t bytes=kind?0:NVM_RECORD_ARRAY_EXECUTION_BYTES-cost.preparation_bytes_reserved;
        uint64_t work=kind?NVM_RECORD_ARRAY_EXECUTION_STEPS-cost.preparation_work_reserved:0;
        CHECK(rme_image_copy_reserved(source,&image,bytes,work).status==NVM_ARRAY_ELIGIBLE);
        NvmRecordMapImageCost exact;CHECK(nvm_record_map_image_cost(image,&exact));
        CHECK(kind?exact.preparation_work_reserved==NVM_RECORD_ARRAY_EXECUTION_STEPS:exact.preparation_bytes_reserved==NVM_RECORD_ARRAY_EXECUTION_BYTES);
        nvm_record_map_image_free(image);image=(void *)(uintptr_t)1;
        CHECK(rme_image_copy_reserved(source,&image,bytes+!kind,work+kind).status==NVM_ARRAY_LIMIT&&image==(void *)(uintptr_t)1);
        CHECK(ra_bytes==source_bytes&&ra_live==source_live);
    }
    size_t calls=ra_calls;image=(void *)(uintptr_t)1;
    CHECK(rme_image_copy_reserved(source,&image,UINT64_MAX,0).status==NVM_ARRAY_LIMIT&&image==(void *)(uintptr_t)1&&calls==ra_calls);
    CHECK(rme_image_copy_reserved(source,&image,0,UINT64_MAX).status==NVM_ARRAY_LIMIT&&image==(void *)(uintptr_t)1&&calls==ra_calls);
    /* Reproof reads mutated snapshot bytes, not the still-valid source facts. */
    for(unsigned bad=0;bad<6;bad++) {
        NvmModule *m=&source->carrier->snapshot;uint8_t *byte=bad==0?&m->code[0]:bad==1?&m->code[1]:bad==2?&m->ownership_data[0]:&m->code[m->code_size-1];
        if(bad>=4) {
            bool found=false;
            for(uint32_t pc=0;pc<m->code_size;) {
                DecodedInstruction in;uint32_t size=isa_decode(m->code+pc,m->code_size-pc,&in);CHECK(size);
                if(bad==4&&in.opcode==OP_PUSH_I64) {byte=&m->code[pc];found=true;break;}
                if(bad==5&&in.opcode==OP_LOAD_LOCAL&&size==3&&m->code[pc+1]==1&&m->code[pc+2]==0) {byte=&m->code[pc+1];found=true;break;}
                pc+=size;
            }
            CHECK(found);
        }
        uint8_t saved=*byte;*byte=bad==0?OP_HM_DELETE:bad==1?TAG_BOOL:bad==2?255:bad==3?OP_POP:bad==4?OP_PUSH_F64:0;
        image=(void *)(uintptr_t)1;CHECK(nvm_record_map_image_copy(source,&image).status!=NVM_ARRAY_ELIGIBLE&&image==(void *)(uintptr_t)1);
        uint8_t key=99,value=98;CHECK(nvm_record_map_execution_map_field(source,0,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
        *byte=saved;CHECK(ra_bytes==source_bytes&&ra_live==source_live);
        CHECK(nvm_record_map_image_copy(source,&image).status==NVM_ARRAY_ELIGIBLE);nvm_record_map_image_free(image);
    }
    printf("I retained %zu independent-image allocation prefixes; overlap %llu bytes/%llu work.\n",allocations,(unsigned long long)cost.preparation_bytes_reserved,(unsigned long long)cost.preparation_work_reserved);
    nvm_record_map_execution_free(source);CHECK(!ra_bytes&&!ra_live);
#endif
}
int main(int argc,char **argv) {
    old_map_execution_main(argc,argv);image_cost_cases();image_lifetimes();image_failure_controls();
    printf("PASS %u retained map image cost checks; public boundaries checked\n",checks);return 0;
}
