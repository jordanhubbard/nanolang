#define UFE_MAIN retained_union_recipe_controls
#include "test_union_field_execution.c"
#include "../../src/nanoisa/union_field_snapshot_private.h"
static int corrupt_variant;
static bool target_variant(const NvmUnionFieldImage *p,uint32_t i,uint16_t j,NvmUnionVariantFact *out) {
 bool ok=nvm_union_field_image_variant(p,i,j,out);if(ok&&corrupt_variant&&j==1)out->field_count=1;return ok;
}
#define nvm_union_field_image_variant target_variant
#include "../../src/nanoisa/nvm2c_union_target_private.inc"
#undef nvm_union_field_image_variant
#ifdef UFP_ALLOC
static void target_status(NvmUnionFieldImage *image,NvmArrayEligibilityStatus expected) {
 Nvm2cUnionTargetContext output;memset(&output,0xa5,sizeof output);Nvm2cUnionTargetContext before=output;
 Nvm2cUnionTargetResult r=nvm2c_union_target(image,UFN_WORK_LIMIT,&output);
 if(r.eligibility.status!=expected)fprintf(stderr,"target expected%u actual%u work%llu: %s\n",expected,r.eligibility.status,(unsigned long long)r.work,r.eligibility.message);
 CHECK(r.eligibility.status==expected);if(expected!=NVM_ARRAY_ELIGIBLE)CHECK(!memcmp(&output,&before,sizeof output));
}
static void controls(void) {
 for(unsigned op=0;op<256;op++) {
  NvmRecordArrayExecutionInstruction a={0},b={0};int x=ufe_recipe((uint8_t)op,&a),y=ufn_recipe((uint8_t)op,&b);
  CHECK(x==y);if(x)CHECK(a.recipe==b.recipe&&a.obligations==b.obligations);
 }
 UnionCase v;uint8_t code[128];execution_case(&v,code);
 NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);NvmUnionFieldImage *image=NULL;
 CHECK(nvm_union_field_image_copy(p,&image).status==NVM_ARRAY_ELIGIBLE);
 nvm_union_field_execution_free(p);memset(&v,0,sizeof v);memset(code,0,sizeof code);
 target_status(image,NVM_ARRAY_ELIGIBLE);
 Nvm2cUnionTargetContext output;memset(&output,0xa5,sizeof output);Nvm2cUnionTargetContext prior=output;
 Nvm2cUnionTargetResult budget=nvm2c_union_target(image,UFN_WORK_LIMIT,&output);CHECK(budget.eligibility.status==NVM_ARRAY_ELIGIBLE);
 uint64_t exact=budget.work;CHECK(exact>0);output=prior;
 budget=nvm2c_union_target(image,exact-1,&output);CHECK(budget.eligibility.status==NVM_ARRAY_LIMIT&&!memcmp(&output,&prior,sizeof output));
 budget=nvm2c_union_target(image,exact,&output);CHECK(budget.eligibility.status==NVM_ARRAY_ELIGIBLE&&budget.work==exact);
 NvmUnionFieldExecutionPlan *q=image->plan;
 uint32_t code_size=q->carrier->snapshot.code_size;q->carrier->snapshot.code_size=UINT32_MAX;
 target_status(image,NVM_ARRAY_INVALID);q->carrier->snapshot.code_size=code_size;target_status(image,NVM_ARRAY_ELIGIBLE);
 q->sites[2].reserved[0]=1;target_status(image,NVM_ARRAY_INVALID);q->sites[2].reserved[0]=0;
 uint64_t saved=q->sites[2].result.origins;q->sites[2].result.origins=UINT64_C(1)<<63;
 target_status(image,NVM_ARRAY_INVALID);q->sites[2].result.origins=saved;
 q->carrier->instructions[0].operand_bits[1]=1;target_status(image,NVM_ARRAY_INVALID);q->carrier->instructions[0].operand_bits[1]=0;
 uint32_t next=q->carrier->instructions[0].successors[0];q->carrier->instructions[0].successors[0]=1;
 target_status(image,NVM_ARRAY_INVALID);q->carrier->instructions[0].successors[0]=next;
 uint8_t count=q->carrier->instructions[0].successor_count;q->carrier->instructions[0].successor_count=2;
 target_status(image,NVM_ARRAY_INVALID);q->carrier->instructions[0].successor_count=count;
 corrupt_variant=1;target_status(image,NVM_ARRAY_INVALID);corrupt_variant=0;
 q->carrier->snapshot.import_count=1;target_status(image,NVM_ARRAY_UNRESOLVED);
 q->sites[2].reserved[1]=1;target_status(image,NVM_ARRAY_INVALID);q->sites[2].reserved[1]=0;q->carrier->snapshot.import_count=0;
 uint16_t tags=q->proof->origins[0].member.tags;q->proof->origins[0].member.tags=(uint16_t)(1u<<15);
 target_status(image,NVM_ARRAY_INVALID);q->proof->origins[0].member.tags=tags;
 CHECK(q->proof->counts.origins==2);
 q->proof->origins=ra_test_realloc(q->proof->origins,3*sizeof *q->proof->origins);CHECK(q->proof->origins);
 q->proof->origins[2]=(NvmUnionFieldOrigin){.function=0,.pc=0,.kind=NVM_UNION_FIELD_ORIGIN,.variant=1,.layout_index=0};
 q->proof->counts.origins=3;target_status(image,NVM_ARRAY_INVALID); /* unused scalar origin */
 q->proof->origins[2].pc=1;target_status(image,NVM_ARRAY_INVALID); /* middle of PUSH_I64 */
 q->proof->origins[2]=q->proof->origins[0];target_status(image,NVM_ARRAY_INVALID); /* duplicate construction */
 q->proof->counts.origins=2;target_status(image,NVM_ARRAY_ELIGIBLE);

 nvm_union_field_image_free(image);CHECK(!ra_live&&!ra_bytes);
 /* A genuine independently constructed dormant None allocation, not tampering. */
 union_case(&v);size_t n=0;proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
 u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
 proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_AGG_TAG);u8(code,&n,OP_RET);proof_header(&v,code,n);
 p=execution_query(&v.c.module);image=NULL;CHECK(nvm_union_field_image_copy(p,&image).status==NVM_ARRAY_ELIGIBLE);
 NvmUnionFieldExecutionInstruction dormant;CHECK(nvm_union_field_image_instruction(image,4,&dormant));
 CHECK(dormant.site.state==UFS_NOT_ESTABLISHED&&dormant.site.operation==UFS_VARIANT_PACK);
 target_status(image,NVM_ARRAY_UNRESOLVED);nvm_union_field_image_free(image);nvm_union_field_execution_free(p);CHECK(!ra_live&&!ra_bytes);

 execution_case(&v,code);UnionCase before=v;uint8_t code_before[128];memcpy(code_before,code,sizeof code);
 ra_calls=0;ra_fail=SIZE_MAX;ra_peak=0;
 Nvm2cUnionTargetResult r=nvm2c_prepare_union_target_source(&v.c.module);CHECK(r.eligibility.status==NVM_ARRAY_ELIGIBLE);
 size_t positions=ra_calls,peak=ra_peak;CHECK(positions>0&&!ra_live&&!ra_bytes);
 for(int persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<positions;fail++) {
  ra_calls=0;ra_fail=fail;ra_persistent=persistent;
  r=nvm2c_prepare_union_target_source(&v.c.module);
  CHECK(r.eligibility.status==NVM_ARRAY_MEMORY);CHECK(!ra_live&&!ra_bytes);
  CHECK(!memcmp(&v,&before,sizeof v)&&!memcmp(code,code_before,sizeof code));
  ra_fail=SIZE_MAX;ra_persistent=0;ra_calls=0;
  r=nvm2c_prepare_union_target_source(&v.c.module);CHECK(r.eligibility.status==NVM_ARRAY_ELIGIBLE&&!ra_live&&!ra_bytes);
 }
 printf("wrapper positions=%zu modes=2 peak=%zu live=%zu bytes=%zu\n",positions,peak,ra_live,ra_bytes);
}
#endif
#ifndef UFP_ALLOC
static void target_original(const char *path) {
 FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long size=ftell(f);CHECK(size>0);rewind(f);
 uint8_t *bytes=malloc((size_t)size);CHECK(bytes);CHECK(fread(bytes,1,(size_t)size,f)==(size_t)size);CHECK(!fclose(f));
 NvmV2Module wire={0};CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&wire)==NVM_V2_OK);
 NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&wire,&m)==NVM_V2_OK);
 Nvm2cUnionTargetResult result=nvm2c_prepare_union_target_source(m);
 printf("target %s status=%u work=%llu scratch=%llu message=%s\n",path,result.eligibility.status,
  (unsigned long long)result.work,(unsigned long long)result.scratch_reserved,result.eligibility.message);fflush(stdout);
 CHECK(result.eligibility.status==NVM_ARRAY_ELIGIBLE);
 NvmUnionFieldExecutionPlan *plan=execution_query(m);NvmUnionFieldImage *image=NULL;
 CHECK(nvm_union_field_image_copy(plan,&image).status==NVM_ARRAY_ELIGIBLE);
 Nvm2cUnionTargetContext context;memset(&context,0xa5,sizeof context);Nvm2cUnionTargetContext prior=context;
 result=nvm2c_union_target(image,UFN_WORK_LIMIT,&context);CHECK(result.eligibility.status==NVM_ARRAY_ELIGIBLE);
 uint64_t exact=result.work;CHECK(context.record_layout==1&&context.union_layout==0);
 context=prior;result=nvm2c_union_target(image,exact-1,&context);
 CHECK(result.eligibility.status==NVM_ARRAY_LIMIT&&!memcmp(&context,&prior,sizeof context));
 result=nvm2c_union_target(image,exact,&context);CHECK(result.eligibility.status==NVM_ARRAY_ELIGIBLE&&result.work==exact);
 context=prior;result=nvm2c_union_target(NULL,UFN_WORK_LIMIT,&context);
 CHECK(result.eligibility.status==NVM_ARRAY_INVALID&&!memcmp(&context,&prior,sizeof context));
 /* Internal image module corruption, restored before independent valid retry. */
 NvmModule *copy=(NvmModule *)nvm_union_field_image_module_internal(image);
 uint32_t saved=copy->function_count;copy->function_count=0;
 result=nvm2c_union_target(image,UFN_WORK_LIMIT,&context);
 CHECK(result.eligibility.status==NVM_ARRAY_INVALID&&!memcmp(&context,&prior,sizeof context));copy->function_count=saved;
 result=nvm2c_union_target(image,UFN_WORK_LIMIT,&context);CHECK(result.eligibility.status==NVM_ARRAY_ELIGIBLE);
 UfnOrigins origins={0};NvmRecordValueOrigins value={0};value.tags=(uint16_t)(1u<<15);
 CHECK(!ufn_origins_value(&origins,value));value.tags=(uint16_t)(1u<<TAG_INT);value.origins=1;CHECK(!ufn_origins_value(&origins,value));
 nvm_union_field_image_free(image);nvm_union_field_execution_free(plan);nvm_module_free(m);nvm_v2_module_free(&wire);free(bytes);
}
#endif
int main(int argc,char **argv) {
#ifdef UFP_ALLOC
 (void)argv;CHECK(argc==1);controls();printf("PASS %u observed native target checks; originals=0\n",checks);
#else
 CHECK(argc==1||argc==7);UnionCase v;uint8_t code[128];execution_case(&v,code);
 Nvm2cUnionTargetResult r=nvm2c_prepare_union_target_source(&v.c.module);CHECK(r.eligibility.status==NVM_ARRAY_ELIGIBLE);
 for(int i=1;i<argc;i++)target_original(argv[i]);
 printf("PASS %u native target checks; originals=%d; no emission\n",checks,argc-1);
#endif
 return 0;
}
