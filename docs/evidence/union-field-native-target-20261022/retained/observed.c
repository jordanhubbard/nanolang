#define UFE_MAIN retained_union_recipe_controls
#include "/Users/jkh/Src/nanolang-51-platform-qualification/tests/nanoisa/test_union_field_execution.c"
#include "/Users/jkh/Src/nanolang-51-platform-qualification/src/nanoisa/union_field_snapshot_private.h"
static int corrupt_variant;
static bool target_variant(const NvmUnionFieldImage *p,uint32_t i,uint16_t j,NvmUnionVariantFact *out) {
 bool ok=nvm_union_field_image_variant(p,i,j,out);if(ok&&corrupt_variant&&j==1)out->field_count=1;return ok;
}
#define nvm_union_field_image_variant target_variant
#include "/Users/jkh/Src/nanolang-51-platform-qualification/src/nanoisa/nvm2c_union_target_private.inc"
#undef nvm_union_field_image_variant
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
 NvmUnionFieldExecutionPlan *q=image->plan;
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
int main(void) {controls();printf("PASS %u observed native target checks\n",checks);return 0;}
