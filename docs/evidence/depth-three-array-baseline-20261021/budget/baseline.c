#define main origin_fixture_main
#include "/Users/jkh/Src/nanolang-51-owned-float-design/tests/nanoisa/test_record_array_origins.c"
#undef main
int main(void) {
 printf("analysis_size=%zu\n",sizeof(Analysis));
 for(unsigned mode=0;mode<6;mode++) {
  Input c;unsigned leaves[]={TAG_INT,TAG_INT,TAG_STRING,TAG_U8,TAG_FLOAT,TAG_BOOL};unsigned leaf=leaves[mode];
  if(!mode)basic(&c,TAG_INT,false);
  else {
   init(&c,(uint8_t)leaf,false);array(&c,TAG_ARRAY);arg16(&c,OP_STORE_LOCAL,0);
   array(&c,(uint8_t)leaf);arg16(&c,OP_STORE_LOCAL,1);
   arg16(&c,OP_LOAD_LOCAL,0);record(&c,0);op(&c,OP_POP);
   arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_LOAD_LOCAL,1);op(&c,OP_ARR_PUSH);op(&c,OP_POP);
   finish(&c);nested_authority(&c);c.ownership[c.o-36]=(uint8_t)leaf;
  }
  ra_calls=0;ra_peak=0;NvmRecordArrayOrigins *p=query(&c,NVM_ARRAY_ELIGIBLE);
  NvmRecordArrayOriginCounts n;CHECK(nvm_record_array_origins_counts(p,&n));
  printf("mode=%u calls=%zu actual_peak=%zu reserved_peak=%llu work=%llu\n",mode,ra_calls,ra_peak,(unsigned long long)n.peak_bytes_reserved,(unsigned long long)n.work_reserved);
  nvm_record_array_origins_free(p);CHECK(!ra_live&&!ra_bytes);
 }
 return 0;
}
