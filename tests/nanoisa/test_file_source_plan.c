/* I test descriptive C ownership only; validated NSI and complete source facts
 * remain caller preconditions. I execute no File service. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../src/nanoisa/file_source_plan.h"
static size_t checks;
#define CHECK(x) do { checks++; if(!(x)){fprintf(stderr,"I failed line %d: %s\n",__LINE__,#x);abort();} } while(0)
#ifdef SOURCE_PLAN_INSTRUMENT
static size_t calls,live,last_size;
static bool fail_alloc;
static void *source_malloc(size_t n){calls++;last_size=n;if(fail_alloc)return NULL;void *p=malloc(n);if(p)live++;return p;}
static void source_free(void *p){if(p){CHECK(live==1);live--;}free(p);}
#define malloc source_malloc
#define free source_free
#include "../../src/nanoisa/file_source_plan.c"
#undef malloc
#undef free
#endif
static NlFileSourceText text(const char *s){return (NlFileSourceText){s,strlen(s)};}
static void bindings(NlFileSourceBinding *b,uint32_t base){
 for(uint32_t i=0;i<13;i++){uint32_t k=i>=8,o=k?i-8:i;b[i]=(NlFileSourceBinding){base+i,k,o,text(nl_file_source_catalog_string(k+1,o,1,0))};}
}
static void expect(NlFileSourceRequest *q,size_t nq,NlFileSourceAlias *a,size_t na,NlFileSourceOrdinary *o,size_t no,NlFileSourceStatus status){
 NlFileSourcePlan *sentinel=(NlFileSourcePlan *)(uintptr_t)1,*p=sentinel;
 CHECK(nl_file_source_plan_build(q,nq,a,na,o,no,&p)==status);
 if(status==NL_FILE_SOURCE_OK){CHECK(p!=sentinel);nl_file_source_plan_free(p);}else CHECK(p==sentinel);
}
int main(void){
 char catalog[32768];size_t needed=0;CHECK(nl_file_source_catalog_view(NULL,0,&needed));CHECK(needed>100 && needed<=sizeof catalog);
 memset(catalog,0xa5,sizeof catalog);size_t unchanged=17;
 CHECK(!nl_file_source_catalog_view(catalog,needed-1,&unchanged));CHECK(unchanged==17);CHECK((unsigned char)catalog[0]==0xa5);
 CHECK(nl_file_source_catalog_view(catalog,sizeof catalog,&needed));CHECK(needed==strlen(catalog)+1);
 for(int k=-1;k<4;k++)for(int i=-1;i<10;i++){
  if(k<0||k>2||i<0||(k==0&&i!=0)||(k==1&&i>=8)||(k==2&&i>=5)){
   CHECK(!strcmp(nl_file_source_catalog_string(k,i,0,0),""));CHECK(nl_file_source_catalog_number(k,i,0,0)==-1);
  }
 }
 CHECK(!strcmp(nl_file_source_catalog_string(1,0,99,0),""));CHECK(nl_file_source_catalog_number(2,0,99,0)==-1);
 NlFileSourceBinding b[13];bindings(b,1);
 NlFileSourceRequest q={text("module"),text("nsi:nanolang/filesystem"),text(catalog),1,1,2,b,13};
 expect(&q,1,NULL,0,NULL,0,NL_FILE_SOURCE_OK);
 CHECK(nl_file_source_plan_build(&q,1,NULL,0,NULL,0,NULL)==NL_FILE_SOURCE_INVALID);
 expect(NULL,1,NULL,0,NULL,0,NL_FILE_SOURCE_INVALID);expect(&q,0,NULL,0,NULL,0,NL_FILE_SOURCE_INVALID);
 NlFileSourceRequest many[17];for(size_t i=0;i<17;i++)many[i]=q;
 expect(many,17,NULL,0,NULL,0,NL_FILE_SOURCE_LIMIT);
 NlFileSourceAlias dummy={text("a"),text("a"),100,1};NlFileSourceOrdinary od={text("b"),text("b"),101};
 NlFileSourceAlias too_many_aliases[65];NlFileSourceOrdinary too_many_ordinary[257];
 for(size_t i=0;i<65;i++)too_many_aliases[i]=dummy;
 for(size_t i=0;i<257;i++)too_many_ordinary[i]=od;
 expect(&q,1,too_many_aliases,65,NULL,0,NL_FILE_SOURCE_LIMIT);expect(&q,1,NULL,0,too_many_ordinary,257,NL_FILE_SOURCE_LIMIT);
 expect(&q,1,NULL,1,NULL,0,NL_FILE_SOURCE_INVALID);expect(&q,1,NULL,0,NULL,1,NL_FILE_SOURCE_INVALID);
 for(size_t i=0;i<q.catalog_view.size;i++){
  char saved=catalog[i];catalog[i]=saved=='x'?'y':'x';expect(&q,1,NULL,0,NULL,0,NL_FILE_SOURCE_UNRESOLVED);catalog[i]=saved;
 }
 /* I copy all output strings before the caller mutates and frees inputs. */
 char *module=malloc(7),*name=malloc(5);CHECK(module&&name);memcpy(module,"module",7);memcpy(name,"File",5);
 q.module=text(module);b[0].name=text(name);NlFileSourcePlan *p=NULL;
 CHECK(nl_file_source_plan_build(&q,1,NULL,0,NULL,0,&p)==NL_FILE_SOURCE_OK);
 size_t owned=0;for(size_t i=0;i<13;i++)owned+=q.module.size+1+b[i].name.size+1;
 CHECK(nl_file_source_plan_bytes(p)==2*sizeof(size_t)+13*sizeof(NlFileSourceRow)+owned);
 memset(module,'z',6);memset(name,'z',4);free(module);free(name);
 NlFileSourceRow row;CHECK(nl_file_source_plan_row(p,0,&row));CHECK(row.module.size==6&&!memcmp(row.module.data,"module",6));CHECK(row.name.size==4&&!memcmp(row.name.data,"File",4));
 NlFileSourceRow before=row;CHECK(!nl_file_source_plan_row(p,13,&row));CHECK(!memcmp(&before,&row,sizeof row));
 CHECK(!nl_file_source_plan_row(NULL,0,&row));CHECK(!memcmp(&before,&row,sizeof row));CHECK(!nl_file_source_plan_row(p,0,NULL));
 CHECK(row.module.data[row.module.size]==0 && row.name.data[row.name.size]==0);
 nl_file_source_plan_free(p);CHECK(nl_file_source_plan_count(NULL)==0&&nl_file_source_plan_bytes(NULL)==0);nl_file_source_plan_free(NULL);
 bindings(b,1);q.module=text("module");
#ifdef SOURCE_PLAN_INSTRUMENT
 size_t prior=calls;fail_alloc=true;expect(&q,1,NULL,0,NULL,0,NL_FILE_SOURCE_MEMORY);CHECK(calls==prior+1&&live==0);fail_alloc=false;
 expect(&q,1,NULL,0,NULL,0,NL_FILE_SOURCE_OK);CHECK(calls==prior+2&&live==0&&last_size>0);
 /* I check the precise internal budget boundary, independently of reachability. */
 size_t budget=NL_FILE_SOURCE_TEXT_BUDGET-2;CHECK(add_text(&budget,text("x")));CHECK(budget==NL_FILE_SOURCE_TEXT_BUDGET);CHECK(!add_text(&budget,text("x")));CHECK(budget==NL_FILE_SOURCE_TEXT_BUDGET);
#endif
 /* I reach the public logical budget with ordinary facts, not oversized fields. */
 char large[4097];memset(large,'m',4096);large[4096]=0;NlFileSourceOrdinary ordinary[256];char names[256][16];
 size_t base=q.module.size+1+q.interface_id.size+1+q.catalog_view.size+1;
 for(size_t i=0;i<13;i++)base+=b[i].name.size+1+q.module.size+1+b[i].name.size+1;
 size_t used=base,count=0;
 while(count<256){snprintf(names[count],sizeof names[count],"n%zu",count);size_t n=strlen(names[count])+1;
  size_t remain=NL_FILE_SOURCE_TEXT_BUDGET-used;if(remain<=n+1)break;
  size_t len=remain-n-1;if(len>4096)len=4096;ordinary[count]=(NlFileSourceOrdinary){{large,len},text(names[count]),(uint32_t)(1000+count)};used+=len+1+n;count++;
  if(used==NL_FILE_SOURCE_TEXT_BUDGET)break;
 }
 CHECK(used==NL_FILE_SOURCE_TEXT_BUDGET&&count<=256);expect(&q,1,NULL,0,ordinary,count,NL_FILE_SOURCE_OK);
 CHECK(ordinary[count-1].module.size<4096);ordinary[count-1].module.size++;expect(&q,1,NULL,0,ordinary,count,NL_FILE_SOURCE_LIMIT);
#ifdef SOURCE_PLAN_INSTRUMENT
 CHECK(live==0);
#endif
 printf("PASS source C ownership %zu checks\n",checks);return 0;
}
