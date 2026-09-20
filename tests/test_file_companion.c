/* I observe selected provider allocations and real input descriptors. Injected
 * close failure occurs AFTER actual close: I model reporting, not libc failure. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "file_source_input.h"
#include "file_companion_bridge.h"
#ifdef COMPANION_GRAPH
#include "file_source_resolution.h"
#endif
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static size_t checks;
#define CHECK(x) do { ++checks; if(!(x)){fprintf(stderr,"I failed %s:%d: %s\n",__FILE__,__LINE__,#x);exit(1);} } while(0)
static void *owned[100000];
static size_t live,attempts,fail_at,fail_hits;
static int allow_external_free;
static int transient;
static size_t allocations_at_success;
static size_t opens,closes,reads,stats;
static int fail_open,fail_stat,fail_read,fail_close,short_eof,extra_byte,interrupts;
static void remember(void *p){if(p){CHECK(live<100000);owned[live++]=p;}}
static void forget(void *p){if(!p)return;for(size_t i=0;i<live;i++)if(owned[i]==p){owned[i]=owned[--live];return;}CHECK(allow_external_free);}
static int failing(void){++attempts;if(fail_at&&attempts>=fail_at&&(!transient||attempts==fail_at)){++fail_hits;return 1;}return 0;}
void *fc_test_malloc(size_t n){if(failing())return NULL;void *p=malloc(n);remember(p);return p;}
void *fc_test_calloc(size_t n,size_t s){if(failing())return NULL;void *p=calloc(n,s);remember(p);return p;}
void *fc_test_realloc(void *p,size_t n){if(failing())return NULL;size_t index=live;for(size_t i=0;i<live;i++)if(owned[i]==p){index=i;break;}CHECK(!p||index<live);void *q=realloc(p,n);if(q){if(index<live)owned[index]=q;else remember(q);}return q;}
void fc_test_free(void *p){forget(p);free(p);}
char *fc_test_strdup(const char *s){size_t n=strlen(s)+1;char *p=fc_test_malloc(n);if(p)memcpy(p,s,n);return p;}
int fc_test_open(const char *p,int flags,...){++opens;if(fail_open){errno=EACCES;return -1;}return open(p,flags);}
int fc_test_openat(int fd,const char *p,int flags,...){++opens;if(fail_open){errno=EACCES;return -1;}return openat(fd,p,flags);}
int fc_test_fstat(int fd,struct stat *s){++stats;if(fail_stat){errno=EIO;return -1;}return fstat(fd,s);}
ssize_t fc_test_read(int fd,void *p,size_t n){++reads;if(interrupts){--interrupts;errno=EINTR;return -1;}if(fail_read){errno=EIO;return -1;}if(short_eof)return 0;ssize_t got=read(fd,p,n);if(!got&&extra_byte){*(char *)p='x';return 1;}return got;}
int fc_test_close(int fd){++closes;int rc=close(fd);if(fail_close){CHECK(rc==0);errno=EIO;return -1;}return rc;}
static void reset(void){CHECK(live==0);attempts=fail_at=fail_hits=0;transient=0;opens=closes=reads=stats=0;fail_open=fail_stat=fail_read=fail_close=short_eof=extra_byte=interrupts=0;}
static void write_bytes(const char *path,const void *p,size_t n){FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(p,1,n,f)==n);CHECK(fclose(f)==0);}
static NlFileSourceText text(const char *s){return (NlFileSourceText){s,strlen(s)};}
static void release(void *p){
#ifdef COMPANION_INSTRUMENT
fc_test_free(p);
#else
free(p);
#endif
}
static void input_controls(const char *path){
 write_bytes(path,"abc",3);char *sentinel=(char *)(uintptr_t)1,*out=sentinel;size_t n=999;
 NlFileCompanionReport r=nl_file_source_input_read(path,4,&out,&n);
 CHECK(r.status==NL_FILE_COMPANION_OK);CHECK(n==3);CHECK(!memcmp(out,"abc",4));CHECK(r.peak_heap_bytes_reserved==4);release(out);CHECK(live==0);
 out=sentinel;n=999;r=nl_file_source_input_read(path,3,&out,&n);CHECK(r.status==NL_FILE_COMPANION_LIMIT);CHECK(out==sentinel&&n==999);
 const unsigned char invalid[][4]={{'a',0,'b',0},{0xc0,0xaf,0,0},{0xed,0xa0,0x80,0},{0xf4,0x90,0x80,0x80}};
 const size_t sizes[]={3,2,3,4};
 for(size_t i=0;i<4;i++){write_bytes(path,invalid[i],sizes[i]);r=nl_file_source_input_read(path,16,&out,&n);CHECK(r.status==NL_FILE_COMPANION_INVALID);CHECK(out==sentinel&&n==999);CHECK(live==0);}
 const unsigned char valid[]={0xc3,0xa9};write_bytes(path,valid,2);r=nl_file_source_input_read(path,3,&out,&n);CHECK(r.status==0&&n==2);CHECK(!memcmp(out,valid,2));release(out);
#ifdef COMPANION_INSTRUMENT
 write_bytes(path,"abc",3);
 for(int mode=0;mode<8;mode++){
  reset();out=sentinel;n=999;
  if(mode==0){fail_open=1;}if(mode==1){fail_stat=1;}if(mode==2){fail_read=1;}if(mode==3){short_eof=1;}
  if(mode==4){extra_byte=1;}if(mode==5){fail_close=1;}if(mode==6){fail_read=1;fail_close=1;}if(mode==7)fail_at=1;
  r=nl_file_source_input_read(path,4,&out,&n);CHECK(r.status!=0);CHECK(out==sentinel&&n==999);CHECK(closes==(mode==0?0u:1u));CHECK(live==0);
  if(mode==6){CHECK(r.stage==NL_FILE_COMPANION_READ);CHECK(r.system_error==EIO&&r.close_error==EIO);}
  if(mode==7)CHECK(r.status==NL_FILE_COMPANION_MEMORY);
 }
 reset();interrupts=64;r=nl_file_source_input_read(path,4,&out,&n);CHECK(r.status==0);CHECK(reads==66);release(out);
 reset();interrupts=65;out=sentinel;n=999;r=nl_file_source_input_read(path,4,&out,&n);CHECK(r.status==NL_FILE_COMPANION_IO);CHECK(reads==65);CHECK(closes==1);CHECK(out==sentinel&&n==999);
#endif
 reset();
 out=sentinel;n=999;r=nl_file_source_input_read(path,0,&out,&n);CHECK(r.status==NL_FILE_COMPANION_INVALID);CHECK(out==sentinel&&n==999);
 r=nl_file_source_input_read(path,(size_t)NL_FILE_COMPANION_BYTES+1,&out,&n);CHECK(r.status==NL_FILE_COMPANION_INVALID);CHECK(out==sentinel&&n==999);
 int fd=open(path,O_WRONLY|O_TRUNC);CHECK(fd>=0);CHECK(ftruncate(fd,(off_t)NL_FILE_COMPANION_BYTES)==0);CHECK(close(fd)==0);
 r=nl_file_source_input_read(path,(size_t)NL_FILE_COMPANION_BYTES,&out,&n);CHECK(r.status==NL_FILE_COMPANION_LIMIT);CHECK(out==sentinel&&n==999);CHECK(live==0);
 write_bytes(path,"abc",3);reset();
}
static void snapshot_controls(const char *module,const char *document){
 NlFileCompanionRequest request={text(module),text("interface.nsi.json"),text("nsi:nanolang/filesystem"),1,2,3};
 NlFileCompanionSet *sentinel=(NlFileCompanionSet *)(uintptr_t)1,*set=sentinel;
 reset();NlFileCompanionReport r=nl_file_companions_prepare(&request,1,&set);CHECK(r.status==0);CHECK(set!=sentinel);CHECK(nl_file_companions_count(set)==1);
 size_t count=attempts;allocations_at_success=count;NlFileCompanionView v;CHECK(nl_file_companions_view(set,0,&v));CHECK(v.original_document.size>0);CHECK(v.generated_source.size>0);CHECK(v.catalog_view.size>0);CHECK(v.request.line==2&&v.request.column==3);
 char *copy=malloc(v.original_document.size);CHECK(copy);memcpy(copy,v.original_document.data,v.original_document.size);
 /* I prove snapshot immutability after the original file is replaced. */
 write_bytes(document,"{}",2);CHECK(!memcmp(v.original_document.data,copy,v.original_document.size));
 write_bytes(document,copy,v.original_document.size);free(copy);
 NlFileCompanionView untouched=v;CHECK(!nl_file_companions_view(set,1,&untouched));CHECK(!memcmp(&untouched,&v,sizeof(v)));
 size_t generated_size=v.generated_source.size,catalog_size=v.catalog_view.size;
 char *generated=malloc(generated_size),*catalog=malloc(catalog_size);CHECK(generated&&catalog);
 memcpy(generated,v.generated_source.data,generated_size);memcpy(catalog,v.catalog_view.data,catalog_size);
 nl_file_companions_free(set);CHECK(live==0);
#ifdef COMPANION_INSTRUMENT
 for(int one=0;one<2;one++)for(size_t prefix=1;prefix<=count;prefix++){
  reset();fail_at=prefix;transient=one;set=sentinel;r=nl_file_companions_prepare(&request,1,&set);CHECK(fail_hits>0);
  printf("ALLOCATION transient=%d index=%zu status=%d hits=%zu\n",one,prefix,r.status,fail_hits);
  if(r.status==0){CHECK(nl_file_companions_view(set,0,&v));CHECK(v.generated_source.size==generated_size&&v.catalog_view.size==catalog_size);CHECK(!memcmp(v.generated_source.data,generated,generated_size));CHECK(!memcmp(v.catalog_view.data,catalog,catalog_size));nl_file_companions_free(set);}
  else CHECK(set==sentinel);
  CHECK(live==0);
 }
#else
 CHECK(count==0);
#endif
 free(generated);free(catalog);
 reset();request.companion_path=text("../interface.nsi.json");set=sentinel;r=nl_file_companions_prepare(&request,1,&set);CHECK(r.status==NL_FILE_COMPANION_INVALID);CHECK(set==sentinel);CHECK(opens==0);CHECK(live==0);
}
static void snapshot_boundaries(const char *module){
 NlFileCompanionRequest valid={text(module),text("interface.nsi.json"),text("nsi:nanolang/filesystem"),1,1,1};
 NlFileCompanionSet *sentinel=(NlFileCompanionSet *)(uintptr_t)1,*out=sentinel;
 reset();NlFileCompanionReport r=nl_file_companions_prepare(&valid,17,&out);CHECK(r.status==NL_FILE_COMPANION_LIMIT);CHECK(out==sentinel);CHECK(opens==0&&attempts==0);
 const char *bad_paths[]={"","/absolute",".","..","a/../b","a/./b","a//b","a/"};
 for(size_t i=0;i<sizeof(bad_paths)/sizeof(*bad_paths);i++){
  reset();NlFileCompanionRequest q=valid;q.companion_path=text(bad_paths[i]);
  r=nl_file_companions_prepare(&q,1,&out);CHECK(r.status==NL_FILE_COMPANION_INVALID);CHECK(out==sentinel);CHECK(opens==0&&live==0);
 }
 for(int mode=0;mode<5;mode++){
  reset();NlFileCompanionRequest q[2]={valid,valid};
  if(mode==0){q[1].catalog_version=2;}if(mode==1){q[1].line=0;}if(mode==2){q[1].column=0;}
  if(mode==3){q[1].interface_id=text("nsi:nanolang/wrong");}if(mode==4){q[1].module_path=text("relative.nano");}
  r=nl_file_companions_prepare(q,2,&out);CHECK(r.status==NL_FILE_COMPANION_INVALID);CHECK(r.request==1);CHECK(out==sentinel);CHECK(opens==0&&live==0);
 }
#ifdef COMPANION_INSTRUMENT
 for(int mode=0;mode<5;mode++){
  reset();if(mode==0){fail_open=1;}if(mode==1){fail_stat=1;}if(mode==2){fail_read=1;}if(mode==3){fail_close=1;}if(mode==4){fail_read=1;fail_close=1;}
  r=nl_file_companions_prepare(&valid,1,&out);CHECK(r.status==NL_FILE_COMPANION_IO);CHECK(out==sentinel&&live==0);
  CHECK(closes==(mode==0?0u:2u));
  if(mode==4){CHECK(r.system_error==EIO&&r.close_error==EIO);CHECK(r.stage==NL_FILE_COMPANION_READ);}
 }
 reset();interrupts=64;r=nl_file_companions_prepare(&valid,1,&out);CHECK(r.status==0);CHECK(closes==2);nl_file_companions_free(out);CHECK(live==0);
 reset();interrupts=65;out=sentinel;r=nl_file_companions_prepare(&valid,1,&out);CHECK(r.status==NL_FILE_COMPANION_IO);CHECK(reads==65&&closes==2);CHECK(out==sentinel&&live==0);
#endif
 reset();
}
static void bridge_controls(const char *path){
 reset();int64_t first=nl_file_companion_source(path,4096);CHECK(first>0);CHECK(nl_file_companion_number(first,-1,0)==0);
 CHECK(nl_file_companion_source(path,4096)==0);CHECK(nl_file_companion_number(first,0,3)==3);CHECK(!strcmp(nl_file_companion_text(first,0,3),"abc"));
 CHECK(nl_file_companion_destroy(first));CHECK(!nl_file_companion_destroy(first));CHECK(nl_file_companion_number(first,-1,0)==-1);
 int64_t next=nl_file_companion_source(path,4096);CHECK(next>first);CHECK(nl_file_companion_destroy(next));CHECK(live==0);
 int64_t bad=nl_file_companion_open("0;trailing",10);CHECK(bad>next);CHECK(nl_file_companion_number(bad,-1,0)!=0);CHECK(nl_file_companion_destroy(bad));CHECK(live==0);
}

#ifdef COMPANION_GRAPH
static uint64_t hash_span(uint64_t h,NlFileSourceText s){for(size_t i=0;i<s.size;i++){h^=(unsigned char)s.data[i];h*=UINT64_C(1099511628211);}return (h^s.size)*UINT64_C(1099511628211);}
static uint64_t graph_facts(NlFileResolution *r){
 uint64_t hash=UINT64_C(1469598103934665603);
 for(size_t i=0;i<nl_file_source_resolution_visibility(r);i++){
  NlFileVisibility v;CHECK(nl_file_source_resolution_row(r,i,&v));
  hash=hash_span(hash,v.origin);hash=hash_span(hash,v.qualifier);hash=hash_span(hash,v.name);hash=hash_span(hash,v.target_origin);hash=hash_span(hash,v.target_name);
  const uint32_t fields[]={v.id,v.target,v.kind,v.ordinal,v.exported,v.service};
  for(size_t j=0;j<6;j++)hash=(hash^fields[j])*UINT64_C(1099511628211);
 }
 for(size_t i=0;i<nl_file_source_resolution_plan_rows(r);i++){
  NlFileSourceRow v;CHECK(nl_file_source_resolution_plan_row(r,i,&v));hash=hash_span(hash,v.module);hash=hash_span(hash,v.name);
  const uint32_t fields[]={v.id,v.target,v.request,v.kind,v.ordinal,v.category,v.input_mode,v.result_ordinal,v.global_layout,v.import_index,v.line,v.column};
  for(size_t j=0;j<12;j++)hash=(hash^fields[j])*UINT64_C(1099511628211);
 }
 return hash;
}
static void graph_allocations(const char *path){
 reset();allow_external_free=1;
 NlFileResolution *sentinel=(NlFileResolution *)(uintptr_t)1,*out=sentinel;
 NlFileResolutionReport report=nl_file_source_resolve(path,&out);CHECK(report.status==NL_FILE_RESOLUTION_PREPARED);
 size_t count=attempts,rows=nl_file_source_resolution_visibility(out),plans=nl_file_source_resolution_plan_rows(out);uint64_t expected=graph_facts(out);
 nl_file_source_resolution_free(out);CHECK(live==0);
 for(int one=0;one<2;one++)for(size_t prefix=1;prefix<=count;prefix++){
  reset();fail_at=prefix;transient=one;out=sentinel;report=nl_file_source_resolve(path,&out);CHECK(fail_hits>0);
  printf("GRAPH_ALLOCATION transient=%d index=%zu status=%d hits=%zu\n",one,prefix,report.status,fail_hits);
  if(report.status==NL_FILE_RESOLUTION_PREPARED){CHECK(nl_file_source_resolution_visibility(out)==rows);CHECK(nl_file_source_resolution_plan_rows(out)==plans);CHECK(graph_facts(out)==expected);nl_file_source_resolution_free(out);}
  else CHECK(out==sentinel);
  CHECK(live==0);
 }
 reset();allow_external_free=0;
}
#endif
int main(int argc,char **argv){
#ifdef COMPANION_GRAPH
CHECK(argc==5);graph_allocations(argv[4]);
#else
CHECK(argc==4);
#endif
input_controls(argv[1]);snapshot_controls(argv[2],argv[3]);snapshot_boundaries(argv[2]);write_bytes(argv[1],"abc",3);bridge_controls(argv[1]);printf("PASS companion checks=%zu allocations=%zu\n",checks,allocations_at_success);return 0;}
