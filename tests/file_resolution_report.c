/* I serialize original counted tuples, never just encoded namespace keys. */
#include "file_source_resolution.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"I failed line %d: %s\n",__LINE__,#x);return 1;}}while(0)
static void span(NlFileSourceText t){printf("%zu:",t.size);for(size_t i=0;i<t.size;i++)printf("%02x",(unsigned char)t.data[i]);}
int main(int argc,char **argv){
 CHECK(argc==2);NlFileResolution *sentinel=(NlFileResolution *)(uintptr_t)1,*r=sentinel;
 NlFileResolutionReport report=nl_file_source_resolve(argv[1],&r);printf("STATUS %u\n",(unsigned)report.status);
 if(report.status!=NL_FILE_RESOLUTION_PREPARED){CHECK(r==sentinel);return 0;}
 CHECK(r&&r!=sentinel);printf("COUNTS %zu %zu %zu\n",nl_file_source_resolution_origins(r),nl_file_source_resolution_visibility(r),nl_file_source_resolution_plan_rows(r));
 for(size_t i=0;i<nl_file_source_resolution_origins(r);i++){
  NlFileSourceText path,source;CHECK(nl_file_source_resolution_origin(r,i,&path,&source));printf("ORIGIN ");span(path);printf(" ");span(source);puts("");
 }
 for(size_t i=0;i<nl_file_source_resolution_visibility(r);i++){
  NlFileVisibility v;CHECK(nl_file_source_resolution_row(r,i,&v));printf("ROW ");span(v.origin);printf(" ");span(v.qualifier);printf(" ");span(v.name);printf(" ");span(v.target_origin);printf(" ");span(v.target_name);
  printf(" %u %u %u %u %u %u\n",v.id,v.target,v.kind,v.ordinal,v.exported,v.service);
 }
 for(size_t i=0;i<nl_file_source_resolution_plan_rows(r);i++){
  NlFileSourceRow v;CHECK(nl_file_source_resolution_plan_row(r,i,&v));printf("PLAN ");span(v.module);printf(" ");span(v.name);
  printf(" %u %u %u %u %u %u %u %u %u %u %u %u\n",v.id,v.target,v.request,v.kind,v.ordinal,v.category,v.input_mode,v.result_ordinal,v.global_layout,v.import_index,v.line,v.column);
 }
 for(size_t i=0;i<report.services;i++){
  NlFileCompanionView v;CHECK(nl_file_source_resolution_snapshot(r,i,&v));printf("SNAPSHOT ");span(v.request.module_path);printf(" ");span(v.original_document);printf(" ");span(v.canonical_document);printf(" ");span(v.generated_source);printf(" ");span(v.catalog_view);puts("");
 }
 NlFileVisibility old;memset(&old,0x5a,sizeof(old));NlFileVisibility unchanged;memcpy(&unchanged,&old,sizeof(old));
 CHECK(!nl_file_source_resolution_row(r,nl_file_source_resolution_visibility(r),&old));CHECK(!memcmp(&old,&unchanged,sizeof(old)));
 nl_file_source_resolution_free(r);return 0;
}
