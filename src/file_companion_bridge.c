#define _POSIX_C_SOURCE 200809L
#include "file_companion_snapshot.h"
#include "file_companion_bridge.h"
#include "file_source_input.h"
#include <stdlib.h>
#include <limits.h>
#include <string.h>
typedef struct {
 const char *wire;size_t size,at;
 NlFileCompanionRequest requests[NL_FILE_SOURCE_REQUESTS];
} CompanionTransport;
static struct {
 int64_t next,token;
 NlFileCompanionSet *set;
 char *source;size_t source_size,source_lines;
 NlFileCompanionReport report;
} companion_bridge;
static bool cb_number(CompanionTransport *p,char end,uint32_t *out) {
 uint32_t v=0;size_t start=p->at;
 while(p->at<p->size && p->wire[p->at]!=end) {
  unsigned char c=(unsigned char)p->wire[p->at++];
  if(c<'0'||c>'9'||v>(UINT32_MAX-(uint32_t)(c-'0'))/10)return false;
  v=v*10+(uint32_t)(c-'0');
 }
 if(p->at==start || p->at==p->size)return false;
 p->at++;*out=v;return true;
}
static bool cb_text(CompanionTransport *p,NlFileSourceText *out) {
 uint32_t size;
 if(!cb_number(p,':',&size)||!size||size>NL_FILE_COMPANION_PATH||size>p->size-p->at)return false;
 *out=(NlFileSourceText){p->wire+p->at,size};p->at+=size;return true;
}
int64_t nl_file_companion_open(const char *wire,int64_t size) {
 if(companion_bridge.token || companion_bridge.next==INT64_MAX)return 0;
 companion_bridge.token=++companion_bridge.next;
 companion_bridge.report=(NlFileCompanionReport){NL_FILE_COMPANION_INVALID,
  NL_FILE_COMPANION_INPUT,UINT32_MAX,0,0,0,0,0,0};
 if(!wire||size<=0||size>200000||strnlen(wire,(size_t)size+1)!=(uint64_t)size)return companion_bridge.token;
 CompanionTransport p={0};p.wire=wire;p.size=(size_t)size;
 uint32_t count;
 if(!cb_number(&p,';',&count)||!count||count>NL_FILE_SOURCE_REQUESTS)return companion_bridge.token;
 for(uint32_t i=0;i<count;i++) {
  NlFileCompanionRequest *r=&p.requests[i];
  if(!cb_number(&p,';',&r->catalog_version)||!cb_number(&p,';',&r->line)||
     !cb_number(&p,';',&r->column)||!cb_text(&p,&r->module_path)||
     !cb_text(&p,&r->companion_path)||!cb_text(&p,&r->interface_id))return companion_bridge.token;
 }
 if(p.at!=p.size)return companion_bridge.token;
 companion_bridge.report=nl_file_companions_prepare(p.requests,count,&companion_bridge.set);
 return companion_bridge.token;
}
int64_t nl_file_companion_source(const char *path,int64_t maximum) {
 if(companion_bridge.token||companion_bridge.next==INT64_MAX)return 0;
 companion_bridge.token=++companion_bridge.next;
 companion_bridge.report=(NlFileCompanionReport){NL_FILE_COMPANION_INVALID,NL_FILE_COMPANION_INPUT,UINT32_MAX,0,0,0,0,0,0};
 if(maximum>0&&(uint64_t)maximum<=NL_FILE_COMPANION_BYTES)
  companion_bridge.report=nl_file_source_input_read(path,(size_t)maximum,&companion_bridge.source,&companion_bridge.source_size);
 if(companion_bridge.source) {
  companion_bridge.source_lines=1;
  for(size_t i=0;i<companion_bridge.source_size;i++)if(companion_bridge.source[i]=='\n')companion_bridge.source_lines++;
  companion_bridge.report.work_reserved+=companion_bridge.source_size+1;
 }
 return companion_bridge.token;
}
static bool cb_view(int64_t token,int64_t row,NlFileCompanionView *out) {
 return token>0 && token==companion_bridge.token && row>=0 &&
  nl_file_companions_view(companion_bridge.set,(size_t)row,out);
}
static NlFileSourceText cb_span(NlFileCompanionView v,int64_t field) {
 switch(field) {
  case 0:return v.request.module_path;
  case 1:return v.request.companion_path;
  case 2:return v.request.interface_id;
  case 3:return v.original_document;
  case 4:return v.canonical_document;
  case 5:return v.generated_source;
  case 6:return v.catalog_view;
  default:return (NlFileSourceText){"",0};
 }
}
int64_t nl_file_companion_number(int64_t token,int64_t row,int64_t field) {
 if(token<=0||token!=companion_bridge.token)return -1;
 if(row==-1) {
  NlFileCompanionReport r=companion_bridge.report;
  switch(field) {
   case 0:return r.status;case 1:return r.stage;case 2:return r.request;
   case 3:return r.system_error;case 4:return r.close_error;
   case 5:return (int64_t)r.peak_heap_bytes_reserved;
   case 6:return (int64_t)r.work_reserved;
   case 7:return companion_bridge.source?1:(int64_t)nl_file_companions_count(companion_bridge.set);
   case 8:return (int64_t)r.catalog_buffer_bytes;
   case 9:return (int64_t)r.parent_buffer_bytes;
   case 10:return (int64_t)sizeof(CompanionTransport);
   default:return -1;
  }
 }
 if(companion_bridge.source) {
  if(row==0&&field==3)return (int64_t)companion_bridge.source_size;
  if(row==0&&field==4)return (int64_t)companion_bridge.source_lines;
  return -1;
 }
 NlFileCompanionView v;
 if(!cb_view(token,row,&v))return -1;
 if(field>=0&&field<=6)return (int64_t)cb_span(v,field).size;
 switch(field) {case 7:return v.request.catalog_version;
  case 8:return v.request.line;case 9:return v.request.column;default:return -1;}
}
const char *nl_file_companion_text(int64_t token,int64_t row,int64_t field) {
 if(token>0&&token==companion_bridge.token&&companion_bridge.source)
  return row==0&&field==3?companion_bridge.source:"";
 NlFileCompanionView v;
 return cb_view(token,row,&v)?cb_span(v,field).data:"";
}
bool nl_file_companion_destroy(int64_t token) {
 if(token<=0||token!=companion_bridge.token)return false;
 nl_file_companions_free(companion_bridge.set);companion_bridge.set=NULL;
 free(companion_bridge.source);companion_bridge.source=NULL;companion_bridge.source_size=0;companion_bridge.source_lines=0;
 companion_bridge.token=0;memset(&companion_bridge.report,0,sizeof companion_bridge.report);
 return true;
}
