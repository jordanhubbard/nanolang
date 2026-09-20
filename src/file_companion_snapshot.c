#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "file_companion_snapshot.h"
#include "nsi_file_binding.h"
#include "utf8.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* I reserve the existing strict reader's bounded scans separately from my own
 * traversals: two decodes plus render passes, bounded key/typed-array scans.
 * This is a conservative operation reservation, not measured CPU instructions. */
#define FC_STRICT_WORK (UINT64_C(8)*NL_FILE_BINDING_MAX_TOKENS*NL_FILE_BINDING_MAX_LEXEME* \
 (NL_FILE_BINDING_MAX_DEPTH+NL_FILE_BINDING_MAX_MEMBERS+NL_FILE_BINDING_MAX_ELEMENTS+16u))
typedef struct {
 NlFileCompanionRequest request;
 unsigned char *document;
 size_t size;
 NlFileBindingPlan *binding;
} Companion;
struct NlFileCompanionSet {
 size_t count, catalog_size;
 uint64_t bytes;
 NlFileCompanionReport report;
 Companion rows[NL_FILE_SOURCE_REQUESTS];
 char catalog[32768];
};
static bool fc_work(NlFileCompanionSet *s,uint64_t n) {
 if(n>NL_FILE_COMPANION_WORK-s->report.work_reserved) {
  s->report.status=NL_FILE_COMPANION_LIMIT;return false;
 }
 s->report.work_reserved+=n;return true;
}
static bool fc_reserve(NlFileCompanionSet *s,uint64_t n) {
 if(n>NL_FILE_COMPANION_BYTES-s->bytes) {
  s->report.status=NL_FILE_COMPANION_LIMIT;return false;
 }
 s->bytes+=n;
 if(s->bytes>s->report.peak_bytes_reserved)s->report.peak_bytes_reserved=s->bytes;
 return true;
}
static void *fc_allocate(NlFileCompanionSet *s,size_t n) {
 if(!fc_reserve(s,n))return NULL;
 void *p=malloc(n);
 if(!p)s->report.status=NL_FILE_COMPANION_MEMORY;
 return p;
}
static bool fc_text(NlFileCompanionSet *s,NlFileSourceText t,size_t maximum,bool utf8) {
 if(!t.data || !t.size || t.size>maximum)return false;
 if(!fc_work(s,(uint64_t)t.size*(utf8?2u:1u)))return false;
 return !memchr(t.data,0,t.size) && (!utf8 || nl_utf8_validate(t.data,t.size,NULL));
}
static bool fc_path(NlFileCompanionSet *s,NlFileSourceText t,bool absolute) {
 if(!fc_text(s,t,NL_FILE_COMPANION_PATH,!absolute) ||
    (t.data[0]=='/')!=absolute || (absolute && t.size==1))return false;
 if(!fc_work(s,t.size))return false;
 size_t start=absolute?1:0;
 for(size_t i=start;i<=t.size;i++) {
  if(i<t.size && t.data[i]!='/')continue;
  size_t n=i-start;
  if(!n || (n==1 && t.data[start]=='.') ||
     (n==2 && t.data[start]=='.' && t.data[start+1]=='.'))return false;
  start=i+1;
 }
 return true;
}
static bool fc_copy_text(NlFileCompanionSet *s,NlFileSourceText from,NlFileSourceText *to) {
 if(!fc_work(s,from.size+1))return false;
 char *p=fc_allocate(s,from.size+1);if(!p)return false;
 memcpy(p,from.data,from.size);p[from.size]=0;
 *to=(NlFileSourceText){p,from.size};return true;
}
void nl_file_companions_free(NlFileCompanionSet *s) {
 if(!s)return;
 for(size_t i=0;i<s->count;i++) {
  Companion *r=&s->rows[i];
  free((void *)r->request.module_path.data);free((void *)r->request.companion_path.data);
  free((void *)r->request.interface_id.data);free(r->document);nl_file_binding_free(r->binding);
 }
 free(s);
}
static bool fc_retry(NlFileCompanionSet *s,unsigned *interruptions) {
 if(errno!=EINTR || *interruptions==NL_FILE_COMPANION_RETRIES)return false;
 ++*interruptions;return fc_work(s,1);
}
static bool fc_io_failure(NlFileCompanionSet *s) {
 if(s->report.status!=NL_FILE_COMPANION_LIMIT)s->report.status=NL_FILE_COMPANION_IO;
 s->report.system_error=errno;return false;
}
/* I close once, including EINTR. I never retry an ambiguously consumed fd. */
static bool fc_close(NlFileCompanionSet *s,int *fd,bool primary) {
 if(*fd<0)return true;
 int owned=*fd;*fd=-1;
 if(close(owned)==0)return true;
 if(primary && s->report.status==NL_FILE_COMPANION_OK) {
  s->report.stage=NL_FILE_COMPANION_CLOSE;s->report.status=NL_FILE_COMPANION_IO;
  s->report.system_error=errno;
 } else if(!s->report.close_error)s->report.close_error=errno;
 return false;
}
static NlFileCompanionStatus fc_document_status(NlFileBindingStatus status) {
 switch(status) {
 case NL_FILE_BINDING_OK:return NL_FILE_COMPANION_OK;
 case NL_FILE_BINDING_INVALID:return NL_FILE_COMPANION_INVALID;
 case NL_FILE_BINDING_LIMIT:return NL_FILE_COMPANION_LIMIT;
 case NL_FILE_BINDING_MEMORY:return NL_FILE_COMPANION_MEMORY;
 default:return NL_FILE_COMPANION_UNRESOLVED;
 }
}
static bool fc_read(NlFileCompanionSet *s,Companion *r) {
 char parent[NL_FILE_COMPANION_PATH+1];
 int directory=-1,fd=-1;unsigned interruptions=0;bool ok=false;
 size_t n=r->request.module_path.size;
 if(!fc_work(s,n*2u+1))return false;
 memcpy(parent,r->request.module_path.data,n+1);
 while(n && parent[n-1]!='/')--n;
 parent[n==1?1:n-1]=0;
 s->report.stage=NL_FILE_COMPANION_PARENT;
 do {directory=open(parent,O_RDONLY|O_DIRECTORY|O_CLOEXEC);} while(directory<0 && fc_retry(s,&interruptions));
 if(directory<0)return fc_io_failure(s);
 s->report.stage=NL_FILE_COMPANION_OPEN;
 do {fd=openat(directory,r->request.companion_path.data,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);}
 while(fd<0 && fc_retry(s,&interruptions));
 if(fd<0){fc_io_failure(s);goto done;}
 s->report.stage=NL_FILE_COMPANION_STAT;
 struct stat before,after;int status;
 do {status=fstat(fd,&before);} while(status<0 && fc_retry(s,&interruptions));
 if(status<0){fc_io_failure(s);goto done;}
 if(!S_ISREG(before.st_mode) || before.st_size<0) {
  s->report.status=NL_FILE_COMPANION_INVALID;goto done;
 }
 if((uint64_t)before.st_size>NL_FILE_BINDING_MAX_BYTES) {
  s->report.status=NL_FILE_COMPANION_LIMIT;goto done;
 }
 s->report.stage=NL_FILE_COMPANION_READ;
 while(r->size<NL_FILE_BINDING_MAX_BYTES+1u) {
  if(!fc_work(s,1))goto done;
  ssize_t got=read(fd,r->document+r->size,NL_FILE_BINDING_MAX_BYTES+1u-r->size);
  if(got<0) {if(fc_retry(s,&interruptions))continue;fc_io_failure(s);goto done;}
  if(!got)break;
  if(!fc_work(s,(uint64_t)got))goto done;
  r->size+=(size_t)got;
 }
 if(r->size>NL_FILE_BINDING_MAX_BYTES){s->report.status=NL_FILE_COMPANION_LIMIT;goto done;}
 if(r->size!=(uint64_t)before.st_size){s->report.status=NL_FILE_COMPANION_IO;s->report.system_error=EIO;goto done;}
 s->report.stage=NL_FILE_COMPANION_STAT;
 do {status=fstat(fd,&after);} while(status<0 && fc_retry(s,&interruptions));
 if(status<0){fc_io_failure(s);goto done;}
 if(before.st_dev!=after.st_dev || before.st_ino!=after.st_ino ||
    before.st_size!=after.st_size || before.st_mtime!=after.st_mtime || before.st_ctime!=after.st_ctime) {
  s->report.status=NL_FILE_COMPANION_IO;s->report.system_error=EIO;goto done;
 }
 r->document[r->size]=0;ok=true;
done:
 if(!fc_close(s,&fd,ok))ok=false;
 if(!fc_close(s,&directory,ok))ok=false;
 return ok;
}
NlFileCompanionReport nl_file_companions_prepare(const NlFileCompanionRequest *requests,size_t count,
                                                NlFileCompanionSet **out) {
 NlFileCompanionReport invalid={NL_FILE_COMPANION_INVALID,NL_FILE_COMPANION_INPUT,UINT32_MAX,0,0,0,0};
 if(!out || !requests || !count)return invalid;
 if(count>NL_FILE_SOURCE_REQUESTS){invalid.status=NL_FILE_COMPANION_LIMIT;return invalid;}
 if(sizeof(NlFileCompanionSet)+16384u>NL_FILE_COMPANION_BYTES) {
  invalid.status=NL_FILE_COMPANION_LIMIT;return invalid;
 }
 NlFileCompanionSet *s=calloc(1,sizeof *s);
 if(!s){invalid.status=NL_FILE_COMPANION_MEMORY;return invalid;}
 s->report=invalid;s->report.status=NL_FILE_COMPANION_OK;s->count=count;
 /* My owning size and fixed automatic scratch are known before allocation.
  * I retain reservations for every live input and strict reader overlap. */
 if(!fc_reserve(s,sizeof *s+16384u) || !fc_work(s,sizeof *s+16384u))goto fail;
 size_t catalog_size;
 if(!fc_work(s,sizeof s->catalog*4u))goto fail;
 if(!nl_file_source_catalog_view(s->catalog,sizeof s->catalog,&catalog_size)) {
  s->report.status=NL_FILE_COMPANION_UNRESOLVED;goto fail;
 }
 s->catalog_size=catalog_size-1;
 /* I validate and own every request before the first filesystem operation. */
 for(size_t i=0;i<count;i++) {
  const NlFileCompanionRequest *q=&requests[i];Companion *r=&s->rows[i];
  s->report.request=(uint32_t)i;s->report.stage=NL_FILE_COMPANION_INPUT;
  if(!fc_path(s,q->module_path,true) || !fc_path(s,q->companion_path,false) ||
     !fc_text(s,q->interface_id,4095,true) || q->catalog_version!=1 || !q->line || !q->column ||
     q->interface_id.size!=sizeof("nsi:nanolang/filesystem")-1 ||
     memcmp(q->interface_id.data,"nsi:nanolang/filesystem",sizeof("nsi:nanolang/filesystem")-1)) {
   if(s->report.status==NL_FILE_COMPANION_OK)s->report.status=NL_FILE_COMPANION_INVALID;
   goto fail;
  }
  s->report.stage=NL_FILE_COMPANION_STORAGE;
  r->request.catalog_version=q->catalog_version;r->request.line=q->line;r->request.column=q->column;
  if(!fc_copy_text(s,q->module_path,&r->request.module_path) ||
     !fc_copy_text(s,q->companion_path,&r->request.companion_path) ||
     !fc_copy_text(s,q->interface_id,&r->request.interface_id))goto fail;
  r->document=fc_allocate(s,NL_FILE_BINDING_MAX_BYTES+1u);if(!r->document)goto fail;
 }
 for(size_t i=0;i<count;i++) {
  Companion *r=&s->rows[i];s->report.request=(uint32_t)i;
  if(!fc_read(s,r))goto fail;
  size_t bound;
  s->report.stage=NL_FILE_COMPANION_DOCUMENT;
  if(!nl_file_binding_allocation_bound(&bound) || !fc_reserve(s,bound) || !fc_work(s,FC_STRICT_WORK)) {
   s->report.status=NL_FILE_COMPANION_LIMIT;goto fail;
  }
  s->report.status=fc_document_status(nl_file_binding_prepare(r->document,r->size,&r->binding));
  if(s->report.status!=NL_FILE_COMPANION_OK)goto fail;
  size_t retained=nl_file_binding_storage_size(r->binding);
  if(retained>bound){s->report.status=NL_FILE_COMPANION_UNRESOLVED;goto fail;}
  s->bytes-=bound-retained;
 }
 s->report.stage=NL_FILE_COMPANION_DONE;s->report.request=UINT32_MAX;
 *out=s;return s->report;
fail:
 invalid=s->report;nl_file_companions_free(s);return invalid;
}
size_t nl_file_companions_count(const NlFileCompanionSet *s){return s?s->count:0;}
bool nl_file_companions_view(const NlFileCompanionSet *s,size_t i,NlFileCompanionView *out) {
 if(!s || !out || i>=s->count)return false;
 const Companion *r=&s->rows[i];size_t canonical_size=0,source_size=0;
 const unsigned char *canonical=nl_file_binding_interface_bytes(r->binding,&canonical_size);
 const unsigned char *source=nl_file_binding_source_bytes(r->binding,&source_size);
 if(!canonical || !source)return false;
 NlFileCompanionView v={r->request,{(const char *)r->document,r->size},
  {(const char *)canonical,canonical_size},{(const char *)source,source_size},{s->catalog,s->catalog_size}};
 *out=v;return true;
}
