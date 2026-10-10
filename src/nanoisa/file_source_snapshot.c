#include "file_source_snapshot.h"
#include "nsi_socket_binding.h"
#include "utf8.h"
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
typedef struct {
 char *path;
 unsigned char *bytes;
 size_t path_size,size;
 int64_t catalog;
 union { NlFileBindingPlan *file; NlSocketBindingPlan *socket; } plan;
} SourceSnapshot;
struct NlFileSourceSnapshots {
 size_t count,storage,peak;
 SourceSnapshot entries[NL_FILE_SOURCE_SNAPSHOT_LIMIT];
};
static bool snapshot_span(const char *p,size_t n) {
 return p && n && n<=NL_FILE_BINDING_MAX_BYTES && !memchr(p,0,n) && nl_utf8_validate(p,n,NULL);
}
NlFileBindingStatus nl_file_source_snapshots_new(NlFileSourceSnapshots **out) {
 if(!out)return NL_FILE_BINDING_INVALID;
 NlFileSourceSnapshots *p=calloc(1,sizeof(*p));
 if(!p)return NL_FILE_BINDING_MEMORY;
 p->storage=p->peak=sizeof(*p);*out=p;return NL_FILE_BINDING_OK;
}
static void snapshot_dispose(SourceSnapshot *p) {
 if(p->catalog==NL_SOURCE_CATALOG_FILE)nl_file_binding_free(p->plan.file);
 else if(p->catalog==NL_SOURCE_CATALOG_SOCKET)nl_socket_binding_free(p->plan.socket);
 free(p->bytes);free(p->path);
}
void nl_file_source_snapshots_free(NlFileSourceSnapshots *p) {
 if(!p)return;
 for(size_t i=0;i<p->count;i++)snapshot_dispose(&p->entries[i]);
 free(p);
}
NlBindingStatus nl_service_source_snapshot_open(NlFileSourceSnapshots *p,int64_t catalog,
 const char *origin,size_t no,const char *relative,size_t nr,size_t *index) {
 if((catalog!=NL_SOURCE_CATALOG_FILE && catalog!=NL_SOURCE_CATALOG_SOCKET) ||
    !p || !index || !snapshot_span(origin,no) || !snapshot_span(relative,nr) ||
    origin[0]!='/' || relative[0]=='/' || origin[no-1]=='/')return NL_FILE_BINDING_INVALID;
 if(p->count==NL_FILE_SOURCE_SNAPSHOT_LIMIT)return NL_FILE_BINDING_LIMIT;
 size_t parent=no;
 while(parent && origin[parent-1]!='/')parent--;
 if(nr>NL_FILE_BINDING_MAX_BYTES-parent)return NL_FILE_BINDING_LIMIT;
 size_t path_size=parent+nr,prepare_bound;
 bool bounded=catalog==NL_SOURCE_CATALOG_FILE?nl_file_binding_allocation_bound(&prepare_bound):
              nl_socket_binding_allocation_bound(&prepare_bound);
 if(!bounded)return NL_FILE_BINDING_LIMIT;
 /* I count the retained context, new path/read allocation and the complete
  * nested preparation bound once. Its final plan is already in that bound. */
 size_t extra=path_size+1u+NL_FILE_BINDING_MAX_BYTES+1u;
 if(extra>NL_FILE_SOURCE_SNAPSHOT_BUDGET || prepare_bound>NL_FILE_SOURCE_SNAPSHOT_BUDGET-extra ||
    p->storage>NL_FILE_SOURCE_SNAPSHOT_BUDGET-extra-prepare_bound)return NL_FILE_BINDING_LIMIT;
 size_t peak=p->storage+extra+prepare_bound;
 if(peak>p->peak)p->peak=peak;
 SourceSnapshot next={0};next.catalog=catalog;next.path_size=path_size;
 next.path=malloc(path_size+1u);
 if(!next.path)return NL_FILE_BINDING_MEMORY;
 memcpy(next.path,origin,parent);memcpy(next.path+parent,relative,nr);next.path[path_size]=0;
 int fd=open(next.path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
 NlFileBindingStatus status=NL_FILE_BINDING_IO;
 if(fd<0)goto done;
 struct stat st;
 if(fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<0)goto close_input;
 if((uintmax_t)st.st_size>NL_FILE_BINDING_MAX_BYTES){status=NL_FILE_BINDING_LIMIT;goto close_input;}
 next.size=(size_t)st.st_size;
 next.bytes=malloc(next.size+1u);
 if(!next.bytes){status=NL_FILE_BINDING_MEMORY;goto close_input;}
 size_t read_size=0;
 while(read_size<next.size) {
  ssize_t n=read(fd,next.bytes+read_size,next.size-read_size);
  if(n<0 && errno==EINTR)continue;
  if(n<=0)goto close_input;
  read_size+=(size_t)n;
 }
 unsigned char trailing;
 ssize_t n;
 do { n=read(fd,&trailing,1); } while(n<0 && errno==EINTR);
 if(n!=0)goto close_input;
 next.bytes[next.size]=0;status=NL_FILE_BINDING_OK;
close_input:
 /* I never retry close: an error does not preserve ownership of its number. */
 if(close(fd) && status==NL_FILE_BINDING_OK)status=NL_FILE_BINDING_IO;
 if(status!=NL_FILE_BINDING_OK)goto done;
 status=catalog==NL_SOURCE_CATALOG_FILE?nl_file_binding_prepare(next.bytes,next.size,&next.plan.file):
        nl_socket_binding_prepare(next.bytes,next.size,&next.plan.socket);
 if(status!=NL_FILE_BINDING_OK)goto done;
 p->storage+=path_size+1u+next.size+1u+(catalog==NL_SOURCE_CATALOG_FILE?nl_file_binding_storage_size(next.plan.file):nl_socket_binding_storage_size(next.plan.socket));
 p->entries[p->count]=next;*index=p->count++;return NL_FILE_BINDING_OK;
done:
 snapshot_dispose(&next);return status;
}
size_t nl_file_source_snapshot_count(const NlFileSourceSnapshots *p){return p?p->count:0;}
size_t nl_file_source_snapshot_storage(const NlFileSourceSnapshots *p){return p?p->storage:0;}
size_t nl_file_source_snapshot_peak_bound(const NlFileSourceSnapshots *p){return p?p->peak:0;}
const unsigned char *nl_file_source_snapshot_bytes(const NlFileSourceSnapshots *p,size_t i,unsigned kind,size_t *size) {
 if(!p || !size || i>=p->count)return NULL;
 const SourceSnapshot *s=&p->entries[i];
 if(kind==0){*size=s->path_size;return (const unsigned char *)s->path;}
 if(kind==1){*size=s->size;return s->bytes;}
 if(kind==2)return s->catalog==NL_SOURCE_CATALOG_FILE?nl_file_binding_interface_bytes(s->plan.file,size):nl_socket_binding_interface_bytes(s->plan.socket,size);
 if(kind==3)return s->catalog==NL_SOURCE_CATALOG_FILE?nl_file_binding_source_bytes(s->plan.file,size):nl_socket_binding_source_bytes(s->plan.socket,size);
 return NULL;
}

NlFileBindingStatus nl_file_source_snapshot_open(NlFileSourceSnapshots *p,
 const char *origin,size_t no,const char *relative,size_t nr,size_t *index) {
 return nl_service_source_snapshot_open(p,NL_SOURCE_CATALOG_FILE,origin,no,relative,nr,index);
}
int64_t nl_service_source_snapshot_catalog(const NlFileSourceSnapshots *p,size_t index) {
 return p && index<p->count?p->entries[index].catalog:NL_SOURCE_CATALOG_NONE;
}
