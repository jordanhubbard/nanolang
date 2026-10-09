#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static size_t alloc_calls,fail_at,live,closes;
static int io_fault;
static void *tracked_malloc(size_t n){alloc_calls++;if(alloc_calls==fail_at)return NULL;void *p=malloc(n);if(p)live++;return p;}
static void *tracked_calloc(size_t n,size_t w){alloc_calls++;if(alloc_calls==fail_at)return NULL;void *p=calloc(n,w);if(p)live++;return p;}
static void tracked_free(void *p){if(p){assert(live);live--;}free(p);}
static ssize_t checked_read(int fd,void *p,size_t n){if(io_fault==1)return 0;if(io_fault==2){io_fault=0;errno=EINTR;return -1;}return read(fd,p,n);}
static int checked_close(int fd){closes++;int r=close(fd);if(io_fault==3){errno=EIO;return -1;}return r;}
#define malloc tracked_malloc
#define calloc tracked_calloc
#define free tracked_free
#define read checked_read
#define close checked_close
#include "../src/nanoisa/file_source_snapshot.c"
#undef malloc
#undef calloc
#undef free
#undef read
#undef close
static void put(const char *path,const unsigned char *data,size_t n){FILE *f=fopen(path,"wb");assert(f);assert(fwrite(data,1,n,f)==n);assert(!fclose(f));}
static NlFileBindingStatus acquire(NlFileSourceSnapshots *p,const char *origin,const char *name,size_t *i){return nl_file_source_snapshot_open(p,origin,strlen(origin),name,strlen(name),i);}
int main(int argc,char **argv){
 assert(argc==2);FILE *f=fopen(argv[1],"rb");assert(f);assert(!fseek(f,0,SEEK_END));long length=ftell(f);assert(length>0);rewind(f);unsigned char *bytes=malloc((size_t)length);assert(bytes);assert(fread(bytes,1,(size_t)length,f)==(size_t)length);assert(!fclose(f));
 char temporary[]="/tmp/nano-file-companion-XXXXXX";assert(mkdtemp(temporary));
 char *directory=realpath(temporary,NULL);assert(directory);
 char path[512],origin[512],link[512],fifo[512],folder[512];
 snprintf(path,sizeof path,"%s/interface.json",directory);snprintf(origin,sizeof origin,"%s/binding.nano",directory);snprintf(link,sizeof link,"%s/link.json",directory);snprintf(fifo,sizeof fifo,"%s/pipe.json",directory);snprintf(folder,sizeof folder,"%s/folder",directory);
 put(path,bytes,(size_t)length);assert(!symlink(path,link));assert(!mkfifo(fifo,0600));assert(!mkdir(folder,0700));
 NlFileSourceSnapshots *p=NULL;assert(nl_file_source_snapshots_new(&p)==NL_FILE_BINDING_OK);
 size_t i=99;assert(acquire(p,origin,"interface.json",&i)==NL_FILE_BINDING_OK && i==0);
 size_t n=99;const unsigned char *raw=nl_file_source_snapshot_bytes(p,0,1,&n);assert(n==(size_t)length && !memcmp(raw,bytes,n));
 put(path,(const unsigned char *)"{}",2);assert(!memcmp(raw,bytes,n));
 size_t prior=nl_file_source_snapshot_storage(p),unchanged=99;assert(acquire(p,origin,"interface.json",&unchanged)!=NL_FILE_BINDING_OK && unchanged==99);assert(nl_file_source_snapshot_storage(p)==prior && nl_file_source_snapshot_count(p)==1);
 put(path,bytes,(size_t)length);
 assert(acquire(p,origin,"link.json",&i)==NL_FILE_BINDING_IO);
 assert(acquire(p,origin,"pipe.json",&i)==NL_FILE_BINDING_IO);
 assert(acquire(p,origin,"folder",&i)==NL_FILE_BINDING_IO);
 assert(acquire(p,origin,"absent.json",&i)==NL_FILE_BINDING_IO);
 int large=open(path,O_WRONLY|O_TRUNC);assert(large>=0);assert(!ftruncate(large,NL_FILE_BINDING_MAX_BYTES+1u));assert(!close(large));
 unchanged=99;assert(acquire(p,origin,"interface.json",&unchanged)==NL_FILE_BINDING_LIMIT && unchanged==99);put(path,bytes,(size_t)length);
 unsigned char *maximum=malloc(NL_FILE_BINDING_MAX_BYTES);assert(maximum);memset(maximum,' ',NL_FILE_BINDING_MAX_BYTES);memcpy(maximum,bytes,(size_t)length);put(path,maximum,NL_FILE_BINDING_MAX_BYTES);free(maximum);
 assert(acquire(p,origin,"interface.json",&i)==NL_FILE_BINDING_OK);size_t full_size=0;assert(nl_file_source_snapshot_bytes(p,i,1,&full_size) && full_size==NL_FILE_BINDING_MAX_BYTES);put(path,bytes,(size_t)length);
 assert(acquire(p,"relative.nano","interface.json",&i)==NL_FILE_BINDING_INVALID);
 assert(acquire(p,origin,path,&i)==NL_FILE_BINDING_INVALID);
 char nul[]="interface.json\0suffix";assert(nl_file_source_snapshot_open(p,origin,strlen(origin),nul,sizeof nul-1,&i)==NL_FILE_BINDING_INVALID);
 const char invalid[]={ (char)0xff };assert(nl_file_source_snapshot_open(p,origin,strlen(origin),invalid,1,&i)==NL_FILE_BINDING_INVALID);
 for(int mode=1;mode<=3;mode++){
  io_fault=mode;size_t before=closes,index=99;NlFileBindingStatus result=acquire(p,origin,"interface.json",&index);assert(closes==before+1);
  if(mode==2)assert(result==NL_FILE_BINDING_OK);else assert(result==NL_FILE_BINDING_IO && index==99);
 }io_fault=0;
 n=99;assert(!nl_file_source_snapshot_bytes(p,999,1,&n) && n==99);assert(!nl_file_source_snapshot_bytes(p,0,4,&n) && n==99);
 assert(nl_file_source_snapshot_bytes(p,0,2,&n) && n>0);assert(nl_file_source_snapshot_bytes(p,0,3,&n) && n>0);
 while(nl_file_source_snapshot_count(p)<NL_FILE_SOURCE_SNAPSHOT_LIMIT)assert(acquire(p,origin,"interface.json",&i)==NL_FILE_BINDING_OK);
 unchanged=99;assert(acquire(p,origin,"interface.json",&unchanged)==NL_FILE_BINDING_LIMIT && unchanged==99);
 assert(nl_file_source_snapshot_storage(p)<=nl_file_source_snapshot_peak_bound(p));assert(nl_file_source_snapshot_peak_bound(p)<=NL_FILE_SOURCE_SNAPSHOT_BUDGET);
 nl_file_source_snapshots_free(p);assert(!live);
 for(size_t failure=1;failure<=3;failure++){
  alloc_calls=0;fail_at=failure;p=NULL;i=99;NlFileBindingStatus r=nl_file_source_snapshots_new(&p);
  if(failure==1)assert(r==NL_FILE_BINDING_MEMORY && p==NULL);
  else {assert(r==NL_FILE_BINDING_OK);assert(acquire(p,origin,"interface.json",&i)==NL_FILE_BINDING_MEMORY && i==99);assert(nl_file_source_snapshot_count(p)==0);}
  nl_file_source_snapshots_free(p);assert(!live);
 }fail_at=0;
 assert(!unlink(path));assert(!unlink(link));assert(!unlink(fifo));assert(!rmdir(folder));assert(!rmdir(directory));free(directory);free(bytes);
 puts("PASS immutable companion, counted paths, strict catalog, I/O/allocator failures, capacity and bounds");return 0;
}
