/* I exercise private real reads and copied roots, not bytecode admission. */
#include "portable_read_managed.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static unsigned checks;
#define CHECK(x) do { ++checks; if(!(x)){fprintf(stderr,"I failed %s:%d: %s\n",__FILE__,__LINE__,#x);abort();} } while(0)
#ifdef READ_LLVM
extern int32_t pr_llvm_read(void *,const uint8_t *,uint32_t,uint8_t *,uint32_t,uint32_t *);
#define READ pr_llvm_read
#else
#define READ npr_file_read
#endif

typedef struct {void *p;size_t n;} Allocation;
static Allocation live[256];static size_t live_count,live_bytes,requests;
static long fail_at=-1;static bool persistent;static unsigned failures;
static bool fail(void){size_t i=requests++;bool f=fail_at>=0&&(persistent?i>=(size_t)fail_at:i==(size_t)fail_at);if(f)++failures;return f;}
static void remember(void *p,size_t n){if(p){CHECK(live_count<256);live[live_count++]=(Allocation){p,n};live_bytes+=n;}}
void *pr_test_malloc(size_t n){if(fail())return NULL;void *p=malloc(n);remember(p,n);return p;}
void *pr_test_calloc(size_t n,size_t w){CHECK(!w||n<=SIZE_MAX/w);if(fail())return NULL;void *p=calloc(n,w);remember(p,n*w);return p;}
void pr_test_free(void *p){if(!p)return;size_t i=0;while(i<live_count&&live[i].p!=p)++i;CHECK(i<live_count);live_bytes-=live[i].n;live[i]=live[--live_count];free(p);}
static NprFileHost *active_probe;static unsigned active_refusals;
static unsigned opens,closes,successful_opens;static bool read_error,close_error,read_progress;static int last_fd=-1;
FILE *pr_test_fopen(const char *p,const char *m){++opens;if(active_probe){CHECK(npr_file_host_destroy(active_probe)==NPR_INVALID);++active_refusals;}FILE *f=fopen(p,m);if(f){++successful_opens;last_fd=fileno(f);}return f;}
size_t pr_test_fread(void *p,size_t s,size_t n,FILE *f){if(read_error&&n>1)n=1;size_t k=fread(p,s,n,f);if(read_error&&k)read_progress=true;return k;}
int pr_test_ferror(FILE *f){return read_error&&read_progress?1:ferror(f);}
int pr_test_fclose(FILE *f){++closes;int r=fclose(f);if(close_error){errno=EIO;return EOF;}return r;}
static void io_reset(void){opens=closes=successful_opens=0;last_fd=-1;read_error=close_error=read_progress=false;}
static void view_is(NmsRuntime *r,NmsHandle h,const void *p,size_t n){NmsView v;CHECK(nms_view(r,h,&v)==NMS_OK);CHECK(v.length==n);CHECK(!n||!memcmp(v.data,p,n));}
static void runtime_start(NmsRuntime *r){nms_init(r,NULL,0);CHECK(nms_bind_records(r,NULL,0)==NMS_OK);CHECK(nms_begin(r)==NMS_OK);}
static void runtime_end(NmsRuntime *r){CHECK(!r->live_objects&&!r->live_bytes);CHECK(nms_finish(r,NMS_OK,0)==0);CHECK(nms_dispose(r)==NMS_OK);}
static NprFileHost *host_for(const char *path){NprPath p={(const uint8_t *)path,(uint32_t)strlen(path)};NprFileHost *h=NULL;CHECK(npr_file_host_create(&p,1,&h)==NPR_OK);CHECK(h);return h;}
static char directory[3072];static unsigned file_index;
static void make_file(char path[4097],const uint8_t *data,size_t n){int k=snprintf(path,4097,"%s/vector-%u.bin",directory,file_index++);CHECK(k>0&&k<4097);FILE *f=fopen(path,"wb");CHECK(f);CHECK(!n||fwrite(data,1,n,f)==n);CHECK(fclose(f)==0);}
static void vector(const uint8_t *data,size_t n,bool empty,NprStatus expected){
 char path[4097];make_file(path,data,n);NprFileHost *host=host_for(path);NprHostBinding b={READ,host};NmsRuntime r;runtime_start(&r);
 NmsHandle arg=0;CHECK(nms_create(&r,(const uint8_t *)path,strlen(path),&arg)==NMS_OK);CHECK(nms_retain(&r,arg)==NMS_OK);CHECK(nms_retain(&r,arg)==NMS_OK);
 uint64_t objects=r.live_objects,bytes=r.live_bytes;io_reset();
 NprManagedResult got=npr_read_managed(&r,arg,&b);CHECK(got.host_status==expected&&got.managed_status==NMS_OK);
 if(expected==NPR_OK){CHECK(got.value&&got.value!=arg);view_is(&r,got.value,data,empty?0:n);CHECK(nms_release(&r,got.value)==NMS_OK);}else CHECK(!got.value);
 CHECK(r.live_objects==objects&&r.live_bytes==bytes);view_is(&r,arg,path,strlen(path));
 CHECK(r.slots[arg&~NMS_DYNAMIC].references==3);
#ifdef READ_OBSERVED
 CHECK(opens==1&&closes==1&&successful_opens==1);CHECK(fcntl(last_fd,F_GETFD)==-1&&errno==EBADF);
#endif
 CHECK(nms_release(&r,arg)==NMS_OK);CHECK(nms_release(&r,arg)==NMS_OK);CHECK(nms_release(&r,arg)==NMS_OK);runtime_end(&r);CHECK(npr_file_host_destroy(host)==NPR_OK);
}
static void real_vectors(void){
 static const uint8_t unicode[]={0xe2,0x82,0xac,0xf0,0x9f,0x92,0xa9};static const uint8_t partial[]={0xe2,0x82};static const uint8_t nul[]={'a',0,'b'};
 vector(NULL,0,false,NPR_OK);vector((const uint8_t *)"x",1,false,NPR_OK);vector(unicode,sizeof unicode,false,NPR_OK);vector(partial,sizeof partial,false,NPR_OK);vector(nul,sizeof nul,true,NPR_OK);
 uint8_t *big=malloc(NPR_TEXT_LIMIT+1u);CHECK(big);memset(big,'z',NPR_TEXT_LIMIT+1u);vector(big,NPR_TEXT_LIMIT,false,NPR_OK);vector(big,NPR_TEXT_LIMIT+1u,false,NPR_LIMIT);free(big);
}
static void context_and_ranges(void){
 char path[4097];make_file(path,(const uint8_t *)"abc",3);NprFileHost *h=host_for(path);uint8_t buffer[32];uint32_t length=91;io_reset();
 CHECK(READ(NULL,(uint8_t *)path,strlen(path),buffer,32,&length)==NPR_DENIED&&length==91);
 CHECK(READ(h,(uint8_t *)"denied",6,buffer,32,&length)==NPR_DENIED&&length==91);
 CHECK(READ(h,(uint8_t *)path,strlen(path),(uint8_t *)path,3,&length)==NPR_INVALID&&length==91);
 CHECK(READ(h,(uint8_t *)path,strlen(path),(uint8_t *)&length,4,&length)==NPR_INVALID&&length==91);
 CHECK(READ(h,(uint8_t *)&length,4,buffer,32,&length)==NPR_INVALID&&length==91);
 CHECK(READ(h,(uint8_t *)h,1,buffer,32,&length)==NPR_INVALID&&length==91);
 CHECK(READ(h,(uint8_t *)path,strlen(path),(uint8_t *)h,1,&length)==NPR_INVALID&&length==91);
 CHECK(READ(h,(uint8_t *)path,strlen(path),buffer,32,(uint32_t *)h)==NPR_INVALID);
 CHECK(READ(h,(uint8_t *)(uintptr_t)(UINTPTR_MAX-1),4,buffer,32,&length)==NPR_INVALID&&length==91);
 CHECK(READ(h,(uint8_t *)path,strlen(path),buffer,NPR_TEXT_LIMIT+1u,&length)==NPR_LIMIT&&length==91);
#ifdef READ_OBSERVED
 CHECK(!opens&&!closes);
#endif
 CHECK(READ(h,(uint8_t *)path,strlen(path),buffer,32,&length)==NPR_OK&&length==3&&!memcmp(buffer,"abc",3));
 length=91;CHECK(READ(h,(uint8_t *)path,strlen(path),NULL,0,&length)==NPR_LIMIT&&length==91);
 CHECK(npr_file_host_destroy(h)==NPR_OK);
 char empty[4097];make_file(empty,NULL,0);h=host_for(empty);CHECK(READ(h,(uint8_t *)empty,strlen(empty),NULL,0,&length)==NPR_OK&&length==0);CHECK(npr_file_host_destroy(h)==NPR_OK);
 NprFileHost *out=(NprFileHost *)(uintptr_t)1;NprPath invalid={(uint8_t *)"a\0b",3};CHECK(npr_file_host_create(&invalid,1,&out)==NPR_INVALID&&out==(void *)(uintptr_t)1);
 CHECK(npr_file_host_create(NULL,65,&out)==NPR_INVALID&&out==(void *)(uintptr_t)1);
 CHECK(npr_file_host_create(&invalid,65,&out)==NPR_LIMIT&&out==(void *)(uintptr_t)1);
 CHECK(npr_file_host_destroy(NULL)==NPR_OK);CHECK(npr_file_host_create(NULL,0,&h)==NPR_OK);CHECK(READ(h,(uint8_t *)empty,strlen(empty),buffer,32,&length)==NPR_DENIED);CHECK(npr_file_host_destroy(h)==NPR_OK);
}
static unsigned callback_mode,callback_count;
static const uint8_t *callback_expected_path;static uint32_t callback_expected_length;
static int32_t bad_callback(void *ctx,const uint8_t *p,uint32_t n,uint8_t *dst,uint32_t cap,uint32_t *len){
 (void)ctx;++callback_count;if(callback_expected_path){CHECK(n==callback_expected_length);CHECK(!memcmp(p,callback_expected_path,n));}
 switch(callback_mode){case 0:return 99;case 1:*len=cap+1;return NPR_OK;case 2:return NPR_OK;case 3:dst[0]=0;*len=1;return NPR_OK;case 4:*len=999;return NPR_DENIED;default:dst[0]='q';*len=1;return NPR_OK;}
}
static void copied_allowlist_and_filename(void){
 char path[4097];int k=snprintf(path,sizeof path,"%s/caf\xc3\xa9-\xe2\x82\xac.bin",directory);CHECK(k>0&&k<4097);
 FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite("copy",1,4,f)==4);CHECK(fclose(f)==0);
 uint8_t *storage=malloc(64u*4097u);CHECK(storage);NprPath rows[64];
 for(unsigned i=0;i<64;i++){char *name=(char *)storage+i*4097u;int n=i==63?snprintf(name,4097,"%s",path):snprintf(name,4097,"%s/allow-%u",directory,i);CHECK(n>0&&n<=4096);rows[i]=(NprPath){(uint8_t *)name,(uint32_t)n};}
 NprFileHost *h=NULL;CHECK(npr_file_host_create(rows,64,&h)==NPR_OK&&h);NprFileHost *sentinel=(void *)(uintptr_t)1;
 CHECK(npr_file_host_create(rows,65,&sentinel)==NPR_LIMIT&&sentinel==(void *)(uintptr_t)1);
 memset(storage,'!',64u*4097u);free(storage);memset(rows,0,sizeof rows);
 uint8_t data[8];uint32_t n=91;io_reset();
#ifdef READ_OBSERVED
 active_probe=h;active_refusals=0;
#endif
 CHECK(READ(h,(uint8_t *)path,strlen(path),data,8,&n)==NPR_OK&&n==4&&!memcmp(data,"copy",4));
#ifdef READ_OBSERVED
 CHECK(active_refusals==1&&opens==1&&closes==1);active_probe=NULL;
#endif
 CHECK(npr_file_host_destroy(h)==NPR_OK);
}
static void managed_boundaries(void){
 char path[4097];make_file(path,(uint8_t *)"value",5);NprFileHost *h=host_for(path);NprHostBinding real={READ,h},bad={bad_callback,&callback_mode};NmsRuntime r;runtime_start(&r);NmsHandle arg;
 size_t pn=strlen(path);uint8_t prefix[4200];memcpy(prefix,path,pn);prefix[pn]=0;memset(prefix+pn+1,'x',64);CHECK(nms_create(&r,prefix,pn+65,&arg)==NMS_OK);
 NprManagedResult x=npr_read_managed(&r,arg,&real);CHECK(x.host_status==NPR_OK&&x.managed_status==NMS_OK);view_is(&r,x.value,"value",5);CHECK(nms_release(&r,x.value)==NMS_OK);
 for(callback_mode=0;callback_mode<5;callback_mode++){callback_count=0;x=npr_read_managed(&r,arg,&bad);CHECK(callback_count==1&&!x.value&&x.managed_status==NMS_OK);CHECK(x.host_status==(callback_mode==4?NPR_DENIED:NPR_INVALID));}
 callback_mode=5;for(unsigned i=0;i<32;i++){x=npr_read_managed(&r,arg,&bad);CHECK(x.host_status==NPR_OK&&x.managed_status==NMS_OK);view_is(&r,x.value,"q",1);CHECK(nms_release(&r,x.value)==NMS_OK);}
 callback_count=0;x=npr_read_managed(NULL,arg,&bad);CHECK(x.host_status==NPR_OK&&x.managed_status==NMS_STATE&&!x.value);r.active=0;x=npr_read_managed(&r,arg,&bad);CHECK(x.managed_status==NMS_STATE&&!x.value);r.active=1;
 x=npr_read_managed(&r,0,&bad);CHECK(x.managed_status==NMS_STATE&&!x.value);CHECK(!callback_count);
 x=npr_read_managed(&r,arg,NULL);CHECK(x.host_status==NPR_DENIED&&!x.value);
 NprHostBinding absent={NULL,h};x=npr_read_managed(&r,arg,&absent);CHECK(x.host_status==NPR_DENIED&&!x.value);absent=(NprHostBinding){bad_callback,NULL};x=npr_read_managed(&r,arg,&absent);CHECK(x.host_status==NPR_DENIED&&!x.value&&!callback_count);
 NmsHandle array;CHECK(nms_string_array_create(&r,&array)==NMS_OK);x=npr_read_managed(&r,array,&bad);CHECK(x.host_status==NPR_OK&&x.managed_status==NMS_TYPE&&!x.value&&!callback_count);CHECK(nms_release(&r,array)==NMS_OK);
 uint8_t long_path[4097];memset(long_path,'x',sizeof long_path);NmsHandle long_arg;CHECK(nms_create(&r,long_path,sizeof long_path,&long_arg)==NMS_OK);x=npr_read_managed(&r,long_arg,&bad);CHECK(x.host_status==NPR_LIMIT&&!x.value&&!callback_count);CHECK(nms_release(&r,long_arg)==NMS_OK);
 CHECK(nms_create(&r,long_path,4096,&long_arg)==NMS_OK);callback_expected_path=long_path;callback_expected_length=4096;callback_mode=5;
 x=npr_read_managed(&r,long_arg,&bad);CHECK(callback_count==1&&x.host_status==NPR_OK&&x.managed_status==NMS_OK);view_is(&r,x.value,"q",1);CHECK(nms_release(&r,x.value)==NMS_OK);callback_expected_path=NULL;CHECK(nms_release(&r,long_arg)==NMS_OK);
 CHECK(nms_create(&r,NULL,0,&long_arg)==NMS_OK);io_reset();x=npr_read_managed(&r,long_arg,&real);CHECK(x.host_status==NPR_DENIED&&x.managed_status==NMS_OK&&!x.value);
#ifdef READ_OBSERVED
 CHECK(!opens&&!closes);
#endif
 CHECK(nms_release(&r,long_arg)==NMS_OK);
 CHECK(nms_release(&r,arg)==NMS_OK);runtime_end(&r);x=npr_read_managed(&r,0,&bad);CHECK(x.managed_status==NMS_DISPOSED&&!x.value&&x.host_status==NPR_OK);CHECK(npr_file_host_destroy(h)==NPR_OK);
 char missing[4097];CHECK(snprintf(missing,sizeof missing,"%s/missing",directory)>0);h=host_for(missing);uint8_t data[8];uint32_t n=99;CHECK(READ(h,(uint8_t *)missing,strlen(missing),data,8,&n)==NPR_OK&&n==0);CHECK(npr_file_host_destroy(h)==NPR_OK);
}
static void io_faults(void){
#ifdef READ_OBSERVED
 char path[4097];make_file(path,(uint8_t *)"abc",3);NprFileHost *h=host_for(path);uint8_t data[8];uint32_t n;FILE *sentinel=fopen(path,"rb");CHECK(sentinel);int fd=fileno(sentinel);
 for(unsigned mode=0;mode<3;mode++){io_reset();read_error=mode==0;close_error=mode!=0;n=91;int s=READ(h,(uint8_t *)path,strlen(path),data,mode==2?1:8,&n);CHECK(s==(mode==2?NPR_LIMIT:NPR_OK));CHECK(n==(mode==2?91:0));CHECK(opens==1&&closes==1&&successful_opens==1);CHECK(fcntl(last_fd,F_GETFD)==-1&&errno==EBADF);CHECK(fcntl(fd,F_GETFD)!=-1);if(mode==0)CHECK(read_progress);}
 io_reset();n=91;CHECK(READ(h,(uint8_t *)path,strlen(path),data,8,&n)==NPR_OK&&n==3);CHECK(fclose(sentinel)==0);CHECK(npr_file_host_destroy(h)==NPR_OK);
#endif
}
static void allocation_faults(void){
#ifdef READ_OBSERVED
 char path[4097];make_file(path,(uint8_t *)"abc",3);NprPath p={(uint8_t *)path,(uint32_t)strlen(path)};NprFileHost *h=(void *)(uintptr_t)1;
 requests=0;fail_at=0;failures=0;CHECK(npr_file_host_create(&p,1,&h)==NPR_MEMORY&&h==(void *)(uintptr_t)1&&failures==1&&!live_count);fail_at=-1;
 size_t measured=0;
 for(unsigned pass=0;pass<3;pass++)for(size_t point=0;point<(pass?measured:1);point++){
  h=host_for(path);NprHostBinding b={READ,h};NmsRuntime r;runtime_start(&r);NmsHandle roots[8];
  for(unsigned i=0;i<8;i++)CHECK(nms_create(&r,(uint8_t *)path,strlen(path),&roots[i])==NMS_OK);
  CHECK(!r.free_head);CHECK(nms_retain(&r,roots[0])==NMS_OK);CHECK(nms_retain(&r,roots[0])==NMS_OK);CHECK(nms_prepare_collection(&r)==NMS_OK);
  size_t before_count=live_count,before_bytes=live_bytes;uint64_t objects=r.live_objects,bytes=r.live_bytes;requests=failures=0;fail_at=pass?(long)point:-1;persistent=pass==1;io_reset();
  NprManagedResult x=npr_read_managed(&r,roots[0],&b);
  if(!pass){measured=requests;CHECK(measured==4&&x.value&&x.host_status==NPR_OK&&x.managed_status==NMS_OK);CHECK(nms_release(&r,x.value)==NMS_OK);}
  else {CHECK(failures&&requests==point+1&&!x.value);CHECK(x.host_status==(point?NPR_OK:NPR_MEMORY));CHECK(x.managed_status==(point?NMS_MEMORY:NMS_OK));CHECK(live_count==before_count&&live_bytes==before_bytes);CHECK(opens==(point?1u:0u)&&closes==opens);}
  CHECK(r.live_objects==objects&&r.live_bytes==bytes);for(unsigned i=0;i<8;i++)view_is(&r,roots[i],path,strlen(path));CHECK(r.slots[roots[0]&~NMS_DYNAMIC].references==3);
  fail_at=-1;NprManagedResult recovery=npr_read_managed(&r,roots[0],&b);CHECK(recovery.host_status==NPR_OK&&recovery.managed_status==NMS_OK);view_is(&r,recovery.value,"abc",3);CHECK(nms_release(&r,recovery.value)==NMS_OK);
  CHECK(nms_release(&r,roots[0])==NMS_OK);CHECK(nms_release(&r,roots[0])==NMS_OK);for(unsigned i=0;i<8;i++)CHECK(nms_release(&r,roots[i])==NMS_OK);runtime_end(&r);CHECK(npr_file_host_destroy(h)==NPR_OK);CHECK(!live_count&&!live_bytes);
 }
 printf("I checked %zu measured call allocations in persistent and transient modes.\n",measured);
#else
 CHECK(!failures&&!live_count&&!live_bytes);
#endif
}
int main(int argc,char **argv){CHECK(argc==2);CHECK(strlen(argv[1])<sizeof directory);strcpy(directory,argv[1]);real_vectors();context_and_ranges();copied_allowlist_and_filename();managed_boundaries();io_faults();allocation_faults();CHECK(!live_count&&!live_bytes);printf("I passed %u private read-text adapter checks; no bytecode admission.\n",checks);return 0;}
