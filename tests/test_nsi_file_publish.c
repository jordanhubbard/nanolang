#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/random.h>
#endif
#ifdef __linux__
#include <sys/syscall.h>
#endif
#include "nsi_file_publish.h"
static unsigned checks;
#define CHECK(x) do { checks++; if(!(x)){fprintf(stderr,"CHECK %s:%d: %s\n",__FILE__,__LINE__,#x);abort();} } while(0)
static unsigned serial;
static const char *root;
static NlFileBindingPlan *plan;
static void path(char *out,size_t size,const char *leaf){int n=snprintf(out,size,"%s/%s",root,leaf);CHECK(n>=0&&(size_t)n<size);}
static void destination(char out[4096]){char leaf[64];snprintf(leaf,sizeof leaf,"case-%u",serial++);path(out,4096,leaf);}
static bool exists(const char *name){struct stat s;return lstat(name,&s)==0;}
static void remove_dir(const char *name){
    char p[4096];const char *children[]={"interface.nsi.json","binding.nano","unrelated"};
    for(size_t i=0;i<3;i++){int n=snprintf(p,sizeof p,"%s/%s",name,children[i]);CHECK(n>=0&&(size_t)n<sizeof p);if(unlink(p)!=0)CHECK(errno==ENOENT);}
    if(rmdir(name)!=0)CHECK(errno==ENOENT);
}
static void content(const char *directory,const char *leaf,const unsigned char *expected,size_t size){
    char p[4096];int n=snprintf(p,sizeof p,"%s/%s",directory,leaf);CHECK(n>=0&&(size_t)n<sizeof p);
    FILE *f=fopen(p,"rb");CHECK(f);unsigned char *bytes=malloc(size+1);CHECK(bytes);
    CHECK(fread(bytes,1,size+1,f)==size);CHECK(!ferror(f));CHECK(fclose(f)==0);CHECK(!memcmp(bytes,expected,size));free(bytes);
}
static void complete(const char *directory){size_t n;const unsigned char *b=nl_file_binding_interface_bytes(plan,&n);content(directory,"interface.nsi.json",b,n);b=nl_file_binding_source_bytes(plan,&n);content(directory,"binding.nano",b,n);}
#ifdef PUBLISH_INSTRUMENT
static unsigned events,fail_at,hit,secondary,secondary_hits,secondary_skip,short_reads,eintr_left,eintr_attempts,short_writes,zero_write;
static const char *eintr_op;
static const char *trace[256];
static bool active[8192];
static unsigned opened,closed,live,rename_calls;
static int rename_error;
static bool mismatch,extra_child,rename_parent,fixed_entropy,interleave;
static unsigned mkdir_attempts;
static unsigned mismatch_hits;
static bool event(const char *op){
    CHECK(events<256);trace[events]=op;events++;
    if(eintr_op&&!strcmp(op,eintr_op)){eintr_attempts++;if(eintr_left&&(!interleave||(eintr_attempts%2))){eintr_left--;errno=EINTR;return true;}}
    if(fail_at&&events==fail_at){hit++;errno=EIO;return true;}
    if(hit&&secondary&&((secondary==1&&!strcmp(op,"unlink"))||(secondary==2&&!strcmp(op,"close")))){if(secondary_skip){secondary_skip--;}else{secondary=0;secondary_hits++;errno=EBUSY;return true;}}
    return false;
}
static int own(int fd){if(fd>=0){CHECK((unsigned)fd<8192&&!active[fd]);active[fd]=true;opened++;live++;}return fd;}
static int h_open(const char *p,int flags,...){if(event("open"))return -1;return own(open(p,flags));}
static int h_openat(int d,const char *p,int flags,...){mode_t mode=0;if(flags&O_CREAT){va_list a;va_start(a,flags);mode=(mode_t)va_arg(a,int);va_end(a);}if(event("openat"))return -1;return own(openat(d,p,flags,mode));}
static int h_close(int fd){CHECK((unsigned)fd<8192&&active[fd]);active[fd]=false;live--;closed++;bool fail=event("close");int saved=errno;int r=close(fd);CHECK(r==0);if(fail){errno=saved;return -1;}return 0;}
static int h_mkdir(int d,const char *p,mode_t m){mkdir_attempts++;if(event("mkdir"))return -1;return mkdirat(d,p,m);}
static int h_fstat(int fd,struct stat *s){if(event("fstat"))return -1;return fstat(fd,s);}
static int h_fstatat(int fd,const char *p,struct stat *s,int flags){if(event("fstatat"))return -1;int r=fstatat(fd,p,s,flags);if(r==0&&mismatch&&!strcmp(p,"interface.nsi.json")){s->st_ino++;mismatch_hits++;}return r;}
static ssize_t h_write(int fd,const void *p,size_t n){if(event("write"))return -1;if(zero_write){zero_write--;return 0;}if(short_writes&&n>1){short_writes--;n=1;}return write(fd,p,n);}
static int h_sync(int fd){if(event("sync"))return -1;return fsync(fd);}
static int h_entropy(void *p,size_t n){if(event("entropy"))return -1;if(rename_parent){char moved[4096];int size=snprintf(moved,sizeof moved,"%s-moved",root);CHECK(size>0&&(size_t)size<sizeof moved);CHECK(rename(root,moved)==0);rename_parent=false;}if(fixed_entropy){memset(p,0,n);return 0;}return getentropy(p,n);}
static int h_unlink(int d,const char *p,int flags){if(event("unlink"))return -1;return unlinkat(d,p,flags);}
static int h_rename(int a,const char *b,int c,const char *d,unsigned flags){
    rename_calls++;if(event("rename"))return -1;
    if(rename_error){errno=rename_error;return -1;}
    if(extra_child){int stage=openat(a,b,O_RDONLY|O_DIRECTORY);CHECK(stage>=0);int fd=openat(stage,"unrelated",O_WRONLY|O_CREAT|O_EXCL,0600);CHECK(fd>=0);CHECK(write(fd,"keep",4)==4);CHECK(close(fd)==0);CHECK(close(stage)==0);errno=EIO;return -1;}
#ifdef __linux__
    return (int)syscall(SYS_renameat2,a,b,c,d,flags);
#else
    return renameatx_np(a,b,c,d,flags);
#endif
}
#ifdef __linux__
static long h_syscall(long number,int a,const char *b,int c,const char *d,unsigned flags){CHECK(number==SYS_renameat2);return h_rename(a,b,c,d,flags);}
#define syscall h_syscall
#else
#define renameatx_np h_rename
#endif
#define open h_open
#define openat h_openat
#define close h_close
#define mkdirat h_mkdir
#define fstat h_fstat
#define fstatat h_fstatat
#define write h_write
#define fsync h_sync
#define getentropy h_entropy
#define unlinkat h_unlink
#include "../src/nsi_file_publish.c"
#undef open
#undef openat
#undef close
#undef mkdirat
#undef fstat
#undef fstatat
#undef write
#undef fsync
#undef getentropy
#undef unlinkat
#ifdef __linux__
#undef syscall
#else
#undef renameatx_np
#endif
static void *input_allocation;
static ssize_t h_read(int fd,void *p,size_t n){if(event("read"))return -1;if(short_reads&&n>1){short_reads--;n=1;}return read(fd,p,n);}
static void *h_malloc(size_t n){if(event("malloc"))return NULL;CHECK(!input_allocation);input_allocation=malloc(n);return input_allocation;}
static void h_free(void *p){if(p){CHECK(p==input_allocation);input_allocation=NULL;}free(p);}
#define open h_open
#define close h_close
#define fstat h_fstat
#define read h_read
#define malloc h_malloc
#define free h_free
#define main fixture_cli_main
#include "../src/nsi_file_binding_main.c"
#undef main
#undef open
#undef close
#undef fstat
#undef read
#undef malloc
#undef free
static void reset(void){CHECK(live==0&&!input_allocation);events=fail_at=hit=secondary=secondary_hits=secondary_skip=short_reads=eintr_left=eintr_attempts=short_writes=zero_write=0;eintr_op=NULL;opened=closed=rename_calls=0;rename_error=0;mismatch=extra_child=rename_parent=fixed_entropy=interleave=false;mismatch_hits=mkdir_attempts=0;}
static void leftovers(const NlFileBindingPublishReport *r){if(r->staging_name[0]){char p[4096];path(p,sizeof p,r->staging_name);if(!r->cleanup_pending)CHECK(!exists(p));remove_dir(p);}CHECK(live==0&&opened==closed);}
static void faults(void){
    char dest[4096];NlFileBindingPublishReport r;reset();destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_OK);unsigned baseline=events;const char *names[256];memcpy(names,trace,sizeof names);CHECK(r.published&&r.durable&&!r.cleanup_pending);complete(dest);remove_dir(dest);CHECK(live==0&&opened==closed);
    for(unsigned i=1;i<=baseline;i++){
        reset();fail_at=i;destination(dest);NlFileBindingStatus s=nl_file_binding_publish(plan,dest,&r);CHECK(hit==1&&s==NL_FILE_BINDING_IO&&r.first_errno==EIO);CHECK(!r.durable||r.published);
        if(r.published){complete(dest);remove_dir(dest);}else CHECK(!exists(dest));
        printf("FAULT index=%u operation=%s stage=%u published=%u durable=%u cleanup=%d pending=%u\n",i,names[i-1],r.failed_stage,r.published,r.durable,r.cleanup_errno,r.cleanup_pending);leftovers(&r);
    }
    unsigned first_write=0;for(unsigned i=0;i<baseline;i++)if(!strcmp(names[i],"write")){first_write=i+1;break;}CHECK(first_write);
    for(unsigned kind=1;kind<=2;kind++){reset();fail_at=first_write;secondary=kind;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_IO);CHECK(r.first_errno==EIO&&r.cleanup_errno==EBUSY&&r.cleanup_pending&&secondary_hits==1&&!r.published);leftovers(&r);}
    unsigned rename_index=0,parent_sync_index=0;for(unsigned i=0;i<baseline;i++){if(!strcmp(names[i],"rename"))rename_index=i+1;if(!strcmp(names[i],"sync"))parent_sync_index=i+1;}CHECK(rename_index&&parent_sync_index>rename_index);
    for(unsigned skip=0;skip<3;skip++){reset();fail_at=rename_index;secondary=1;secondary_skip=skip;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_IO);CHECK(r.first_errno==EIO&&r.cleanup_errno==EBUSY&&secondary_hits==1&&!r.published&&r.cleanup_pending);leftovers(&r);}
    for(unsigned skip=0;skip<3;skip++){reset();fail_at=first_write;secondary=2;secondary_skip=skip;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_IO);CHECK(r.first_errno==EIO&&r.cleanup_errno==EBUSY&&secondary_hits==1&&r.cleanup_pending);leftovers(&r);}
    reset();fail_at=parent_sync_index;secondary=2;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_IO);CHECK(r.published&&!r.durable&&r.first_errno==EIO&&r.cleanup_errno==EBUSY&&secondary_hits==1);complete(dest);remove_dir(dest);leftovers(&r);
    reset();fail_at=first_write;mismatch=true;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_IO);CHECK(r.cleanup_pending&&r.cleanup_errno==ESTALE&&mismatch_hits==1);char stage[4096];path(stage,sizeof stage,r.staging_name);CHECK(exists(stage));leftovers(&r);
    reset();extra_child=true;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_IO);CHECK(!r.published&&r.cleanup_pending&&r.first_errno==EIO);path(stage,sizeof stage,r.staging_name);content(stage,"unrelated",(const unsigned char *)"keep",4);leftovers(&r);
    const int errors[]={ENOSYS,ENOTSUP,EINVAL,EEXIST,ENOTEMPTY,EACCES};
    for(size_t i=0;i<sizeof errors/sizeof errors[0];i++){reset();rename_error=errors[i];destination(dest);NlFileBindingStatus expected=i<3?NL_FILE_BINDING_UNSUPPORTED:(i<5?NL_FILE_BINDING_EXISTS:NL_FILE_BINDING_IO);CHECK(nl_file_binding_publish(plan,dest,&r)==expected);CHECK(!r.published&&r.first_errno==errors[i]&&rename_calls==1);leftovers(&r);}
    for(unsigned which=0;which<2;which++)for(unsigned count=64;count<=65;count++){
        reset();eintr_op=which?"sync":"write";eintr_left=count;destination(dest);NlFileBindingStatus s=nl_file_binding_publish(plan,dest,&r);
        if(count==64){CHECK(s==NL_FILE_BINDING_OK&&r.published&&r.durable);complete(dest);remove_dir(dest);}else CHECK(s==NL_FILE_BINDING_IO&&r.first_errno==EINTR&&!r.published);
        CHECK(eintr_left==0);printf("EINTR operation=%s interrupts=%u attempts=%u status=%u\n",eintr_op,count,eintr_attempts,s);leftovers(&r);
    }
    reset();interleave=true;eintr_op="write";eintr_left=65;short_writes=128;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_IO&&r.first_errno==EINTR&&!r.published);CHECK(eintr_left==0&&eintr_attempts==129);leftovers(&r);
    reset();rename_parent=true;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_OK);CHECK(!exists(dest));char moved[4096];int moved_size=snprintf(moved,sizeof moved,"%s-moved",root);CHECK(moved_size>0&&(size_t)moved_size<sizeof moved);CHECK(rename(moved,root)==0);complete(dest);remove_dir(dest);leftovers(&r);
    reset();fixed_entropy=true;char collision[4096];path(collision,sizeof collision,".nsi-file-binding-00000000000000000000000000000000");CHECK(mkdir(collision,0700)==0);destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_IO&&r.first_errno==EEXIST&&mkdir_attempts==64&&!r.published&&!r.staging_name[0]);CHECK(exists(collision));CHECK(rmdir(collision)==0);leftovers(&r);
    reset();short_writes=8;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_OK);CHECK(short_writes==0);complete(dest);remove_dir(dest);leftovers(&r);
    reset();zero_write=1;destination(dest);CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_IO&&r.first_errno==EIO&&!r.published);leftovers(&r);
    reset();memset(&r,0x5a,sizeof r);CHECK(nl_file_binding_publish(NULL,"bad",&r)==NL_FILE_BINDING_INVALID);CHECK(events==0&&!r.published);CHECK(nl_file_binding_publish(plan,"bad",NULL)==NL_FILE_BINDING_INVALID&&events==0);
    const char *bad[]={"","/",".","..","x/","x/.","x/.."};for(size_t i=0;i<sizeof bad/sizeof bad[0];i++)CHECK(nl_file_binding_publish(plan,bad[i],&r)==NL_FILE_BINDING_INVALID&&events==0);
    char longpath[4097];memset(longpath,'x',sizeof longpath);longpath[4096]=0;CHECK(nl_file_binding_publish(plan,longpath,&r)==NL_FILE_BINDING_LIMIT&&events==0);
    longpath[256]=0;CHECK(nl_file_binding_publish(plan,longpath,&r)==NL_FILE_BINDING_LIMIT&&events==0);
}
static void cli_faults(const char *input){
    char dest[4096];char *args[]={"nsi-file-binding",(char *)input,"--file-binding-dir",dest,NULL};
    reset();destination(dest);CHECK(fixture_cli_main(4,args)==0);unsigned baseline=events;const char *names[256];memcpy(names,trace,sizeof names);complete(dest);remove_dir(dest);CHECK(!input_allocation&&live==0);
    unsigned prepublication=0;
    for(unsigned i=0;i<baseline;i++){if(!strcmp(names[i],"close")){prepublication=i+1;break;}}
    CHECK(prepublication>0);
    for(unsigned i=1;i<=prepublication;i++){reset();fail_at=i;destination(dest);CHECK(fixture_cli_main(4,args)==1);CHECK(hit==1&&!input_allocation&&live==0&&opened==closed&&rename_calls==0&&!exists(dest));printf("CLI_FAULT index=%u operation=%s\n",i,names[i-1]);}
    for(unsigned count=64;count<=65;count++){reset();eintr_op="read";eintr_left=count;destination(dest);int rc=fixture_cli_main(4,args);if(count==64){CHECK(rc==0);complete(dest);remove_dir(dest);}else CHECK(rc==1&&rename_calls==0&&!exists(dest));CHECK(!input_allocation&&live==0&&eintr_left==0);printf("CLI_EINTR interrupts=%u attempts=%u result=%d\n",count,eintr_attempts,rc);}
    reset();interleave=true;eintr_op="read";eintr_left=65;short_reads=128;destination(dest);CHECK(fixture_cli_main(4,args)==1&&eintr_left==0&&eintr_attempts==129&&rename_calls==0);CHECK(!input_allocation&&live==0&&!exists(dest));
    reset();short_reads=8;destination(dest);CHECK(fixture_cli_main(4,args)==0&&short_reads==0);complete(dest);remove_dir(dest);CHECK(!input_allocation&&live==0);
    reset();CHECK(fixture_cli_main(1,args)==2&&events==0);
}
#endif
int main(int argc,char **argv){
    CHECK(argc==3);root=argv[2];FILE *f=fopen(argv[1],"rb");CHECK(f);unsigned char bytes[16384];size_t n=fread(bytes,1,sizeof bytes,f);CHECK(n>0&&n<sizeof bytes&&!ferror(f));CHECK(fclose(f)==0);CHECK(nl_file_binding_prepare(bytes,n,&plan)==NL_FILE_BINDING_OK);
    char dest[4096];destination(dest);NlFileBindingPublishReport r;CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_OK);CHECK(r.published&&r.durable&&!r.first_errno&&!r.cleanup_errno&&!r.cleanup_pending);CHECK(exists(dest));complete(dest);
    CHECK(nl_file_binding_publish(plan,dest,&r)==NL_FILE_BINDING_EXISTS);CHECK(!r.published&&!r.durable&&!r.cleanup_pending);complete(dest);remove_dir(dest);
#ifdef PUBLISH_INSTRUMENT
    faults();cli_faults(argv[1]);
#endif
    nl_file_binding_free(plan);printf("PASS File publisher checks=%u mode=%s\n",checks,
#ifdef PUBLISH_INSTRUMENT
    "instrumented"
#else
    "linked"
#endif
    );return 0;
}
