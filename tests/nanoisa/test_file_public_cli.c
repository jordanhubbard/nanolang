/* Real temporary files; close/write failures below model reporting after actual
 * I/O. I do not infer arbitrary libc fclose failure resource behavior. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"I fail CLI check %u\n",__LINE__);exit(1);}}while(0)
static int fault,live,rejected_fd=-1;static unsigned closes;
static void *cli_malloc(size_t n){if(fault==1)return NULL;void *p=malloc(n);if(p)live++;return p;}
static void cli_free(void *p){if(p)live--;free(p);}
static int cli_close(FILE *f){int fd=fileno(f);int r=fclose(f);closes++;REQUIRE(fcntl(fd,F_GETFD)==-1 && errno==EBADF);if(fault==2){errno=EIO;return EOF;}return r;}
static size_t cli_write(const void *p,size_t n,size_t count,FILE *f){
 if(fault==3){size_t r=fwrite(p,n,count?count-1:0,f);errno=ENOSPC;return r;}return fwrite(p,n,count,f);
}
static int cli_flush(FILE *f){int r=fflush(f);if(fault==4){errno=ENOSPC;return EOF;}return r;}
static int cli_rename(const char *a,const char *b){if(fault==5){errno=EACCES;return -1;}return rename(a,b);}
static FILE *cli_fdopen(int fd,const char *mode){if(fault==6){rejected_fd=fd;errno=EMFILE;return NULL;}return fdopen(fd,mode);}
#define malloc cli_malloc
#define free cli_free
#undef fclose
#define fclose cli_close
#undef fwrite
#define fwrite cli_write
#undef fflush
#define fflush cli_flush
#define rename cli_rename
#define fdopen cli_fdopen
#include "../../src/nanoisa/file_cli.c"
#undef malloc
#undef free
#undef fclose
#undef fwrite
#undef fflush
#undef rename
#undef fdopen
static void retained(const char *path){char b[16]={0};FILE *f=fopen(path,"rb");REQUIRE(f);REQUIRE(fread(b,1,9,f)==9 && !memcmp(b,"preserved",9));REQUIRE(!fclose(f));}
static void clean_directory(const char *dir){DIR *d=opendir(dir);REQUIRE(d);struct dirent *e;while((e=readdir(d)))REQUIRE(!strstr(e->d_name,".nano-file-"));REQUIRE(!closedir(d));}
int main(void){
 char directory[]="/tmp/nano-file-cli-XXXXXX";REQUIRE(mkdtemp(directory));char path[256];REQUIRE(snprintf(path,sizeof path,"%s/out.c",directory)>0);
 FILE *f=fopen(path,"wb");REQUIRE(f);REQUIRE(fwrite("preserved",1,9,f)==9 && !fclose(f));
 char error[256];
 for(fault=1;fault<=6;fault++){
  REQUIRE(!nvm_file_cli_write(path,"replacement",error,sizeof error));REQUIRE(error[0] && !live);if(fault==6)REQUIRE(rejected_fd>=0 && fcntl(rejected_fd,F_GETFD)==-1 && errno==EBADF);retained(path);clean_directory(directory);
  int expected=fault==1?ENOMEM:fault==2?EIO:(fault==3 || fault==4)?ENOSPC:fault==5?EACCES:EMFILE;REQUIRE(strstr(error,strerror(expected)));
 }
 fault=0;REQUIRE(nvm_file_cli_write(path,"replacement",error,sizeof error));REQUIRE(!error[0] && !live);clean_directory(directory);
 uint8_t *bytes=(uint8_t *)(uintptr_t)1;size_t n=919;
 fault=1;REQUIRE(!nvm_file_cli_read(path,&bytes,&n,error,sizeof error) && bytes==(uint8_t *)(uintptr_t)1 && n==919 && !live);
 fault=2;unsigned old=closes;REQUIRE(!nvm_file_cli_read(path,&bytes,&n,error,sizeof error) && bytes==(uint8_t *)(uintptr_t)1 && n==919 && !live && closes==old+1);
 fault=0;REQUIRE(nvm_file_cli_read(path,&bytes,&n,error,sizeof error));REQUIRE(n==11 && !memcmp(bytes,"replacement",11) && live==1);cli_free(bytes);REQUIRE(!live);
 REQUIRE(!unlink(path) && !rmdir(directory));puts("PASS real CLI staging and modeled I/O/allocation failure cleanup");return 0;
}
