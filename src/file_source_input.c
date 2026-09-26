#define _POSIX_C_SOURCE 200809L
#include "file_source_input.h"
#include "utf8.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
NlFileCompanionReport nl_file_source_input_read(const char *path,size_t cap,char **out,size_t *size) {
 NlFileCompanionReport r={NL_FILE_COMPANION_INVALID,NL_FILE_COMPANION_INPUT,UINT32_MAX,0,0,0,0,0,0};
 if(!path||!out||!size||!cap||cap>NL_FILE_COMPANION_BYTES)return r;
 int fd=open(path,O_RDONLY|O_NONBLOCK|O_CLOEXEC);
 if(fd<0){r.status=NL_FILE_COMPANION_IO;r.stage=NL_FILE_COMPANION_OPEN;r.system_error=errno;return r;}
 char *data=NULL;size_t n=0,at=0;unsigned interrupted=0;struct stat st;
 r.status=NL_FILE_COMPANION_IO;r.stage=NL_FILE_COMPANION_STAT;
 if(fstat(fd,&st)){r.system_error=errno;goto close_input;}
 if(!S_ISREG(st.st_mode)||st.st_size<0){r.status=NL_FILE_COMPANION_INVALID;goto close_input;}
 if((uint64_t)st.st_size>=cap){r.status=NL_FILE_COMPANION_LIMIT;goto close_input;}
 n=(size_t)st.st_size;r.peak_heap_bytes_reserved=n+1;
 data=malloc(n+1);if(!data){r.status=NL_FILE_COMPANION_MEMORY;r.stage=NL_FILE_COMPANION_STORAGE;goto close_input;}
 r.stage=NL_FILE_COMPANION_READ;
 while(at<n) {
  ssize_t got=read(fd,data+at,n-at);
  if(got<0&&errno==EINTR&&interrupted++<64)continue;
  if(got<=0){r.system_error=got<0?errno:0;goto close_input;}at+=(size_t)got;
 }
 { char extra;ssize_t got;
  do {got=read(fd,&extra,1);}while(got<0&&errno==EINTR&&interrupted++<64);
  if(got!=0){r.system_error=got<0?errno:0;goto close_input;}
 }
 data[n]=0;r.work_reserved=n*3+1;
 if(memchr(data,0,n)||!nl_utf8_validate(data,n,NULL)){r.status=NL_FILE_COMPANION_INVALID;goto close_input;}
 r.status=NL_FILE_COMPANION_OK;r.stage=NL_FILE_COMPANION_DONE;
 close_input:
 if(close(fd)) {
  int err=errno;if(r.status==NL_FILE_COMPANION_OK){r.status=NL_FILE_COMPANION_IO;r.stage=NL_FILE_COMPANION_CLOSE;r.system_error=err;}
  else r.close_error=err;
 }
 if(r.status!=NL_FILE_COMPANION_OK){free(data);return r;}
 *out=data;*size=n;return r;
}
int64_t nl_file_source_metadata(const char *path,int64_t size) {
 if(!path||size<=0||size>NL_FILE_COMPANION_PATH||strnlen(path,(size_t)size+1)!=(uint64_t)size)return -1;
 const char *slash=strrchr(path,'/');if(!slash)return -1;
 size_t parent=(size_t)(slash-path)+1;char metadata[NL_FILE_COMPANION_PATH+16];
 memcpy(metadata,path,parent);memcpy(metadata+parent,"module.json",12);
 struct stat st;if(!lstat(metadata,&st))return 1;
 return errno==ENOENT?0:-1;
}
