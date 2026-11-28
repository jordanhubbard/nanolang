/* I observe the unchanged shared collector; I grant no executable admission. */
#include "../src/nanovm/vm_decode.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../src/nanoisa/nvm2c_effect_flow.h"
static uint32_t word(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
int main(int argc,char **argv){
 for(int arg=1;arg<argc;arg++){
  FILE *file=fopen(argv[arg],"rb");if(!file)return 2;
  if(fseek(file,0,SEEK_END))return 2;
  long size=ftell(file);if(size<0 || fseek(file,0,SEEK_SET))return 2;
  unsigned char *p=malloc(size?(size_t)size:1);if(!p)return 2;
  if(fread(p,1,(size_t)size,file)!=(size_t)size || fclose(file))return 2;
  NvmModule m={0};Nvm2cHandler *handlers=NULL;size_t count=0;int valid=0;
  if(size>=8){
   uint32_t fc=word(p),sc=word(p+4);size_t pos=8+(size_t)fc*12;
   if(pos<=(size_t)size && sc<=((size_t)size-pos)/4){
    m.functions=calloc(fc?fc:1,sizeof *m.functions);m.strings=calloc(sc?sc:1,sizeof *m.strings);
    if(!m.functions || !m.strings)return 2;
    m.function_count=fc;m.string_count=sc;valid=1;
    for(uint32_t i=0;i<fc;i++){
     const unsigned char *f=p+8+(size_t)i*12;uint32_t locals=word(f+8);
     if(locals>65535){valid=0;break;}
     m.functions[i].code_offset=word(f);m.functions[i].code_length=word(f+4);m.functions[i].local_count=(uint16_t)locals;
    }
    for(uint32_t i=0;i<sc && valid;i++){
     if((size_t)size-pos<4){valid=0;break;}
     uint32_t len=word(p+pos);pos+=4;if(len>(size_t)size-pos){valid=0;break;}
     m.strings[i]=calloc((size_t)len+1,1);if(!m.strings[i])return 2;
     memcpy(m.strings[i],p+pos,len);pos+=len;
    }
    if((size_t)size-pos>UINT32_MAX)valid=0;
    if(valid){m.code=p+pos;m.code_size=(uint32_t)((size_t)size-pos);valid=effect_collect_entries(&m,&handlers,&count);}
   }
  }
  if(!valid)puts("0");else{
   printf("1:%zu",count);
   for(size_t i=0;i<count;i++){
    Nvm2cHandler *h=&handlers[i];printf("|%u,%u,%u,%u,%u:",h->function,h->operation,h->target,h->start,h->count);
    for(uint32_t j=0;j<=m.functions[h->function].code_length;j++)putchar(h->flow[j]?'1':'0');
   }putchar('\n');
  }
  for(size_t i=0;i<count;i++)free(handlers[i].flow);
  for(uint32_t i=0;m.strings && i<m.string_count;i++)free(m.strings[i]);
  free(handlers);free(m.functions);free(m.strings);free(p);
 }
 return 0;
}
