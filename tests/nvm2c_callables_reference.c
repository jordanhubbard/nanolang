/* I expose complete callable targets/projection masks from unchanged C source. */
#include "../src/nanoisa/nvm2c_callables.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
static uint32_t word(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
int main(int argc,char **argv){
 for(int arg=1;arg<argc;arg++){
  FILE *file=fopen(argv[arg],"rb");if(!file)return 2;
  if(fseek(file,0,SEEK_END))return 2;
  long size=ftell(file);if(size<0 || fseek(file,0,SEEK_SET))return 2;
  unsigned char *p=malloc(size?(size_t)size:1);if(!p)return 2;
  if(fread(p,1,(size_t)size,file)!=(size_t)size || fclose(file))return 2;
  NvmModule m={0};int framed=0;
  if(size>=12){
   uint32_t fc=word(p),sc=word(p+4),ic=word(p+8);size_t pos=12+(size_t)fc*24;
   if(pos<=(size_t)size && ic<=((size_t)size-pos)/20 && sc<=((size_t)size-pos)/4){
    m.functions=calloc(fc?fc:1,sizeof *m.functions);m.strings=calloc(sc?sc:1,sizeof *m.strings);
    m.imports=calloc(ic?ic:1,sizeof *m.imports);m.import_param_types=calloc(ic?ic:1,sizeof *m.import_param_types);
    if(!m.functions || !m.strings || !m.imports || !m.import_param_types)return 2;
    m.function_count=fc;m.string_count=sc;m.import_count=ic;framed=1;
    for(uint32_t i=0;i<fc;i++){
     const unsigned char *f=p+12+(size_t)i*24;
     if(word(f+8)>65535 || word(f+12)>65535 || word(f+16)>65535 || word(f+20)>255){framed=0;break;}
     m.functions[i].code_offset=word(f);m.functions[i].code_length=word(f+4);m.functions[i].arity=(uint16_t)word(f+8);
     m.functions[i].local_count=(uint16_t)word(f+12);m.functions[i].upvalue_count=(uint16_t)word(f+16);m.functions[i].result_count=(uint8_t)word(f+20);
    }
    for(uint32_t i=0;i<ic && framed;i++){
     if((size_t)size-pos<20){framed=0;break;}
     const unsigned char *v=p+pos;uint32_t params=word(v+8),present=word(v+16);
     if(params>65535 || word(v+12)>255 || present>1){framed=0;break;}
     m.imports[i].module_name_idx=word(v);m.imports[i].function_name_idx=word(v+4);m.imports[i].param_count=(uint16_t)params;m.imports[i].return_type=(uint8_t)word(v+12);pos+=20;
     if(present){
      if(params>((size_t)size-pos)/4){framed=0;break;}
      m.import_param_types[i]=calloc(params?params:1,1);if(!m.import_param_types[i])return 2;
      for(uint32_t k=0;k<params;k++){uint32_t tag=word(p+pos);pos+=4;if(tag>255){framed=0;break;}m.import_param_types[i][k]=(uint8_t)tag;}
     }
    }
    if(sc>((size_t)size-pos)/4)framed=0;
    for(uint32_t i=0;i<sc && framed;i++){
     if((size_t)size-pos<4){framed=0;break;}uint32_t len=word(p+pos);pos+=4;
     if(len>(size_t)size-pos){framed=0;break;}
     m.strings[i]=calloc((size_t)len+1,1);if(!m.strings[i])return 2;memcpy(m.strings[i],p+pos,len);pos+=len;
    }
    if((size_t)size-pos>UINT32_MAX)framed=0;
    if(framed){m.code=p+pos;m.code_size=(uint32_t)((size_t)size-pos);}
   }
  }
  if(!framed)puts("P");else{
   NvmCallableAnalysis a={0};int valid=nvm_callable_analyze(&m,&a);
   if(!valid)printf("0|%s\n",a.error);else{
    putchar('1');for(uint32_t f=0;f<=m.function_count;f++){
     uint32_t length=f<m.function_count?m.functions[f].code_length:0;
     for(uint32_t pc=0;pc<=length;pc++){
      printf("|%u,%u:%u:",f,pc,nvm_callable_projection_tags(&a,f,pc));NvmShapeId id=nvm_callable_at(&a,f,pc);
      if(!id)putchar('-');else{size_t targets=nvm_shape_function_count(&a.shapes,id);printf("%zu",targets);
       for(size_t k=0;k<targets;k++){uint32_t target;if(!nvm_shape_function_target(&a.shapes,id,k,&target))abort();printf(",%u",target);}}
     }
    }putchar('\n');
   }nvm_callable_destroy(&a);
  }
  for(uint32_t i=0;m.strings && i<m.string_count;i++)free(m.strings[i]);
  for(uint32_t i=0;m.import_param_types && i<m.import_count;i++)free(m.import_param_types[i]);
  free(m.functions);free(m.strings);free(m.imports);free(m.import_param_types);free(p);
 }
 return 0;
}
