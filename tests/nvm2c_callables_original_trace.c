/* I retain every original assertion and capture actual module inputs. */
#include "../src/nanoisa/nvm2c_callables.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static FILE *packets;
static void put_word(uint32_t n){unsigned char b[4]={(unsigned char)n,(unsigned char)(n>>8),(unsigned char)(n>>16),(unsigned char)(n>>24)};if(fwrite(b,1,4,packets)!=4)abort();}
static int observed_analyze(const NvmModule *m,NvmCallableAnalysis *a){
 long start=ftell(packets);if(start<0)abort();put_word(0);
 put_word(m->function_count);put_word(m->string_count);put_word(m->import_count);
 for(uint32_t f=0;f<m->function_count;f++){
  const NvmFunctionEntry *fn=&m->functions[f];put_word(fn->code_offset);put_word(fn->code_length);put_word(fn->arity);put_word(fn->local_count);put_word(fn->upvalue_count);put_word(fn->result_count);
 }
 for(uint32_t i=0;i<m->import_count;i++){
  const NvmImportEntry *p=&m->imports[i];int typed=m->import_param_types && m->import_param_types[i];
  put_word(p->module_name_idx);put_word(p->function_name_idx);put_word(p->param_count);put_word(p->return_type);put_word(typed);
  if(typed)for(uint16_t k=0;k<p->param_count;k++)put_word(m->import_param_types[i][k]);
 }
 for(uint32_t i=0;i<m->string_count;i++){
  size_t length=strlen(m->strings[i]);if(length>UINT32_MAX)abort();put_word((uint32_t)length);if(fwrite(m->strings[i],1,length,packets)!=length)abort();
 }
 if(fwrite(m->code,1,m->code_size,packets)!=m->code_size)abort();
 long end=ftell(packets);if(end<start || (unsigned long)(end-start-4)>UINT32_MAX || fseek(packets,start,SEEK_SET))abort();put_word((uint32_t)(end-start-4));if(fseek(packets,end,SEEK_SET))abort();
 return nvm_callable_analyze(m,a);
}
#define nvm_callable_analyze observed_analyze
#define main original_main
#include "nanoisa/test_nvm2c_callables.c"
#undef main
#undef nvm_callable_analyze
int main(int argc,char **argv){
 if(argc!=2)return 2;packets=fopen(argv[1],"wb");if(!packets)return 2;
 int result=original_main(1,argv);if(fclose(packets))return 2;return result;
}
