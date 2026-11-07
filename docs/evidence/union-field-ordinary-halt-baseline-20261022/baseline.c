#include "nanovm/vm.h"
#include "nanoisa/assembler.h"
#include "nanoisa/verifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc=0;char **g_argv=NULL;
#define REQUIRE(x) do {if(!(x)){fprintf(stderr,"failed %s:%d: %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static void write_file(const char *dir,const char *name,const char *suffix,const void *bytes,size_t n){char path[1024];snprintf(path,sizeof path,"%s/%s.%s",dir,name,suffix);FILE *f=fopen(path,"wb");REQUIRE(f);REQUIRE(fwrite(bytes,1,n,f)==n);REQUIRE(!fclose(f));}
static void one(const char *dir,const char *name,const char *text,int core){
 write_file(dir,name,"nasm",text,strlen(text));AsmResult a;NvmModule *m=asm_assemble_unverified(text,&a);
 printf("CASE %s assemble=%u %s\n",name,a.error,a.message);if(!m)return;
 uint32_t size=0;uint8_t *bytes=nvm_serialize(m,&size);REQUIRE(bytes);write_file(dir,name,"nvm",bytes,size);free(bytes);
 NvmVerifyResult verified=nvm_verify(m);printf("VERIFY %s ok=%d message=%s\n",name,verified.ok,verified.error_msg);
 if(!verified.ok){nvm_module_free(m);return;}
 VmState *v=calloc(1,sizeof *v);REQUIRE(v);vm_init(v,m);REQUIRE(v->verified&&v->last_error==VM_OK);
 VmResult result=VM_OK;int trap=-1;
 if(core){VmCallFrame *f=&v->frames[0];f->fn_idx=0;f->module=m;f->owned_callable=val_void();f->instruction_ip=m->functions[0].code_offset;v->frame_count=1;v->current_fn=0;v->ip=f->instruction_ip;VmTrap t=vm_core_execute(v);trap=t.type;}
 else result=vm_execute(v);
 printf("FIRST %s public=%d result=%d trap=%d error=%d message=%s stack=%u frames=%u fn=%u ip=%u floor=%u refs=%d callee_refs=%d globals=%u capacity=%u objects=%llu\n",name,!core,result,trap,v->last_error,v->error_msg,v->stack_size,v->frame_count,v->current_fn,v->ip,v->activation_floor,v->references.active,v->callee_references.active,v->global_count,v->global_capacity,(unsigned long long)v->heap.stats.num_objects);
 for(uint32_t i=0;i<v->stack_size;i++)printf("STACK %s %u tag=%u int=%lld\n",name,i,v->stack[i].tag,(long long)(v->stack[i].tag==TAG_INT?v->stack[i].as.i64:0));
 for(uint32_t i=0;i<v->global_count;i++)printf("GLOBAL %s %u tag=%u int=%lld\n",name,i,v->globals[i].tag,(long long)(v->globals[i].tag==TAG_INT?v->globals[i].as.i64:0));
 vm_destroy(v);printf("DESTROY %s objects=%llu allocated=%llu freed=%llu intern=%u frames=%u\n",name,(unsigned long long)v->heap.stats.num_objects,(unsigned long long)v->heap.stats.allocated,(unsigned long long)v->heap.stats.freed,v->heap.intern_count,v->frame_count);REQUIRE(!v->heap.stats.num_objects&&!v->heap.intern_count&&!v->frame_count);free(v);nvm_module_free(m);fflush(stdout);
}
int main(int argc,char **argv){REQUIRE(argc==2);
 one(argv[1],"return",".entry 0\n.function main 0 0 0 int 1\nPUSH_I64 7\nRET\n.end\n",0);
 one(argv[1],"root-halt",".entry 0\n.function main 0 1 0 int 1\nPUSH_I64 9\nSTORE_LOCAL 0\nPUSH_I64 11\nHALT\n.end\n",0);
 one(argv[1],"call-halt",".entry 0\n.function main 0 0 0 int 1\nCALL 1\nRET\n.end\n.function helper 0 0 0 int 1\nPUSH_I64 12\nHALT\n.end\n",0);
 one(argv[1],"tail-halt",".entry 0\n.function main 0 0 0 int 1\nTAIL_CALL 1\n.end\n.function helper 0 0 0 int 1\nPUSH_I64 13\nHALT\n.end\n",0);
 one(argv[1],"assert-halt",".entry 0\n.function main 0 0 0 int 1\nPUSH_BOOL 1\nASSERT\nPUSH_I64 14\nHALT\n.end\n",0);
 one(argv[1],"false-assert",".entry 0\n.function main 0 0 0 int 1\nPUSH_BOOL 0\nASSERT\nPUSH_I64 15\nRET\n.end\n",0);
 one(argv[1],"init-halt",".entry 0\n.function main 0 0 0 int 1\nPUSH_I64 99\nSTORE_GLOBAL 0\nPUSH_I64 7\nRET\n.end\n.function __init__ 0 0 0 int 1\nPUSH_I64 16\nHALT\n.end\n",0);
 one(argv[1],"init-return",".entry 0\n.function main 0 0 0 int 1\nPUSH_I64 99\nSTORE_GLOBAL 0\nPUSH_I64 7\nRET\n.end\n.function __init__ 0 0 0 int 1\nPUSH_I64 16\nRET\n.end\n",0);
 one(argv[1],"core-halt",".entry 0\n.function main 0 0 0 int 1\nPUSH_I64 17\nHALT\n.end\n",1);
 return 0;
}
