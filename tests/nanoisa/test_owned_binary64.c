/* I compare already-admitted scalar float operations while real owners stay live. */
#include "affine_bytecode.h"
#include "assembler.h"
#include "retained_layouts.h"
#include "verifier.h"
#include "nvm2c.h"
#include "../../src/nanovm/vm.h"
#include "../../src/binary64_arithmetic.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc=0;char **g_argv=NULL;
static unsigned checks;
#define CHECK(c) do {checks++;if(!(c)){fprintf(stderr,"check line %d: %s\n",__LINE__,#c);abort();}} while(0)
static void word(uint8_t *p,uint32_t v){for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(i*8));}
static void slot(uint8_t *p,uint8_t tag,uint32_t layout){p[0]=tag;word(p+4,layout);}
static void append(char *s,size_t cap,const char *format,...) {
    size_t n=strlen(s);va_list ap;va_start(ap,format);int added=vsnprintf(s+n,cap-n,format,ap);va_end(ap);
    CHECK(added>=0&&(size_t)added<cap-n);
}
static void comparisons(char *source,size_t capacity,unsigned first) {
    const char *values[]={"0.0","-0.0","1.5","-2.5","inf","-inf","nan"};
    const char *ops[]={"EQ","NE","LT","LE","GT","GE","F64_EQ","F64_NE","F64_LT","F64_LE","F64_GT","F64_GE"};
    for(unsigned op=first;op<first+6;op++)for(unsigned a=0;a<7;a++)for(unsigned b=0;b<7;b++) {
        double x=strtod(values[a],NULL),y=strtod(values[b],NULL);int order=x<y?-1:x>y?1:0;
        bool expected=op%6==0?x==y:op%6==1?x!=y:op%6==2?(op<6?order<0:x<y):
            op%6==3?(op<6?order<=0:x<=y):op%6==4?(op<6?order>0:x>y):(op<6?order>=0:x>=y);
        append(source,capacity,"PUSH_F64 %s\nDUP\nPOP\nPUSH_F64 %s\n%s\nPUSH_BOOL %u\nEQ\nASSERT\n",values[a],values[b],ops[op],expected);
    }
}
static NvmModule *fixture(unsigned index) {
    char source[100000]=".types 1 0 0\n.entry 0\n.function main 0 4 0 int 1\n";
    append(source,sizeof(source),"%s\nOWN_PACK 0\nOWN_STORE_LOCAL 0\n",index==5?"PUSH_F64 42.0":"PUSH_I64 42");
    if(index<2)comparisons(source,sizeof(source),index*6);
    else {
        const char *op[]={"F64_ADD","F64_SUB","F64_MUL","F64_DIV"};
        for(unsigned i=0;i<4;i++) {
            double expected=i==0?nano_rt_f64_add(1.5,2.5):i==1?nano_rt_f64_sub(1.5,2.5):i==2?nano_rt_f64_mul(1.5,2.5):nano_rt_f64_div(1.5,2.5);
            append(source,sizeof(source),"PUSH_F64 1.5\nPUSH_F64 2.5\n%s\nPUSH_F64 %.17g\nF64_EQ\nASSERT\n",op[i],expected);
        }
        append(source,sizeof(source),"PUSH_F64 nan\nPUSH_F64 1.0\nF64_ADD\nDUP\nF64_NE\nASSERT\nPUSH_F64 inf\nPUSH_F64 -inf\nF64_ADD\nDUP\nF64_NE\nASSERT\nPUSH_F64 nan\nPUSH_F64 -0.0\nF64_DIV\nPUSH_F64 0.0\nF64_EQ\nASSERT\n");
        append(source,sizeof(source),"PUSH_F64 -2.5\nF64_NEG\nDUP\nPUSH_F64 2.5\nSWAP\nF64_EQ\nASSERT\nPUSH_I64 0\nSTORE_LOCAL 2\nloop:\nLOAD_LOCAL 2\nPUSH_I64 3\nLT\nJMP_FALSE done\nPUSH_F64 1.0\nF64_ADD\nLOAD_LOCAL 2\nPUSH_I64 1\nADD\nSTORE_LOCAL 2\nJMP loop\ndone:\nPUSH_F64 5.5\nEQ\nASSERT\n");
        if(index==7)append(source,sizeof(source),"PUSH_F64 0.0\n");
        append(source,sizeof(source),"CALL 1\n%s\n",index==6?"POP":"ASSERT");
        if(index==3)append(source,sizeof(source),"PUSH_BOOL 0\nASSERT\n");
    }
    append(source,sizeof(source),"OWN_UNPACK_LOCAL 0\n%s\nASSERT\nPUSH_I64 0\nRET\n.end\n.function helper %u 1 0 %s 1\n%sRET\n.end\n",index==5?"PUSH_F64 42.0\nF64_EQ":"PUSH_I64 42\nEQ",index==7?1:0,index==6?"float":"bool",index==6?"PUSH_F64 1.5\n":index==7?"LOAD_LOCAL 0\nPUSH_F64 1.0\nLE\n":"PUSH_F64 nan\nPUSH_F64 1.0\nLE\n");
    if(index==7)append(source,sizeof(source),".parameters 1 float\n");
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m)fprintf(stderr,"%s\n",assembled.message);
    CHECK(m);
    NvmV2LayoutField field={index==5?TAG_FLOAT:TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout row={NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field};NvmV2Layouts layouts={&row,1};
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=84;m->ownership_data=calloc(84,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,2);word(p+4,1);p[8]=3;word(p+12,2);
    p[16]=4;slot(p+20,TAG_INT,NVM_V2_NO_INDEX);
    slot(p+28,TAG_STRUCT,0);slot(p+36,index==4?TAG_FLOAT:TAG_INT,NVM_V2_NO_INDEX);slot(p+44,TAG_INT,NVM_V2_NO_INDEX);slot(p+52,TAG_BOOL,NVM_V2_NO_INDEX);
    p[60]=1;p[62]=index==7?1:0;slot(p+64,index==6?TAG_FLOAT:TAG_BOOL,NVM_V2_NO_INDEX);slot(p+72,index==7?TAG_FLOAT:TAG_INT,NVM_V2_NO_INDEX);
    if(index<4){NvmVerifyResult result=nvm_verify(m);if(!result.ok)fprintf(stderr,"%s\n",result.error_msg);CHECK(result.ok);CHECK(nvm_verify_owned_module(m).ok);}
    return m;
}
static void artifacts(NvmModule *m,const char *dir,unsigned index) {
    NvmV2Module v2={0};size_t length;CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v2,NULL,0,&length)==NVM_V2_OK);uint8_t *bytes=malloc(length);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&v2,bytes,length,NULL)==NVM_V2_OK);
    char path[1024];snprintf(path,sizeof(path),"%s/case%u.nvm",dir,index);FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(bytes,1,length,f)==length);CHECK(!fclose(f));free(bytes);nvm_v2_module_free(&v2);
    char error[256];char *c=nvm2c_emit(m,error,sizeof(error));if(!c)fprintf(stderr,"%s\n",error);CHECK(c);
    snprintf(path,sizeof(path),"%s/case%u.c",dir,index);f=fopen(path,"w");CHECK(f);CHECK(fputs(c,f)>=0);CHECK(!fclose(f));free(c);
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    for(unsigned i=4;i<8;i++) {
        NvmModule *m=fixture(i);bool needs=false;
        CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);
        NvmVerifyResult rejected=nvm_verify_owned_module(m);CHECK(!rejected.ok);
        CHECK(strstr(rejected.error_msg,"owned record fields") || strstr(rejected.error_msg,"entry locals") || strstr(rejected.error_msg,"scalar entry"));
        char error[256];CHECK(!nvm2c_emit(m,error,sizeof(error)));nvm_module_free(m);
    }
    for(unsigned i=0;i<4;i++) {
        NvmModule *m=fixture(i);artifacts(m,argv[1],i);VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<2;repeat++) {
            NanoValue result=val_int(-91);
            VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):api==1?vm_execute(&vm):api==2?vm_call_function(&vm,0,NULL,0):vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
            CHECK(status==(i==3?VM_ERR_ASSERT_FAILED:VM_OK));
            if(i!=3){if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}CHECK(result.tag==TAG_INT&&result.as.i64==0);vm_release(&vm.heap,result);}
            CHECK(!vm.stack_size&&!vm.frame_count);CHECK(vm.heap.stats.num_objects==baseline);
            CHECK(!vm.references.active&&!vm.callee_references.active);
            for(unsigned n=0;n<NVM_OWNED_MAX_FUNCTIONS-2;n++)CHECK(!vm.value_references[n].active);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u %u\n",i,i==3?2:0);
    }
    printf("%u owned binary64 checks passed\n",checks);return 0;
}
