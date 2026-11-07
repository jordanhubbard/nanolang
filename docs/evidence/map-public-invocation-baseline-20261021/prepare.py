from pathlib import Path
r=Path.cwd();w=Path(__file__).parent
s=(r/'tests/nanoisa/test_record_map_runtime.c').read_text();part=s[s.index('static void runtime_alias_failure(void) {'):];part=part[:part.index('    VmRecordMapPrivate *p=')]
preamble='''#define MAP_ORIGIN_MAIN old_map_controls
#include "test_record_map_origins.c"
#include "vm.h"
int g_argc;char **g_argv;
static void runtime_integer(Input*c,int64_t v){op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)b8(c->code,&c->n,(uint8_t)((uint64_t)v>>(8*i)));}
static void runtime_patch(Input*c,size_t p){size_t at=p+1;b32(c->code,&at,(uint32_t)(int32_t)(c->n-p));}
'''
part+='''    c.m.layout_data=NULL;c.m.layout_size=0;c.m.ownership_data=NULL;c.m.ownership_size=0;
    CHECK(nvm_verify(&c.m).ok);VmState vm;vm_init(&vm,&c.m);
    VmResult first=vm_execute(&vm);printf("alias first=%d stack=%u frames=%u globals=%u\\n",first,vm.stack_size,vm.frame_count,vm.global_count);CHECK(first==VM_OK);
    NanoValue result=vm_get_result(&vm);CHECK(result.tag==TAG_STRUCT&&result.as.obj==vm.globals[0].as.obj);
    void *identity=result.as.obj;vm.stack_size--;vm_release(&vm.heap,result);
    VmResult second=vm_execute(&vm);printf("alias second=%d stack=%u frames=%u globals=%u identity=%d\\n",second,vm.stack_size,vm.frame_count,vm.global_count,vm.globals[0].as.obj==identity);
    CHECK(second==VM_ERR_ASSERT_FAILED&&vm.globals[0].as.obj==identity);
    NanoValue value=vm_hashmap_get(vm.globals[0].as.sval->fields[0].as.hashmap,val_string(vm.module_constants.strings[1]));CHECK(value.tag==TAG_INT&&value.as.i64==2);
    vm_destroy(&vm);CHECK(vm.heap.stats.num_objects==0);
}
'''
loader='''static void repeated(const char*path){FILE*f=fopen(path,"rb");CHECK(f);fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);void*b=malloc(n);CHECK(b&&fread(b,1,n,f)==(size_t)n);fclose(f);NvmV2Module v;NvmModule*m=NULL;CHECK(nvm_v2_module_deserialize(b,n,&v)==NVM_V2_OK);CHECK(nvm_v2_to_nvm_module(&v,&m)==NVM_V2_OK);nvm_v2_module_free(&v);free(b);CHECK(nvm_verify(m).ok);VmState vm;vm_init(&vm,m);for(unsigned i=0;i<2;i++){VmResult status=vm_execute(&vm);CHECK(status==VM_OK);NanoValue value=vm_get_result(&vm);CHECK(value.tag==TAG_INT&&value.as.i64==0);printf("%s run%u result0 globals%u\\n",path,i,vm.global_count);vm.stack_size--;vm_release(&vm.heap,value);CHECK(!vm.stack_size&&!vm.frame_count);}vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);nvm_module_free(m);}
int main(int argc,char**argv){runtime_alias_failure();for(int i=1;i<argc;i++)repeated(argv[i]);printf("PASS %u ordinary lifecycle checks\\n",checks);return 0;}
'''
(w/'baseline.c').write_text(preamble+part+loader)
