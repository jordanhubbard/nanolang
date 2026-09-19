/* I observe representation without changing public values or admission. */
#define NANO_OWNER_ARRAY_RUNTIME_MAIN retained_bits_runtime_main
#include "test_private_owned_array_runtime.c"
static const uint64_t patterns[]={UINT64_C(0),UINT64_C(0x8000000000000000),UINT64_C(0x7ff8000000000042),UINT64_C(0xfff8000000000123)};
static VmState *bit_vm;
static VmArray *bit_array;
static unsigned bit_seen,bit_case;
static uint32_t observed_pcs[32];
static unsigned observed_count;
NanoValue bits_actual_array_get(VmArray *,uint32_t);
static uint64_t payload(NanoValue v){uint64_t bits=0;CHECK(v.tag==TAG_FLOAT);_Static_assert(sizeof bits==sizeof v.as.f64,"I require binary64 storage");memcpy(&bits,&v.as.f64,sizeof bits);return bits;}
static uint64_t final_bits(unsigned index){return index==0?patterns[3]:index<4?patterns[index]:patterns[(index-4)%4];}
NanoValue vm_array_get(VmArray *array,uint32_t index) {
    NanoValue result=bits_actual_array_get(array,index);
    if(bit_vm){
        unsigned expected_index,expected_fn;uint64_t expected;
        if(bit_seen<4){expected_index=bit_seen;expected_fn=1;expected=patterns[expected_index];}
        else if(!bit_case && bit_seen<8){expected_index=bit_seen-4;expected_fn=0;expected=patterns[expected_index];}
        else {unsigned offset=bit_seen-(bit_case?4:8);if(!offset){expected_index=0;expected_fn=3;expected=patterns[0];}else{offset--;expected_index=offset%9;expected_fn=offset<9?3:0;expected=final_bits(expected_index);}}
        CHECK(bit_seen<observed_count && index==expected_index && bit_vm->current_fn==expected_fn && bit_vm->ip==observed_pcs[bit_seen]);
        if(!bit_array){bit_array=array;}CHECK(bit_array==array && array->elem_type==TAG_FLOAT);
        CHECK(payload(result)==expected);bit_seen++;
    }
    return result;
}
static void append_text(char *out,size_t size,const char *text){CHECK(strlen(out)+strlen(text)<size);strcat(out,text);}
static void reads(char *out,size_t size,unsigned local,unsigned count){char line[128];for(unsigned n=0;n<count;n++){snprintf(line,sizeof line,"LOAD_LOCAL %u\nPUSH_I64 %u\nARR_GET\nPOP\n",local,n);append_text(out,size,line);}}
static void observer_sites(const NvmModule *m,unsigned fn,unsigned skip,unsigned take){
    unsigned found=0;uint32_t begin=m->functions[fn].code_offset,end=begin+m->functions[fn].code_length;
    for(uint32_t pc=begin;pc<end;){DecodedInstruction in;uint32_t width=isa_decode(m->code+pc,end-pc,&in);CHECK(width);if(in.opcode==OP_ARR_GET || in.opcode==OP_ARR_SET){if(found>=skip && found<skip+take){CHECK(observed_count<32);observed_pcs[observed_count++]=pc+width;}found++;}pc+=width;}
    CHECK(found>=skip+take);
}
static NvmModule *bits_module(unsigned which) {
    char root[4096]="CALL 1\nCALL 2\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nAGG_GET 1\nSTORE_LOCAL 1\n";
    char factory_body[2048]="PUSH_F64 1.0\nPUSH_F64 2.0\nPUSH_F64 3.0\nPUSH_F64 4.0\nARR_LITERAL 3 4\nSTORE_LOCAL 0\n";
    reads(factory_body,sizeof factory_body,0,4);
    append_text(factory_body,sizeof factory_body,"PUSH_I64 7\nOWN_PACK 0\nLOAD_LOCAL 0\nPUSH_STR value\nOWN_PACK 1\nRET\n");
    char consume[4096]="OWN_UNPACK_LOCAL 0\nPOP\nSTORE_LOCAL 1\nOWN_STORE_LOCAL 2\nOWN_UNPACK_LOCAL 2\nPOP\nLOAD_LOCAL 1\nPUSH_I64 0\nPUSH_F64 4.0\nARR_SET\nPOP\n";
    for(unsigned i=0;i<5;i++){char line[96];snprintf(line,sizeof line,"LOAD_LOCAL 1\nPUSH_F64 %u.0\nARR_PUSH\nPOP\n",i%4+1);append_text(consume,sizeof consume,line);}
    reads(consume,sizeof consume,1,9);append_text(consume,sizeof consume,"PUSH_I64 0\nRET\n");
    if(!which)reads(root,sizeof root,1,4);
    append_text(root,sizeof root,"OWN_MOVE_LOCAL 0\nCALL 3\nPOP\nPUSH_I64 7\nPRINTLN\n");
    if(!which)reads(root,sizeof root,1,9);
    else {
        const char *indices[]={"-1","-9223372036854775808","9223372036854775807"};
        unsigned group=(which-1)/3,index=(which-1)%3;char line[256];
        snprintf(line,sizeof line,"LOAD_LOCAL 1\nPUSH_I64 %s\n%s",indices[index],group==2?"PUSH_F64 1.0\nARR_SET\nPOP\n":group==1?"ARR_GET\nPUSH_F64 1.0\nF64_EQ\nPOP\n":"ARR_GET\nPOP\n");append_text(root,sizeof root,line);
    }
    append_text(root,sizeof root,"PUSH_I64 9\nPRINTLN\nPUSH_I64 0\nRET\n");
    Function f[]={{root,0,2,T(TAG_INT),{OWNER(1),T(TAG_ARRAY)}},{factory_body,0,1,OWNER(1),{T(TAG_ARRAY)}},{"OWN_MOVE_LOCAL 0\nRET\n",1,1,OWNER(1),{OWNER(1)}},{consume,1,3,T(TAG_INT),{OWNER(1),T(TAG_ARRAY),OWNER(0)}}};
    NvmModule *m=build(f,4);
    for(uint32_t pc=0;pc<m->code_size;){DecodedInstruction in;uint32_t width=isa_decode(m->code+pc,m->code_size-pc,&in);CHECK(width);if(in.opcode==OP_PUSH_F64){double value=in.operands[0].f64;CHECK(value>=1 && value<=4);unsigned at=(unsigned)value-1;memcpy(&in.operands[0].f64,&patterns[at],8);CHECK(isa_encode(&in,m->code+pc,width)==width);}pc+=width;}
    NvmOwnedArrayPlan *plan=NULL;NvmOwnerAuthorityResult admitted=nvm_owned_array_admit(m,&plan);if(admitted.status!=NVM_OWNER_AUTH_PREPARED)fprintf(stderr,"bits admission %u: %s\n",which,admitted.message);CHECK(admitted.status==NVM_OWNER_AUTH_PREPARED && nvm_verify(m).ok);nvm_owned_array_plan_free(plan);observed_count=0;observer_sites(m,1,0,4);if(!which)observer_sites(m,0,0,4);observer_sites(m,3,0,10);if(!which)observer_sites(m,0,4,9);CHECK(observed_count==(which?14:27));return m;
}
static void bits_run(NvmModule *m,unsigned which,unsigned fused){
    VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);VmDispatchProfile profile={.fuse_load_local_field=fused};vm_set_dispatch_profile(&vm,profile);CHECK(vm.dispatch_module_valid);
    size_t roots=vm.heap.stats.num_objects,bytes=vm.heap.stats.allocated-vm.heap.stats.freed;FILE *out=tmpfile();CHECK(out);vm.output=out;bit_vm=&vm;bit_array=NULL;bit_seen=0;bit_case=which;
    NanoValue result=val_int(-91);VmResult status=runtime_entry(&vm,&result);bit_vm=NULL;unsigned group=which?(which-1)/3:0;VmResult wanted=which&&group==1?VM_ERR_TYPE_ERROR:which&&group==2?VM_ERR_OUT_OF_BOUNDS:VM_OK;
    fprintf(stderr,"bits case=%u api=%u fused=%u status=%d observations=%u\n",which,public_api,fused,status,bit_seen);CHECK(status==wanted && bit_seen==(which?14:27));CHECK(wanted==VM_OK?(result.tag==TAG_INT && result.as.i64==0):runtime_failure_value(result));
    CHECK(!fflush(out));rewind(out);char text[32]={0};size_t n=fread(text,1,sizeof text-1,out);CHECK(!ferror(out) && n<sizeof text-1);CHECK(!strcmp(text,wanted==VM_OK?"7\n9\n":"7\n"));CHECK(!fclose(out));vm.output=NULL;clean(&vm,roots,bytes);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
int main(int argc,char **argv){CHECK(argc==2);for(unsigned which=0;which<10;which++){NvmModule *m=bits_module(which);char error[256],path[1024];char *source=nvm2c_emit(m,error,sizeof error);CHECK(source);snprintf(path,sizeof path,"%s/bits%u.c",argv[1],which);FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(source,1,strlen(source),f)==strlen(source));CHECK(!fclose(f));free(source);for(public_api=0;public_api<4;public_api++)for(unsigned fused=0;fused<2;fused++){bits_run(m,which,fused);bits_run(m,which,fused);}nvm_module_free(m);}printf("%u owned ARRAY bit and boundary checks passed\n",checks);return 0;}
