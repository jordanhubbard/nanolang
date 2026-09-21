/* I retain the unchanged linked VM corpus and capture its actual observations.
 * Each generated product is compiled/replayed later in a separate bounded phase. */
#include "../../src/nanovm/record_array_runtime_private.h"
#include "../../src/nanoisa/nvm2c_record_array_private.h"
#include "../../src/nanoisa/record_array_generated_private.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "../../src/nanoisa/record_array_structure_private.h"
#include "../../src/nanovm/vm.c"
static NvmArrayEligibilityResult capture_create(const NvmModule *,VmRecordArrayPrivate **);
static VmResult capture_run(VmRecordArrayPrivate *);
static void capture_destroy(VmRecordArrayPrivate *);
static bool capture_stats(const VmRecordArrayPrivate *,VmRecordArrayPrivateStats *);
static bool capture_observe(const VmRecordArrayPrivate *,VmRecordArrayRoot,uint32_t,const uint32_t *,uint16_t,VmRecordArrayObservation *);
static bool capture_string(const VmRecordArrayPrivate *,VmRecordArrayRoot,uint32_t,const uint32_t *,uint16_t,uint32_t,uint32_t,void *);
#define vm_record_array_private_create capture_create
#define vm_record_array_private_run capture_run
#define vm_record_array_private_destroy capture_destroy
#define vm_record_array_private_stats capture_stats
#define vm_record_array_private_observe capture_observe
#define vm_record_array_private_string capture_string
#define RECORD_ARRAY_VM_MAIN original_linked_vm_corpus
#include "test_record_array_vm.c"
#undef RECORD_ARRAY_VM_MAIN
#undef vm_record_array_private_create
#undef vm_record_array_private_run
#undef vm_record_array_private_destroy
#undef vm_record_array_private_stats
#undef vm_record_array_private_observe
#undef vm_record_array_private_string

typedef struct {
    VmRecordArrayPrivate *vm;
    FILE *trace;
    VmRecordArrayPrivateStats baseline;
    unsigned number,run_count;
    unsigned runs[512];
} Capture;
static Capture captures[256];
static unsigned products,actions;
static bool retired[256];
static const char *artifact_directory;
static Capture *capture_find(const VmRecordArrayPrivate *p) {
    for(unsigned i=0;i<256;i++)if(captures[i].vm==p && p)return &captures[i];
    return NULL;
}
static unsigned status_value(VmResult s) {
    switch(s) {
    case VM_OK:return NRG_OK;
    case VM_ERR_TYPE_ERROR:return NRG_TYPE;
    case VM_ERR_OUT_OF_BOUNDS:return NRG_BOUNDS;
    case VM_ERR_ASSERT_FAILED:return NRG_ASSERT;
    case VM_ERR_MEMORY:return NRG_MEMORY;
    case VM_ERR_CALL_DEPTH:return NRG_FRAMES_EXHAUSTED;
    case VM_ERR_UNDEFINED_FUNCTION:return NRG_UNDEFINED_FUNCTION;
    default:CHECK(0);return NRG_STATE;
    }
}
static void trace_path(FILE *f,const uint32_t *path,uint16_t count) {
    fprintf(f,"const uint32_t path[%u]={",count?count:1);
    for(uint16_t i=0;i<count;i++)fprintf(f,"%u,",path[i]);
    if(!count)fputs("0",f);
    fputs("};\n",f);
}
static const char replay_prefix[]=
    "#define NANO_RECORD_ARRAY_GENERATED_PRIVATE 1\n"
    "#include \"record_array_generated_private.h\"\n#include <stdint.h>\n"
    "#ifndef __wasm32__\n#include <stdio.h>\n#endif\n"
    "#ifdef NRG_OBSERVED\n#include \"record_array_alloc.h\"\n#endif\n"
    "NrgStatus nrg_generated_create(NrgInstance **);\n"
    "static uint64_t identities[4096][2];static unsigned identity_count,checks;\n"
    "static int identity(uint64_t expected,uint64_t actual){if(!expected||!actual)return 0;"
    "for(unsigned i=0;i<identity_count;i++){if(identities[i][0]==expected)return identities[i][1]==actual;"
    "if(identities[i][1]==actual)return 0;}if(identity_count==4096)return 0;"
    "identities[identity_count][0]=expected;identities[identity_count++][1]=actual;return 1;}\n"
    "#define CHECK(x) do{checks++;if(!(x))return (int)checks;}while(0)\n"
    "static int exercise(void){(void)&identity;NrgInstance *p=(void *)(uintptr_t)1;"
    "CHECK(nrg_generated_create(&p)==NRG_OK&&p!=(void *)(uintptr_t)1);"
    "NrgStats initial;CHECK(nrg_stats(p,&initial));\n";
static NvmArrayEligibilityResult capture_create(const NvmModule *m,VmRecordArrayPrivate **out) {
    char *source=(void *)(uintptr_t)1;size_t length=SIZE_MAX;
    NvmRecordArrayGeneratedCost cost;memset(&cost,0xa5,sizeof cost);
    NvmArrayEligibilityResult generated=nvm2c_record_array_private(m,&source,&length,&cost);
    NvmArrayEligibilityResult actual=vm_record_array_private_create(m,out);
    CHECK(generated.status==actual.status);
    if(actual.status!=NVM_ARRAY_ELIGIBLE) {
        CHECK(source==(void *)(uintptr_t)1&&length==SIZE_MAX);
        return actual;
    }
    CHECK(cost.plan_bytes<=NVM_RECORD_ARRAY_EXECUTION_BYTES&&cost.plan_steps<=NVM_RECORD_ARRAY_EXECUTION_STEPS);
    CHECK(cost.consumer_bytes<=NRG_EXTRA_BYTES&&cost.consumer_steps<=NRG_EXTRA_STEPS);
    Capture *capture=NULL;
    for(unsigned i=0;i<256;i++)if(!captures[i].vm){capture=&captures[i];break;}
    CHECK(capture&&products<4096);
    capture->number=products++;capture->vm=*out;
    CHECK(vm_record_array_private_stats(*out,&capture->baseline));
    vm_profile_enable(&(*out)->vm,true);
    char path[4096];int n=snprintf(path,sizeof path,"%s/product-%04u.c",artifact_directory,capture->number);
    CHECK(n>0&&(size_t)n<sizeof path);
    FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(source,1,length,f)==length);CHECK(!fclose(f));free(source);
    n=snprintf(path,sizeof path,"%s/product-%04u.replay.c",artifact_directory,capture->number);
    CHECK(n>0&&(size_t)n<sizeof path);capture->trace=fopen(path,"wb");CHECK(capture->trace);
    CHECK(fputs(replay_prefix,capture->trace)>=0);
    return actual;
}
static VmResult capture_run(VmRecordArrayPrivate *p) {
    VmResult actual=vm_record_array_private_run(p);Capture *c=capture_find(p);
    if(!c)return actual;
    for(unsigned op=0;op<256;op++)retired[op]|=p->vm.profile.opcode_counts[op]!=0;
    actions++;CHECK(c->run_count<512);c->runs[c->run_count++]=status_value(actual);
    fprintf(c->trace,"{NrgObservation before,after;bool had=nrg_observe(p,false,0,0,0,&before);"
        "NrgStatus s=nrg_run(p);CHECK(s==%u);identity_count=0;"
        "if(s!=NRG_OK&&had){CHECK(nrg_observe(p,false,0,0,0,&after));"
        "CHECK(before.tag==after.tag&&before.identity==after.identity&&before.scalar_bits==after.scalar_bits);}}\n",status_value(actual));
    return actual;
}
static bool capture_stats(const VmRecordArrayPrivate *p,VmRecordArrayPrivateStats *out) {
    bool actual=vm_record_array_private_stats(p,out);Capture *c=capture_find(p);
    if(!c || !actual)return actual;
    actions++;
    fprintf(c->trace,"{NrgStats s;CHECK(nrg_stats(p,&s));"
        "CHECK(s.epoch==UINT64_C(%" PRIu64 ")&&s.frames==%u&&s.maximum_frames==%u&&s.status==%u&&s.has_result==%u);"
        "CHECK(s.preparation_bytes<=NRG_EXTRA_BYTES);",
        out->epoch,out->active_frames,out->maximum_frames,status_value(out->last_status),(unsigned)out->has_result);
    if(out->heap_objects==c->baseline.heap_objects&&out->heap_live_bytes==c->baseline.heap_live_bytes)
        fputs("CHECK(s.live_objects==initial.live_objects&&s.live_bytes==initial.live_bytes);",c->trace);
    fputs("}\n",c->trace);return actual;
}
static bool capture_observe(const VmRecordArrayPrivate *p,VmRecordArrayRoot root,uint32_t index,
    const uint32_t *path,uint16_t count,VmRecordArrayObservation *out) {
    bool actual=vm_record_array_private_observe(p,root,index,path,count,out);Capture *c=capture_find(p);
    if(!c)return actual;
    CHECK(count<=257&&(!count||path));actions++;
    fputs("{",c->trace);trace_path(c->trace,path,count);
    fprintf(c->trace,"NrgObservation o;CHECK(nrg_observe(p,%s,%u,path,%u,&o)==%u);",
        root==VM_RA_GLOBAL?"true":"false",index,count,(unsigned)actual);
    if(actual) {
        fprintf(c->trace,"CHECK(o.epoch==UINT64_C(%" PRIu64 ")&&o.tag==%u&&o.scalar_bits==UINT64_C(%" PRIu64 ")&&o.length==%u&&o.layout==%u&&o.element==%u);",
            out->epoch,out->tag,out->scalar_bits,out->length,out->layout,out->element_tag);
        if(out->identity)fprintf(c->trace,"CHECK(identity(UINT64_C(%" PRIu64 "),o.identity));",out->identity);
        else fputs("CHECK(!o.identity);",c->trace);
    }
    fputs("}\n",c->trace);return actual;
}
static bool capture_string(const VmRecordArrayPrivate *p,VmRecordArrayRoot root,uint32_t index,
    const uint32_t *path,uint16_t count,uint32_t offset,uint32_t length,void *out) {
    bool actual=vm_record_array_private_string(p,root,index,path,count,offset,length,out);Capture *c=capture_find(p);
    if(!c)return actual;
    CHECK(count<=257&&(!count||path)&&length<=65536);actions++;
    fputs("{",c->trace);trace_path(c->trace,path,count);
    fprintf(c->trace,"unsigned char bytes[%u];for(unsigned j=0;j<%u;j++)bytes[j]=165;"
        "CHECK(nrg_string(p,%s,%u,path,%u,%u,%u,%s)==%u);",
        length?length:1,length?length:1,root==VM_RA_GLOBAL?"true":"false",index,count,offset,length,out?"bytes":"0",(unsigned)actual);
    if(actual)for(uint32_t i=0;i<length;i++)fprintf(c->trace,"CHECK(bytes[%u]==%u);",i,((unsigned char *)out)[i]);
    else fprintf(c->trace,"for(unsigned j=0;j<%u;j++)CHECK(bytes[j]==165);",length?length:1);
    fputs("}\n",c->trace);return actual;
}
static void capture_destroy(VmRecordArrayPrivate *p) {
    Capture *c=capture_find(p);
    if(c) {
        FILE *f=c->trace;
        CHECK(fputs("nrg_destroy(p);\n#ifdef NMS_TESTING\nCHECK(!nms_test_live_allocations());\n#endif\nreturn 0;}\n",f)>=0);
        fprintf(f,"#ifdef NRG_OBSERVED\nstatic const unsigned expected[%u]={",c->run_count?c->run_count:1);
        for(unsigned i=0;i<c->run_count;i++)fprintf(f,"%u,",c->runs[i]);
        if(!c->run_count)fputs("0",f);
        fprintf(f,"};\nstatic int sequence(int fault){NrgInstance *p=(void *)(uintptr_t)1;"
            "NrgStatus s=nrg_generated_create(&p);if(s==NRG_MEMORY){CHECK(fault&&p==(void *)(uintptr_t)1);return 0;}"
            "CHECK(s==NRG_OK);NrgStats prepared;CHECK(nrg_stats(p,&prepared));"
            "CHECK(ra_peak<=prepared.preparation_bytes);int failed=0;for(unsigned i=0;i<%u;i++){"
            "NrgObservation before,after;bool had=nrg_observe(p,false,0,0,0,&before);s=nrg_run(p);"
            "if(s==NRG_MEMORY){CHECK(fault);if(had){CHECK(nrg_observe(p,false,0,0,0,&after));"
            "CHECK(before.tag==after.tag&&before.identity==after.identity&&before.scalar_bits==after.scalar_bits);}failed=1;break;}"
            "CHECK(s==expected[i]);}nrg_destroy(p);CHECK(!ra_live&&!ra_bytes&&!nms_test_live_allocations());"
            "CHECK(!fault||failed);return 0;}\n",c->run_count);
        fputs("static int faults(void){CHECK(!ra_live&&!ra_bytes);ra_calls=0;ra_peak=0;ra_fail=SIZE_MAX;ra_persistent=0;"
            "CHECK(!sequence(0));size_t calls=ra_calls,peak=ra_peak;CHECK(calls>0);"
            "for(int mode=0;mode<2;mode++)for(size_t i=0;i<calls;i++){ra_calls=0;ra_peak=0;ra_fail=i;ra_persistent=mode;"
            "CHECK(!sequence(1));CHECK(ra_calls>i&&!ra_live&&!ra_bytes&&!nms_test_live_allocations());"
            "ra_calls=0;ra_peak=0;ra_fail=SIZE_MAX;ra_persistent=0;CHECK(!sequence(0));}"
            "printf(\"I checked %zu actual runtime allocation positions in both modes; peak %zu bytes.\\n\",calls,peak);return 0;}\n#endif\n"
            "#ifdef __wasm32__\nint nano_main(void){return exercise();}\n#else\n"
            "int main(void){int status=exercise();\n#ifdef NRG_OBSERVED\nif(!status)status=faults();\n#endif\n"
            "printf(\"I checked %u generated replay observations; status %d.\\n\",checks,status);return status?1:0;}\n#endif\n",f);
        CHECK(!fclose(f));*c=(Capture){0};
    }
    vm_record_array_private_destroy(p);
}

int main(int argc,char **argv) {
    CHECK(argc==2);artifact_directory=argv[1];
    int status=original_linked_vm_corpus();CHECK(status==0);
    unsigned operations=0;
    for(unsigned i=0;i<256;i++) {
        CHECK(!captures[i].vm);
        CHECK(retired[i]==nvm_record_array_opcode_supported((uint8_t)i));operations+=retired[i];
    }
    CHECK(operations==93);
    puts("I captured all93 actual retired operations and all256 decisions from the unchanged linked corpus.");
    char path[4096];int n=snprintf(path,sizeof path,"%s/corpus-counts.json",artifact_directory);
    CHECK(n>0&&(size_t)n<sizeof path);FILE *f=fopen(path,"wb");CHECK(f);
    fprintf(f,"{\"products\":%u,\"actions\":%u,\"original_vm_checks_plus_capture\":%u}\n",products,actions,checks);
    CHECK(!fclose(f));printf("I retained %u unchanged VM programs and %u actions for generated replay.\n",products,actions);
    return 0;
}
