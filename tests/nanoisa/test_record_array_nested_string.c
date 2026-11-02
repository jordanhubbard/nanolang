/* I qualify exact depth-two STRING fields through actual public consumers. */
#define main retained_origin_fixture_main
#include "test_record_array_origins.c"
#undef main
#include "../../src/nanoisa/record_array_admission.h"
#include "../../src/nanoisa/nvm2c.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/disassembler.h"
#include "../../src/nanovm/vm.h"
#include "../../src/nanovm/record_array_runtime_private.h"
int g_argc=0;char **g_argv=NULL;
#ifdef RA_NESTED_WHITEBOX
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#include "../../src/nanovm/vm.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
#endif
static void pack_nested(Input *c) {
    op(c,OP_AGG_PACK);b8(c->code,&c->n,AGG_RECORD);b32(c->code,&c->n,0);
    b16(c->code,&c->n,0);b16(c->code,&c->n,1);
}
static void integer(Input *c,unsigned value) {
    scalar(c,TAG_INT);c->code[c->n-8]=(uint8_t)value;
}
static void string_value(Input *c) {
    arg32(c,OP_PUSH_STR,1);arg32(c,OP_PUSH_STR,1);op(c,OP_STR_CONCAT);
}
static void string_assert(Input *c) {string_value(c);op(c,OP_STR_EQ);op(c,OP_ASSERT);}
static void fixture(Input *c,unsigned mode) {
    init(c,TAG_STRING,false);c->fn[0].local_count=4;
    array(c,TAG_STRING);arg32(c,OP_PUSH_STR,1);op(c,OP_ARR_PUSH);
    arg32(c,OP_PUSH_STR,1);op(c,OP_ARR_PUSH);arg16(c,OP_STORE_LOCAL,1);
    array(c,TAG_ARRAY);if(mode>=5 && mode<=6)arg32(c,OP_CALL,1);arg16(c,OP_STORE_LOCAL,0);
    arg16(c,OP_LOAD_LOCAL,0);pack_nested(c);arg32(c,OP_STORE_GLOBAL,0);
    for(unsigned i=0;i<2;i++) {
        arg16(c,OP_LOAD_LOCAL,0);arg16(c,OP_LOAD_LOCAL,1);op(c,OP_ARR_PUSH);op(c,OP_POP);
    }
    if(mode==1) {
        arg16(c,OP_LOAD_LOCAL,0);integer(c,0);integer(c,2);op(c,OP_PUSH_BOOL);b8(c->code,&c->n,1);op(c,OP_ASSERT);op(c,OP_ARR_SLICE);arg16(c,OP_STORE_LOCAL,0);
    } else if(mode==2) {
        arg16(c,OP_LOAD_LOCAL,0);op(c,OP_ARR_POP);arg16(c,OP_STORE_LOCAL,1);
    } else if(mode==3) {
        arg16(c,OP_LOAD_LOCAL,0);integer(c,0);op(c,OP_ARR_REMOVE);op(c,OP_POP);
    } else if(mode==4) {
        arg32(c,OP_LOAD_GLOBAL,0);array(c,TAG_ARRAY);op(c,OP_AGG_SET);b16(c->code,&c->n,0);op(c,OP_POP);
    }
    for(unsigned i=0;i<2;i++) {
        arg16(c,OP_LOAD_LOCAL,1);integer(c,i);string_value(c);op(c,OP_ARR_SET);op(c,OP_POP);
    }
    if(mode==7) {
        arg16(c,OP_LOAD_LOCAL,1);integer(c,0);integer(c,2);op(c,OP_PUSH_BOOL);b8(c->code,&c->n,1);op(c,OP_ASSERT);
        op(c,OP_ARR_SLICE);arg16(c,OP_STORE_LOCAL,1);
    }
    arg16(c,OP_LOAD_LOCAL,0);integer(c,0);op(c,OP_ARR_GET);integer(c,0);op(c,OP_ARR_GET);string_assert(c);
    if(mode!=4) {
        arg32(c,OP_LOAD_GLOBAL,0);arg16(c,OP_AGG_GET,0);integer(c,0);op(c,OP_ARR_GET);
        integer(c,0);op(c,OP_ARR_GET);string_assert(c);
    }
    integer(c,0);op(c,OP_RET);c->fn[0].code_length=(uint32_t)c->n;
    if(mode>=5 && mode<=6) {
        c->strings[3]="forward";c->lengths[3]=7;
        c->m.function_count=mode==5?2:3;c->m.function_param_types=c->param_rows;c->params[0]=TAG_ARRAY;
        for(unsigned f=1;f<c->m.function_count;f++) {
            c->fn[f].name_idx=f+1;c->fn[f].arity=c->fn[f].local_count=1;
            c->fn[f].result_count=1;c->fn[f].result_tag=TAG_ARRAY;c->param_rows[f]=c->params;
            c->fn[f].code_offset=(uint32_t)c->n;arg16(c,OP_LOAD_LOCAL,0);
            if(f==1 && mode==6)arg32(c,OP_TAIL_CALL,2);else op(c,OP_RET);
            c->fn[f].code_length=(uint32_t)c->n-c->fn[f].code_offset;
        }
    }
    nested_authority(c);c->ownership[c->o-36]=TAG_STRING;
}
static void execute(const NvmModule *m) {
    VmState vm;vm_init(&vm,m);VmResult result=vm_execute(&vm);
    if(result!=VM_OK)fprintf(stderr,"I failed nested VM: %s\n",vm.error_msg);
    CHECK(result==VM_OK && vm.frame_count==0 && vm.stack_size==1 && vm.stack[0].tag==TAG_INT && vm.stack[0].as.i64==0);
    vm_destroy(&vm);CHECK(vm.heap.stats.num_objects==0);
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    VmRecordArrayPrivate *private=NULL;
    CHECK(vm_record_array_private_create(m,&private).status==NVM_ARRAY_ELIGIBLE);
    for(unsigned attempt=0;attempt<2;attempt++) {
        CHECK(vm_record_array_private_run(private)==VM_OK);
        VmRecordArrayPrivateStats stats;CHECK(vm_record_array_private_stats(private,&stats));
        CHECK(!stats.active_frames && !stats.active_stack && stats.has_result);
        uint32_t path[]={0};VmRecordArrayObservation outer;
        CHECK(vm_record_array_private_observe(private,VM_RA_GLOBAL,0,path,1,&outer));
        CHECK(outer.tag==TAG_ARRAY && outer.element_tag==TAG_ARRAY);
        if(outer.length) {
            uint32_t inner_path[]={0,0,0};VmRecordArrayObservation value;
            CHECK(vm_record_array_private_observe(private,VM_RA_GLOBAL,0,inner_path,3,&value));
            CHECK(value.tag==TAG_STRING && value.length==8);
        }
    }
    vm_record_array_private_destroy(private);
#endif
}
static void admitted(Input *c) {
    NvmRecordArrayExecutionPlan *plan=NULL;
    NvmArrayEligibilityResult result=nvm_record_array_admit(&c->m,&plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"I failed nested proof: %u %s\n",result.status,result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);nvm_record_array_execution_free(plan);CHECK(nvm_verify(&c->m).ok);
}
static void public_routes(Input *c,const char *directory,unsigned id) {
    admitted(c);execute(&c->m);
    char *assembly=disasm_module_styled(&c->m,DISASM_STYLE_CANONICAL);CHECK(assembly);
    AsmResult result;NvmModule *copy=asm_assemble(assembly,&result);
    if(!copy)fprintf(stderr,"I failed nested assembly: %s\n",result.message);
    CHECK(copy);CHECK(copy->ownership_size==c->m.ownership_size && !memcmp(copy->ownership_data,c->m.ownership_data,c->m.ownership_size));
    execute(copy);nvm_module_free(copy);free(assembly);
    NvmV2Module wire;CHECK(nvm_v2_from_nvm_module(&c->m,&wire)==NVM_V2_OK);
    size_t size=0;CHECK(nvm_v2_module_serialize(&wire,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);CHECK(nvm_v2_module_serialize(&wire,bytes,size,NULL)==NVM_V2_OK);
    NvmV2Module decoded;CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);execute(copy);
    nvm_module_free(copy);
    /* I require the decoded transport consumer to rederive nested authority;
     * an already serialized container is not a reusable proof token. */
    uint8_t *retained=(uint8_t *)decoded.ownership_data;
    size_t leaf=decoded.ownership_size-36;CHECK(retained[leaf]==TAG_STRING);
    retained[leaf]=TAG_INT;copy=NULL;
    CHECK(nvm_v2_to_nvm_module(&decoded,&copy)!=NVM_V2_OK && !copy);
    retained[leaf]=TAG_STRING;
    CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);execute(copy);
    nvm_module_free(copy);nvm_v2_module_free(&decoded);nvm_v2_module_free(&wire);free(bytes);
    char error[256];char *native=nvm2c_emit(&c->m,error,sizeof error);
    if(!native)fprintf(stderr,"I failed nested native emission: %s\n",error);
    CHECK(native);char path[1024];snprintf(path,sizeof path,"%s/nested-%u.c",directory,id);
    FILE *file=fopen(path,"w");CHECK(file);CHECK(fputs("#define main fixture_main\n",file)>=0);CHECK(fputs(native,file)>=0);
    CHECK(fputs("\n#undef main\nint main(void) { nmap_collection_budget=1; int result=fixture_main(); nglobal[0]=(nmap_value){0}; nmap_collect(); if(nrec_owned_head || narr_owners || nagg_live_bytes) abort(); return result; }\n",file)>=0);
    CHECK(!fclose(file));free(native);
}
static void refusal_controls(void) {
    for(unsigned variant=0;variant<5;variant++) {
        Input c;fixture(&c,0);size_t row=c.o-36;
        if(variant==0)c.ownership[row]=TAG_INT;
        if(variant==1){size_t at=c.o-40;b32(c.ownership,&at,0);} /* cycle */
        if(variant==2){size_t at=c.o-40;b32(c.ownership,&at,9);} /* invalid referent */
        if(variant==3)c.ownership[row]=TAG_VOID;
        if(variant==4)c.code[1]=TAG_INT; /* known wrong child origin */
        NvmRecordArrayExecutionPlan *plan=(void *)(uintptr_t)1;
        NvmArrayEligibilityResult result=nvm_record_array_admit(&c.m,&plan);
        CHECK(result.status==(variant==0 || variant==4?NVM_ARRAY_UNRESOLVED:NVM_ARRAY_INVALID));
        CHECK(plan==(void *)(uintptr_t)1 && !nvm_verify(&c.m).ok);
        char *text=disasm_module_styled(&c.m,DISASM_STYLE_CANONICAL);CHECK(text);
        AsmResult error;NvmModule *bad=asm_assemble(text,&error);CHECK(!bad);free(text);
        NvmV2Module wire={0};
        CHECK(nvm_v2_from_nvm_module(&c.m,&wire)!=NVM_V2_OK);
        nvm_v2_module_free(&wire);
        fixture(&c,0);admitted(&c);execute(&c.m);
    }
}
static void mixed_leaf_controls(void) {
    for(unsigned conflict=0;conflict<2;conflict++) {
        Input c;fixture(&c,0);CHECK(c.code[c.n-1]==OP_RET);c.n--;
        op(&c,OP_POP);
        if(conflict)arg16(&c,OP_LOAD_LOCAL,0);
        else {
            array(&c,TAG_INT);integer(&c,7);op(&c,OP_ARR_PUSH);
            op(&c,OP_ARR_LITERAL);b8(c.code,&c.n,TAG_ARRAY);b16(c.code,&c.n,1);
        }
        op(&c,OP_AGG_PACK);b8(c.code,&c.n,AGG_RECORD);b32(c.code,&c.n,1);
        b16(c.code,&c.n,0);b16(c.code,&c.n,1);op(&c,OP_POP);integer(&c,0);op(&c,OP_RET);
        c.fn[0].code_length=(uint32_t)c.n;nested_authority(&c);c.ownership[c.o-36]=TAG_STRING;
        size_t extension=c.o-56,types=extension+8,at=types;
        memmove(c.ownership+types+36,c.ownership+types+20,c.o-types-20);c.o+=16;
        b32(c.ownership,&at,4);at=types+20;
        b8(c.ownership,&at,TAG_INT);b8(c.ownership,&at,0);b16(c.ownership,&at,0);b32(c.ownership,&at,NO);
        b8(c.ownership,&at,TAG_ARRAY);b8(c.ownership,&at,0);b16(c.ownership,&at,0);b32(c.ownership,&at,2);
        at=extension+4;b32(c.ownership,&at,64);at=c.o-4;b32(c.ownership,&at,3);
        c.m.ownership_size=(uint32_t)c.o;
        if(conflict) {
            NvmRecordArrayExecutionPlan *plan=(void *)(uintptr_t)1;
            CHECK(nvm_record_array_admit(&c.m,&plan).status==NVM_ARRAY_UNRESOLVED && plan==(void *)(uintptr_t)1);
        } else {admitted(&c);execute(&c.m);}
    }
}
static void bounds_cleanup(void) {
    for(unsigned negative=0;negative<2;negative++) {
        Input c;fixture(&c,0);uint32_t at=0;bool found=false;
        while(at<c.n) {
            DecodedInstruction in;uint32_t width=isa_decode(c.code+at,(uint32_t)c.n-at,&in);CHECK(width);
            if(in.opcode==OP_ARR_SET) {
                CHECK(at>=20 && c.code[at-20]==OP_PUSH_I64);
                memset(c.code+at-19,negative?0xff:0,8);
                if(!negative)c.code[at-19]=2;
                found=true;break;
            }
            at+=width;
        }
        CHECK(found);admitted(&c);
        VmState vm;vm_init(&vm,&c.m);CHECK(vm_execute(&vm)==VM_ERR_OUT_OF_BOUNDS);
        CHECK(!vm.frame_count && !vm.stack_size);vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
        VmRecordArrayPrivate *private=NULL;CHECK(vm_record_array_private_create(&c.m,&private).status==NVM_ARRAY_ELIGIBLE);
        CHECK(vm_record_array_private_run(private)==VM_ERR_OUT_OF_BOUNDS);
        VmRecordArrayPrivateStats stats;CHECK(vm_record_array_private_stats(private,&stats));
        CHECK(!stats.active_frames && !stats.active_stack && !stats.has_result);
        vm_record_array_private_destroy(private);CHECK(!ra_live && !ra_bytes);
#endif
        fixture(&c,0);execute(&c.m);CHECK(!ra_live && !ra_bytes);
    }
}
static void allocation_prefixes(void) {
    Input c;fixture(&c,1);
    CHECK(!ra_live && !ra_bytes);
    NvmRecordArrayExecutionPlan *plan=NULL;ra_calls=0;
    CHECK(nvm_record_array_admit(&c.m,&plan).status==NVM_ARRAY_ELIGIBLE);
    size_t proof_total=ra_calls;nvm_record_array_execution_free(plan);CHECK(!ra_live && !ra_bytes);
    for(int persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<proof_total;fail++) {
        Input before=c;plan=(void *)(uintptr_t)1;ra_calls=0;ra_fail=fail;ra_persistent=persistent;
        NvmArrayEligibilityResult result=nvm_record_array_admit(&c.m,&plan);
        CHECK(ra_calls>fail && result.status==NVM_ARRAY_MEMORY && plan==(void *)(uintptr_t)1);
        CHECK(!memcmp(&before,&c,sizeof c) && !ra_live && !ra_bytes);
        ra_fail=SIZE_MAX;ra_persistent=0;plan=NULL;
        CHECK(nvm_record_array_admit(&c.m,&plan).status==NVM_ARRAY_ELIGIBLE);
        nvm_record_array_execution_free(plan);CHECK(!ra_live && !ra_bytes);
    }
    printf("I checked %zu typed nested proof allocation positions in both failure modes.\n",proof_total);
    VmState vm;vm_init(&vm,&c.m);
    ra_calls=0;CHECK(vm_execute(&vm)==VM_OK);size_t total=ra_calls;
    vm_destroy(&vm);CHECK(total && !ra_live && !ra_bytes);
    for(int persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<total;fail++) {
        ra_fail=SIZE_MAX;ra_persistent=0;vm_init(&vm,&c.m);
        ra_calls=0;ra_fail=fail;ra_persistent=persistent;VmResult result=vm_execute(&vm);
        CHECK(ra_calls>fail);
        bool refused=result==VM_ERR_TYPE_ERROR &&
            !strcmp(vm.error_msg,"I require reference lifetime and ownership instruction verification before execution");
        if(result!=VM_OK && result!=VM_ERR_MEMORY && !refused)
            fprintf(stderr,"I failed nested allocation %zu/%zu mode %d: %u %s\n",fail,total,persistent,result,vm.error_msg);
        CHECK(result==VM_OK || result==VM_ERR_MEMORY || refused);
        CHECK(vm.frame_count==0);if(result!=VM_OK)CHECK(vm.stack_size==0);
        ra_fail=SIZE_MAX;ra_persistent=0;vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
        vm_init(&vm,&c.m);CHECK(vm_execute(&vm)==VM_OK);vm_destroy(&vm);CHECK(!ra_live && !ra_bytes);
    }
    printf("I checked %zu nested allocation positions in transient and persistent modes with independent recovery.\n",total);
}
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
static void maximum_private_release(void) {
    Input c;fixture(&c,0);uint8_t *layouts=calloc(1,4+256*20);CHECK(layouts);
    size_t n=0;b32(layouts,&n,256);
    for(unsigned i=0;i<256;i++) {
        b8(layouts,&n,NVM_V2_LAYOUT_STRUCT);b8(layouts,&n,0);b16(layouts,&n,1);b32(layouts,&n,NO);
        b8(layouts,&n,i?TAG_STRUCT:TAG_ARRAY);b8(layouts,&n,0);b16(layouts,&n,0);
        b32(layouts,&n,i?i-1:NO);b32(layouts,&n,NO);
    }
    c.m.layout_data=layouts;c.m.layout_size=(uint32_t)n;c.m.struct_count=256;
    c.o=0;b32(c.ownership,&c.o,3);b32(c.ownership,&c.o,256);
    for(unsigned i=0;i<256;i++)b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);
    b32(c.ownership,&c.o,1);b16(c.ownership,&c.o,4);b16(c.ownership,&c.o,0);desc(&c,TAG_INT);
    for(unsigned i=0;i<4;i++)desc(&c,TAG_VOID);
    b32(c.ownership,&c.o,4);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b16(c.ownership,&c.o,2);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,36);b32(c.ownership,&c.o,2);
    b8(c.ownership,&c.o,TAG_ARRAY);b8(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b8(c.ownership,&c.o,TAG_STRING);b8(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,NO);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);
    c.m.ownership_size=(uint32_t)c.o;
    VmRecordArrayPrivate *p=NULL;CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);free(layouts);
    CHECK(vm_record_array_private_run(p)==VM_OK && p->layouts[255].rank==256);
    VmHeap *heap=&p->vm.heap;uint64_t objects=heap->stats.num_objects,bytes=heap->stats.allocated-heap->stats.freed;
    VmString *text=vm_string_new(heap,"deep dynamic string",19);CHECK(text);
    VmArray *inner=vm_array_new(heap,TAG_STRING,0),*outer=vm_array_new(heap,TAG_ARRAY,0);CHECK(inner&&outer);
    CHECK(vm_array_push(heap,inner,val_string(text)));vm_release(heap,val_string(text));
    CHECK(vm_array_push(heap,outer,val_array(inner)));vm_release(heap,val_array(inner));
    NanoValue child=val_array(outer);
    for(unsigned i=0;i<256;i++) {
        CHECK(vm_ra_field_accepts(p,i,0,child));VmStruct *owner=vm_struct_new(heap,i,1);CHECK(owner);
        owner->fields[0]=child;child=val_struct(owner);
    }
    size_t calls=ra_calls;ra_fail=calls;ra_persistent=1;vm_release(heap,child);
    CHECK(ra_calls==calls && heap->stats.num_objects==objects && heap->stats.allocated-heap->stats.freed==bytes);
    ra_fail=SIZE_MAX;ra_persistent=0;vm_record_array_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void copied_string_tables(void) {
    VmRecordArrayPrivate *instances[2]={0};
    for(unsigned which=0;which<2;which++) {
        Input *source=malloc(sizeof *source);CHECK(source);fixture(source,0);
        char text[4]={(char)('a'+which),0,'z',0};
        source->strings[1]=text;source->lengths[1]=3;
        CHECK(vm_record_array_private_create(&source->m,&instances[which]).status==NVM_ARRAY_ELIGIBLE);
        memset(text,'x',sizeof text);memset(source,0,sizeof *source);free(source);
    }
    for(unsigned round=0;round<2;round++)for(unsigned which=0;which<2;which++) {
        CHECK(vm_record_array_private_run(instances[which])==VM_OK);
        uint32_t path[]={0,0,0};char actual[6],expected[6]={(char)('a'+which),0,'z',(char)('a'+which),0,'z'};
        CHECK(vm_record_array_private_string(instances[which],VM_RA_GLOBAL,0,path,3,0,6,actual));
        CHECK(!memcmp(actual,expected,6));
    }
    for(unsigned which=0;which<2;which++)vm_record_array_private_destroy(instances[which]);
    CHECK(!ra_live && !ra_bytes);
}
static void private_prefixes(void) {
    Input c;fixture(&c,1);VmRecordArrayPrivate *p=NULL;
    CHECK(!ra_live && !ra_bytes);ra_calls=0;
    CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
    size_t preparation=ra_calls;CHECK(vm_record_array_private_run(p)==VM_OK);
    ra_calls=0;CHECK(vm_record_array_private_run(p)==VM_OK);size_t execution=ra_calls;
    vm_record_array_private_destroy(p);CHECK(!ra_live && !ra_bytes && preparation && execution);
    for(unsigned phase=0;phase<2;phase++)for(int persistent=0;persistent<2;persistent++)
    for(size_t fail=0;fail<(phase?execution:preparation);fail++) {
        ra_fail=SIZE_MAX;ra_persistent=0;p=NULL;
        if(phase) {
            CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
            CHECK(vm_record_array_private_run(p)==VM_OK);
        }
        ra_calls=0;ra_fail=fail;ra_persistent=persistent;
        if(phase) {
            CHECK(vm_record_array_private_run(p)==VM_ERR_MEMORY && ra_calls>fail);
            VmRecordArrayPrivateStats stats;VmRecordArrayObservation prior;
            CHECK(vm_record_array_private_stats(p,&stats) && !stats.active_frames && !stats.active_stack);
            CHECK(vm_record_array_private_observe(p,VM_RA_RESULT,0,NULL,0,&prior) && prior.tag==TAG_INT && prior.scalar_bits==0);
            vm_record_array_private_destroy(p);
        } else {
            p=(void *)(uintptr_t)1;
            CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_MEMORY && p==(void *)(uintptr_t)1 && ra_calls>fail);
        }
        CHECK(!ra_live && !ra_bytes);ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
        CHECK(vm_record_array_private_run(p)==VM_OK);vm_record_array_private_destroy(p);
        CHECK(!ra_live && !ra_bytes);
    }
    printf("I checked %zu private preparation and %zu execution failure positions in both modes.\n",preparation,execution);
}
#endif
#ifdef RA_NESTED_WHITEBOX
static void interned_result_headroom(bool private_mode) {
    Input c;init(&c,TAG_STRING,false);c.strings[2]="leafleaf";c.lengths[2]=8;
    array(&c,TAG_STRING);op(&c,OP_ARR_LITERAL);b8(c.code,&c.n,TAG_ARRAY);b16(c.code,&c.n,1);
    pack_nested(&c);arg32(&c,OP_STORE_GLOBAL,0);
    string_value(&c);op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,1);op(&c,OP_ASSERT);op(&c,OP_POP);
    integer(&c,0);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;
    nested_authority(&c);c.ownership[c.o-36]=TAG_STRING;admitted(&c);
    VmState local,*vm=&local;
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    VmRecordArrayPrivate *p=NULL;
    if(private_mode) {
        CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
        p->busy=true;vm=&p->vm;
    } else
#else
    (void)private_mode;
#endif
    vm_init(vm,&c.m);
    uint32_t active=0;VmOwnedInvocationProof proof={0};
    proof.module=vm->module;proof.ordinary_arrays=true;proof.ordinary_active_root=&active;
    VmCallFrame *frame=&vm->frames[0];memset(frame,0,sizeof *frame);
    frame->module=vm->module;frame->local_count=c.fn[0].local_count;frame->owned_callable=val_void();
    for(uint16_t i=0;i<frame->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;vm->current_fn=0;vm->ip=0;
    VmString *pooled=vm->module_constants.strings[2],*operand=vm->module_constants.strings[1];
    uint32_t refs=pooled->header.ref_count,operand_refs=operand->header.ref_count;
    pooled->header.ref_count=UINT32_MAX;
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    VmTrap trap=vm_core_execute_scoped(vm,p?NULL:&proof,NULL,p);
#else
    VmTrap trap=vm_core_execute_scoped(vm,&proof,NULL,NULL);
#endif
    bool refused=trap.type==TRAP_ERROR && trap.data.error.code==VM_ERR_MEMORY;
    uint32_t observed=pooled->header.ref_count;
    CHECK(operand->header.ref_count==operand_refs);
    pooled->header.ref_count=refs+(refused?0:1);
    if(trap.type==TRAP_ASSERT)vm_release(&vm->heap,trap.data.assert_check.condition);
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    if(p) {vm_ra_unwind(p);p->busy=false;CHECK(vm_record_array_private_run(p)==VM_OK);vm_record_array_private_destroy(p);} else
#endif
    vm_destroy(vm);
    CHECK(!ra_live&&!ra_bytes);
    if(!refused)fprintf(stderr,"I observed public concat pooled references wrap to %u instead of refusing.\n",observed);
    CHECK(refused && observed==UINT32_MAX);
    vm_init(&local,&c.m);CHECK(vm_execute(&local)==VM_OK);vm_destroy(&local);CHECK(!ra_live&&!ra_bytes);
}
static void slice_headroom(bool private_mode,unsigned mode) {
    Input c;fixture(&c,mode);VmState local,*vm=&local;
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    VmRecordArrayPrivate *p=NULL;
    if(private_mode) {
        CHECK(vm_record_array_private_create(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
        CHECK(vm_record_array_private_run(p)==VM_OK);p->busy=true;vm=&p->vm;
    } else
#else
    (void)private_mode;
#endif
    vm_init(vm,&c.m);
    uint32_t active=0;VmOwnedInvocationProof proof={0};
    proof.module=vm->module;proof.ordinary_arrays=true;proof.ordinary_active_root=&active;
    VmCallFrame *frame=&vm->frames[0];memset(frame,0,sizeof *frame);
    frame->module=vm->module;frame->local_count=c.fn[0].local_count;frame->owned_callable=val_void();
    for(uint16_t i=0;i<frame->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;vm->current_fn=0;vm->ip=0;
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
#define STRING_STEP() vm_core_execute_scoped(vm,p?NULL:&proof,NULL,p)
#else
#define STRING_STEP() vm_core_execute_scoped(vm,&proof,NULL,NULL)
#endif
    VmTrap trap=STRING_STEP();CHECK(trap.type==TRAP_ASSERT);
    vm_release(&vm->heap,trap.data.assert_check.condition);
    NanoValue outer=stack_peek(vm,2);CHECK(outer.tag==TAG_ARRAY && outer.as.array->length==2);
    NanoValue child=outer.as.array->elements[0];
    CHECK(child.as.obj==outer.as.array->elements[1].as.obj);
    CHECK(child.tag==(mode==7?TAG_STRING:TAG_ARRAY));
    VmHeapHeader *header=(VmHeapHeader *)child.as.obj;
    uint32_t refs=header->ref_count,ip=vm->ip,stack=vm->stack_size;
    header->ref_count=UINT32_MAX-1;
    trap=STRING_STEP();CHECK(trap.type==TRAP_ERROR && trap.data.error.code==VM_ERR_MEMORY);
    CHECK(vm->stack_size==stack && header->ref_count==UINT32_MAX-1 && outer.as.array->length==2);
    header->ref_count=refs;vm->ip=ip;
    do {
        trap=STRING_STEP();
        if(trap.type==TRAP_ASSERT) {CHECK(val_truthy(trap.data.assert_check.condition));vm_release(&vm->heap,trap.data.assert_check.condition);}
    } while(trap.type==TRAP_ASSERT);
    CHECK(trap.type==TRAP_NONE && !vm->frame_count && vm->stack_size==1 && vm->stack[0].as.i64==0);
#undef STRING_STEP
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    if(p) {vm_ra_unwind(p);p->busy=false;CHECK(vm_record_array_private_run(p)==VM_OK);vm_record_array_private_destroy(p);} else
#endif
    vm_destroy(vm);
    CHECK(!ra_live && !ra_bytes);execute(&c.m);CHECK(!ra_live && !ra_bytes);
}
#endif
int main(int argc,char **argv) {
#ifdef RA_NESTED_WHITEBOX
    interned_result_headroom(false);
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    interned_result_headroom(true);
#endif
    slice_headroom(false,1);slice_headroom(false,7);
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    slice_headroom(true,1);slice_headroom(true,7);
#endif
#endif
    if(argc==2 && !strcmp(argv[1],"allocations")){
        allocation_prefixes();
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
        private_prefixes();
#endif
        return 0;
    }
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    copied_string_tables();maximum_private_release();
#endif
    refusal_controls();mixed_leaf_controls();bounds_cleanup();
    CHECK(argc==2);for(unsigned mode=0;mode<8;mode++) {
        Input c;fixture(&c,mode);
        if(mode==3) {
            NvmRecordArrayExecutionPlan *plan=(void *)(uintptr_t)1;
            CHECK(nvm_record_array_admit(&c.m,&plan).status==NVM_ARRAY_UNRESOLVED && plan==(void *)(uintptr_t)1);
            CHECK(!nvm_verify(&c.m).ok);
        } else public_routes(&c,argv[1],mode);
    }
    printf("PASS %u nested record-array public checks\n",checks);return 0;
}
