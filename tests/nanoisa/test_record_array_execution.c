/* I inspect copied preparation facts. I do not execute an admitted graph. */
#include "../../src/nanoisa/managed_record_array_execution.h"
#define main record_array_origin_controls
#include "test_record_array_origins.c"
#undef main

static NvmRecordArrayExecutionPlan *prepare(Input *c,NvmArrayEligibilityStatus status) {
    Input before=*c;
    NvmRecordArrayExecutionPlan *p=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult r=nvm_prepare_record_array_execution(&c->m,&p);
    CHECK(!memcmp(c,&before,sizeof before));
    if(r.status!=status)fprintf(stderr,"I expected plan status %u, got %u: %s\n",status,r.status,r.message);
    CHECK(r.status==status);
    CHECK(status==NVM_ARRAY_ELIGIBLE?p!=(void *)(uintptr_t)1:p==(void *)(uintptr_t)1);
    return p;
}
static void plan_positive(Input *c) {
    NvmRecordArrayExecutionPlan *p=prepare(c,NVM_ARRAY_ELIGIBLE);
    nvm_record_array_execution_free(p);
}
static void snapshot_controls(void) {
    for(uint8_t tag=TAG_INT;tag<=TAG_STRING;tag++) {
        Input *c=malloc(sizeof *c);CHECK(c);basic(c,tag,true);
        char leaf[4]={'a',0,'z',0};c->strings[1]=leaf;c->lengths[1]=3;
        NvmMetadataEntry metadata={.key_idx=0,.value_idx=1};
        NvmDebugEntry debug={.bytecode_offset=0,.source_line=7,.source_col=11};
        c->m.metadata=&metadata;c->m.metadata_count=c->m.metadata_capacity=1;
        c->m.debug_entries=&debug;c->m.debug_count=c->m.debug_capacity=1;
        c->m.section_count=c->m.header.section_count=1;
        c->m.sections[0]=(NvmSectionEntry){.type=NVM_SECTION_CODE,.offset=64,.size=(uint32_t)c->n};
        NvmHeader header=c->m.header;
        uint8_t code[32768],layouts[2048],ownership[4096];
        memcpy(code,c->code,c->n);memcpy(layouts,c->layouts,c->l);memcpy(ownership,c->ownership,c->o);
        uint32_t sizes[]={(uint32_t)c->n,(uint32_t)c->l,(uint32_t)c->o,3};
        NvmRecordArrayExecutionPlan *p=prepare(c,NVM_ARRAY_ELIGIBLE);
        NvmRecordArrayExecutionPlan *independent=prepare(c,NVM_ARRAY_ELIGIBLE);CHECK(p!=independent);
        memset(c,0,sizeof *c);free(c);memset(leaf,0xcc,sizeof leaf);
        memset(&metadata,0xcc,sizeof metadata);memset(&debug,0xcc,sizeof debug);
        nvm_record_array_execution_free(independent);
        NvmRecordArrayExecutionCounts n;CHECK(nvm_record_array_execution_counts(p,&n));
        CHECK(n.functions==1&&n.entry==0&&n.initializer==NO&&n.frame_limit==1024&&n.records==2);
        CHECK(n.strings==4&&n.sections==1&&n.metadata==1&&n.debug==1&&n.globals==0);
        CHECK(n.peak_bytes_reserved<=NVM_RECORD_ARRAY_EXECUTION_BYTES&&n.work_reserved<=NVM_RECORD_ARRAY_EXECUTION_STEPS);
        NvmHeader h;CHECK(nvm_record_array_execution_header(p,&h));
        CHECK(!memcmp(h.magic,header.magic,4)&&h.flags==header.flags&&h.format_version==header.format_version);
        NvmSectionEntry section;CHECK(nvm_record_array_execution_section(p,0,&section)&&section.type==NVM_SECTION_CODE&&section.offset==64&&section.size==sizes[0]);
        NvmMetadataEntry m;CHECK(nvm_record_array_execution_metadata(p,0,&m)&&m.key_idx==0&&m.value_idx==1);
        NvmDebugEntry d;CHECK(nvm_record_array_execution_debug(p,0,&d)&&d.bytecode_offset==0&&d.source_line==7&&d.source_col==11);
        const uint8_t expected_string[]={ 'a',0,'z' };
        const void *expected[]={code,layouts,ownership,expected_string};
        for(unsigned k=0;k<4;k++) {
            uint32_t size=NO;unsigned index=k==3?1:0;uint8_t copy[32768];
            CHECK(nvm_record_array_execution_size(p,(NvmRecordArraySnapshotKind)k,index,&size)&&size==sizes[k]);
            CHECK(nvm_record_array_execution_bytes(p,(NvmRecordArraySnapshotKind)k,index,0,size,copy));
            CHECK(!memcmp(copy,expected[k],size));
            memset(copy,0xa5,sizeof copy);
            CHECK(!nvm_record_array_execution_bytes(p,(NvmRecordArraySnapshotKind)k,index,size,1,copy)&&copy[0]==0xa5);
            CHECK(!nvm_record_array_execution_bytes(p,(NvmRecordArraySnapshotKind)k,index,UINT32_MAX,2,copy)&&copy[0]==0xa5);
            CHECK(nvm_record_array_execution_bytes(p,(NvmRecordArraySnapshotKind)k,index,size,0,copy)&&copy[0]==0xa5);
        }
        NvmRecordArrayExecutionFunction f;CHECK(nvm_record_array_execution_function(p,0,&f));
        CHECK(f.signature.result_tag==TAG_INT&&f.signature.local_count==3&&f.instruction_start==0&&f.instruction_count==7&&f.maximum_stack==2&&!f.parameter_tags_present);
        CHECK(n.instructions==7);
        NvmRecordArrayExecutionInstruction row;
        CHECK(nvm_record_array_execution_instruction(p,0,&row)&&row.opcode==OP_ARR_NEW&&row.operand_bits[0]==tag&&row.recipe==NVM_RA_ROOT_CONSTRUCT);
        CHECK((row.obligations&(NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS))==(NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS));
        CHECK(nvm_record_array_execution_instruction(p,2,&row)&&row.opcode==OP_ARR_PUSH&&row.pops==2&&row.pushes==1&&row.recipe==NVM_RA_ROOT_ARRAY_PUSH);
        CHECK(nvm_record_array_execution_instruction(p,3,&row)&&row.opcode==OP_STRUCT_LITERAL&&row.operand_bits[0]==0&&row.operand_bits[1]==1&&row.pops==1&&row.pushes==1);
        CHECK(nvm_record_array_execution_instruction(p,6,&row)&&row.opcode==OP_RET&&row.pops==1&&!row.pushes&&!row.successor_count);
        NvmRecordArrayExecutionDescriptor descriptor;
        for(unsigned i=0;i<2;i++)CHECK(nvm_record_array_execution_descriptor(p,i,&descriptor)&&descriptor.ordinal==i&&descriptor.layout==i+1&&descriptor.fields==1);
        NvmRecordArrayOriginCounts origins;CHECK(nvm_record_array_execution_origin_counts(p,&origins)&&origins.origins==2&&origins.fields==1);
        NvmRecordHeapOrigin origin;CHECK(nvm_record_array_execution_origin(p,0,&origin)&&origin.declared_tag==tag);
        CHECK(nvm_record_array_execution_origin(p,1,&origin)&&origin.record_ordinal==0&&origin.layout_index==1);
        NvmRecordValueOrigins value;CHECK(nvm_record_array_execution_field_value(p,0,&value)&&value.origins==1&&value.tags==MASK(TAG_ARRAY));
        uint16_t mask;CHECK(nvm_record_array_execution_required_elements(p,0,&mask)&&mask==MASK(tag));
        NvmDeclarationCounts dc;CHECK(nvm_record_array_execution_declaration_counts(p,&dc)&&dc.layouts==3&&dc.bindings==2);
        NvmDeclarationLayout dl;CHECK(nvm_record_array_execution_layout(p,1,&dl)&&dl.kind==NVM_V2_LAYOUT_STRUCT);
        NvmV2LayoutField field;CHECK(nvm_record_array_execution_field(p,1,0,&field)&&field.type_tag==TAG_ARRAY);
        NvmOrdinaryArrayType type;CHECK(nvm_record_array_execution_type(p,0,&type)&&type.tag==tag);
        NvmOrdinaryArrayBinding binding;CHECK(nvm_record_array_execution_binding(p,1,&binding)&&binding.layout==2&&binding.element_type==0);
        NvmUnionVariantFact variant;CHECK(nvm_record_array_execution_variant(p,0,0,&variant)&&variant.field_count==1);
        UNCHANGED(NvmRecordArrayExecutionCounts,nvm_record_array_execution_counts(NULL,&out));
        UNCHANGED(NvmRecordArrayExecutionFunction,nvm_record_array_execution_function(p,1,&out));
        UNCHANGED(NvmRecordArrayExecutionInstruction,nvm_record_array_execution_instruction(p,n.instructions,&out));
        UNCHANGED(uint8_t,nvm_record_array_execution_parameter(p,0,0,&out));
        UNCHANGED(uint32_t,nvm_record_array_execution_size(p,NVM_RA_SNAPSHOT_CODE,1,&out));
        UNCHANGED(uint32_t,nvm_record_array_execution_size(p,NVM_RA_SNAPSHOT_STRING,4,&out));
        UNCHANGED(uint32_t,nvm_record_array_execution_size(p,(NvmRecordArraySnapshotKind)99,0,&out));
        UNCHANGED(uint8_t,nvm_record_array_execution_bytes(p,NVM_RA_SNAPSHOT_CODE,1,0,1,&out));
        UNCHANGED(uint8_t,nvm_record_array_execution_bytes(p,NVM_RA_SNAPSHOT_STRING,4,0,1,&out));
        UNCHANGED(uint8_t,nvm_record_array_execution_bytes(p,(NvmRecordArraySnapshotKind)99,0,0,1,&out));
        UNCHANGED(NvmHeader,nvm_record_array_execution_header(NULL,&out));
        UNCHANGED(NvmSectionEntry,nvm_record_array_execution_section(p,1,&out));
        UNCHANGED(NvmMetadataEntry,nvm_record_array_execution_metadata(p,1,&out));
        UNCHANGED(NvmDebugEntry,nvm_record_array_execution_debug(p,1,&out));
        UNCHANGED(NvmRecordArrayExecutionDescriptor,nvm_record_array_execution_descriptor(p,2,&out));
        UNCHANGED(NvmRecordArrayOriginCounts,nvm_record_array_execution_origin_counts(NULL,&out));
        UNCHANGED(NvmRecordHeapOrigin,nvm_record_array_execution_origin(p,origins.origins,&out));
        UNCHANGED(NvmRecordValueOrigins,nvm_record_array_execution_field_value(p,origins.fields,&out));
        UNCHANGED(uint16_t,nvm_record_array_execution_required_elements(p,origins.origins,&out));
        UNCHANGED(NvmDeclarationCounts,nvm_record_array_execution_declaration_counts(NULL,&out));
        UNCHANGED(NvmDeclarationLayout,nvm_record_array_execution_layout(p,3,&out));
        UNCHANGED(NvmV2LayoutField,nvm_record_array_execution_field(p,1,1,&out));
        UNCHANGED(NvmOrdinaryArrayType,nvm_record_array_execution_type(p,1,&out));
        UNCHANGED(NvmOrdinaryArrayBinding,nvm_record_array_execution_binding(p,2,&out));
        UNCHANGED(NvmUnionVariantFact,nvm_record_array_execution_variant(p,0,1,&out));
        CHECK(!nvm_record_array_execution_counts(p,NULL)&&!nvm_record_array_execution_function(p,0,NULL));
        CHECK(!nvm_record_array_execution_instruction(p,0,NULL)&&!nvm_record_array_execution_parameter(p,0,0,NULL));
        CHECK(!nvm_record_array_execution_bytes(p,NVM_RA_SNAPSHOT_CODE,0,0,0,NULL)&&!nvm_record_array_execution_size(p,NVM_RA_SNAPSHOT_CODE,0,NULL));
        CHECK(!nvm_record_array_execution_descriptor(p,0,NULL)&&!nvm_record_array_execution_header(p,NULL));
        CHECK(!nvm_record_array_execution_section(p,0,NULL)&&!nvm_record_array_execution_metadata(p,0,NULL)&&!nvm_record_array_execution_debug(p,0,NULL));
        CHECK(!nvm_record_array_execution_origin_counts(p,NULL)&&!nvm_record_array_execution_origin(p,0,NULL));
        CHECK(!nvm_record_array_execution_field_value(p,0,NULL)&&!nvm_record_array_execution_required_elements(p,0,NULL));
        CHECK(!nvm_record_array_execution_declaration_counts(p,NULL)&&!nvm_record_array_execution_layout(p,0,NULL));
        CHECK(!nvm_record_array_execution_field(p,1,0,NULL)&&!nvm_record_array_execution_type(p,0,NULL));
        CHECK(!nvm_record_array_execution_binding(p,0,NULL)&&!nvm_record_array_execution_variant(p,0,0,NULL));
        nvm_record_array_execution_free(p);
    }
}
static void initializer_input(Input *c) {
    init(c,TAG_INT,true);arg32(c,OP_LOAD_GLOBAL,0);op(c,OP_POP);finish(c);
    c->m.function_count=2;c->fn[1].name_idx=3;c->fn[1].code_offset=(uint32_t)c->n;
    c->fn[1].result_tag=TAG_VOID;c->fn[1].result_count=0;
    scalar(c,TAG_INT);arg32(c,OP_STORE_GLOBAL,0);op(c,OP_RET);
    c->fn[1].code_length=(uint32_t)c->n-c->fn[1].code_offset;authority(c);
}
static void initializer_and_call_controls(void) {
    Input c;initializer_input(&c);positive(&c);
    NvmRecordArrayExecutionPlan *p=prepare(&c,NVM_ARRAY_ELIGIBLE);
    NvmRecordArrayExecutionCounts n;CHECK(nvm_record_array_execution_counts(p,&n)&&n.initializer==1&&n.entry==0&&n.globals==1);
    NvmRecordArrayExecutionFunction f;CHECK(nvm_record_array_execution_function(p,1,&f)&&f.signature.result_count==0&&f.instruction_count==3);
    NvmRecordArrayExecutionInstruction row;CHECK(nvm_record_array_execution_instruction(p,f.instruction_start+2,&row)&&row.opcode==OP_RET&&!row.pops&&!row.pushes);
    nvm_record_array_execution_free(p);
    char counted_init[]="__init__\0tail";c.strings[3]=counted_init;c.lengths[3]=sizeof counted_init-1;
    /* The old C-string selector still sees an initializer; only this new plan
     * refuses the ambiguous counted identity before calling that analysis. */
    positive(&c);prepare(&c,NVM_ARRAY_UNRESOLVED);positive(&c);
    basic(&c,TAG_INT,false);char counted_main[]="main\0suffix";c.strings[0]=counted_main;c.lengths[0]=sizeof counted_main-1;
    prepare(&c,NVM_ARRAY_UNRESOLVED);
    init(&c,TAG_INT,true);array(&c,TAG_INT);arg32(&c,OP_CALL,1);record(&c,0);op(&c,OP_POP);finish(&c);
    c.m.function_count=2;c.fn[1].name_idx=2;c.fn[1].code_offset=(uint32_t)c.n;
    c.fn[1].local_count=c.fn[1].arity=1;c.fn[1].result_count=1;c.fn[1].result_tag=TAG_ARRAY;
    c.params[0]=TAG_ARRAY;c.param_rows[1]=c.params;c.m.function_param_types=c.param_rows;
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;authority(&c);
    p=prepare(&c,NVM_ARRAY_ELIGIBLE);c.params[0]=TAG_FLOAT;
    uint8_t tag=0;CHECK(nvm_record_array_execution_parameter(p,1,0,&tag)&&tag==TAG_ARRAY);
    CHECK(nvm_record_array_execution_function(p,1,&f)&&f.parameter_tags_present&&f.signature.arity==1);
    CHECK(nvm_record_array_execution_instruction(p,1,&row)&&row.opcode==OP_CALL&&row.callee==1&&row.pops==1&&row.pushes==1&&row.successor_count==1&&row.successors[0]==row.next_pc);
    nvm_record_array_execution_free(p);
    /* I distinguish a present empty row from an absent parameter table. */
    basic(&c,TAG_INT,false);c.param_rows[0]=c.params;c.m.function_param_types=c.param_rows;
    p=prepare(&c,NVM_ARRAY_ELIGIBLE);CHECK(nvm_record_array_execution_function(p,0,&f)&&f.parameter_tags_present);
    UNCHANGED(uint8_t,nvm_record_array_execution_parameter(p,0,0,&out));nvm_record_array_execution_free(p);
}
static void all_copy_domains(Input *c) {
    init(c,TAG_STRING,true);arg32(c,OP_LOAD_GLOBAL,0);op(c,OP_POP);
    array(c,TAG_STRING);scalar(c,TAG_STRING);op(c,OP_ARR_PUSH);
    arg32(c,OP_CALL,1);record(c,0);op(c,OP_POP);finish(c);
    c->m.function_count=3;c->fn[1].name_idx=2;c->fn[1].code_offset=(uint32_t)c->n;
    c->fn[1].local_count=c->fn[1].arity=1;c->fn[1].result_count=1;c->fn[1].result_tag=TAG_ARRAY;
    c->params[0]=TAG_ARRAY;c->param_rows[0]=c->params;c->param_rows[1]=c->params;c->m.function_param_types=c->param_rows;
    arg16(c,OP_LOAD_LOCAL,0);op(c,OP_RET);c->fn[1].code_length=(uint32_t)c->n-c->fn[1].code_offset;
    c->fn[2].name_idx=3;c->fn[2].code_offset=(uint32_t)c->n;c->fn[2].result_tag=TAG_VOID;
    scalar(c,TAG_INT);arg32(c,OP_STORE_GLOBAL,0);op(c,OP_RET);
    c->fn[2].code_length=(uint32_t)c->n-c->fn[2].code_offset;authority(c);
}
static void function_boundary(void) {
    for(unsigned count=256;count<=257;count++) {
        Input c;init(&c,TAG_INT,false);c.fn[0].local_count=0;finish(&c);
        uint8_t tail[256];CHECK(c.o>=28&&c.o-28<=sizeof tail);
        size_t tail_size=c.o-28;memcpy(tail,c.ownership+28,tail_size);
        NvmFunctionEntry *functions=calloc(count,sizeof *functions);CHECK(functions);
        functions[0]=c.fn[0];
        for(unsigned i=1;i<count;i++) {
            functions[i]=functions[0];functions[i].code_offset=(uint32_t)c.n;
            scalar(&c,TAG_INT);op(&c,OP_RET);
        }
        c.o=12;b32(c.ownership,&c.o,count);
        for(unsigned i=0;i<count;i++){b16(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);desc(&c,TAG_INT);}
        CHECK(c.o+tail_size<=sizeof c.ownership);memcpy(c.ownership+c.o,tail,tail_size);c.o+=tail_size;
        c.m.ownership_size=(uint32_t)c.o;c.m.code_size=(uint32_t)c.n;c.m.function_count=count;c.m.functions=functions;
        NvmRecordArrayExecutionPlan *plan=prepare(&c,count==256?NVM_ARRAY_ELIGIBLE:NVM_ARRAY_LIMIT);
        if(count==256) {
            NvmRecordArrayExecutionCounts n;CHECK(nvm_record_array_execution_counts(plan,&n)&&n.functions==256&&n.frame_limit==1024);
            NvmRecordArrayExecutionFunction f;CHECK(nvm_record_array_execution_function(plan,255,&f)&&f.signature.code_offset==2550&&f.instruction_count==2);
            nvm_record_array_execution_free(plan);
        }
        free(functions);
    }
}
static void counted_string_boundary(void) {
    Input c;basic(&c,TAG_INT,false);
    char **strings=calloc(NVM_MAX_STRINGS,sizeof *strings);
    uint32_t *lengths=calloc(NVM_MAX_STRINGS,sizeof *lengths);CHECK(strings&&lengths);
    for(unsigned i=0;i<NVM_MAX_STRINGS;i++){strings[i]=i<4?c.strings[i]:"";lengths[i]=i<4?c.lengths[i]:0;}
    c.m.strings=strings;c.m.string_lengths=lengths;c.m.string_count=NVM_MAX_STRINGS;
    NvmRecordArrayExecutionPlan *p=prepare(&c,NVM_ARRAY_ELIGIBLE);
    free(strings);free(lengths);NvmRecordArrayExecutionCounts n;
    CHECK(nvm_record_array_execution_counts(p,&n)&&n.strings==NVM_MAX_STRINGS);
    uint32_t length=NO;CHECK(nvm_record_array_execution_size(p,NVM_RA_SNAPSHOT_STRING,NVM_MAX_STRINGS-1,&length)&&length==0);
    nvm_record_array_execution_free(p);
}
static void control_edges(void) {
    Input c;init(&c,TAG_INT,false);
    scalar(&c,TAG_BOOL);size_t branch=c.n;arg32(&c,OP_JMP_FALSE,0);
    array(&c,TAG_INT);size_t jump=c.n;arg32(&c,OP_JMP,0);
    size_t other=c.n;array(&c,TAG_INT);size_t join=c.n;record(&c,0);op(&c,OP_POP);finish(&c);
    size_t at=branch+1;b32(c.code,&at,(uint32_t)(other-branch));at=jump+1;b32(c.code,&at,(uint32_t)(join-jump));
    NvmRecordArrayExecutionPlan *p=prepare(&c,NVM_ARRAY_ELIGIBLE);NvmRecordArrayExecutionInstruction row;
    CHECK(nvm_record_array_execution_instruction(p,1,&row)&&row.pc==branch&&row.successor_count==2&&row.successors[0]==other&&row.successors[1]==branch+5);
    CHECK(nvm_record_array_execution_instruction(p,3,&row)&&row.pc==jump&&row.successor_count==1&&row.successors[0]==join);
    NvmRecordValueOrigins value;CHECK(nvm_record_array_execution_field_value(p,0,&value)&&value.origins==3);nvm_record_array_execution_free(p);
    /* A real decoded negative backedge reaches the same state; no execution. */
    init(&c,TAG_INT,false);scalar(&c,TAG_BOOL);arg32(&c,OP_JMP_TRUE,(uint32_t)-2);finish(&c);
    p=prepare(&c,NVM_ARRAY_ELIGIBLE);CHECK(nvm_record_array_execution_instruction(p,1,&row)&&row.successors[0]==0&&row.successors[1]==7);nvm_record_array_execution_free(p);
    /* Exact scalar bits, independent of host floating-point conversions. */
    init(&c,TAG_INT,false);op(&c,OP_PUSH_F64);const uint64_t bits=UINT64_C(0x8000000000000000);
    for(unsigned i=0;i<8;i++)b8(c.code,&c.n,(uint8_t)(bits>>(8*i)));
    op(&c,OP_POP);finish(&c);p=prepare(&c,NVM_ARRAY_ELIGIBLE);
    CHECK(nvm_record_array_execution_instruction(p,0,&row)&&row.operand_bits[0]==bits&&row.next_pc==9);nvm_record_array_execution_free(p);
}
static void plan_refusals(void) {
    Input c;NvmRecordArrayExecutionPlan *p=(void *)(uintptr_t)1;
    CHECK(nvm_prepare_record_array_execution(NULL,&p).status==NVM_ARRAY_INVALID&&p==(void *)(uintptr_t)1);
    basic(&c,TAG_INT,false);CHECK(nvm_prepare_record_array_execution(&c.m,NULL).status==NVM_ARRAY_INVALID);nvm_record_array_execution_free(NULL);
    for(unsigned kind=0;kind<9;kind++) {
        basic(&c,TAG_INT,false);
        switch(kind) {
        case 0:c.m.service_data=(void *)(uintptr_t)1;break;
        case 1:c.m.passive_data=(void *)(uintptr_t)1;break;
        case 2:c.m.import_count=1;break;
        case 3:c.m.module_ref_count=1;break;
        case 4:c.m.callback_contract_count=1;break;
        case 5:c.m.call_descriptors=(void *)(uintptr_t)1;break;
        case 6:c.m.call_descriptor_count=1;break;
        case 7:c.m.service_size=1;break;
        default:c.m.passive_size=1;break;
        }
        prepare(&c,NVM_ARRAY_UNRESOLVED);
    }
    basic(&c,TAG_INT,false);c.m.function_count=257;prepare(&c,NVM_ARRAY_LIMIT);
    basic(&c,TAG_INT,false);c.m.string_count=NVM_MAX_STRINGS+1;prepare(&c,NVM_ARRAY_LIMIT);
    basic(&c,TAG_INT,false);c.m.code_size=16u*1024u*1024u+1;prepare(&c,NVM_ARRAY_LIMIT);
    basic(&c,TAG_INT,false);c.m.function_count=0;prepare(&c,NVM_ARRAY_INVALID);
    basic(&c,TAG_INT,false);c.fn[0].name_idx=4;prepare(&c,NVM_ARRAY_INVALID);
    basic(&c,TAG_INT,false);c.m.layout_size--;prepare(&c,NVM_ARRAY_INVALID);
    basic(&c,TAG_INT,false);c.fn[0].local_count=257;prepare(&c,NVM_ARRAY_LIMIT);
    for(unsigned explicit_return=0;explicit_return<2;explicit_return++) {
        init(&c,TAG_INT,false);op(&c,explicit_return?OP_RET:OP_NOP);c.fn[0].code_length=(uint32_t)c.n;authority(&c);prepare(&c,NVM_ARRAY_INVALID);
    }
    init(&c,TAG_INT,false);array(&c,TAG_ARRAY);op(&c,OP_POP);finish(&c);prepare(&c,NVM_ARRAY_UNRESOLVED);
    basic(&c,TAG_INT,false);NvmVerifyResult before=nvm_verify(&c.m);plan_positive(&c);NvmVerifyResult after=nvm_verify(&c.m);
    CHECK(before.ok==after.ok&&!strcmp(before.error_msg,after.error_msg));
}
#ifdef RA_WHITEBOX
/* This independent manifest covers all 92 operations in the reviewed profile.
 * Decoder-only rows below are not claims that arbitrary operand stacks verify. */
#define BASE (NVM_RA_FIRST_ERROR_CLEANUP|NVM_RA_CHECK_TAGS)
#define SAFE (NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS)
#define KEEP NVM_RA_RETAIN_BEFORE_RELEASE
#define BOUND NVM_RA_CHECK_BOUNDS
#define NOM NVM_RA_CHECK_NOMINAL
#define SPEC(op,kind,flags) {OP_##op,NVM_RA_ROOT_##kind,(flags)}
static const struct {uint8_t opcode,recipe;uint16_t flags;} operations[]={
    SPEC(NOP,SCALAR,NVM_RA_FIRST_ERROR_CLEANUP),
    SPEC(PUSH_I64,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),SPEC(PUSH_U8,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),
    SPEC(PUSH_F64,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),SPEC(PUSH_BOOL,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),
    SPEC(PUSH_VOID,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),SPEC(PUSH_STR,CONSTANT,BASE|KEEP),
    SPEC(I64_ADD,SCALAR,BASE),SPEC(I64_SUB,SCALAR,BASE),SPEC(I64_MUL,SCALAR,BASE),SPEC(I64_DIV_S,SCALAR,BASE),SPEC(I64_REM_S,SCALAR,BASE),SPEC(I64_NEG,SCALAR,BASE),
    SPEC(F64_TO_BITS,SCALAR,BASE),SPEC(F64_FROM_BITS,SCALAR,BASE),SPEC(F64_ADD,SCALAR,BASE),SPEC(F64_SUB,SCALAR,BASE),SPEC(F64_MUL,SCALAR,BASE),SPEC(F64_DIV,SCALAR,BASE),SPEC(F64_NEG,SCALAR,BASE),
    SPEC(CAST_INT,SCALAR,BASE),SPEC(CAST_FLOAT,SCALAR,BASE),SPEC(CAST_BOOL,SCALAR,BASE),SPEC(TYPE_CHECK,SCALAR,BASE),
    SPEC(EQ,SCALAR,BASE),SPEC(NE,SCALAR,BASE),SPEC(LT,SCALAR,BASE),SPEC(LE,SCALAR,BASE),SPEC(GT,SCALAR,BASE),SPEC(GE,SCALAR,BASE),
    SPEC(I64_EQ,SCALAR,BASE),SPEC(I64_NE,SCALAR,BASE),SPEC(I64_LT_S,SCALAR,BASE),SPEC(I64_LE_S,SCALAR,BASE),SPEC(I64_GT_S,SCALAR,BASE),SPEC(I64_GE_S,SCALAR,BASE),
    SPEC(F64_EQ,SCALAR,BASE),SPEC(F64_NE,SCALAR,BASE),SPEC(F64_LT,SCALAR,BASE),SPEC(F64_LE,SCALAR,BASE),SPEC(F64_GT,SCALAR,BASE),SPEC(F64_GE,SCALAR,BASE),
    SPEC(BOOL_AND,SCALAR,BASE),SPEC(BOOL_OR,SCALAR,BASE),SPEC(BOOL_NOT,SCALAR,BASE),SPEC(AND,SCALAR,BASE),SPEC(OR,SCALAR,BASE),SPEC(NOT,SCALAR,BASE),
    SPEC(STR_LEN,SCALAR,BASE),SPEC(STR_CHAR_AT,SCALAR,BASE),SPEC(ARR_LEN,SCALAR,BASE),SPEC(STR_EQ,SCALAR,BASE),SPEC(STR_CONTAINS,SCALAR,BASE),SPEC(STR_STARTS_WITH,SCALAR,BASE),SPEC(STR_ENDS_WITH,SCALAR,BASE),
    SPEC(CAST_STRING,STRING,BASE|SAFE),SPEC(STR_CONCAT,STRING,BASE|SAFE),SPEC(STR_SUBSTR,STRING,BASE|SAFE),SPEC(STR_TRIM,STRING,BASE|SAFE),SPEC(STR_TO_LOWER,STRING,BASE|SAFE),SPEC(STR_TO_UPPER,STRING,BASE|SAFE),SPEC(STR_REPLACE,STRING,BASE|SAFE),SPEC(STR_FROM_INT,STRING,BASE|SAFE),SPEC(STR_FROM_FLOAT,STRING,BASE|SAFE),SPEC(STR_SPLIT,STRING,BASE|SAFE),
    SPEC(LOAD_LOCAL,LOAD,BASE|KEEP),SPEC(LOAD_GLOBAL,LOAD,BASE|KEEP),SPEC(STORE_LOCAL,STORE,BASE|KEEP),SPEC(STORE_GLOBAL,STORE,BASE|KEEP),
    SPEC(DUP,DUP,BASE|KEEP),SPEC(SWAP,SWAP,BASE),SPEC(POP,DROP,BASE),SPEC(ASSERT,DROP,BASE),
    SPEC(JMP,BRANCH,BASE),SPEC(JMP_TRUE,BRANCH,BASE),SPEC(JMP_FALSE,BRANCH,BASE),
    SPEC(CALL,CALL,BASE|NVM_RA_STAGE_OPERANDS),SPEC(RET,RETURN,BASE|NVM_RA_STAGE_OPERANDS),
    SPEC(STRUCT_NEW,CONSTRUCT,BASE|NOM|SAFE),SPEC(STRUCT_LITERAL,CONSTRUCT,BASE|NOM|SAFE),SPEC(AGG_PACK,CONSTRUCT,BASE|NOM|SAFE),
    SPEC(ARR_NEW,CONSTRUCT,BASE|SAFE),SPEC(ARR_LITERAL,CONSTRUCT,BASE|SAFE),
    SPEC(STRUCT_GET,GET,BASE|NOM|BOUND|KEEP),SPEC(AGG_GET,GET,BASE|NOM|BOUND|KEEP),SPEC(ARR_GET,GET,BASE|BOUND|KEEP),
    SPEC(STRUCT_SET,SET,BASE|NOM|BOUND|KEEP),SPEC(AGG_SET,SET,BASE|NOM|BOUND|KEEP),SPEC(ARR_SET,SET,BASE|SAFE|BOUND|KEEP),
    SPEC(ARR_PUSH,ARRAY_PUSH,BASE|SAFE|KEEP),SPEC(ARR_POP,ARRAY_POP,BASE|BOUND|KEEP),SPEC(ARR_SLICE,ARRAY_COPY,BASE|SAFE|BOUND)
};
#undef SPEC
#undef BASE
#undef SAFE
#undef KEEP
#undef BOUND
#undef NOM
static void complete_opcode_table(void) {
    bool listed[256]={false};size_t count=sizeof operations/sizeof operations[0];CHECK(count==92);
    for(size_t i=0;i<count;i++) {
        uint8_t opcode=operations[i].opcode;CHECK(!listed[opcode]);listed[opcode]=true;
        NvmRecordArrayExecutionInstruction recipe={0};CHECK(rae_recipe(opcode,&recipe));
        CHECK(recipe.recipe==operations[i].recipe&&recipe.obligations==operations[i].flags);
        Input c;init(&c,TAG_INT,false);op(&c,opcode);
        const InstructionInfo *info=isa_get_info(opcode);CHECK(info);
        uint64_t bits[MAX_OPERANDS]={0};
        for(unsigned j=0;j<info->operand_count;j++) {
            unsigned width=isa_operand_size(info->operands[j]);CHECK(width&&width<=8);
            /* Arbitrary encoded immediate bits are retained even in dead rows.
             * Branch/call/global fields need valid decoder-level destinations. */
            bits[j]=width==8?UINT64_C(0xfff8000012345678):3;
            if(opcode==OP_CALL||opcode==OP_JMP||opcode==OP_JMP_TRUE||opcode==OP_JMP_FALSE)bits[j]=0;
            for(unsigned k=0;k<width;k++)b8(c.code,&c.n,(uint8_t)(bits[j]>>(8*k)));
        }
        c.fn[0].arity=2;c.fn[0].code_length=(uint32_t)c.n;c.m.code_size=(uint32_t)c.n;
        NvmRecordArrayExecutionPlan plan={0};plan.snapshot=c.m;plan.counts.initializer=NO;
        CHECK(rae_instructions(&plan)&&plan.counts.instructions==1);
        NvmRecordArrayExecutionInstruction *row=plan.instructions;
        CHECK(row->opcode==opcode&&row->pc==0&&row->next_pc==c.n&&row->function==0);
        CHECK(row->recipe==operations[i].recipe&&row->obligations==operations[i].flags&&row->operand_count==info->operand_count);
        for(unsigned j=0;j<info->operand_count;j++)CHECK(row->operand_types[j]==info->operands[j]&&row->operand_bits[j]==bits[j]);
        int pops=info->pop_count,pushes=info->push_count;
        if(opcode==OP_ARR_LITERAL||opcode==OP_STRUCT_LITERAL||opcode==OP_AGG_PACK){pops=3;pushes=1;}
        if(opcode==OP_CALL){pops=2;pushes=1;CHECK(row->callee==0);}
        else CHECK(row->callee==NO);
        if(opcode==OP_RET){pops=1;pushes=0;}
        CHECK(row->pops==pops&&row->pushes==pushes);
        if(opcode==OP_RET)CHECK(row->successor_count==0);
        else if(opcode==OP_JMP)CHECK(row->successor_count==1&&row->successors[0]==0);
        else if(opcode==OP_JMP_TRUE||opcode==OP_JMP_FALSE)CHECK(row->successor_count==2&&row->successors[0]==0&&row->successors[1]==c.n);
        else CHECK(row->successor_count==1&&row->successors[0]==c.n);
        ra_test_free(plan.instructions);CHECK(!ra_live&&!ra_bytes);
    }
    for(unsigned i=0;i<256;i++) {
        NvmRecordArrayExecutionInstruction row={0};CHECK(!!rae_recipe((uint8_t)i,&row)==listed[i]);
        CHECK(!!nvm_record_array_opcode_supported((uint8_t)i)==listed[i]);
    }
    printf("I checked %zu explicit opcode recipes and every numeric opcode\n",count);
}
static void plan_accounting(void) {
    for(unsigned over=0;over<2;over++) {
        NvmRecordArrayExecutionPlan p={0};p.bytes=NVM_RECORD_ARRAY_EXECUTION_BYTES-16+over;
        size_t before=ra_calls;void *v=rae_zero(&p,4,4);
        if(over)CHECK(!v&&p.failure==NVM_ARRAY_LIMIT&&ra_calls==before);
        else {CHECK(v&&p.bytes==NVM_RECORD_ARRAY_EXECUTION_BYTES);ra_test_free(v);}
        memset(&p,0,sizeof p);p.work=NVM_RECORD_ARRAY_EXECUTION_STEPS-16+over;
        uint8_t original[16]={0};before=ra_calls;v=rae_copy(&p,original,4,4);
        if(over)CHECK(!v&&p.failure==NVM_ARRAY_LIMIT&&ra_calls==before);
        else {CHECK(v&&p.work==NVM_RECORD_ARRAY_EXECUTION_STEPS&&!memcmp(v,original,16));ra_test_free(v);}
        CHECK(!ra_live&&!ra_bytes);
    }
    NvmRecordArrayExecutionPlan p={0};size_t before=ra_calls;
    CHECK(!rae_zero(&p,SIZE_MAX,2)&&p.failure==NVM_ARRAY_LIMIT&&ra_calls==before);
    memset(&p,0,sizeof p);CHECK(!rae_copy(&p,"x",SIZE_MAX,2)&&p.failure==NVM_ARRAY_LIMIT&&ra_calls==before);
    memset(&p,0,sizeof p);p.bytes=UINT64_MAX;CHECK(!rae_charge(&p,0,0)&&p.failure==NVM_ARRAY_LIMIT);
    memset(&p,0,sizeof p);p.work=UINT64_MAX;CHECK(!rae_charge(&p,0,0)&&p.failure==NVM_ARRAY_LIMIT);
    /* I exercise the actual independent decoder's exact row cap. This is not
     * a claim that 65536 unreachable NOPs form an executable entry point. */
    for(unsigned count=65536;count<=65537;count++) {
        Input c;init(&c,TAG_INT,false);uint8_t *code=calloc(count,1);CHECK(code);
        c.m.code=code;c.m.code_size=count;c.fn[0].code_length=count;
        memset(&p,0,sizeof p);p.snapshot=c.m;p.counts.initializer=NO;before=ra_calls;
        int ok=rae_instructions(&p);
        if(count==65536)CHECK(ok&&p.counts.instructions==count&&p.instructions[count-1].next_pc==count);
        else CHECK(!ok&&p.failure==NVM_ARRAY_LIMIT&&ra_calls==before&&!p.instructions);
        ra_test_free(p.instructions);free(code);CHECK(!ra_live&&!ra_bytes);
    }
    /* Every defined scalar/byte comparison is against the original module.
     * Mutations are restored after each rejected comparison. */
    Input c;basic(&c,TAG_INT,true);NvmRecordArrayExecutionPlan *plan=prepare(&c,NVM_ARRAY_ELIGIBLE);
    CHECK(rae_same(plan,&c.m));
#define DIFFER(field) do{(field)^=1;CHECK(!rae_same(plan,&c.m));(field)^=1;CHECK(rae_same(plan,&c.m));}while(0)
    DIFFER(c.m.header.flags);DIFFER(c.m.header.entry_point);DIFFER(c.m.header.checksum);
    DIFFER(c.m.function_capacity);DIFFER(c.m.string_capacity);DIFFER(c.m.code_capacity);
    DIFFER(c.fn[0].local_count);DIFFER(c.fn[0].result_tag);DIFFER(c.fn[0].code_length);
    DIFFER(c.code[0]);DIFFER(c.layouts[0]);DIFFER(c.ownership[0]);DIFFER(c.lengths[1]);
#undef DIFFER
    nvm_record_array_execution_free(plan);CHECK(!ra_live&&!ra_bytes);
    all_copy_domains(&c);
    NvmMetadataEntry metadata={.key_idx=0,.value_idx=1};
    NvmDebugEntry debug={.bytecode_offset=0,.source_line=3,.source_col=9};
    c.m.metadata=&metadata;c.m.metadata_count=c.m.metadata_capacity=1;
    c.m.debug_entries=&debug;c.m.debug_count=c.m.debug_capacity=1;
    ra_calls=ra_peak=0;
    plan=prepare(&c,NVM_ARRAY_ELIGIBLE);size_t measured=ra_calls,peak=ra_peak;
    NvmRecordArrayExecutionCounts n;CHECK(nvm_record_array_execution_counts(plan,&n)&&peak<=n.peak_bytes_reserved);
    nvm_record_array_execution_free(plan);CHECK(!ra_live&&!ra_bytes&&measured>20);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<measured;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;
        plan=(void *)(uintptr_t)1;Input saved=c;NvmArrayEligibilityResult r=nvm_prepare_record_array_execution(&c.m,&plan);
        CHECK(r.status!=NVM_ARRAY_ELIGIBLE&&plan==(void *)(uintptr_t)1&&!memcmp(&saved,&c,sizeof c));
        CHECK(!ra_live&&!ra_bytes);ra_fail=SIZE_MAX;ra_persistent=0;
        plan_positive(&c);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I measured %zu plan allocation positions and %zu peak payload bytes; each one-shot/persistent refusal recovered\n",measured,peak);
}
#endif
int main(void) {
    CHECK(record_array_origin_controls()==0);
    puts("I begin owned execution snapshots");snapshot_controls();
    puts("I begin initializer identity and call signatures");initializer_and_call_controls();
    puts("I begin exact branch and instruction facts");control_edges();
    puts("I begin the 256-function and counted-string boundaries");function_boundary();counted_string_boundary();
    Input all;all_copy_domains(&all);plan_positive(&all);
    puts("I begin closed preparation refusal boundaries");plan_refusals();
#ifdef RA_WHITEBOX
    puts("I begin complete opcode recipes");complete_opcode_table();
    puts("I begin plan allocation/work boundaries");plan_accounting();
#endif
    printf("PASS %u record-array execution-plan checks; no runtime admission\n",checks);return 0;
}
