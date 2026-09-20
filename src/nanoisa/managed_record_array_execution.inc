/* I share this translation unit's exact origin whitelist and owned report.
 * No old selector or verifier calls this new non-admitting entry. */
#include "managed_record_array_execution.h"
struct NvmRecordArrayExecutionPlan {
    NvmModule snapshot;
    NvmRecordArrayOrigins *origins;
    NvmRecordArrayExecutionCounts counts;
    NvmRecordArrayExecutionFunction functions[256];
    uint16_t maximum_stacks[256];
    NvmRecordArrayExecutionDescriptor descriptors[256];
    NvmRecordArrayExecutionInstruction *instructions;
    uint64_t bytes, work;
    NvmArrayEligibilityStatus failure;
};
static int rae_charge(NvmRecordArrayExecutionPlan *p,uint64_t bytes,uint64_t work) {
    if(p->bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES || p->work>NVM_RECORD_ARRAY_EXECUTION_STEPS ||
       bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES-p->bytes || work>NVM_RECORD_ARRAY_EXECUTION_STEPS-p->work) {
        p->failure=NVM_ARRAY_LIMIT;return 0;
    }
    p->bytes+=bytes;p->work+=work;return 1;
}
static void *rae_copy(NvmRecordArrayExecutionPlan *p,const void *source,size_t n,size_t width) {
    if(width && n>SIZE_MAX/width){p->failure=NVM_ARRAY_LIMIT;return NULL;}
    size_t bytes=n*width;
    if(!bytes)return NULL;
    if(!source){p->failure=NVM_ARRAY_INVALID;return NULL;}
    if(!rae_charge(p,bytes,bytes))return NULL;
    void *copy=malloc(bytes);
    if(!copy){p->failure=NVM_ARRAY_MEMORY;return NULL;}
    memcpy(copy,source,bytes);return copy;
}
static void *rae_zero(NvmRecordArrayExecutionPlan *p,size_t n,size_t width) {
    if(width && n>SIZE_MAX/width){p->failure=NVM_ARRAY_LIMIT;return NULL;}
    size_t bytes=n*width;if(!bytes)return NULL;
    if(!rae_charge(p,bytes,bytes))return NULL;
    void *v=calloc(n,width);if(!v)p->failure=NVM_ARRAY_MEMORY;return v;
}
void nvm_record_array_execution_free(NvmRecordArrayExecutionPlan *p) {
    if(!p)return;
    NvmModule *m=&p->snapshot;
    if(m->strings)for(uint32_t i=0;i<m->string_count;i++)free(m->strings[i]);
    if(m->function_param_types)for(uint32_t i=0;i<m->function_count;i++)free(m->function_param_types[i]);
    free(m->strings);free(m->string_lengths);free(m->functions);free(m->function_param_types);
    free(m->code);free(m->ownership_data);free(m->layout_data);free(m->metadata);free(m->debug_entries);
    nvm_record_array_origins_free(p->origins);free(p->instructions);free(p);
}
static int rae_snapshot(NvmRecordArrayExecutionPlan *p,const NvmModule *m) {
    /* Closed-profile absence is checked before any unsupported pointer is read. */
    if(m->service_data || m->service_size || m->passive_data || m->passive_size ||
       m->import_count || m->module_ref_count || m->callback_contract_count ||
       m->call_descriptors || m->call_descriptor_count) {
        p->failure=NVM_ARRAY_UNRESOLVED;return 0;
    }
    if(m->function_count>256 || m->string_count>NVM_MAX_STRINGS || m->code_size>16u*1024u*1024u) {
        p->failure=NVM_ARRAY_LIMIT;return 0;
    }
    if(!m->function_count || !m->functions || m->section_count>NVM_MAX_SECTIONS ||
       m->header.section_count>NVM_MAX_SECTIONS ||
       (m->string_count && (!m->strings || !m->string_lengths)))return 0;
    NvmModule *s=&p->snapshot;
    /* Only logical rows are copied; capacity values are descriptive scalars,
     * never permission for a consumer to append to the frozen arrays. */
    s->header=m->header;memcpy(s->sections,m->sections,sizeof s->sections);
    s->section_count=m->section_count;s->string_count=m->string_count;s->string_capacity=m->string_capacity;
    s->function_count=m->function_count;s->function_capacity=m->function_capacity;
    s->code_size=m->code_size;s->code_capacity=m->code_capacity;
    s->ownership_size=m->ownership_size;s->layout_size=m->layout_size;
    s->metadata_count=m->metadata_count;s->metadata_capacity=m->metadata_capacity;
    s->debug_count=m->debug_count;s->debug_capacity=m->debug_capacity;s->source_file_idx=m->source_file_idx;
    s->import_capacity=m->import_capacity;s->module_ref_capacity=m->module_ref_capacity;
    s->callback_contract_capacity=m->callback_contract_capacity;
    s->struct_count=m->struct_count;s->union_count=m->union_count;s->enum_count=m->enum_count;
#define RAE_COPY(member,count) do { s->member=rae_copy(p,m->member,(count),sizeof *s->member); if((count) && !s->member)return 0; } while(0)
    RAE_COPY(functions,m->function_count);RAE_COPY(code,m->code_size);
    RAE_COPY(ownership_data,m->ownership_size);RAE_COPY(layout_data,m->layout_size);
    RAE_COPY(metadata,m->metadata_count);RAE_COPY(debug_entries,m->debug_count);
    RAE_COPY(string_lengths,m->string_count);
#undef RAE_COPY
    s->strings=rae_zero(p,m->string_count,sizeof *s->strings);
    if(m->string_count && !s->strings)return 0;
    for(uint32_t i=0;i<m->string_count;i++) {
        if((uint64_t)m->string_lengths[i]+1>SIZE_MAX){p->failure=NVM_ARRAY_LIMIT;return 0;}
        s->strings[i]=rae_copy(p,m->strings[i],(size_t)m->string_lengths[i]+1,1);
        if(!s->strings[i])return 0;
    }
    if(m->function_param_types) {
        s->function_param_types=rae_zero(p,m->function_count,sizeof *s->function_param_types);
        if(!s->function_param_types)return 0;
    }
    for(uint32_t i=0;i<m->function_count;i++) {
        if(m->functions[i].arity>256 || m->functions[i].local_count>256){p->failure=NVM_ARRAY_LIMIT;return 0;}
        if(m->function_param_types && m->function_param_types[i]) {
            /* Preserve a present zero-arity row as distinct from absent tags. */
            size_t n=m->functions[i].arity;
            s->function_param_types[i]=n?rae_copy(p,m->function_param_types[i],n,1):rae_zero(p,1,1);
            if(!s->function_param_types[i])return 0;
        }
    }
    return 1;
}
static int rae_equal_bytes(NvmRecordArrayExecutionPlan *p,const void *a,const void *b,size_t n) {
    if(!rae_charge(p,0,n))return 0;
    return !n || (a && b && !memcmp(a,b,n));
}
static int rae_same(NvmRecordArrayExecutionPlan *p,const NvmModule *m) {
    const NvmModule *s=&p->snapshot;
    if(!rae_charge(p,0,1024))return 0;
#define SAME(field) if(s->field!=m->field)return 0
    SAME(header.format_version);SAME(header.flags);SAME(header.entry_point);SAME(header.section_count);
    SAME(header.string_pool_offset);SAME(header.string_pool_length);SAME(header.checksum);
    SAME(section_count);SAME(string_count);SAME(string_capacity);SAME(function_count);SAME(function_capacity);
    SAME(code_size);SAME(code_capacity);SAME(ownership_size);SAME(layout_size);SAME(service_size);SAME(passive_size);
    SAME(metadata_count);SAME(metadata_capacity);SAME(debug_count);SAME(debug_capacity);SAME(source_file_idx);
    SAME(import_count);SAME(import_capacity);SAME(module_ref_count);SAME(module_ref_capacity);
    SAME(callback_contract_count);SAME(callback_contract_capacity);SAME(call_descriptor_count);
    SAME(struct_count);SAME(enum_count);SAME(union_count);
#undef SAME
    if(m->service_data || m->passive_data || m->call_descriptors ||
       !!s->function_param_types!=!!m->function_param_types)return 0;
    if(!rae_equal_bytes(p,s->header.magic,m->header.magic,4))return 0;
    for(uint32_t i=0;i<s->section_count;i++)
        if(s->sections[i].type!=m->sections[i].type || s->sections[i].offset!=m->sections[i].offset ||
           s->sections[i].size!=m->sections[i].size)return 0;
    for(uint32_t i=0;i<s->function_count;i++) {
        if(!rae_charge(p,0,16))return 0;
        const NvmFunctionEntry *a=&s->functions[i],*b=&m->functions[i];
#define FEQ(x) if(a->x!=b->x)return 0
        FEQ(name_idx);FEQ(arity);FEQ(code_offset);FEQ(code_length);FEQ(local_count);FEQ(upvalue_count);FEQ(result_tag);FEQ(result_count);
#undef FEQ
        if(s->function_param_types) {
            if(!!s->function_param_types[i]!=!!m->function_param_types[i])return 0;
            if(s->function_param_types[i] && !rae_equal_bytes(p,s->function_param_types[i],m->function_param_types[i],a->arity))return 0;
        }
    }
    for(uint32_t i=0;i<s->string_count;i++) {
        if(!rae_charge(p,0,1) || s->string_lengths[i]!=m->string_lengths[i] ||
           !rae_equal_bytes(p,s->strings[i],m->strings[i],(size_t)s->string_lengths[i]+1))return 0;
    }
    for(uint32_t i=0;i<s->metadata_count;i++) {
        if(!rae_charge(p,0,2) || s->metadata[i].key_idx!=m->metadata[i].key_idx || s->metadata[i].value_idx!=m->metadata[i].value_idx)return 0;
    }
    for(uint32_t i=0;i<s->debug_count;i++) {
        if(!rae_charge(p,0,3) || s->debug_entries[i].bytecode_offset!=m->debug_entries[i].bytecode_offset ||
           s->debug_entries[i].source_line!=m->debug_entries[i].source_line || s->debug_entries[i].source_col!=m->debug_entries[i].source_col)return 0;
    }
    return rae_equal_bytes(p,s->code,m->code,s->code_size) &&
        rae_equal_bytes(p,s->layout_data,m->layout_data,s->layout_size) &&
        rae_equal_bytes(p,s->ownership_data,m->ownership_data,s->ownership_size);
}
/* I enumerate every existing supported opcode. A later origin-whitelist change
 * cannot silently acquire an unspecified root recipe. */
static int rae_recipe(uint8_t op,NvmRecordArrayExecutionInstruction *r) {
    r->obligations=NVM_RA_FIRST_ERROR_CLEANUP|NVM_RA_CHECK_TAGS;
    switch(op) {
    case OP_PUSH_I64: case OP_PUSH_U8: case OP_PUSH_F64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
        r->recipe=NVM_RA_ROOT_CONSTANT;r->obligations=NVM_RA_FIRST_ERROR_CLEANUP;break;
    case OP_PUSH_STR:
        r->recipe=NVM_RA_ROOT_CONSTANT;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_NOP:r->recipe=NVM_RA_ROOT_SCALAR;r->obligations=NVM_RA_FIRST_ERROR_CLEANUP;break;
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S:
    case OP_I64_REM_S: case OP_I64_NEG: case OP_F64_TO_BITS: case OP_F64_FROM_BITS:
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV: case OP_F64_NEG:
    case OP_CAST_INT: case OP_CAST_FLOAT: case OP_CAST_BOOL: case OP_TYPE_CHECK:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S: case OP_I64_GT_S: case OP_I64_GE_S:
    case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
    case OP_BOOL_AND: case OP_BOOL_OR: case OP_BOOL_NOT: case OP_AND: case OP_OR: case OP_NOT:
    case OP_STR_LEN: case OP_STR_CHAR_AT: case OP_ARR_LEN:
    case OP_STR_EQ: case OP_STR_CONTAINS: case OP_STR_STARTS_WITH: case OP_STR_ENDS_WITH:
        r->recipe=NVM_RA_ROOT_SCALAR;break;
    case OP_CAST_STRING: case OP_STR_CONCAT: case OP_STR_SUBSTR: case OP_STR_TRIM:
    case OP_STR_TO_LOWER: case OP_STR_TO_UPPER: case OP_STR_REPLACE:
    case OP_STR_FROM_INT: case OP_STR_FROM_FLOAT: case OP_STR_SPLIT:
        r->recipe=NVM_RA_ROOT_STRING;
        r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;break;
    case OP_LOAD_LOCAL: case OP_LOAD_GLOBAL:r->recipe=NVM_RA_ROOT_LOAD;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_STORE_LOCAL: case OP_STORE_GLOBAL:r->recipe=NVM_RA_ROOT_STORE;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_DUP:r->recipe=NVM_RA_ROOT_DUP;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_SWAP:r->recipe=NVM_RA_ROOT_SWAP;break;
    case OP_POP: case OP_ASSERT:r->recipe=NVM_RA_ROOT_DROP;break;
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE:r->recipe=NVM_RA_ROOT_BRANCH;break;
    case OP_CALL:r->recipe=NVM_RA_ROOT_CALL;r->obligations|=NVM_RA_STAGE_OPERANDS;break;
    case OP_RET:r->recipe=NVM_RA_ROOT_RETURN;r->obligations|=NVM_RA_STAGE_OPERANDS;break;
    case OP_STRUCT_NEW: case OP_STRUCT_LITERAL: case OP_AGG_PACK:
        r->recipe=NVM_RA_ROOT_CONSTRUCT;
        r->obligations|=NVM_RA_CHECK_NOMINAL|NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;break;
    case OP_ARR_NEW: case OP_ARR_LITERAL:
        r->recipe=NVM_RA_ROOT_CONSTRUCT;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;break;
    case OP_STRUCT_GET: case OP_AGG_GET:
        r->recipe=NVM_RA_ROOT_GET;r->obligations|=NVM_RA_CHECK_NOMINAL|NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_GET:r->recipe=NVM_RA_ROOT_GET;r->obligations|=NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_STRUCT_SET: case OP_AGG_SET:
        r->recipe=NVM_RA_ROOT_SET;r->obligations|=NVM_RA_CHECK_NOMINAL|NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_SET:r->recipe=NVM_RA_ROOT_SET;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS|NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_PUSH:r->recipe=NVM_RA_ROOT_ARRAY_PUSH;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_POP:r->recipe=NVM_RA_ROOT_ARRAY_POP;r->obligations|=NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_SLICE:r->recipe=NVM_RA_ROOT_ARRAY_COPY;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS|NVM_RA_CHECK_BOUNDS;break;
    default:return 0;
    }
    return 1;
}
static int rae_instructions(NvmRecordArrayExecutionPlan *p) {
    const NvmModule *m=&p->snapshot;uint32_t total=0;
    for(uint32_t fi=0;fi<m->function_count;fi++) {
        const NvmFunctionEntry *f=&m->functions[fi];
        if(!rae_charge(p,0,(uint64_t)f->code_length+32))return 0;
        NvmRecordArrayExecutionFunction *copy=&p->functions[fi];copy->signature=*f;
        copy->maximum_stack=p->maximum_stacks[fi];copy->instruction_start=total;
        copy->parameter_tags_present=m->function_param_types && m->function_param_types[fi];
        for(uint32_t pc=0;pc<f->code_length;) {
            DecodedInstruction in;uint32_t size=isa_decode(m->code+f->code_offset+pc,f->code_length-pc,&in);
            if(!size || total==65536){p->failure=size?NVM_ARRAY_LIMIT:NVM_ARRAY_INVALID;return 0;}
            total++;copy->instruction_count++;pc+=size;
        }
        uint32_t name=f->name_idx;
        if(!rae_charge(p,0,8))return 0;
        if(p->counts.initializer==UINT32_MAX && name<m->string_count && m->string_lengths[name]==8 &&
           !memcmp(m->strings[name],"__init__",8))p->counts.initializer=fi;
    }
    p->counts.instructions=total;
    p->instructions=rae_zero(p,total,sizeof *p->instructions);
    if(total && !p->instructions)return 0;
    for(uint32_t fi=0;fi<m->function_count;fi++) {
        const NvmFunctionEntry *f=&m->functions[fi];
        if(!rae_charge(p,0,(uint64_t)f->code_length+p->functions[fi].instruction_count*64u))return 0;
        uint8_t *boundaries=rae_zero(p,(size_t)f->code_length+1,1);
        if(!boundaries)return 0;
        uint32_t index=p->functions[fi].instruction_start;
        for(uint32_t pc=0;pc<f->code_length;) {
            DecodedInstruction in;uint32_t size=isa_decode(m->code+f->code_offset+pc,f->code_length-pc,&in);
            if(!size){free(boundaries);return 0;}
            NvmRecordArrayExecutionInstruction *r=&p->instructions[index++];
            const InstructionInfo *info=isa_get_info(in.opcode);
            if(!info || !nvm_record_array_opcode_supported(in.opcode) || !rae_recipe(in.opcode,r)) {
                free(boundaries);p->failure=NVM_ARRAY_UNRESOLVED;return 0;
            }
            boundaries[pc]=1;r->function=fi;r->pc=pc;r->next_pc=pc+size;r->opcode=in.opcode;r->callee=UINT32_MAX;
            r->operand_count=info->operand_count;uint32_t offset=1;
            for(uint8_t j=0;j<info->operand_count;j++) {
                uint32_t width=isa_operand_size(info->operands[j]);
                if(width>8 || offset>size || width>size-offset){free(boundaries);return 0;}
                r->operand_types[j]=(uint8_t)info->operands[j];
                for(uint32_t k=0;k<width;k++)r->operand_bits[j]|=(uint64_t)m->code[f->code_offset+pc+offset+k]<<(k*8u);
                offset+=width;
            }
            if(offset!=size){free(boundaries);return 0;}
            int pops=info->pop_count,pushes=info->push_count;
            if(in.opcode==OP_ARR_LITERAL || in.opcode==OP_STRUCT_LITERAL){pops=in.operands[1].u16;pushes=1;}
            if(in.opcode==OP_AGG_PACK){pops=in.operands[3].u16;pushes=1;}
            if(in.opcode==OP_CALL) {
                r->callee=in.operands[0].u32;
                if(r->callee>=m->function_count){free(boundaries);return 0;}
                pops=m->functions[r->callee].arity;pushes=m->functions[r->callee].result_count;
            }
            if(in.opcode==OP_RET){pops=f->result_count;pushes=0;}
            if(pops<0 || pushes<0 || pops>256 || pushes>256){free(boundaries);return 0;}
            r->pops=(uint16_t)pops;r->pushes=(uint16_t)pushes;
            if(in.opcode==OP_JMP || in.opcode==OP_JMP_TRUE || in.opcode==OP_JMP_FALSE) {
                int64_t target=(int64_t)pc+in.operands[0].i32;
                if(target<0 || (uint64_t)target>f->code_length){free(boundaries);return 0;}
                r->successors[r->successor_count++]=(uint32_t)target;
            }
            if(in.opcode!=OP_JMP && in.opcode!=OP_RET)r->successors[r->successor_count++]=pc+size;
            if(in.opcode==OP_LOAD_GLOBAL || in.opcode==OP_STORE_GLOBAL) {
                uint32_t count=in.operands[0].u32+1;
                if(count>256){free(boundaries);return 0;}
                if(count>p->counts.globals)p->counts.globals=count;
            }
            pc+=size;
        }
        boundaries[f->code_length]=1;
        /* A separate charged scan validates every published edge, even dead code. */
        if(!rae_charge(p,0,(uint64_t)p->functions[fi].instruction_count*3u)){free(boundaries);return 0;}
        for(uint32_t i=p->functions[fi].instruction_start;i<index;i++) {
            NvmRecordArrayExecutionInstruction *r=&p->instructions[i];
            for(uint8_t j=0;j<r->successor_count;j++)if(!boundaries[r->successors[j]]){free(boundaries);return 0;}
        }
        free(boundaries); /* Reservation remains conservative across functions. */
    }
    return 1;
}
NvmArrayEligibilityResult nvm_prepare_record_array_execution(const NvmModule *m,NvmRecordArrayExecutionPlan **out) {
    NvmArrayEligibilityResult r={.status=NVM_ARRAY_INVALID};
    snprintf(r.message,sizeof r.message,"I require a stable original module and plan output.");
    if(!m || !out)return r;
    NvmRecordArrayExecutionPlan *p=calloc(1,sizeof *p);
    if(!p){r.status=NVM_ARRAY_MEMORY;return r;}
    p->failure=NVM_ARRAY_INVALID;p->counts.initializer=UINT32_MAX;
    /* Query peak/work includes its transient allocation and publication overlap.
     * My own copies/temporaries never borrow that reserved64MiB/16M allowance. */
    p->bytes=UINT64_C(64)*1024*1024;p->work=UINT64_C(16777216);
    if(!rae_charge(p,65536u+sizeof *p,65536u+sizeof *p) || !rae_snapshot(p,m))goto fail;
    r=analyze(&p->snapshot,NULL,NULL,NULL,&p->origins,p->maximum_stacks);
    if(r.status!=NVM_ARRAY_ELIGIBLE){nvm_record_array_execution_free(p);return r;}
    /* Entire numeric opcode space is a fixed, charged completeness check. */
    if(!rae_charge(p,0,256))goto fail;
    for(uint32_t op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction recipe={0};
        if(!!nvm_record_array_opcode_supported((uint8_t)op)!=!!rae_recipe((uint8_t)op,&recipe)) {
            p->failure=NVM_ARRAY_UNRESOLVED;goto fail;
        }
    }
    NvmDeclarationCounts declarations;
    if(!nvm_record_array_declaration_counts(p->origins,&declarations) || declarations.layouts>256)goto fail;
    if(!rae_charge(p,0,(uint64_t)declarations.layouts*4u))goto fail;
    for(uint32_t i=0;i<declarations.layouts;i++) {
        NvmDeclarationLayout row;
        if(!nvm_record_array_declaration_layout(p->origins,i,&row))goto fail;
        if(row.kind==NVM_V2_LAYOUT_STRUCT) {
            uint32_t ordinal=p->counts.records++;
            p->descriptors[ordinal]=(NvmRecordArrayExecutionDescriptor){ordinal,i,row.fields};
        }
    }
    if(p->counts.records!=p->snapshot.struct_count)goto fail;
    if(!rae_instructions(p) || !rae_same(p,m))goto fail;
    p->counts.strings=p->snapshot.string_count;p->counts.sections=p->snapshot.section_count;
    p->counts.metadata=p->snapshot.metadata_count;p->counts.debug=p->snapshot.debug_count;
    p->counts.functions=p->snapshot.function_count;p->counts.entry=p->snapshot.header.entry_point;
    p->counts.frame_limit=NVM_RECORD_ARRAY_EXECUTION_FRAMES;
    p->counts.peak_bytes_reserved=p->bytes;p->counts.work_reserved=p->work;
    *out=p;r.status=NVM_ARRAY_ELIGIBLE;
    snprintf(r.message,sizeof r.message,"I copied complete execution facts; I grant no executable admission.");return r;
fail:
    r.status=p->failure;snprintf(r.message,sizeof r.message,"I could not prepare bounded exact mixed execution facts.");
    nvm_record_array_execution_free(p);return r;
}
bool nvm_record_array_execution_counts(const NvmRecordArrayExecutionPlan *p,NvmRecordArrayExecutionCounts *out) {
    if(!p || !out)return false;
    *out=p->counts;return true;
}
bool nvm_record_array_execution_function(const NvmRecordArrayExecutionPlan *p,uint32_t i,NvmRecordArrayExecutionFunction *out) {
    if(!p || !out || i>=p->counts.functions)return false;
    *out=p->functions[i];return true;
}
bool nvm_record_array_execution_instruction(const NvmRecordArrayExecutionPlan *p,uint32_t i,NvmRecordArrayExecutionInstruction *out) {
    if(!p || !out || i>=p->counts.instructions)return false;
    *out=p->instructions[i];return true;
}
bool nvm_record_array_execution_parameter(const NvmRecordArrayExecutionPlan *p,uint32_t fi,uint16_t i,uint8_t *out) {
    if(!p || !out || fi>=p->counts.functions || !p->functions[fi].parameter_tags_present || i>=p->snapshot.functions[fi].arity)return false;
    *out=p->snapshot.function_param_types[fi][i];return true;
}
bool nvm_record_array_execution_bytes(const NvmRecordArrayExecutionPlan *p,NvmRecordArraySnapshotKind kind,
        uint32_t index,uint32_t offset,uint32_t count,void *out) {
    if(!p || !out)return false;
    const void *source=NULL;uint32_t size=0;
    switch(kind) {
    case NVM_RA_SNAPSHOT_CODE:if(index)return false;source=p->snapshot.code;size=p->snapshot.code_size;break;
    case NVM_RA_SNAPSHOT_LAYOUTS:if(index)return false;source=p->snapshot.layout_data;size=p->snapshot.layout_size;break;
    case NVM_RA_SNAPSHOT_OWNERSHIP:if(index)return false;source=p->snapshot.ownership_data;size=p->snapshot.ownership_size;break;
    case NVM_RA_SNAPSHOT_STRING:if(index>=p->snapshot.string_count)return false;source=p->snapshot.strings[index];size=p->snapshot.string_lengths[index];break;
    default:return false;
    }
    if(offset>size || count>size-offset)return false;
    if(count)memcpy(out,(const uint8_t *)source+offset,count);
    return true;
}
#define RAE_FORWARD(name,query,type) \
 bool name(const NvmRecordArrayExecutionPlan *p,uint32_t i,type *out){return p && query(p->origins,i,out);}
RAE_FORWARD(nvm_record_array_execution_origin,nvm_record_array_origin,NvmRecordHeapOrigin)
RAE_FORWARD(nvm_record_array_execution_field_value,nvm_record_array_field_value,NvmRecordValueOrigins)
RAE_FORWARD(nvm_record_array_execution_required_elements,nvm_record_array_required_elements,uint16_t)
RAE_FORWARD(nvm_record_array_execution_layout,nvm_record_array_declaration_layout,NvmDeclarationLayout)
RAE_FORWARD(nvm_record_array_execution_type,nvm_record_array_declaration_type,NvmOrdinaryArrayType)
RAE_FORWARD(nvm_record_array_execution_binding,nvm_record_array_declaration_binding,NvmOrdinaryArrayBinding)
#undef RAE_FORWARD
bool nvm_record_array_execution_declaration_counts(const NvmRecordArrayExecutionPlan *p,NvmDeclarationCounts *out) {
    return p && nvm_record_array_declaration_counts(p->origins,out);
}
bool nvm_record_array_execution_field(const NvmRecordArrayExecutionPlan *p,uint32_t i,uint16_t field,NvmV2LayoutField *out) {
    return p && nvm_record_array_declaration_field(p->origins,i,field,out);
}
bool nvm_record_array_execution_variant(const NvmRecordArrayExecutionPlan *p,uint32_t i,uint16_t variant,NvmUnionVariantFact *out) {
    return p && nvm_record_array_declaration_variant(p->origins,i,variant,out);
}
bool nvm_record_array_execution_header(const NvmRecordArrayExecutionPlan *p,NvmHeader *out) {
    if(!p || !out)return false;
    *out=p->snapshot.header;return true;
}
bool nvm_record_array_execution_section(const NvmRecordArrayExecutionPlan *p,uint32_t i,NvmSectionEntry *out) {
    if(!p || !out || i>=p->snapshot.section_count)return false;
    *out=p->snapshot.sections[i];return true;
}
bool nvm_record_array_execution_metadata(const NvmRecordArrayExecutionPlan *p,uint32_t i,NvmMetadataEntry *out) {
    if(!p || !out || i>=p->snapshot.metadata_count)return false;
    *out=p->snapshot.metadata[i];return true;
}
bool nvm_record_array_execution_debug(const NvmRecordArrayExecutionPlan *p,uint32_t i,NvmDebugEntry *out) {
    if(!p || !out || i>=p->snapshot.debug_count)return false;
    *out=p->snapshot.debug_entries[i];return true;
}
bool nvm_record_array_execution_size(const NvmRecordArrayExecutionPlan *p,NvmRecordArraySnapshotKind kind,uint32_t index,uint32_t *out) {
    if(!p || !out)return false;
    uint32_t value;
    switch(kind) {
    case NVM_RA_SNAPSHOT_CODE:if(index)return false;value=p->snapshot.code_size;break;
    case NVM_RA_SNAPSHOT_LAYOUTS:if(index)return false;value=p->snapshot.layout_size;break;
    case NVM_RA_SNAPSHOT_OWNERSHIP:if(index)return false;value=p->snapshot.ownership_size;break;
    case NVM_RA_SNAPSHOT_STRING:if(index>=p->snapshot.string_count)return false;value=p->snapshot.string_lengths[index];break;
    default:return false;
    }
    *out=value;return true;
}
bool nvm_record_array_execution_origin_counts(const NvmRecordArrayExecutionPlan *p,NvmRecordArrayOriginCounts *out) {
    return p && nvm_record_array_origins_counts(p->origins,out);
}

bool nvm_record_array_execution_descriptor(const NvmRecordArrayExecutionPlan *p,uint32_t i,NvmRecordArrayExecutionDescriptor *out) {
    if(!p || !out || i>=p->counts.records)return false;
    *out=p->descriptors[i];return true;
}
