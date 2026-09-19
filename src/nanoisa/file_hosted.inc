#include "file_hosted.h"
struct NvmFileHostedPlan {
    NvmFileBodyReport *body;
    NvmFileHostedStartup startup;
    NvmFileHostedFunction functions[NVM_FILE_FLOW_FUNCTIONS];
};
typedef struct {
    uint32_t counts[NVM_V2_SECTION_TYPE_MAX+1];
    size_t section_bytes[NVM_V2_SECTION_TYPE_MAX+1];
    size_t decoded, fields, strings;
    uint16_t maximum_params;
} FileHostedCounts;
static bool file_hosted_add(size_t *total,size_t count,size_t width) {
    if(count && width>SIZE_MAX/count)return false;
    size_t add=count*width;
    if(*total>NVM_FILE_HOSTED_BYTES || add>NVM_FILE_HOSTED_BYTES-*total)return false;
    *total+=add;return true;
}
static bool file_hosted_take(NvmV2Cursor *c,size_t count) {
    const uint8_t *ignored;return nvm_v2_take(c,count,&ignored)==NVM_V2_OK;
}
static bool file_hosted_array(size_t *total,size_t count,size_t initial,size_t width) {
    size_t capacity=initial;
    while(capacity<count){if(capacity>SIZE_MAX/2)return false;capacity*=2;}
    /* Conservative old+new replacement peak for every growing bridge array. */
    return file_hosted_add(total,capacity,width) && file_hosted_add(total,capacity,width);
}
static NvmFileFlowStatus file_hosted_section(const uint8_t *data,NvmV2SectionEntry e,FileHostedCounts *a) {
    a->section_bytes[e.type]=(size_t)e.size;
    if(e.type==NVM_V2_SECTION_CODE)
        return e.size && e.size<=NVM_FILE_CODE_BYTES?NVM_FILE_FLOW_OK:NVM_FILE_FLOW_LIMIT;
    if(e.type==NVM_V2_SECTION_OWNERSHIP)return e.size?NVM_FILE_FLOW_OK:NVM_FILE_FLOW_INVALID;
    if(e.type==NVM_V2_SECTION_SERVICE_BINDINGS) {
        NvmFileNominalBindings bindings;
        return nvm_file_nominal_decode(data+e.offset,(size_t)e.size,&bindings)==NVM_SERVICE_OK?
            NVM_FILE_FLOW_OK:NVM_FILE_FLOW_INVALID;
    }
    if(e.type==NVM_V2_SECTION_PASSIVE)return NVM_FILE_FLOW_UNRESOLVED;
    NvmV2Cursor c;nvm_v2_cursor_init(&c,data+e.offset,(size_t)e.size);uint32_t count;
    if(nvm_v2_u32(&c,&count)!=NVM_V2_OK)return NVM_FILE_FLOW_INVALID;
    a->counts[e.type]=count;
    size_t wire=0,width=0;
    switch(e.type) {
    case NVM_V2_SECTION_GLOBALS:case NVM_V2_SECTION_LINKS:case NVM_V2_SECTION_CALLBACKS:
        if(count)return NVM_FILE_FLOW_UNRESOLVED;
        break;
    case NVM_V2_SECTION_FUNCTIONS:
        if(!count || count>NVM_FILE_FLOW_FUNCTIONS)return NVM_FILE_FLOW_LIMIT;
        wire=32;width=sizeof(NvmV2Function);break;
    case NVM_V2_SECTION_IMPORTS:
        if(count!=NVM_SERVICE_BINDING_COUNT)return NVM_FILE_FLOW_INVALID;
        wire=16;width=sizeof(NvmV2Import);break;
    case NVM_V2_SECTION_METADATA:wire=8;width=sizeof(NvmV2MetadataEntry);break;
    case NVM_V2_SECTION_DEBUG:wire=16;width=sizeof(NvmV2DebugEntry);break;
    case NVM_V2_SECTION_CONSTANTS:
        if(count>(c.size-c.pos)/8)return NVM_FILE_FLOW_INVALID;
        width=sizeof(NvmV2Constant);
        for(uint32_t i=0;i<count;i++) {
            uint8_t tag,pad;uint16_t pad2;uint32_t length;
            if(nvm_v2_u8(&c,&tag)!=NVM_V2_OK || nvm_v2_u8(&c,&pad)!=NVM_V2_OK ||
               nvm_v2_u16(&c,&pad2)!=NVM_V2_OK || nvm_v2_u32(&c,&length)!=NVM_V2_OK ||
               pad || pad2 || tag!=TAG_STRING || !file_hosted_take(&c,length) ||
               nvm_v2_align4(&c)!=NVM_V2_OK)return NVM_FILE_FLOW_INVALID;
            if(!file_hosted_add(&a->strings,(size_t)length+1,1))return NVM_FILE_FLOW_LIMIT;
        }
        break;
    case NVM_V2_SECTION_SIGNATURES:
        if(count>(c.size-c.pos)/4)return NVM_FILE_FLOW_INVALID;
        width=sizeof(NvmV2Signature);
        for(uint32_t i=0;i<count;i++) {
            uint16_t params,results;
            if(nvm_v2_u16(&c,&params)!=NVM_V2_OK || nvm_v2_u16(&c,&results)!=NVM_V2_OK ||
               !file_hosted_take(&c,params) || nvm_v2_align4(&c)!=NVM_V2_OK ||
               !file_hosted_take(&c,results) || nvm_v2_align4(&c)!=NVM_V2_OK)return NVM_FILE_FLOW_INVALID;
            if(params>a->maximum_params)a->maximum_params=params;
        }
        break;
    case NVM_V2_SECTION_LAYOUTS:
        if(count<NVM_FILE_NOMINAL_TYPES || count>NVM_FILE_NOMINAL_MAX_LAYOUTS)return NVM_FILE_FLOW_LIMIT;
        if(count>(c.size-c.pos)/8)return NVM_FILE_FLOW_INVALID;
        width=sizeof(NvmV2Layout);
        for(uint32_t i=0;i<count;i++) {
            uint8_t kind,pad;uint16_t fields;uint32_t ignored;
            if(nvm_v2_u8(&c,&kind)!=NVM_V2_OK || nvm_v2_u8(&c,&pad)!=NVM_V2_OK ||
               nvm_v2_u16(&c,&fields)!=NVM_V2_OK || nvm_v2_u32(&c,&ignored)!=NVM_V2_OK ||
               pad || kind>NVM_V2_LAYOUT_KIND_MAX || fields>(c.size-c.pos)/12)return NVM_FILE_FLOW_INVALID;
            if(!file_hosted_add(&a->fields,fields,sizeof(NvmV2LayoutField)))return NVM_FILE_FLOW_LIMIT;
            for(uint32_t j=0;j<fields;j++) {
                uint32_t nested;
                if(!file_hosted_take(&c,4) || nvm_v2_u32(&c,&nested)!=NVM_V2_OK ||
                   !file_hosted_take(&c,4) || (nested!=NVM_V2_NO_INDEX && nested>=i))return NVM_FILE_FLOW_INVALID;
            }
        }
        break;
    default:return NVM_FILE_FLOW_INVALID;
    }
    if(wire && (count>(c.size-c.pos)/wire || !file_hosted_take(&c,(size_t)count*wire)))return NVM_FILE_FLOW_INVALID;
    if(!nvm_v2_cursor_exhausted(&c))return NVM_FILE_FLOW_INVALID;
    if(!file_hosted_add(&a->decoded,count,width))return NVM_FILE_FLOW_LIMIT;
    return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus file_hosted_preflight(const uint8_t *data,size_t size,NvmV2Header *header,size_t *bound) {
    if(!data)return NVM_FILE_FLOW_INVALID;
    if(size>NVM_FILE_HOSTED_INPUT_BYTES)return NVM_FILE_FLOW_LIMIT;
    if(nvm_v2_validate(data,size)!=NVM_V2_OK || nvm_v2_read_header(data,size,header)!=NVM_V2_OK ||
       header->total_size!=size || header->isa_version!=NVM_V2_ISA_VERSION)return NVM_FILE_FLOW_INVALID;
    uint32_t required=NVM_V2_FEATURE_FFI|NVM_V2_FEATURE_RETAINED_LAYOUTS|
                      NVM_V2_FEATURE_OWNERSHIP|NVM_V2_FEATURE_SERVICE_BINDINGS;
    if((header->feature_bits & required)!=required)return NVM_FILE_FLOW_INVALID;
    if(header->feature_bits & ~(required|NVM_V2_FEATURE_DEBUG))return NVM_FILE_FLOW_UNRESOLVED;
    FileHostedCounts a={0};
    for(uint32_t i=0;i<header->section_count;i++) {
        NvmV2SectionEntry e;
        if(nvm_v2_read_section(data,size,header,i,&e)!=NVM_V2_OK)return NVM_FILE_FLOW_INVALID;
        NvmFileFlowStatus status=file_hosted_section(data,e,&a);if(status!=NVM_FILE_FLOW_OK)return status;
    }
    uint32_t constants=a.counts[NVM_V2_SECTION_CONSTANTS],functions=a.counts[NVM_V2_SECTION_FUNCTIONS];
    uint32_t layouts=a.counts[NVM_V2_SECTION_LAYOUTS];
    size_t layout=a.section_bytes[NVM_V2_SECTION_LAYOUTS],nominal=0,total=0;
    if(!functions || header->entry_point>=functions || !a.section_bytes[NVM_V2_SECTION_SERVICE_BINDINGS] ||
       !a.section_bytes[NVM_V2_SECTION_OWNERSHIP] || !a.section_bytes[NVM_V2_SECTION_CODE])return NVM_FILE_FLOW_INVALID;
    if(!nvm_file_nominal_storage_bound(layouts,&nominal))return NVM_FILE_FLOW_LIMIT;
    /* Sum conservative overlapping phase maxima, never just wire length:
     * copied envelope + decoded tables + private body full budget + plan;
     * bridge capacities (including old/new replacement), payloads/tag copies;
     * retained-layout decoder replay and nominal-v2 validation temporary view.
     * Prior-only preflight prevents the layout decoder's forward DFS allocation. */
    if(!file_hosted_add(&total,size,1) || !file_hosted_add(&total,a.decoded,1) ||
       !file_hosted_add(&total,a.fields,2) || !file_hosted_add(&total,layouts,sizeof(NvmV2Layout)) ||
       !file_hosted_add(&total,1,NVM_FILE_FLOW_BYTES) || !file_hosted_add(&total,1,sizeof(NvmFileHostedPlan)) ||
       !file_hosted_add(&total,2,sizeof(NvmModule)) || !file_hosted_add(&total,1,sizeof(NvmV2Module)) ||
       !file_hosted_add(&total,1,sizeof(FileHostedCounts)) ||
       !file_hosted_array(&total,constants,64,sizeof(char *)+sizeof(uint32_t)) ||
       !file_hosted_array(&total,functions,32,sizeof(NvmFunctionEntry)+sizeof(uint8_t *)) ||
       !file_hosted_array(&total,a.section_bytes[NVM_V2_SECTION_CODE],4096,1) ||
       !file_hosted_array(&total,a.counts[NVM_V2_SECTION_DEBUG],256,sizeof(NvmDebugEntry)) ||
       !file_hosted_array(&total,a.counts[NVM_V2_SECTION_METADATA],8,sizeof(NvmMetadataEntry)) ||
       !file_hosted_add(&total,32,sizeof(NvmImportEntry)+sizeof(uint8_t *)) ||
       !file_hosted_add(&total,16,sizeof(NvmModuleRefEntry)) ||
       !file_hosted_add(&total,a.strings,1) ||
       !file_hosted_add(&total,(size_t)functions+NVM_SERVICE_BINDING_COUNT,a.maximum_params) ||
       !file_hosted_add(&total,layout,2) ||
       !file_hosted_add(&total,a.section_bytes[NVM_V2_SECTION_OWNERSHIP],1) ||
       !file_hosted_add(&total,NVM_FILE_NOMINAL_BYTES,1) ||
       !file_hosted_add(&total,constants,sizeof(char *)+sizeof(uint32_t)) ||
       !file_hosted_add(&total,functions,sizeof(NvmFunctionEntry)+sizeof(uint8_t *)) ||
       !file_hosted_add(&total,nominal,1))return NVM_FILE_FLOW_LIMIT;
    *bound=total;return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus file_hosted_codec_status(NvmV2Result r) {
    /* Allocating readers/bridge also use INDEX_RANGE or SECTION_TYPE after
     * allocation-dependent validation. No legacy failure proves a malformed
     * input or OOM here. Precise INVALID/LIMIT belongs to nonalloc preflight. */
    (void)r;return NVM_FILE_FLOW_UNRESOLVED;
}
static NvmFileFlowStatus file_hosted_start(NvmFileHostedPlan *p,const NvmV2Module *wire,const NvmModule *m) {
    NvmFileCodePlan *code=p->body->plan;
    p->startup.entry=wire->entry_point;p->startup.initializer=NVM_V2_NO_INDEX;p->startup.functions=code->count;
    if(!(m->header.flags & NVM_FLAG_HAS_MAIN) || m->header.entry_point!=wire->entry_point)return NVM_FILE_FLOW_INVALID;
    const NvmFileFlowFunction *entry=&code->functions[p->startup.entry].declaration;
    if(entry->parameters || entry->result_count!=1 || entry->result.mode ||
       entry->result.global_index!=NVM_V2_NO_INDEX ||
       (entry->result.tag!=TAG_INT && entry->result.tag!=TAG_BOOL))return NVM_FILE_FLOW_UNRESOLVED;
    for(uint32_t f=0;f<code->count;f++) {
        uint32_t index=m->functions[f].name_idx;
        if(index>=m->string_count || !m->strings[index] || memchr(m->strings[index],0,m->string_lengths[index]))return NVM_FILE_FLOW_INVALID;
        if(p->startup.initializer==NVM_V2_NO_INDEX && !strcmp(m->strings[index],"__init__")) {
            const NvmFileFlowFunction *init=&code->functions[f].declaration;
            if(init->parameters || init->result_count || init->result.tag!=TAG_VOID ||
               init->result.mode || init->result.global_index!=NVM_V2_NO_INDEX)return NVM_FILE_FLOW_UNRESOLVED;
            p->startup.initializer=f;
        }
    }
    return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus file_hosted_bounds(NvmFileHostedPlan *p,const NvmV2Module *wire) {
    NvmFileCodePlan *code=p->body->plan;
    for(uint32_t rank=0;rank<code->count;rank++) {
        uint32_t f=code->function_order[rank];NvmFileHostedFunction *fact=&p->functions[f];
        fact->code=code->functions[f];
        fact->declared_stack=wire->functions.items[f].max_stack;
        fact->locals=code->functions[f].declaration.locals;fact->staging_slots=1;fact->frames=1;
        for(uint16_t i=0;i<code->functions[f].instruction_count;i++) {
            const NvmFileBodyInstruction *body=&p->body->facts[code->starts[f]+i];
            if(!body->reachable)continue;
            if(body->input_stack>fact->operand_peak)fact->operand_peak=body->input_stack;
            if(body->output_stack>fact->operand_peak)fact->operand_peak=body->output_stack;
            if(body->has_obligation && body->obligation.kind==NVM_FILE_FLOW_CALL &&
               (uint32_t)body->obligation.parameters+1>fact->staging_slots)
                fact->staging_slots=(uint16_t)(body->obligation.parameters+1);
        }
        if(fact->declared_stack && fact->declared_stack<fact->operand_peak)return NVM_FILE_FLOW_INVALID;
        uint64_t frame=(uint64_t)fact->locals+fact->operand_peak+fact->staging_slots;
        fact->vm_value_slots=fact->native_value_slots=frame;
        for(uint16_t i=0;i<code->functions[f].instruction_count;i++) {
            const NvmFileBodyInstruction *body=&p->body->facts[code->starts[f]+i];
            if(!body->reachable || !body->has_obligation || body->obligation.kind!=NVM_FILE_FLOW_CALL)continue;
            const NvmFileFlowObligation *call=&body->obligation;
            if(call->target>=code->count || call->borrowed_inputs>call->parameters)return NVM_FILE_FLOW_INVALID;
            uint16_t supplied=(uint16_t)(call->parameters-call->borrowed_inputs);
            if(supplied>body->input_stack)return NVM_FILE_FLOW_INVALID;
            const NvmFileHostedFunction *callee=&p->functions[call->target];
            /* VM suffix locals overlap consumed value args; native frame arrays
             * remain live during a helper. Both bounds retain argument/result
             * staging until the callee's checked transfer can publish. */
            uint64_t vm=(uint64_t)fact->locals+body->input_stack-supplied+fact->staging_slots+callee->vm_value_slots;
            uint64_t native=frame+callee->native_value_slots;
            if(vm>fact->vm_value_slots)fact->vm_value_slots=vm;
            if(native>fact->native_value_slots)fact->native_value_slots=native;
            if((uint32_t)callee->frames+1>fact->frames)fact->frames=(uint16_t)(callee->frames+1);
        }
        /* Caps are inherited: at most64 frames, each <=256locals+256operands+
         * 257 staged args/result. Widened sums cannot approach UINT64_MAX. */
        fact->reference_slots=(uint32_t)fact->frames*NVM_FILE_FLOW_REFERENCES;
        fact->region_slots=(uint32_t)fact->frames*NVM_FILE_FLOW_REFERENCES;
        if(fact->frames>NVM_FILE_FLOW_FUNCTIONS || fact->vm_value_slots>UINT64_C(64)*769 ||
           fact->native_value_slots>UINT64_C(64)*769)return NVM_FILE_FLOW_LIMIT;
    }
    uint32_t roots[2]={p->startup.entry,p->startup.initializer};
    for(unsigned i=0;i<2;i++)if(roots[i]!=NVM_V2_NO_INDEX) {
        const NvmFileHostedFunction *f=&p->functions[roots[i]];
        if(f->vm_value_slots>p->startup.vm_value_slots)p->startup.vm_value_slots=f->vm_value_slots;
        if(f->native_value_slots>p->startup.native_value_slots)p->startup.native_value_slots=f->native_value_slots;
        if(f->frames>p->startup.frames)p->startup.frames=f->frames;
        if(f->reference_slots>p->startup.reference_slots)p->startup.reference_slots=f->reference_slots;
        if(f->region_slots>p->startup.region_slots)p->startup.region_slots=f->region_slots;
    }
    return NVM_FILE_FLOW_OK;
}
void nvm_file_hosted_free(NvmFileHostedPlan *p){if(p){nvm_file_body_free(p->body);free(p);}}
NvmFileFlowStatus nvm_file_hosted_prepare(const uint8_t *data,size_t size,NvmFileHostedPlan **out) {
    if(!out)return NVM_FILE_FLOW_INVALID;
    NvmV2Header header;size_t bound;
    NvmFileFlowStatus status=file_hosted_preflight(data,size,&header,&bound);
    if(status!=NVM_FILE_FLOW_OK)return status;
    uint8_t *copy=malloc(size);if(!copy)return NVM_FILE_FLOW_MEMORY;memcpy(copy,data,size);
    NvmV2Module wire={0};NvmModule *module=NULL;NvmFileHostedPlan *plan=NULL;
    NvmV2Result decoded=nvm_v2_module_deserialize(copy,size,&wire);
    if(decoded!=NVM_V2_OK){status=file_hosted_codec_status(decoded);goto done;}
    decoded=nvm_v2_to_nvm_module(&wire,&module);
    if(decoded!=NVM_V2_OK){status=file_hosted_codec_status(decoded);goto done;}
    plan=calloc(1,sizeof *plan);if(!plan){status=NVM_FILE_FLOW_MEMORY;goto done;}
    plan->startup.features=header.feature_bits;plan->startup.allocation_bound=bound;
    status=nvm_file_body_analyze(module,&plan->body);if(status!=NVM_FILE_FLOW_OK)goto done;
    status=file_hosted_start(plan,&wire,module);if(status!=NVM_FILE_FLOW_OK)goto done;
    status=file_hosted_bounds(plan,&wire);if(status!=NVM_FILE_FLOW_OK)goto done;
    *out=plan;plan=NULL;
done:
    nvm_file_hosted_free(plan);nvm_module_free(module);nvm_v2_module_free(&wire);free(copy);return status;
}
bool nvm_file_hosted_startup(const NvmFileHostedPlan *p,NvmFileHostedStartup *out) {
    if(!p || !out)return false;
    *out=p->startup;return true;
}
bool nvm_file_hosted_function(const NvmFileHostedPlan *p,uint32_t f,NvmFileHostedFunction *out) {
    if(!p || !out || f>=p->startup.functions)return false;
    *out=p->functions[f];return true;
}
bool nvm_file_hosted_local(const NvmFileHostedPlan *p,uint32_t f,uint16_t i,NvmFileFlowDeclaration *out) {
    return p && nvm_file_body_local(p->body,f,i,out);
}
bool nvm_file_hosted_instruction(const NvmFileHostedPlan *p,uint32_t f,uint16_t i,
                                NvmFileCodeInstruction *instruction,NvmFileBodyInstruction *fact) {
    return p && nvm_file_body_instruction(p->body,f,i,instruction,fact);
}
