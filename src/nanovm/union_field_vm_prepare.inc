/* Private storage preparation only; no executor or heap-policy change. */
#include "union_field_runtime_private.h"
#include "../nanoisa/union_field_vm_image_private.h"
#define VM_UF_ROOT_CAPACITY (VM_MAX_FRAMES*512u+1u)
_Static_assert(VM_MAX_FRAMES==NVM_RECORD_ARRAY_EXECUTION_FRAMES,"unchanged frame bound");
struct VmUnionFieldPrivate {
    VmState vm;
    NvmUnionFieldImage *image;
    NvmRecordArrayExecutionCounts counts;
    NvmRecordArrayExecutionFunction functions[256];
    NvmDeclarationLayout layouts[2];
    NvmUnionVariantFact variants[2];
    NvmV2LayoutField fields[2];
    uint32_t record_layout,union_layout;
    NvmUnionFieldExecutionInstruction *instructions;
    VmUnionFieldPreparation statistics;
    bool prepared;
};
typedef struct {uint64_t bytes,work;NvmArrayEligibilityStatus status;} VmUfBudget;
static bool vm_uf_add(VmUfBudget *b,uint64_t bytes,uint64_t work) {
    if(b->bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES || bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES-b->bytes ||
       b->work>NVM_RECORD_ARRAY_EXECUTION_STEPS || work>NVM_RECORD_ARRAY_EXECUTION_STEPS-b->work) {
        b->status=NVM_ARRAY_LIMIT;return false;
    }
    b->bytes+=bytes;b->work+=work;return true;
}
static bool vm_uf_term(VmUfBudget *b,uint64_t n,uint64_t width,bool memory) {
    if(width && (n>UINT64_MAX/width || (memory && n>SIZE_MAX/width))) {
        b->status=NVM_ARRAY_LIMIT;return false;
    }
    return vm_uf_add(b,memory?n*width:0,n*width);
}
static bool vm_uf_measure(const NvmUnionFieldExecutionPlan *source,VmUfBudget *b,
                          NvmRecordArrayExecutionCounts *out) {
    NvmRecordArrayExecutionCounts c;NvmUnionFieldProofCounts proof;
    /* Four-probe getters and fixed count guards precede every variable scan. */
    if(!vm_uf_add(b,0,64) || !nvm_union_field_execution_counts(source,&c) ||
       !nvm_union_field_execution_proof_counts(source,&proof))return false;
    if(!c.functions || c.functions>256 || c.instructions>65536 || c.globals>256 ||
       c.strings>NVM_MAX_STRINGS || c.records!=1 || c.frame_limit!=VM_MAX_FRAMES ||
       c.entry>=c.functions || (c.initializer!=UINT32_MAX && c.initializer>=c.functions) ||
       proof.origins>64 || proof.functions!=c.functions || proof.globals!=c.globals)return false;
    if(!vm_uf_term(b,c.functions,32,false) || !vm_uf_term(b,c.strings,8,false))return false;
    uint64_t declaration_width=sizeof(NvmDeclarationLayout);
    if(declaration_width<sizeof(NvmUnionVariantFact))declaration_width=sizeof(NvmUnionVariantFact);
    if(declaration_width<sizeof(NvmV2LayoutField))declaration_width=sizeof(NvmV2LayoutField);
    if(declaration_width<sizeof(uint32_t))declaration_width=sizeof(uint32_t);
    if(!vm_uf_term(b,1,sizeof(VmUnionFieldPrivate),true) ||
       !vm_uf_term(b,c.instructions,sizeof(NvmUnionFieldExecutionInstruction),true) ||
       !vm_uf_term(b,VM_UF_ROOT_CAPACITY,sizeof(NanoValue),true) ||
       !vm_uf_term(b,c.globals,sizeof(NanoValue),true) ||
       !vm_uf_term(b,c.functions,sizeof(VmDecodedFunction)+sizeof(VmDispatchFunction),true) ||
       !vm_uf_term(b,1,65536,true) || /* Provisional automatic scratch, not heap. */
       !vm_uf_add(b,0,64+256) ||
       !vm_uf_term(b,c.functions,16+sizeof(NvmRecordArrayExecutionFunction)+64,false) ||
       !vm_uf_term(b,c.instructions,4+sizeof(NvmUnionFieldExecutionInstruction),false) ||
       !vm_uf_term(b,8,4+declaration_width,false))return false;
    uint32_t total=0;
    for(uint32_t f=0;f<c.functions;f++) {
        NvmRecordArrayExecutionFunction fn;
        if(!nvm_union_field_execution_function(source,f,&fn) || fn.signature.local_count>256 ||
           fn.maximum_stack>256 || fn.signature.arity>fn.signature.local_count || fn.signature.upvalue_count ||
           fn.signature.result_count>1 || fn.instruction_start!=total || fn.instruction_count>c.instructions-total ||
           fn.signature.code_length>16u*1024u*1024u)return false;
        total+=fn.instruction_count;
        uint64_t length=fn.signature.code_length;
        if(!vm_uf_term(b,length+1,1+2*sizeof(uint32_t),true) ||
           !vm_uf_term(b,fn.instruction_count,sizeof(VmDispatchInstruction),true) ||
           !vm_uf_add(b,0,4*length+64u*(uint64_t)fn.instruction_count+2*(length+1)) ||
           !vm_uf_term(b,fn.instruction_count,256+sizeof(NvmUnionFieldExecutionInstruction)+
                sizeof(VmDecodedInstruction)+sizeof(VmDispatchInstruction),false))return false;
        for(uint64_t cap=16;fn.instruction_count;cap*=2) {
            if(!vm_uf_term(b,cap,sizeof(VmDecodedInstruction),true))return false;
            if(cap>=fn.instruction_count)break;
        }
    }
    if(total!=c.instructions)return false;
    if(!vm_uf_term(b,c.strings,sizeof(VmString *),true) ||
       !vm_uf_term(b,256,sizeof(VmString *),true))return false;
    /* Growth happens before inserting the next distinct key. S==threshold
     * alone does not trigger it. This preserves the actual zero-string case. */
    for(uint64_t cap=256;c.strings && ((uint64_t)c.strings-1)*4>=cap*3;cap*=2) {
        if(!vm_uf_add(b,0,cap+c.strings) || !vm_uf_term(b,cap*2,sizeof(VmString *),true))return false;
    }
    for(uint32_t s=0;s<c.strings;s++) {
        uint32_t length;
        if(!nvm_union_field_execution_size(source,NVM_RA_SNAPSHOT_STRING,s,&length) ||
           !vm_uf_term(b,1,(uint64_t)sizeof(VmString)+length+1,true) ||
           !vm_uf_term(b,(uint64_t)length+1,(uint64_t)s+8,false) || !vm_uf_add(b,0,16))return false;
    }
    *out=c;return true;
}
static bool vm_uf_sensitive(uint8_t op) {
    return op==OP_STRUCT_NEW || op==OP_STRUCT_LITERAL || op==OP_STRUCT_GET || op==OP_STRUCT_SET ||
        op==OP_AGG_PACK || op==OP_AGG_GET || op==OP_AGG_SET || op==OP_AGG_TAG || op==OP_HALT;
}
static bool vm_uf_recipe(uint8_t op,NvmRecordArrayExecutionInstruction *r) {
    if(!vm_uf_sensitive(op))return nvm_record_array_opcode_supported(op) && vm_rm_recipe(op,r);
    r->recipe=NVM_UF_INVALID_BASE_RECIPE;r->obligations=NVM_RA_FIRST_ERROR_CLEANUP;
    if(op==OP_HALT)return true;
    r->obligations|=NVM_RA_CHECK_TAGS|NVM_RA_CHECK_NOMINAL;
    if(op==OP_STRUCT_NEW || op==OP_STRUCT_LITERAL || op==OP_AGG_PACK)
        r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;
    else if(op!=OP_AGG_TAG)r->obligations|=NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;
    return true;
}
static bool vm_uf_value_same(NvmRecordValueOrigins a,NvmRecordValueOrigins b) {
    return a.tags==b.tags && a.origins==b.origins && a.unknown==b.unknown;
}
/* Image rows already passed independent reproof and full site correspondence.
 * I compare copied site fields to that immutable owner; no row is repaired. */
static bool vm_uf_site_same(const NvmUnionFieldSite *a,const NvmUnionFieldSite *b) {
    return a->function==b->function && a->pc==b->pc && a->nominal==b->nominal && a->layout==b->layout &&
        a->member==b->member && a->variant==b->variant && a->operation==b->operation && a->state==b->state &&
        !a->reserved[0] && !a->reserved[1] && !a->reserved[2] &&
        !b->reserved[0] && !b->reserved[1] && !b->reserved[2] &&
        vm_uf_value_same(a->receiver,b->receiver) && vm_uf_value_same(a->input,b->input) && vm_uf_value_same(a->result,b->result);
}
static bool vm_uf_layouts(VmUnionFieldPrivate *p) {
    if(!nvm_union_field_image_nominal(p->image,TAG_STRUCT,0,&p->record_layout) ||
       !nvm_union_field_image_nominal(p->image,TAG_UNION,0,&p->union_layout) ||
       p->record_layout!=1 || p->union_layout!=0)return false;
    for(uint32_t i=0;i<2;i++) {
        if(!nvm_union_field_image_layout(p->image,i,&p->layouts[i]) ||
           !nvm_union_field_image_variant(p->image,p->union_layout,(uint16_t)i,&p->variants[i]) ||
           !nvm_union_field_image_field(p->image,i,0,&p->fields[i]))return false;
    }
    if(p->layouts[0].kind!=NVM_V2_LAYOUT_UNION || p->layouts[1].kind!=NVM_V2_LAYOUT_STRUCT ||
       p->layouts[0].flags!=NVM_LAYOUT_COMPLETE || p->layouts[1].flags!=NVM_LAYOUT_COMPLETE ||
       p->layouts[0].fields!=1 || p->layouts[1].fields!=1 ||
       p->fields[0].type_tag!=TAG_INT || p->fields[0].nested_idx!=NVM_V2_NO_INDEX ||
       p->fields[1].type_tag!=TAG_UNION || p->fields[1].nested_idx!=p->union_layout)return false;
    return p->variants[0].layout==p->union_layout && p->variants[1].layout==p->union_layout &&
        p->variants[0].field_offset==0 && p->variants[1].field_offset==p->variants[0].field_count &&
        p->variants[0].field_count+p->variants[1].field_count==1 &&
        p->variants[0].field_count<=1 && p->variants[1].field_count<=1;
}

/* All traversal work below is prepaid by vm_uf_measure before image allocation. */
static bool vm_uf_coverage(VmUnionFieldPrivate *p) {
    const NvmModule *m=nvm_union_field_image_module_internal(p->image);
    const VmDecodedModule *dm=&p->vm.decoded_module;
    const VmDispatchModule *dispatch=&p->vm.dispatch_module;
    if(p->counts.functions>256 || p->counts.instructions>65536 || !m || m->function_count!=p->counts.functions ||
       p->vm.dispatch_profile.fuse_load_local_field || !dm->functions || !dispatch->functions ||
       (p->counts.instructions && !p->instructions) || m!=p->vm.module || m!=p->vm.root_module ||
       dm->function_count!=p->counts.functions || dispatch->function_count!=p->counts.functions)return false;
    uint32_t supported=0;
    for(uint32_t op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction expected={0};
        bool active=vm_uf_recipe((uint8_t)op,&expected)!=0;
        if(active!=(nvm_record_array_opcode_supported((uint8_t)op) || op==OP_AGG_TAG || op==OP_HALT))return false;
        supported+=active;
    }
    if(supported!=96)return false;
    uint32_t total=0;
    for(uint32_t fi=0;fi<p->counts.functions;fi++) {
        const NvmRecordArrayExecutionFunction *fn=&p->functions[fi];
        const VmDecodedFunction *decoded=&dm->functions[fi];
        const VmDispatchFunction *body=&dispatch->functions[fi];
        NvmRecordArrayExecutionFunction image_fn;
        if(!nvm_union_field_image_function(p->image,fi,&image_fn) ||
           !vm_rm_signature_same(&fn->signature,&image_fn.signature) ||
           fn->instruction_start!=image_fn.instruction_start || fn->instruction_count!=image_fn.instruction_count ||
           fn->maximum_stack!=image_fn.maximum_stack || fn->parameter_tags_present!=image_fn.parameter_tags_present)return false;
        if(!vm_rm_signature_same(&fn->signature,&m->functions[fi]) || fn->instruction_start!=total || total>p->counts.instructions || fn->instruction_count>p->counts.instructions-total ||
           decoded->instruction_count!=fn->instruction_count || body->instruction_count!=fn->instruction_count ||
           decoded->code_size!=fn->signature.code_length || body->code_size!=fn->signature.code_length ||
           (fn->instruction_count && (!decoded->instructions || !body->instructions)) ||
           !decoded->boundaries || !decoded->instruction_indices || !body->offset_to_index ||
           !decoded->boundaries[decoded->code_size])return false;
        uint32_t expected_index=0;
        for(uint32_t pc=0;pc<=decoded->code_size;pc++) {
            bool start=expected_index<fn->instruction_count &&
                p->instructions[total+expected_index].base.pc==pc;
            uint32_t encoded=start?expected_index+1:0;
            if(decoded->boundaries[pc]!=(uint8_t)(start || pc==decoded->code_size) ||
               decoded->instruction_indices[pc]!=encoded || body->offset_to_index[pc]!=encoded)return false;
            expected_index+=start;
        }
        for(uint32_t i=0;i<fn->instruction_count;i++) {
            const NvmRecordArrayExecutionInstruction *fact=&p->instructions[total+i].base;
            const VmDecodedInstruction *in=&decoded->instructions[i];
            const VmDispatchInstruction *row=&body->instructions[i];
            NvmRecordArrayExecutionInstruction expected={0};
            const InstructionInfo *info=isa_get_info(fact->opcode);
            if(fact->operand_count>MAX_OPERANDS || fact->successor_count>2 || !info || !vm_uf_recipe(fact->opcode,&expected) || fact->recipe!=expected.recipe ||
               fact->obligations!=expected.obligations || fact->function!=fi ||
               fact->pc!=in->byte_offset || fact->next_pc!=in->next_byte_offset ||
               fact->pc!=row->byte_offset || fact->next_pc!=row->next_byte_offset ||
               !vm_rm_instruction_same(fact,&in->instruction) || !vm_rm_instruction_same(fact,&row->instruction) ||
               in->call_handle.resolved || row->call_handle.resolved || in->call_handle.module || row->call_handle.module ||
               in->call_handle.function || row->call_handle.function || in->call_handle.function_index || row->call_handle.function_index ||
               row->super_op!=VM_SUPER_NONE || row->super_operand)return false;
            NvmUnionFieldExecutionInstruction image_row;
            if(!nvm_union_field_image_instruction(p->image,total+i,&image_row) ||
               !vm_uf_site_same(&p->instructions[total+i].site,&image_row.site))return false;
            const NvmRecordArrayExecutionInstruction *original=&image_row.base;
#define VM_UF_EQ(field) do {if(fact->field!=original->field)return false;}while(0)
            VM_UF_EQ(function);VM_UF_EQ(pc);VM_UF_EQ(next_pc);VM_UF_EQ(callee);
            VM_UF_EQ(pops);VM_UF_EQ(pushes);VM_UF_EQ(obligations);VM_UF_EQ(opcode);
            VM_UF_EQ(recipe);VM_UF_EQ(successor_count);VM_UF_EQ(operand_count);
            for(unsigned j=0;j<MAX_OPERANDS;j++){VM_UF_EQ(operand_types[j]);VM_UF_EQ(operand_bits[j]);}
            for(unsigned j=0;j<2;j++)VM_UF_EQ(successors[j]);
#undef VM_UF_EQ
            uint32_t next=i+1<fn->instruction_count?i+1:VM_DISPATCH_NO_INDEX;
            if(row->next_index!=next)return false;
            int pops=info->pop_count,pushes=info->push_count;
            uint32_t callee=UINT32_MAX,resolved=UINT32_MAX,branch=UINT32_MAX,target=UINT32_MAX;
            uint8_t edges=0;uint32_t successors[2]={0,0};
            switch(fact->opcode) {
            case OP_ARR_LITERAL:case OP_STRUCT_LITERAL:pops=in->instruction.operands[1].u16;pushes=1;break;
            case OP_AGG_PACK:pops=in->instruction.operands[3].u16;pushes=1;break;
            case OP_CALL: case OP_TAIL_CALL:
                callee=in->instruction.operands[0].u32;
                if(callee>=p->counts.functions)return false;
                pops=m->functions[callee].arity;pushes=fact->opcode==OP_TAIL_CALL?0:m->functions[callee].result_count;resolved=callee;break;
            case OP_RET:pops=fn->signature.result_count;pushes=0;break;
            case OP_HALT:pops=0;pushes=0;break;
            case OP_JMP:case OP_JMP_TRUE:case OP_JMP_FALSE: {
                int64_t dest=(int64_t)fact->pc+in->instruction.operands[0].i32;
                if(dest<0 || (uint64_t)dest>decoded->code_size)return false;
                successors[edges++]=(uint32_t)dest;
                resolved=fn->signature.code_offset+(uint32_t)dest;target=resolved;
                branch=dest==decoded->code_size?UINT32_MAX:body->offset_to_index[dest]-1;
                break;
            }
            default:break;
            }
            if(fact->opcode!=OP_JMP && fact->opcode!=OP_RET && fact->opcode!=OP_TAIL_CALL && fact->opcode!=OP_HALT)successors[edges++]=fact->next_pc;
            if(pops<0 || pushes<0 || fact->pops!=pops || fact->pushes!=pushes || fact->callee!=callee ||
               fact->successor_count!=edges || in->resolved_target!=resolved || row->call_target!=callee ||
               row->branch_target!=branch || row->branch_target_offset!=target)return false;
            for(uint8_t j=0;j<edges;j++)if(fact->successors[j]!=successors[j])return false;
        }
        total+=fn->instruction_count;
    }
    return total==p->counts.instructions;
}
void vm_union_field_private_destroy(VmUnionFieldPrivate *p) {
    if(!p || !pthread_equal(p->vm.owner_thread,pthread_self()))return;
    vm_destroy(&p->vm);free(p->instructions);
    nvm_union_field_image_free(p->image);free(p);
}
NvmArrayEligibilityResult vm_union_field_private_prepare(const NvmModule *module,VmUnionFieldPrivate **out) {
    NvmArrayEligibilityResult result={NVM_ARRAY_INVALID,0,0,"I require a stable module and private preparation output."};
    if(!module || !out)return result;
    NvmUnionFieldExecutionPlan *source=NULL;
    result=nvm_prepare_union_field_execution(module,&source);
    if(result.status!=NVM_ARRAY_ELIGIBLE)return result;
    VmUfBudget budget={0,0,NVM_ARRAY_INVALID};NvmRecordArrayExecutionCounts counts;
    if(!vm_uf_measure(source,&budget,&counts)) {
        nvm_union_field_execution_free(source);result.status=budget.status;
        snprintf(result.message,sizeof result.message,"I require bounded union preparation storage and correspondence.");return result;
    }
    NvmUnionFieldImage *image=NULL;
    result=nvm_union_field_image_copy_reserved(source,budget.bytes,budget.work,&image);
    nvm_union_field_execution_free(source);source=NULL;
    if(result.status!=NVM_ARRAY_ELIGIBLE)return result;
    VmUnionFieldPrivate *p=calloc(1,sizeof *p);
    if(!p) {
        nvm_union_field_image_free(image);result.status=NVM_ARRAY_MEMORY;
        snprintf(result.message,sizeof result.message,"I could not allocate my private union preparation owner.");return result;
    }
    p->image=image;p->counts=counts;p->vm.owner_thread=pthread_self();
    p->vm.cop_in_fd=-1;p->vm.cop_out_fd=-1;p->vm.cop_pid=-1;
    p->vm.cop_sig_send_fd=-1;p->vm.cop_sig_recv_fd=-1;
    const NvmModule *m=nvm_union_field_image_module_internal(image);
    p->vm.module=m;p->vm.root_module=m;
    NvmArrayEligibilityStatus failure=NVM_ARRAY_INVALID;
    char error[VM_DECODE_ERROR_SIZE]={0};
    NvmUnionFieldImageCost cost;
    if(!m || !nvm_union_field_image_cost(image,&cost))goto invalid;
    for(uint32_t f=0;f<counts.functions;f++)
        if(!nvm_union_field_image_function(image,f,&p->functions[f]))goto invalid;
    if(counts.instructions) {
        p->instructions=calloc(counts.instructions,sizeof *p->instructions);
        if(!p->instructions){failure=NVM_ARRAY_MEMORY;goto invalid;}
    }
    for(uint32_t i=0;i<counts.instructions;i++)
        if(!nvm_union_field_image_instruction(image,i,&p->instructions[i]))goto invalid;
    if(!vm_uf_layouts(p))goto invalid;
    if(!vm_decode_module(m,&p->vm.decoded_module,error)) {
        failure=strstr(error,"allocation failed")?NVM_ARRAY_MEMORY:NVM_ARRAY_INVALID;goto invalid;
    }
    p->vm.decoded_module_valid=true;p->vm.dispatch_profile=vm_dispatch_profile_none();
    if(!vm_dispatch_build_module(&p->vm.decoded_module,p->vm.dispatch_profile,&p->vm.dispatch_module,error)) {
        failure=strstr(error,"allocation failed")?NVM_ARRAY_MEMORY:NVM_ARRAY_INVALID;goto invalid;
    }
    p->vm.dispatch_module_valid=true;
    if(!vm_uf_coverage(p))goto invalid;
    p->vm.stack=calloc(VM_UF_ROOT_CAPACITY,sizeof *p->vm.stack);
    if(!p->vm.stack){failure=NVM_ARRAY_MEMORY;goto invalid;}
    p->vm.stack_capacity=VM_UF_ROOT_CAPACITY;
    if(counts.globals) {
        p->vm.globals=calloc(counts.globals,sizeof *p->vm.globals);
        if(!p->vm.globals){failure=NVM_ARRAY_MEMORY;goto invalid;}
    }
    p->vm.global_count=counts.globals;p->vm.global_capacity=counts.globals;
    vm_heap_init(&p->vm.heap);
    if(!p->vm.heap.intern_buckets || !vm_module_constants_build(&p->vm,m,&p->vm.module_constants)) {
        failure=NVM_ARRAY_MEMORY;goto invalid;
    }
    p->statistics=(VmUnionFieldPreparation){budget.bytes,budget.work,
        cost.preparation_bytes_reserved,cost.preparation_work_reserved,
        counts.peak_bytes_reserved,counts.work_reserved,sizeof *p,sizeof *p->instructions,
        counts.functions,counts.instructions,counts.globals,VM_UF_ROOT_CAPACITY};
    p->prepared=true;*out=p;
    snprintf(result.message,sizeof result.message,"I prepared independent union storage without execution admission.");
    return result;
invalid:
    vm_union_field_private_destroy(p);result.status=failure;
    snprintf(result.message,sizeof result.message,"I could not prepare independent union storage and correspondence: %.120s",
        error[0]?error:"private storage or copied rows failed");return result;
}
bool vm_union_field_private_preparation(const VmUnionFieldPrivate *p,VmUnionFieldPreparation *out) {
    if(!p || !out || !p->prepared || !pthread_equal(p->vm.owner_thread,pthread_self()))return false;
    *out=p->statistics;return true;
}
