/* I share checked transfer code, not old mutable histories or admission. */
#include "file_cyclic.h"
typedef struct {
    NvmFileCyclicStateInfo info;
    NvmFileFlowValue locals[NVM_FILE_FLOW_LOCALS], stack[NVM_FILE_FLOW_STACK];
    NvmFileFlowReference references[NVM_FILE_FLOW_REFERENCES];
    uint64_t regions[NVM_FILE_FLOW_REFERENCES];
} FileCyclicState;
typedef struct {
    FileCyclicState state;
    NvmFileCyclicVariant fact;
    bool processed;
} FileCyclicNode;
typedef struct {
    uint8_t count;
    uint16_t component;
    FileCyclicNode *nodes[NVM_FILE_CYCLIC_ALTERNATIVES];
} FileCyclicSite;
struct NvmFileCyclicReport {
    NvmFileCodePlan *plan;
    FileCyclicSite *sites;
    NvmFileCyclicSummary summary;
};
typedef struct { uint16_t instruction; uint8_t variant; } FileCyclicWork;
typedef struct {
    NvmFileBodyReport body;
    NvmFileFlowState transfer;
    FileCyclicState canonical;
    FileCyclicWork queue[NVM_FILE_CYCLIC_FUNCTION_PAIRS];
    uint32_t head, tail;
    uint64_t owners[NVM_FILE_FLOW_OWNERS];
    uint64_t reach[NVM_FILE_CODE_FUNCTION_INSTRUCTIONS][4];
    uint16_t todo[NVM_FILE_CODE_FUNCTION_INSTRUCTIONS];
} FileCyclicWorkspace;

static bool file_cyclic_room(NvmFileCyclicReport *r,size_t count,size_t width) {
    size_t bytes=r->summary.storage_peak;
    if(!file_code_budget(&bytes,count,width) || bytes>NVM_FILE_CYCLIC_BYTES)return false;
    r->summary.storage_peak=bytes;return true;
}
static bool file_cyclic_value_equal(NvmFileFlowValue a,NvmFileFlowValue b) {
    return value_same(a,b) && a.arm==b.arm;
}
static bool file_cyclic_equal(const FileCyclicState *a,const FileCyclicState *b) {
    if(a->info.locals!=b->info.locals || a->info.stack!=b->info.stack ||
       a->info.regions!=b->info.regions || a->info.owners!=b->info.owners ||
       a->info.references!=b->info.references)return false;
    for(uint16_t i=0;i<a->info.locals;i++)if(!file_cyclic_value_equal(a->locals[i],b->locals[i]))return false;
    for(uint16_t i=0;i<a->info.stack;i++)if(!file_cyclic_value_equal(a->stack[i],b->stack[i]))return false;
    for(uint16_t i=0;i<a->info.regions;i++)if(a->regions[i]!=b->regions[i])return false;
    for(uint16_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++)if(!reference_same(a->references[i],b->references[i]))return false;
    return true;
}
static NvmFileFlowStatus file_cyclic_value(FileCyclicWorkspace *w,NvmFileFlowValue v,
                                          NvmFileFlowValue *out,bool local) {
    if(v.type.mode && (!local || v.type.mode!=2 || v.type.category!=NVM_FILE_CATEGORY_FILE))
        return NVM_FILE_FLOW_INVALID;
    if(!v.initialized) {
        if(v.owner || v.type.mode)return NVM_FILE_FLOW_INVALID;
        v=initial_value(v.type,false,0);
    } else {
        if(!local && v.type.mode)return NVM_FILE_FLOW_INVALID;
        if(owned(v.type)!=(v.owner!=0))return NVM_FILE_FLOW_INVALID;
        if(result_type(v.type)) {
            if(v.arm!=NVM_FILE_FLOW_ARM_UNKNOWN && v.arm!=NVM_FILE_FLOW_ARM_OK &&
               v.arm!=NVM_FILE_FLOW_ARM_ERROR)return NVM_FILE_FLOW_INVALID;
        } else if(v.arm!=NVM_FILE_FLOW_ARM_NONE)return NVM_FILE_FLOW_INVALID;
    }
    if(v.owner) {
        uint16_t count=w->canonical.info.owners;
        if(count==NVM_FILE_FLOW_OWNERS)return NVM_FILE_FLOW_LIMIT;
        for(uint16_t i=0;i<count;i++)if(w->owners[i]==v.owner)return NVM_FILE_FLOW_INVALID;
        w->owners[count]=v.owner;v.owner=(uint64_t)count+1;
        w->canonical.info.owners++;
    }
    *out=v;return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus file_cyclic_canonicalize(FileCyclicWorkspace *w) {
    NvmFileFlowState *s=&w->transfer;FileCyclicState *c=&w->canonical;
    if(!s->reachable || s->stack_count>NVM_FILE_FLOW_STACK || s->region_count>NVM_FILE_FLOW_REFERENCES)
        return NVM_FILE_FLOW_INVALID;
    memset(c,0,sizeof *c);memset(w->owners,0,sizeof w->owners);
    c->info.locals=s->declarations->functions[s->function].locals;
    c->info.stack=s->stack_count;c->info.regions=s->region_count;
    for(uint16_t i=0;i<c->info.locals;i++) {
        if(!same(s->locals[i].type,s->declarations->locals[s->declarations->offsets[s->function]+i]))
            return NVM_FILE_FLOW_INVALID;
        NvmFileFlowStatus status=file_cyclic_value(w,s->locals[i],&c->locals[i],true);
        if(status!=NVM_FILE_FLOW_OK)return status;
    }
    for(uint16_t i=0;i<c->info.stack;i++) {
        if(!s->stack[i].initialized)return NVM_FILE_FLOW_INVALID;
        NvmFileFlowStatus status=file_cyclic_value(w,s->stack[i],&c->stack[i],false);
        if(status!=NVM_FILE_FLOW_OK)return status;
    }
    for(uint16_t i=0;i<s->region_count;i++) {
        if(!s->regions[i])return NVM_FILE_FLOW_INVALID;
        for(uint16_t j=0;j<i;j++)if(s->regions[i]==s->regions[j])return NVM_FILE_FLOW_INVALID;
        c->regions[i]=UINT64_C(513)+i;
    }
    for(uint16_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++) {
        NvmFileFlowReference ref=s->references[i];
        if(!ref.live)continue;
        NvmFileFlowDeclaration actual;
        if(!ref.owner || !ref.identity || !live_reference(s,i,&actual))return NVM_FILE_FLOW_INVALID;
        for(uint16_t j=0;j<i;j++)if(s->references[j].live &&
            (s->references[j].owner==ref.owner || s->references[j].identity==ref.identity))
            return NVM_FILE_FLOW_INVALID;
        if(ref.formal) {
            if(i>=s->declarations->functions[s->function].parameters || ref.local!=i || ref.region ||
               ref.owner!=UINT64_C(257)+i || ref.identity!=ref.owner)return NVM_FILE_FLOW_INVALID;
        } else {
            uint16_t owner=0,region=s->region_count;
            for(uint16_t j=0;j<c->info.owners;j++)if(w->owners[j]==ref.owner)owner=(uint16_t)(j+1);
            for(uint16_t j=0;j<s->region_count;j++)if(s->regions[j]==ref.region)region=j;
            if(!owner || region==s->region_count)return NVM_FILE_FLOW_INVALID;
            ref.owner=owner;ref.identity=UINT64_C(769)+i;ref.region=UINT64_C(513)+region;
        }
        c->references[i]=ref;c->info.references++;
    }
    for(uint16_t i=0;i<c->info.locals;i++)if(c->locals[i].type.mode &&
        (!c->references[i].live || !c->references[i].formal))return NVM_FILE_FLOW_INVALID;
    return NVM_FILE_FLOW_OK;
}
/* Each one-instruction scratch uses its own declaration allocation. Its label
 * allocator cannot modify the retained plan or an old logical state object. */
static void file_cyclic_scratch(FileCyclicWorkspace *w,NvmFileFlowDeclarations *scratch,
                                uint32_t function,const FileCyclicState *input) {
    memset(&w->transfer,0,sizeof w->transfer);
    NvmFileFlowState *s=&w->transfer;s->declarations=scratch;s->function=function;
    s->family=1;s->reachable=true;scratch->next_identity=1025;
    s->stack_count=input->info.stack;s->region_count=input->info.regions;
    memcpy(s->locals,input->locals,sizeof s->locals);memcpy(s->stack,input->stack,sizeof s->stack);
    memcpy(s->references,input->references,sizeof s->references);memcpy(s->regions,input->regions,sizeof s->regions);
}
static NvmFileFlowStatus file_cyclic_intern(NvmFileCyclicReport *r,FileCyclicWorkspace *w,
    uint32_t function,uint16_t instruction,uint8_t *ordinal) {
    FileCyclicSite *site=&r->sites[r->plan->starts[function]+instruction];
    for(uint8_t i=0;i<site->count;i++)if(file_cyclic_equal(&site->nodes[i]->state,&w->canonical)) {
        *ordinal=i;return NVM_FILE_FLOW_OK;
    }
    if(site->count==NVM_FILE_CYCLIC_ALTERNATIVES || w->tail==NVM_FILE_CYCLIC_FUNCTION_PAIRS ||
       r->summary.variants==NVM_FILE_CYCLIC_PAIRS || !file_cyclic_room(r,1,sizeof(FileCyclicNode)))
        return NVM_FILE_FLOW_LIMIT;
    FileCyclicNode *node=calloc(1,sizeof *node);if(!node)return NVM_FILE_FLOW_MEMORY;
    node->state=w->canonical;node->fact.input=w->canonical.info;
    node->fact.edge_variants[0]=node->fact.edge_variants[1]=NVM_FILE_CYCLIC_NO_VARIANT;
    uint8_t v=site->count;site->nodes[v]=node;site->count++;r->summary.variants++;
    w->queue[w->tail++]=(FileCyclicWork){instruction,v};*ordinal=v;
    return NVM_FILE_FLOW_OK;
}
/* Bounded reachability bitsets give SCC representatives without recursion.
 * A component is named by its least original instruction index. */
static void file_cyclic_components(NvmFileCyclicReport *r,FileCyclicWorkspace *w,uint32_t f) {
    uint16_t count=r->plan->functions[f].instruction_count;
    memset(w->reach,0,sizeof w->reach);
    for(uint16_t origin=0;origin<count;origin++) {
        uint16_t used=0;w->todo[used++]=origin;
        w->reach[origin][origin/64]|=UINT64_C(1)<<(origin%64);
        while(used) {
            uint16_t i=w->todo[--used];
            const NvmFileCodeInstruction *in=&r->plan->instructions[r->plan->starts[f]+i];
            for(uint8_t e=0;e<in->successor_count;e++) {
                uint16_t next=in->successors[e];uint64_t bit=UINT64_C(1)<<(next%64);
                if(!(w->reach[origin][next/64]&bit)) {
                    w->reach[origin][next/64]|=bit;w->todo[used++]=next;
                }
            }
        }
    }
    for(uint16_t i=0;i<count;i++) {
        uint16_t representative=i;
        for(uint16_t j=0;j<i;j++)if((w->reach[i][j/64]&(UINT64_C(1)<<(j%64))) &&
            (w->reach[j][i/64]&(UINT64_C(1)<<(i%64)))){representative=j;break;}
        r->sites[r->plan->starts[f]+i].component=representative;
    }
}
static NvmFileFlowStatus file_cyclic_function_analyze(NvmFileCyclicReport *r,FileCyclicWorkspace *w,
                                                    NvmFileFlowDeclarations *scratch,uint32_t f) {
    w->head=w->tail=0;file_cyclic_components(r,w,f);
    memset(&w->canonical,0,sizeof w->canonical);
    const NvmFileFlowFunction *fn=&r->plan->functions[f].declaration;
    w->canonical.info.locals=fn->locals;
    uint64_t next=1;
    for(uint16_t i=0;i<fn->locals;i++) {
        NvmFileFlowDeclaration d=scratch->locals[scratch->offsets[f]+i];bool initialized=i<fn->parameters;
        w->canonical.locals[i]=initial_value(d,initialized,initialized && owned(d)?next++:0);
        if(initialized && d.mode)w->canonical.references[i]=(NvmFileFlowReference){true,true,i,
            UINT64_C(257)+i,UINT64_C(257)+i,0};
    }
    file_cyclic_scratch(w,scratch,f,&w->canonical);
    NvmFileFlowStatus status=file_cyclic_canonicalize(w);if(status!=NVM_FILE_FLOW_OK)return status;
    uint8_t ordinal;status=file_cyclic_intern(r,w,f,0,&ordinal);if(status!=NVM_FILE_FLOW_OK)return status;
    bool returned=false;
    while(w->head<w->tail) {
        FileCyclicWork item=w->queue[w->head++];
        FileCyclicSite *site=&r->sites[r->plan->starts[f]+item.instruction];
        FileCyclicNode *node=site->nodes[item.variant];
        if(node->processed || r->summary.transfers==NVM_FILE_CYCLIC_PAIRS)return NVM_FILE_FLOW_LIMIT;
        r->summary.transfers++;
        file_cyclic_scratch(w,scratch,f,&node->state);
        const NvmFileCodeInstruction *in=&r->plan->instructions[r->plan->starts[f]+item.instruction];
        NvmFileBodyInstruction *fact=&node->fact.body;
        fact->reachable=true;fact->input_stack=w->transfer.stack_count;
        fact->cleanup_local=NVM_FILE_FLOW_NO_REFERENCE;fact->pending_checks=NVM_FILE_FLOW_CHECK_CLEANUP;
        status=file_body_transfer(&w->body,&w->transfer,in,fact);if(status!=NVM_FILE_FLOW_OK)return status;
        fact->output_stack=w->transfer.stack_count;returned|=fact->exit_checked;
        status=file_cyclic_canonicalize(w);if(status!=NVM_FILE_FLOW_OK)return status;
        node->fact.output=w->canonical.info;
        NvmFileFlowArm prior=NVM_FILE_FLOW_ARM_NONE;
        uint16_t refine=0;
        if(in->decoded.opcode==OP_FILE_RESULT_BRANCH) {
            refine=in->decoded.operands[0].u16;
            NvmFileFlowValue value=w->transfer.locals[refine];
            if(!value.initialized || !result_type(value.type) || value.type.mode)return NVM_FILE_FLOW_INVALID;
            prior=value.arm;fact->refinement=true;
        }
        for(uint8_t edge=0;edge<in->successor_count;edge++) {
            if(fact->refinement) {
                NvmFileFlowArm arm=edge?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK;
                if(prior!=NVM_FILE_FLOW_ARM_UNKNOWN && prior!=arm)continue;
                w->transfer.locals[refine].arm=arm;
            }
            if(r->summary.edges==NVM_FILE_CYCLIC_EDGES)return NVM_FILE_FLOW_LIMIT;
            r->summary.edges++;
            status=file_cyclic_canonicalize(w);if(status!=NVM_FILE_FLOW_OK)return status;
            status=file_cyclic_intern(r,w,f,in->successors[edge],&ordinal);
            if(status!=NVM_FILE_FLOW_OK)return status;
            node->fact.edge_variants[edge]=ordinal;node->fact.edge_mask|=(uint8_t)(1u<<edge);
        }
        node->processed=true;
    }
    if(!returned)return NVM_FILE_FLOW_INVALID;
    w->body.checked[f]=true;return NVM_FILE_FLOW_OK;
}
void nvm_file_cyclic_free(NvmFileCyclicReport *r) {
    if(!r)return;
    if(r->sites)for(uint32_t i=0;i<r->plan->instruction_count;i++)
        for(uint8_t j=0;j<r->sites[i].count;j++)free(r->sites[i].nodes[j]);
    free(r->sites);nvm_file_code_free(r->plan);free(r);
}
static bool file_cyclic_complete(const NvmFileCyclicReport *r) {
    uint32_t count=0;
    for(uint32_t f=0;f<r->plan->count;f++)for(uint16_t i=0;i<r->plan->functions[f].instruction_count;i++) {
        const NvmFileCodeInstruction *in=&r->plan->instructions[r->plan->starts[f]+i];
        const FileCyclicSite *site=&r->sites[r->plan->starts[f]+i];
        if(site->count>NVM_FILE_CYCLIC_ALTERNATIVES || site->component>=r->plan->functions[f].instruction_count)
            return false;
        for(uint8_t v=0;v<site->count;v++) {
            const FileCyclicNode *n=site->nodes[v];
            if(!n || !n->processed || n->fact.body.input_stack!=n->fact.input.stack ||
               n->fact.body.output_stack!=n->fact.output.stack ||
               (n->fact.edge_mask>>in->successor_count))return false;
            for(uint8_t e=0;e<2;e++) {
                if(n->fact.edge_mask&(1u<<e)) {
                    if(e>=in->successor_count || in->successors[e]>=r->plan->functions[f].instruction_count ||
                       n->fact.edge_variants[e]>=r->sites[r->plan->starts[f]+in->successors[e]].count)return false;
                } else if(n->fact.edge_variants[e]!=NVM_FILE_CYCLIC_NO_VARIANT)return false;
            }
            count++;
        }
    }
    return count==r->summary.variants && count==r->summary.transfers;
}
NvmFileFlowStatus nvm_file_cyclic_analyze(const NvmModule *m,NvmFileCyclicReport **out) {
    if(!out)return NVM_FILE_FLOW_INVALID;
    uint32_t instructions;size_t bytes;
    NvmFileFlowStatus status=file_code_preflight(m,&instructions,&bytes);
    if(status!=NVM_FILE_FLOW_OK)return status;
    size_t locals=0;
    for(uint32_t i=0;i<m->function_count;i++)locals+=m->functions[i].local_count;
    size_t declaration_bytes=sizeof(NvmFileFlowDeclarations)+locals*sizeof(NvmFileFlowDeclaration);
    if(!file_code_budget(&bytes,1,sizeof(NvmFileCyclicReport)) ||
       !file_code_budget(&bytes,instructions,sizeof(FileCyclicSite)) ||
       !file_code_budget(&bytes,1,sizeof(FileCyclicWorkspace)) ||
       !file_code_budget(&bytes,1,declaration_bytes) || bytes>NVM_FILE_CYCLIC_BYTES)return NVM_FILE_FLOW_LIMIT;
    NvmFileCodePlan *p=NULL;status=file_code_prepare_mode(m,&p,false);
    if(status!=NVM_FILE_FLOW_OK)return status;
    NvmFileCyclicReport *r=calloc(1,sizeof *r);
    if(!r){nvm_file_code_free(p);return NVM_FILE_FLOW_MEMORY;}
    r->plan=p;r->summary=(NvmFileCyclicSummary){NVM_FILE_CYCLIC_REVISION,p->count,instructions,0,0,0,bytes,false};
    r->sites=calloc(instructions,sizeof *r->sites);
    FileCyclicWorkspace *w=calloc(1,sizeof *w);
    NvmFileFlowDeclarations *scratch=malloc(declaration_bytes);
    if(!r->sites || !w || !scratch){status=NVM_FILE_FLOW_MEMORY;goto done;}
    memcpy(scratch,p->declarations,declaration_bytes);scratch->references=1;
    w->body.plan=p;
    for(uint32_t rank=0;rank<p->count;rank++) {
        status=file_cyclic_function_analyze(r,w,scratch,p->function_order[rank]);
        if(status!=NVM_FILE_FLOW_OK)break;
    }
    if(status==NVM_FILE_FLOW_OK && !file_cyclic_complete(r))status=NVM_FILE_FLOW_INVALID;
done:
    /* The scratch borrows the report-owned nominal plan; it never releases it. */
    free(scratch);free(w);
    if(status!=NVM_FILE_FLOW_OK){nvm_file_cyclic_free(r);return status;}
    *out=r;return NVM_FILE_FLOW_OK;
}
static const FileCyclicSite *file_cyclic_site(const NvmFileCyclicReport *r,uint32_t f,uint16_t i) {
    if(!r || f>=r->plan->count || i>=r->plan->functions[f].instruction_count)return NULL;
    return &r->sites[r->plan->starts[f]+i];
}
static const FileCyclicNode *file_cyclic_node(const NvmFileCyclicReport *r,uint32_t f,uint16_t i,uint8_t v) {
    const FileCyclicSite *site=file_cyclic_site(r,f,i);
    return site && v<site->count?site->nodes[v]:NULL;
}
bool nvm_file_cyclic_summary(const NvmFileCyclicReport *r,NvmFileCyclicSummary *out) {
    if(!r || !out)return false;
    *out=r->summary;return true;
}
bool nvm_file_cyclic_function(const NvmFileCyclicReport *r,uint32_t f,NvmFileCodeFunction *out) {
    return r && nvm_file_code_function(r->plan,f,out);
}
bool nvm_file_cyclic_local(const NvmFileCyclicReport *r,uint32_t f,uint16_t l,NvmFileFlowDeclaration *out) {
    return r && nvm_file_code_local(r->plan,f,l,out);
}
bool nvm_file_cyclic_instruction(const NvmFileCyclicReport *r,uint32_t f,uint16_t i,NvmFileCodeInstruction *out) {
    return r && nvm_file_code_instruction(r->plan,f,i,out);
}
bool nvm_file_cyclic_component(const NvmFileCyclicReport *r,uint32_t f,uint16_t i,uint16_t *out) {
    const FileCyclicSite *site=file_cyclic_site(r,f,i);if(!site || !out)return false;
    *out=site->component;return true;
}
bool nvm_file_cyclic_variant_count(const NvmFileCyclicReport *r,uint32_t f,uint16_t i,uint8_t *out) {
    const FileCyclicSite *site=file_cyclic_site(r,f,i);if(!site || !out)return false;
    *out=site->count;return true;
}
bool nvm_file_cyclic_variant(const NvmFileCyclicReport *r,uint32_t f,uint16_t i,uint8_t v,NvmFileCyclicVariant *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out)return false;
    *out=node->fact;return true;
}
bool nvm_file_cyclic_input_local(const NvmFileCyclicReport *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,NvmFileFlowValue *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out || l>=node->state.info.locals)return false;
    *out=node->state.locals[l];return true;
}
bool nvm_file_cyclic_input_stack(const NvmFileCyclicReport *r,uint32_t f,uint16_t i,uint8_t v,uint16_t s,NvmFileFlowValue *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out || s>=node->state.info.stack)return false;
    *out=node->state.stack[s];return true;
}
bool nvm_file_cyclic_input_reference(const NvmFileCyclicReport *r,uint32_t f,uint16_t i,uint8_t v,uint16_t ref,NvmFileFlowReference *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out || ref>=NVM_FILE_FLOW_REFERENCES)return false;
    *out=node->state.references[ref];return true;
}
bool nvm_file_cyclic_input_region(const NvmFileCyclicReport *r,uint32_t f,uint16_t i,uint8_t v,uint16_t region,uint64_t *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out || region>=node->state.info.regions)return false;
    *out=node->state.regions[region];return true;
}
bool nvm_file_cyclic_type(const NvmFileCyclicReport *r,uint32_t ordinal,NvmFileNominalLayout *out) {
    return r && nvm_file_nominal_type(r->plan->declarations->nominal,ordinal,out);
}
bool nvm_file_cyclic_import(const NvmFileCyclicReport *r,uint32_t ordinal,uint32_t *out) {
    return r && nvm_file_nominal_import(r->plan->declarations->nominal,ordinal,out);
}
