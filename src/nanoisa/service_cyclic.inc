/* I share checked transfer code, not old mutable histories or admission. */
typedef struct {
    SERVICE_TYPE(CyclicStateInfo) info;
    SERVICE_TYPE(FlowValue) locals[SERVICE_CONST(FLOW_LOCALS)], stack[SERVICE_CONST(FLOW_STACK)];
    SERVICE_TYPE(FlowReference) references[SERVICE_CONST(FLOW_REFERENCES)];
    uint64_t regions[SERVICE_CONST(FLOW_REFERENCES)];
} FileCyclicState;
typedef struct {
    SERVICE_TYPE(CyclicStateInfo) info;
    SERVICE_TYPE(FlowValue) *locals, *stack;
    SERVICE_TYPE(FlowReference) *references;
    uint64_t *regions;
    uint16_t reference_extent;
} FileCyclicStoredState;
typedef struct {
    FileCyclicStoredState state;
    SERVICE_TYPE(CyclicVariant) fact;
    bool processed;
} FileCyclicNode;
typedef struct {
    uint8_t count;
    uint16_t component;
    FileCyclicNode *nodes[SERVICE_CONST(CYCLIC_ALTERNATIVES)];
} FileCyclicSite;
struct SERVICE_TYPE(CyclicReport) {
    SERVICE_TYPE(CodePlan) *plan;
    FileCyclicSite *sites;
    SERVICE_TYPE(CyclicSummary) summary;
};
typedef struct { uint16_t instruction; uint8_t variant; } FileCyclicWork;
typedef struct {
    SERVICE_TYPE(BodyReport) body;
    SERVICE_TYPE(FlowState) transfer;
    FileCyclicState canonical;
    FileCyclicWork queue[SERVICE_CONST(CYCLIC_FUNCTION_PAIRS)];
    uint32_t head, tail;
    uint64_t owners[SERVICE_CONST(FLOW_OWNERS)];
    uint64_t reach[SERVICE_CONST(CODE_FUNCTION_INSTRUCTIONS)][4];
    uint16_t todo[SERVICE_CONST(CODE_FUNCTION_INSTRUCTIONS)];
} FileCyclicWorkspace;

/* Only my new private composition constructs this internal extension. Existing
 * cyclic entries pass NULL and retain their original storage and transfers. */
typedef struct {
    size_t extra_bytes;
    const uint64_t *graph;
    void *context;
    SERVICE_TYPE(FlowStatus) (*transfer)(void *,SERVICE_TYPE(BodyReport) *,SERVICE_TYPE(FlowState) *,
                                  const SERVICE_TYPE(CodeInstruction) *,SERVICE_TYPE(BodyInstruction) *);
} FileCyclicExtension;

static bool file_cyclic_room(SERVICE_TYPE(CyclicReport) *r,size_t count,size_t width) {
    size_t bytes=r->summary.storage_peak;
    if(!file_code_budget(&bytes,count,width) || bytes>SERVICE_CONST(CYCLIC_BYTES))return false;
    r->summary.storage_peak=bytes;return true;
}
static bool file_cyclic_value_equal(SERVICE_TYPE(FlowValue) a,SERVICE_TYPE(FlowValue) b) {
    return value_same(a,b) && a.arm==b.arm;
}
static bool file_cyclic_equal(const FileCyclicState *a,const FileCyclicState *b) {
    if(a->info.locals!=b->info.locals || a->info.stack!=b->info.stack ||
       a->info.regions!=b->info.regions || a->info.owners!=b->info.owners ||
       a->info.references!=b->info.references)return false;
    for(uint16_t i=0;i<a->info.locals;i++)if(!file_cyclic_value_equal(a->locals[i],b->locals[i]))return false;
    for(uint16_t i=0;i<a->info.stack;i++)if(!file_cyclic_value_equal(a->stack[i],b->stack[i]))return false;
    for(uint16_t i=0;i<a->info.regions;i++)if(a->regions[i]!=b->regions[i])return false;
    for(uint16_t i=0;i<SERVICE_CONST(FLOW_REFERENCES);i++)if(!reference_same(a->references[i],b->references[i]))return false;
    return true;
}
static bool file_cyclic_stored_equal(const FileCyclicStoredState *a,const FileCyclicState *b) {
    if(a->info.locals!=b->info.locals || a->info.stack!=b->info.stack ||
       a->info.regions!=b->info.regions || a->info.owners!=b->info.owners ||
       a->info.references!=b->info.references)return false;
    for(uint16_t i=0;i<a->info.locals;i++)if(!file_cyclic_value_equal(a->locals[i],b->locals[i]))return false;
    for(uint16_t i=0;i<a->info.stack;i++)if(!file_cyclic_value_equal(a->stack[i],b->stack[i]))return false;
    for(uint16_t i=0;i<a->info.regions;i++)if(a->regions[i]!=b->regions[i])return false;
    for(uint16_t i=0;i<SERVICE_CONST(FLOW_REFERENCES);i++)if(!reference_same(i<a->reference_extent?a->references[i]:(SERVICE_TYPE(FlowReference)){0},b->references[i]))return false;
    return true;
}
static SERVICE_TYPE(FlowStatus) file_cyclic_value(FileCyclicWorkspace *w,SERVICE_TYPE(FlowValue) v,
                                          SERVICE_TYPE(FlowValue) *out,bool local) {
    if(v.type.mode && (!local || (v.type.mode!=1 && v.type.mode!=2) || v.type.category!=NOMINAL_CATEGORY_OWNER))
        return SERVICE_CONST(FLOW_INVALID);
    if(!v.initialized) {
        if(v.owner || v.type.mode)return SERVICE_CONST(FLOW_INVALID);
        v=initial_value(v.type,false,0);
    } else {
        if(!local && v.type.mode)return SERVICE_CONST(FLOW_INVALID);
        if(owned(v.type)!=(v.owner!=0))return SERVICE_CONST(FLOW_INVALID);
        if(result_type(v.type)) {
            if(v.arm!=SERVICE_CONST(FLOW_ARM_UNKNOWN) && v.arm!=SERVICE_CONST(FLOW_ARM_OK) &&
               v.arm!=SERVICE_CONST(FLOW_ARM_ERROR))return SERVICE_CONST(FLOW_INVALID);
        } else if(v.arm!=SERVICE_CONST(FLOW_ARM_NONE))return SERVICE_CONST(FLOW_INVALID);
    }
    if(v.owner) {
        uint16_t count=w->canonical.info.owners;
        if(count==SERVICE_CONST(FLOW_OWNERS))return SERVICE_CONST(FLOW_LIMIT);
        for(uint16_t i=0;i<count;i++)if(w->owners[i]==v.owner)return SERVICE_CONST(FLOW_INVALID);
        w->owners[count]=v.owner;v.owner=(uint64_t)count+1;
        w->canonical.info.owners++;
    }
    *out=v;return SERVICE_CONST(FLOW_OK);
}
static SERVICE_TYPE(FlowStatus) file_cyclic_canonicalize(FileCyclicWorkspace *w) {
    SERVICE_TYPE(FlowState) *s=&w->transfer;FileCyclicState *c=&w->canonical;
    if(!s->reachable || s->stack_count>SERVICE_CONST(FLOW_STACK) || s->region_count>SERVICE_CONST(FLOW_REFERENCES))
        return SERVICE_CONST(FLOW_INVALID);
    memset(c,0,sizeof *c);memset(w->owners,0,sizeof w->owners);
    c->info.locals=s->declarations->functions[s->function].locals;
    c->info.stack=s->stack_count;c->info.regions=s->region_count;
    for(uint16_t i=0;i<c->info.locals;i++) {
        if(!same(s->locals[i].type,s->declarations->locals[s->declarations->offsets[s->function]+i]))
            return SERVICE_CONST(FLOW_INVALID);
        SERVICE_TYPE(FlowStatus) status=file_cyclic_value(w,s->locals[i],&c->locals[i],true);
        if(status!=SERVICE_CONST(FLOW_OK))return status;
    }
    for(uint16_t i=0;i<c->info.stack;i++) {
        if(!s->stack[i].initialized)return SERVICE_CONST(FLOW_INVALID);
        SERVICE_TYPE(FlowStatus) status=file_cyclic_value(w,s->stack[i],&c->stack[i],false);
        if(status!=SERVICE_CONST(FLOW_OK))return status;
    }
    for(uint16_t i=0;i<s->region_count;i++) {
        if(!s->regions[i])return SERVICE_CONST(FLOW_INVALID);
        for(uint16_t j=0;j<i;j++)if(s->regions[i]==s->regions[j])return SERVICE_CONST(FLOW_INVALID);
        c->regions[i]=UINT64_C(513)+i;
    }
    for(uint16_t i=0;i<SERVICE_CONST(FLOW_REFERENCES);i++) {
        SERVICE_TYPE(FlowReference) ref=s->references[i];
        if(!ref.live)continue;
        SERVICE_TYPE(FlowDeclaration) actual;
        if(!ref.owner || !ref.identity || !live_reference(s,i,&actual))return SERVICE_CONST(FLOW_INVALID);
        for(uint16_t j=0;j<i;j++)if(s->references[j].live &&
            ((s->references[j].owner==ref.owner && (!ref.shared || !s->references[j].shared)) || s->references[j].identity==ref.identity))
            return SERVICE_CONST(FLOW_INVALID);
        if(ref.formal) {
            if(i>=s->declarations->functions[s->function].parameters || ref.local!=i || ref.region ||
               ref.owner!=UINT64_C(257)+i || ref.identity!=ref.owner)return SERVICE_CONST(FLOW_INVALID);
        } else {
            uint16_t owner=0,region=s->region_count;
            for(uint16_t j=0;j<c->info.owners;j++)if(w->owners[j]==ref.owner)owner=(uint16_t)(j+1);
            for(uint16_t j=0;j<s->region_count;j++)if(s->regions[j]==ref.region)region=j;
            if(!owner || region==s->region_count)return SERVICE_CONST(FLOW_INVALID);
            ref.owner=owner;ref.identity=UINT64_C(769)+i;ref.region=UINT64_C(513)+region;
        }
        c->references[i]=ref;c->info.references++;
    }
    for(uint16_t i=0;i<c->info.locals;i++)if(c->locals[i].type.mode &&
        (!c->references[i].live || !c->references[i].formal))return SERVICE_CONST(FLOW_INVALID);
    return SERVICE_CONST(FLOW_OK);
}
/* Each one-instruction scratch uses its own declaration allocation. Its label
 * allocator cannot modify the retained plan or an old logical state object. */
static void file_cyclic_scratch(FileCyclicWorkspace *w,SERVICE_TYPE(FlowDeclarations) *scratch,
                                uint32_t function,const FileCyclicState *input) {
    memset(&w->transfer,0,sizeof w->transfer);
    SERVICE_TYPE(FlowState) *s=&w->transfer;s->declarations=scratch;s->function=function;
    s->family=1;s->reachable=true;scratch->next_identity=1025;
    s->stack_count=input->info.stack;s->region_count=input->info.regions;
    memcpy(s->locals,input->locals,sizeof s->locals);memcpy(s->stack,input->stack,sizeof s->stack);
    memcpy(s->references,input->references,sizeof s->references);memcpy(s->regions,input->regions,sizeof s->regions);
}
static SERVICE_TYPE(FlowStatus) file_cyclic_intern(SERVICE_TYPE(CyclicReport) *r,FileCyclicWorkspace *w,
    uint32_t function,uint16_t instruction,uint8_t *ordinal) {
    FileCyclicSite *site=&r->sites[r->plan->starts[function]+instruction];
    for(uint8_t i=0;i<site->count;i++)if(file_cyclic_stored_equal(&site->nodes[i]->state,&w->canonical)) {
        *ordinal=i;return SERVICE_CONST(FLOW_OK);
    }
    /* Canonical counts are bounded before interning. I retain all declared
     * locals and occupied stack/region slots; absent references stay zero.
     * Each segment has uint64_t alignment through its element type/size. */
    uint16_t extent=0;
    for(uint16_t i=0;i<SERVICE_CONST(FLOW_REFERENCES);i++)if(w->canonical.references[i].live)extent=i+1;
    size_t bytes=sizeof(FileCyclicNode)+
        ((size_t)w->canonical.info.locals+w->canonical.info.stack)*sizeof(SERVICE_TYPE(FlowValue))+
        (size_t)extent*sizeof(SERVICE_TYPE(FlowReference))+
        (size_t)w->canonical.info.regions*sizeof(uint64_t);
    if(site->count==SERVICE_CONST(CYCLIC_ALTERNATIVES) || w->tail==SERVICE_CONST(CYCLIC_FUNCTION_PAIRS) ||
       r->summary.variants==SERVICE_CONST(CYCLIC_PAIRS) || !file_cyclic_room(r,1,bytes))
        return SERVICE_CONST(FLOW_LIMIT);
    FileCyclicNode *node=calloc(1,bytes);if(!node)return SERVICE_CONST(FLOW_MEMORY);
    node->state.info=w->canonical.info;node->state.reference_extent=extent;
    node->state.locals=(SERVICE_TYPE(FlowValue) *)(node+1);
    node->state.stack=node->state.locals+w->canonical.info.locals;
    node->state.references=(SERVICE_TYPE(FlowReference) *)(node->state.stack+w->canonical.info.stack);
    node->state.regions=(uint64_t *)(node->state.references+extent);
    memcpy(node->state.locals,w->canonical.locals,(size_t)w->canonical.info.locals*sizeof(SERVICE_TYPE(FlowValue)));
    memcpy(node->state.stack,w->canonical.stack,(size_t)w->canonical.info.stack*sizeof(SERVICE_TYPE(FlowValue)));
    memcpy(node->state.references,w->canonical.references,(size_t)extent*sizeof(SERVICE_TYPE(FlowReference)));
    memcpy(node->state.regions,w->canonical.regions,(size_t)w->canonical.info.regions*sizeof(uint64_t));
    node->fact.input=w->canonical.info;
    node->fact.edge_variants[0]=node->fact.edge_variants[1]=SERVICE_CONST(CYCLIC_NO_VARIANT);
    uint8_t v=site->count;site->nodes[v]=node;site->count++;r->summary.variants++;
    w->queue[w->tail++]=(FileCyclicWork){instruction,v};*ordinal=v;
    return SERVICE_CONST(FLOW_OK);
}
/* Bounded reachability bitsets give SCC representatives without recursion.
 * A component is named by its least original instruction index. */
static void file_cyclic_components(SERVICE_TYPE(CyclicReport) *r,FileCyclicWorkspace *w,uint32_t f) {
    uint16_t count=r->plan->functions[f].instruction_count;
    memset(w->reach,0,sizeof w->reach);
    for(uint16_t origin=0;origin<count;origin++) {
        uint16_t used=0;w->todo[used++]=origin;
        w->reach[origin][origin/64]|=UINT64_C(1)<<(origin%64);
        while(used) {
            uint16_t i=w->todo[--used];
            const SERVICE_TYPE(CodeInstruction) *in=&r->plan->instructions[r->plan->starts[f]+i];
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
static SERVICE_TYPE(FlowStatus) file_cyclic_function_analyze(SERVICE_TYPE(CyclicReport) *r,FileCyclicWorkspace *w,
                                                    SERVICE_TYPE(FlowDeclarations) *scratch,uint32_t f,const FileCyclicExtension *extension) {
    w->head=w->tail=0;file_cyclic_components(r,w,f);
    memset(&w->canonical,0,sizeof w->canonical);
    const SERVICE_TYPE(FlowFunction) *fn=&r->plan->functions[f].declaration;
    w->canonical.info.locals=fn->locals;
    uint64_t next=1;
    for(uint16_t i=0;i<fn->locals;i++) {
        SERVICE_TYPE(FlowDeclaration) d=scratch->locals[scratch->offsets[f]+i];bool initialized=i<fn->parameters;
        w->canonical.locals[i]=initial_value(d,initialized,initialized && owned(d)?next++:0);
        if(initialized && d.mode)w->canonical.references[i]=(SERVICE_TYPE(FlowReference)){true,true,i,
            UINT64_C(257)+i,UINT64_C(257)+i,0,d.mode==1};
    }
    file_cyclic_scratch(w,scratch,f,&w->canonical);
    SERVICE_TYPE(FlowStatus) status=file_cyclic_canonicalize(w);if(status!=SERVICE_CONST(FLOW_OK))return status;
    uint8_t ordinal;status=file_cyclic_intern(r,w,f,0,&ordinal);if(status!=SERVICE_CONST(FLOW_OK))return status;
    bool returned=false;
    while(w->head<w->tail) {
        FileCyclicWork item=w->queue[w->head++];
        FileCyclicSite *site=&r->sites[r->plan->starts[f]+item.instruction];
        FileCyclicNode *node=site->nodes[item.variant];
        if(node->processed || r->summary.transfers==SERVICE_CONST(CYCLIC_PAIRS))return SERVICE_CONST(FLOW_LIMIT);
        r->summary.transfers++;
        memset(&w->canonical,0,sizeof w->canonical);
        w->canonical.info=node->state.info;
        memcpy(w->canonical.locals,node->state.locals,(size_t)node->state.info.locals*sizeof(SERVICE_TYPE(FlowValue)));
        memcpy(w->canonical.stack,node->state.stack,(size_t)node->state.info.stack*sizeof(SERVICE_TYPE(FlowValue)));
        memcpy(w->canonical.references,node->state.references,(size_t)node->state.reference_extent*sizeof(SERVICE_TYPE(FlowReference)));
        memcpy(w->canonical.regions,node->state.regions,(size_t)node->state.info.regions*sizeof(uint64_t));
        file_cyclic_scratch(w,scratch,f,&w->canonical);
        const SERVICE_TYPE(CodeInstruction) *in=&r->plan->instructions[r->plan->starts[f]+item.instruction];
        SERVICE_TYPE(BodyInstruction) *fact=&node->fact.body;
        fact->reachable=true;fact->input_stack=w->transfer.stack_count;
        fact->cleanup_local=SERVICE_CONST(FLOW_NO_REFERENCE);fact->pending_checks=SERVICE_CONST(FLOW_CHECK_CLEANUP);
        status=extension?extension->transfer(extension->context,&w->body,&w->transfer,in,fact):
                         file_body_transfer(&w->body,&w->transfer,in,fact);
        if(status!=SERVICE_CONST(FLOW_OK))return status;
        fact->output_stack=w->transfer.stack_count;returned|=fact->exit_checked;
        status=file_cyclic_canonicalize(w);if(status!=SERVICE_CONST(FLOW_OK))return status;
        node->fact.output=w->canonical.info;
        SERVICE_TYPE(FlowArm) prior=SERVICE_CONST(FLOW_ARM_NONE);
        uint16_t refine=0;
        if(in->decoded.opcode==OP_FILE_RESULT_BRANCH) {
            refine=in->decoded.operands[0].u16;
            SERVICE_TYPE(FlowValue) value=w->transfer.locals[refine];
            if(!value.initialized || !result_type(value.type) || value.type.mode)return SERVICE_CONST(FLOW_INVALID);
            prior=value.arm;fact->refinement=true;
        }
        for(uint8_t edge=0;edge<in->successor_count;edge++) {
            if(fact->refinement) {
                SERVICE_TYPE(FlowArm) arm=edge?SERVICE_CONST(FLOW_ARM_ERROR):SERVICE_CONST(FLOW_ARM_OK);
                if(prior!=SERVICE_CONST(FLOW_ARM_UNKNOWN) && prior!=arm)continue;
                w->transfer.locals[refine].arm=arm;
            }
            if(r->summary.edges==SERVICE_CONST(CYCLIC_EDGES))return SERVICE_CONST(FLOW_LIMIT);
            r->summary.edges++;
            status=file_cyclic_canonicalize(w);if(status!=SERVICE_CONST(FLOW_OK))return status;
            status=file_cyclic_intern(r,w,f,in->successors[edge],&ordinal);
            if(status!=SERVICE_CONST(FLOW_OK))return status;
            node->fact.edge_variants[edge]=ordinal;node->fact.edge_mask|=(uint8_t)(1u<<edge);
        }
        node->processed=true;
    }
    if(!returned)return SERVICE_CONST(FLOW_INVALID);
    w->body.checked[f]=true;return SERVICE_CONST(FLOW_OK);
}
void SERVICE_FN(cyclic_free)(SERVICE_TYPE(CyclicReport) *r) {
    if(!r)return;
    if(r->sites)for(uint32_t i=0;i<r->plan->instruction_count;i++)
        for(uint8_t j=0;j<r->sites[i].count;j++)free(r->sites[i].nodes[j]);
    free(r->sites);SERVICE_FN(code_free)(r->plan);free(r);
}
static bool file_cyclic_complete(const SERVICE_TYPE(CyclicReport) *r) {
    uint32_t count=0;
    for(uint32_t f=0;f<r->plan->count;f++)for(uint16_t i=0;i<r->plan->functions[f].instruction_count;i++) {
        const SERVICE_TYPE(CodeInstruction) *in=&r->plan->instructions[r->plan->starts[f]+i];
        const FileCyclicSite *site=&r->sites[r->plan->starts[f]+i];
        if(site->count>SERVICE_CONST(CYCLIC_ALTERNATIVES) || site->component>=r->plan->functions[f].instruction_count)
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
                } else if(n->fact.edge_variants[e]!=SERVICE_CONST(CYCLIC_NO_VARIANT))return false;
            }
            count++;
        }
    }
    return count==r->summary.variants && count==r->summary.transfers;
}
static SERVICE_TYPE(FlowStatus) file_cyclic_budget(const NvmModule *m,size_t extra,
                                             uint32_t *instructions,size_t *bytes,
                                             size_t *declaration_bytes) {
    SERVICE_TYPE(FlowStatus) status=file_code_preflight(m,instructions,bytes);
    if(status!=SERVICE_CONST(FLOW_OK))return status;
    size_t locals=0;
    for(uint32_t i=0;i<m->function_count;i++)locals+=m->functions[i].local_count;
    *declaration_bytes=sizeof(SERVICE_TYPE(FlowDeclarations))+locals*sizeof(SERVICE_TYPE(FlowDeclaration));
    if(!file_code_budget(bytes,1,sizeof(SERVICE_TYPE(CyclicReport))) ||
       !file_code_budget(bytes,*instructions,sizeof(FileCyclicSite)) ||
       !file_code_budget(bytes,1,sizeof(FileCyclicWorkspace)) ||
       !file_code_budget(bytes,1,*declaration_bytes) ||
       !file_code_budget(bytes,1,extra) || *bytes>SERVICE_CONST(CYCLIC_BYTES))return SERVICE_CONST(FLOW_LIMIT);
    return SERVICE_CONST(FLOW_OK);
}
static SERVICE_TYPE(FlowStatus) file_cyclic_analyze_extended(const NvmModule *m,SERVICE_TYPE(CyclicReport) **out,
                                                      const FileCyclicExtension *extension) {
    if(!out)return SERVICE_CONST(FLOW_INVALID);
    uint32_t instructions;size_t bytes,declaration_bytes;
    SERVICE_TYPE(FlowStatus) status=file_cyclic_budget(m,extension?extension->extra_bytes:0,
                                               &instructions,&bytes,&declaration_bytes);
    if(status!=SERVICE_CONST(FLOW_OK))return status;
    SERVICE_TYPE(CodePlan) *p=NULL;status=file_code_prepare_profile(m,&p,false,extension!=NULL);
    if(status!=SERVICE_CONST(FLOW_OK))return status;
    if(extension) {
        uint64_t done=0;
        for(uint32_t rank=0;rank<p->count;rank++) {
            uint32_t f;
            for(f=0;f<p->count;f++)
                if(!(done&(UINT64_C(1)<<f)) && !(extension->graph[f]&~done))break;
            if(f==p->count){SERVICE_FN(code_free)(p);return SERVICE_CONST(FLOW_UNRESOLVED);}
            p->function_order[rank]=f;done|=UINT64_C(1)<<f;
        }
    }
    SERVICE_TYPE(CyclicReport) *r=calloc(1,sizeof *r);
    if(!r){SERVICE_FN(code_free)(p);return SERVICE_CONST(FLOW_MEMORY);}
    r->plan=p;r->summary=(SERVICE_TYPE(CyclicSummary)){SERVICE_CONST(CYCLIC_REVISION),p->count,instructions,0,0,0,bytes,false};
    r->sites=calloc(instructions,sizeof *r->sites);
    FileCyclicWorkspace *w=calloc(1,sizeof *w);
    SERVICE_TYPE(FlowDeclarations) *scratch=malloc(declaration_bytes);
    if(!r->sites || !w || !scratch){status=SERVICE_CONST(FLOW_MEMORY);goto done;}
    memcpy(scratch,p->declarations,declaration_bytes);scratch->references=1;
    w->body.plan=p;
    for(uint32_t rank=0;rank<p->count;rank++) {
        status=file_cyclic_function_analyze(r,w,scratch,p->function_order[rank],extension);
        if(status!=SERVICE_CONST(FLOW_OK))break;
    }
    if(status==SERVICE_CONST(FLOW_OK) && !file_cyclic_complete(r))status=SERVICE_CONST(FLOW_INVALID);
done:
    /* The scratch borrows the report-owned nominal plan; it never releases it. */
    free(scratch);free(w);
    if(status!=SERVICE_CONST(FLOW_OK)){SERVICE_FN(cyclic_free)(r);return status;}
    *out=r;return SERVICE_CONST(FLOW_OK);
}
SERVICE_TYPE(FlowStatus) SERVICE_FN(cyclic_analyze)(const NvmModule *m,SERVICE_TYPE(CyclicReport) **out) {
    return file_cyclic_analyze_extended(m,out,NULL);
}
static const FileCyclicSite *file_cyclic_site(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i) {
    if(!r || f>=r->plan->count || i>=r->plan->functions[f].instruction_count)return NULL;
    return &r->sites[r->plan->starts[f]+i];
}
static const FileCyclicNode *file_cyclic_node(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i,uint8_t v) {
    const FileCyclicSite *site=file_cyclic_site(r,f,i);
    return site && v<site->count?site->nodes[v]:NULL;
}
bool SERVICE_FN(cyclic_summary)(const SERVICE_TYPE(CyclicReport) *r,SERVICE_TYPE(CyclicSummary) *out) {
    if(!r || !out)return false;
    *out=r->summary;return true;
}
bool SERVICE_FN(cyclic_function)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,SERVICE_TYPE(CodeFunction) *out) {
    return r && SERVICE_FN(code_function)(r->plan,f,out);
}
bool SERVICE_FN(cyclic_local)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t l,SERVICE_TYPE(FlowDeclaration) *out) {
    return r && SERVICE_FN(code_local)(r->plan,f,l,out);
}
bool SERVICE_FN(cyclic_instruction)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i,SERVICE_TYPE(CodeInstruction) *out) {
    return r && SERVICE_FN(code_instruction)(r->plan,f,i,out);
}
bool SERVICE_FN(cyclic_component)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i,uint16_t *out) {
    const FileCyclicSite *site=file_cyclic_site(r,f,i);if(!site || !out)return false;
    *out=site->component;return true;
}
bool SERVICE_FN(cyclic_variant_count)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i,uint8_t *out) {
    const FileCyclicSite *site=file_cyclic_site(r,f,i);if(!site || !out)return false;
    *out=site->count;return true;
}
bool SERVICE_FN(cyclic_variant)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i,uint8_t v,SERVICE_TYPE(CyclicVariant) *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out)return false;
    *out=node->fact;return true;
}
bool SERVICE_FN(cyclic_input_local)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,SERVICE_TYPE(FlowValue) *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out || l>=node->state.info.locals)return false;
    *out=node->state.locals[l];return true;
}
bool SERVICE_FN(cyclic_input_stack)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i,uint8_t v,uint16_t s,SERVICE_TYPE(FlowValue) *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out || s>=node->state.info.stack)return false;
    *out=node->state.stack[s];return true;
}
bool SERVICE_FN(cyclic_input_reference)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i,uint8_t v,uint16_t ref,SERVICE_TYPE(FlowReference) *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out || ref>=SERVICE_CONST(FLOW_REFERENCES))return false;
    *out=ref<node->state.reference_extent?node->state.references[ref]:(SERVICE_TYPE(FlowReference)){0};return true;
}
bool SERVICE_FN(cyclic_input_region)(const SERVICE_TYPE(CyclicReport) *r,uint32_t f,uint16_t i,uint8_t v,uint16_t region,uint64_t *out) {
    const FileCyclicNode *node=file_cyclic_node(r,f,i,v);if(!node || !out || region>=node->state.info.regions)return false;
    *out=node->state.regions[region];return true;
}
bool SERVICE_FN(cyclic_type)(const SERVICE_TYPE(CyclicReport) *r,uint32_t ordinal,SERVICE_TYPE(NominalLayout) *out) {
    return r && SERVICE_FN(nominal_type)(r->plan->declarations->nominal,ordinal,out);
}
bool SERVICE_FN(cyclic_import)(const SERVICE_TYPE(CyclicReport) *r,uint32_t ordinal,uint32_t *out) {
    return r && SERVICE_FN(nominal_import)(r->plan->declarations->nominal,ordinal,out);
}
