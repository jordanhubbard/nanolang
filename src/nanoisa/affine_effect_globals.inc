/* I prove initialization over exact bounded activation/handler stacks. I merge
 * only identical control states; lexical owners are never approximated by a
 * function name. The ordinary affine pass separately proves values/ownership. */
#include "../nanovm/vm.h"
#define EG_NONE UINT32_MAX
#define EG_BUCKETS 65536u
#define EG_LIMIT 262144u
#ifdef NVM_AFFINE_EFFECT_TEST_LIMIT
extern uint32_t NVM_AFFINE_EFFECT_TEST_LIMIT(uint32_t limit);
#endif

typedef struct {
    uint32_t parent, function, continuation, handlers, lexical, depth, next;
    bool effect;
} EGFrame;
typedef struct {
    uint32_t parent, owner, key, target, depth, next;
} EGHandler;
typedef struct {
    uint32_t frame, pc, handlers, next, queued_next;
    bool queued;
    uint8_t globals[NVM_OWNERSHIP_MAX_GLOBALS];
} EGPoint;
typedef struct {
    const NvmModule *module;
    const NvmOwnershipGlobal *globals;
    uint32_t global_count;
    VmDecodedFunction decoded[NVM_OWNED_MAX_FUNCTIONS];
    EGFrame *frames; EGHandler *handlers; EGPoint *points;
    uint32_t frame_count, handler_count, point_count;
    uint32_t frame_capacity, handler_capacity, point_capacity;
    uint32_t *frame_buckets, *handler_buckets, *point_buckets;
    uint32_t head, tail;
    const char *error;
} EGProof;

static uint32_t eg_hash(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
    uint32_t h=2166136261u;
    h=(h^a)*16777619u; h=(h^b)*16777619u; h=(h^c)*16777619u;
    h=(h^d)*16777619u; h=(h^e)*16777619u;
    return h&(EG_BUCKETS-1u);
}
static void *eg_grow(void *data, uint32_t *capacity, uint32_t count, size_t width) {
    if(count<*capacity)return data;
    uint32_t limit=EG_LIMIT;
#ifdef NVM_AFFINE_EFFECT_TEST_LIMIT
    limit=NVM_AFFINE_EFFECT_TEST_LIMIT(limit);
    if(limit>EG_LIMIT)limit=EG_LIMIT;
#endif
    if(count>=limit)return NULL;
    uint32_t next=*capacity?*capacity*2u:32u;
    if(next>limit)next=limit;
    void *grown=realloc(data,(size_t)next*width);
    if(!grown)return NULL;
    *capacity=next;return grown;
}
static uint32_t eg_frame(EGProof *p, uint32_t parent, uint32_t function,
                          uint32_t continuation, uint32_t handlers, uint32_t lexical) {
    uint32_t bucket=eg_hash(parent,function,continuation,handlers,lexical);
    for(uint32_t n=p->frame_buckets[bucket];n;n=p->frames[n-1].next) {
        const EGFrame *f=&p->frames[n-1];
        if(f->parent==parent && f->function==function && f->continuation==continuation &&
           f->handlers==handlers && f->effect==(lexical!=EG_NONE) && f->lexical==(lexical==EG_NONE?n-1:lexical))return n-1;
    }
    void *grown=eg_grow(p->frames,&p->frame_capacity,p->frame_count,sizeof(*p->frames));
    if(!grown) {
        p->error="I cannot retain bounded effect/global activation states";return EG_NONE;
    }
    p->frames=grown;
    uint32_t index=p->frame_count++;
    p->frames[index]=(EGFrame){parent,function,continuation,handlers,
        lexical==EG_NONE?index:lexical,parent==EG_NONE?1:p->frames[parent].depth+1,
        p->frame_buckets[bucket],lexical!=EG_NONE};
    p->frame_buckets[bucket]=index+1;return index;
}
static uint32_t eg_handler(EGProof *p, uint32_t parent, uint32_t owner,
                            uint32_t key, uint32_t target) {
    uint32_t bucket=eg_hash(parent,owner,key,target,0);
    for(uint32_t n=p->handler_buckets[bucket];n;n=p->handlers[n-1].next) {
        const EGHandler *h=&p->handlers[n-1];
        if(h->parent==parent && h->owner==owner && h->key==key && h->target==target)return n-1;
    }
    void *grown=eg_grow(p->handlers,&p->handler_capacity,p->handler_count,sizeof(*p->handlers));
    if(!grown) {
        p->error="I cannot retain bounded effect/global handler states";return EG_NONE;
    }
    p->handlers=grown;
    uint32_t index=p->handler_count++;
    p->handlers[index]=(EGHandler){parent,owner,key,target,
        parent==EG_NONE?1:p->handlers[parent].depth+1,p->handler_buckets[bucket]};
    p->handler_buckets[bucket]=index+1;return index;
}
static bool eg_queue(EGProof *p, uint32_t frame, uint32_t pc, uint32_t handlers,
                       const uint8_t *globals) {
    if(frame>=p->frame_count || pc>=p->decoded[p->frames[frame].function].instruction_count) {
        p->error="I require an exact effect/global continuation";return false;
    }
    uint32_t bucket=eg_hash(frame,pc,handlers,0,0),index=EG_NONE;
    for(uint32_t n=p->point_buckets[bucket];n;n=p->points[n-1].next)
        if(p->points[n-1].frame==frame && p->points[n-1].pc==pc &&
           p->points[n-1].handlers==handlers){index=n-1;break;}
    bool changed=false;
    if(index==EG_NONE) {
        void *grown=eg_grow(p->points,&p->point_capacity,p->point_count,sizeof(*p->points));
        if(!grown) {
            p->error="I exceeded my effect/global control-state storage budget";return false;
        }
        p->points=grown;
        index=p->point_count++;
        p->points[index]=(EGPoint){.frame=frame,.pc=pc,.handlers=handlers,
            .next=p->point_buckets[bucket],.queued_next=EG_NONE};
        p->point_buckets[bucket]=index+1;changed=true;
    }
    EGPoint *point=&p->points[index];
    for(uint32_t g=0;g<p->global_count;g++) {
        uint8_t joined=point->globals[g]|globals[g];
        if(joined!=point->globals[g]){point->globals[g]=joined;changed=true;}
    }
    if(changed && !point->queued) {
        point->queued=true;point->queued_next=EG_NONE;
        if(p->tail==EG_NONE)p->head=index;
        else p->points[p->tail].queued_next=index;
        p->tail=index;
    }
    return true;
}
static bool eg_return(EGProof *p, uint32_t frame, const uint8_t *globals) {
    EGFrame f=p->frames[frame];
    return f.parent==EG_NONE || eg_queue(p,f.parent,f.continuation,f.handlers,globals);
}
static uint32_t eg_target(EGProof *p, uint32_t function, uint32_t offset) {
    uint32_t base=p->module->functions[function].code_offset;
    const VmDecodedInstruction *in=offset>=base?
        vm_decoded_function_at(&p->decoded[function],offset-base):NULL;
    if(!in){p->error="I require an exact effect/global branch target";return EG_NONE;}
    return (uint32_t)(in-p->decoded[function].instructions);
}
static bool eg_step(EGProof *p, EGPoint state) {
    EGFrame frame=p->frames[state.frame];
    const VmDecodedInstruction *decoded=&p->decoded[frame.function].instructions[state.pc];
    const DecodedInstruction *in=&decoded->instruction;
    uint32_t next=state.pc+1;
    switch(in->opcode) {
    case OP_HALT:return true;
    case OP_RET:return eg_return(p,frame.lexical,state.globals);
    case OP_EFFECT_RESUME:
        if(!frame.effect){p->error="I require an active effect/global resumption";return false;}
        return eg_return(p,state.frame,state.globals);
    case OP_LOAD_GLOBAL: case OP_STORE_GLOBAL: {
        uint32_t g=in->operands[0].u32;
        if(g>=p->global_count){p->error="I require a declared effect/global slot";return false;}
        uint8_t permitted=in->opcode==OP_LOAD_GLOBAL?2u:(p->globals[g].mutable?3u:1u);
        if(!(nvm_affine_global_require(state.globals[g],permitted)&1u)) {
            p->error="I require global initialization before reads and no repeated immutable store across effects";
            return false;
        }
        if(in->opcode==OP_STORE_GLOBAL)state.globals[g]=nvm_affine_global_compose(state.globals[g],10u);
        break;
    }
    case OP_CALL: {
        uint32_t target=in->operands[0].u32;
        if(!target || target>=p->module->function_count){p->error="I require a declared effect/global callee";return false;}
        /* I preserve the runtime depth trap as a terminal path, not a resume. */
        if(frame.depth==VM_MAX_FRAMES)return true;
        uint32_t child=eg_frame(p,state.frame,target,next,state.handlers,EG_NONE);
        return child!=EG_NONE && eg_queue(p,child,0,state.handlers,state.globals);
    }
    case OP_HANDLER_PUSH: {
        if(state.handlers!=EG_NONE && p->handlers[state.handlers].depth==VM_MAX_FRAMES)return true;
        uint32_t target=eg_target(p,frame.function,decoded->resolved_target);
        if(target==EG_NONE)return false;
        state.handlers=eg_handler(p,state.handlers,state.frame,in->operands[0].u32,target);
        if(state.handlers==EG_NONE)return false;
        break;
    }
    case OP_HANDLER_POP:
        for(uint16_t i=0;i<in->operands[0].u16;i++) {
            if(state.handlers==EG_NONE || p->handlers[state.handlers].owner!=state.frame) {
                p->error="I cannot pop an absent or inherited effect/global handler";return false;
            }
            state.handlers=p->handlers[state.handlers].parent;
        }
        break;
    case OP_PERFORM: {
        uint32_t selected=state.handlers;
        while(selected!=EG_NONE && !effect_key_equal(p->module,in->operands[0].u32,p->handlers[selected].key))
            selected=p->handlers[selected].parent;
        if(selected==EG_NONE || frame.depth==VM_MAX_FRAMES)return true;
        EGHandler handler=p->handlers[selected];EGFrame owner=p->frames[handler.owner];
        uint32_t child=eg_frame(p,state.frame,owner.function,next,state.handlers,owner.lexical);
        return child!=EG_NONE && eg_queue(p,child,handler.target,state.handlers,state.globals);
    }
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE: case OP_MATCH_TAG: {
        uint32_t target=eg_target(p,frame.function,decoded->resolved_target);
        if(target==EG_NONE || !eg_queue(p,state.frame,target,state.handlers,state.globals))return false;
        if(in->opcode==OP_JMP)return true;
        break;
    }
    default:break;
    }
    return eg_queue(p,state.frame,next,state.handlers,state.globals);
}
static NvmAffineAnalysis effect_global_proof(const NvmModule *m, const AnalysisCalls *calls) {
    NvmAffineAnalysis result={0};EGProof p={.module=m,.globals=calls->globals,
        .global_count=calls->global_count,.head=EG_NONE,.tail=EG_NONE};
    p.frame_buckets=calloc(EG_BUCKETS,sizeof(uint32_t));
    p.handler_buckets=calloc(EG_BUCKETS,sizeof(uint32_t));
    p.point_buckets=calloc(EG_BUCKETS,sizeof(uint32_t));
    if(!p.frame_buckets || !p.handler_buckets || !p.point_buckets) {
        p.error="I cannot allocate effect/global control indexes";goto done;
    }
    for(uint32_t f=0;f<m->function_count;f++) {
        char error[VM_DECODE_ERROR_SIZE];
        if(!vm_decode_function(m,f,&p.decoded[f],error)) {
            p.error="I require decodable effect/global control flow";goto done;
        }
    }
    uint8_t initial[NVM_OWNERSHIP_MAX_GLOBALS];memset(initial,1,p.global_count);
    uint32_t root=eg_frame(&p,EG_NONE,0,EG_NONE,EG_NONE,EG_NONE);
    if(root==EG_NONE || !eg_queue(&p,root,0,EG_NONE,initial))goto done;
    while(p.head!=EG_NONE) {
        uint32_t index=p.head;EGPoint state=p.points[index];
        p.head=state.queued_next;if(p.head==EG_NONE)p.tail=EG_NONE;
        p.points[index].queued=false;
        result.byte_offset=p.decoded[p.frames[state.frame].function].instructions[state.pc].byte_offset;
        result.visits++;
        if(!eg_step(&p,state))goto done;
    }
    result.ok=true;result.reachable=p.point_count;result.byte_offset=0;
 done:
    if(p.error)snprintf(result.message,sizeof(result.message),"%s",p.error);
    for(uint32_t f=0;f<m->function_count;f++)vm_decoded_function_free(&p.decoded[f]);
    free(p.frames);free(p.handlers);free(p.points);
    free(p.frame_buckets);free(p.handler_buckets);free(p.point_buckets);
    return result;
}
