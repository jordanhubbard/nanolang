/* I check actual bytecode; no host operation executes in this test. */
#define main websocket_flow_fixture_main
#include "test_websocket_flow.c"
#undef main
#include "../../src/nanoisa/websocket_body.h"
#include "../../src/nanoisa/websocket_indirect_flow.h"
typedef struct {uint8_t bytes[4096];uint32_t n;} Body;
typedef struct {
    Fixture f;Body code;NvmFunctionEntry functions[2];uint8_t *params[2];
    uint32_t connect,send,receive,close,pack,literal,loop;
} Program;
static void op(Body *c,uint8_t b){CHECK(c->n<sizeof c->bytes);c->bytes[c->n++]=b;}
static void u16(Body *c,uint16_t n){op(c,(uint8_t)n);op(c,(uint8_t)(n>>8));}
static void u32(Body *c,uint32_t n){for(unsigned i=0;i<4;i++)op(c,(uint8_t)(n>>(i*8)));}
static void integer(Body *c){op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)op(c,0);}
static void one(Body *c,uint8_t code,uint16_t n){op(c,code);u16(c,n);}
static void service(Program *p,unsigned method,uint16_t ref){op(&p->code,OP_FILE_SERVICE);u32(&p->code,p->f.bindings.imports[method]);u16(&p->code,ref);}
static uint32_t branch(Body *c,uint8_t code,uint16_t slot){uint32_t at=c->n;op(c,code);if(code==OP_FILE_RESULT_BRANCH)u16(c,slot);u32(c,0);return at;}
static void target(Body *c,uint32_t at,uint32_t dest){wr32(c->bytes+at+(c->bytes[at]==OP_FILE_RESULT_BRANCH?3:1),dest-at);}
static void take(Body *c,uint16_t local,uint8_t arm){one(c,OP_FILE_RESULT_TAKE,local);op(c,arm);}
static void text_value(Body *c,bool indirect){
    if(indirect){op(c,OP_FUNCREF);u32(c,1);op(c,OP_CALL_INDIRECT);u16(c,0);u16(c,1);}
    else one(c,OP_LOAD_LOCAL,0);
}
static void build(Program *p,bool indirect,bool loop,bool permute){
    memset(p,0,sizeof *p);fixture_locals(&p->f,permute);Fixture *f=&p->f;Body *c=&p->code;
    /* My entry takes a string in the acyclic profile; the indirect profile
     * produces strings through a checked same-module callable instead. */
    descriptor(f->ownership+32,TAG_STRING,0,NVM_V2_NO_INDEX);
    descriptor(f->ownership+32+6*8,TAG_STRUCT,0,f->bindings.layouts[0]);
    f->function.arity=indirect?0:1;wr16(f->ownership+22,f->function.arity);
    text_value(c,indirect);integer(c);p->connect=c->n;service(p,0,UINT16_MAX);one(c,OP_OWN_STORE_LOCAL,1);
    uint32_t failed=branch(c,OP_FILE_RESULT_BRANCH,1);
    take(c,1,0);one(c,OP_OWN_STORE_LOCAL,6);op(c,OP_REGION_BEGIN);
    op(c,OP_BORROW_LOCAL_EXCLUSIVE);u16(c,20);u16(c,6);
    p->loop=c->n;op(c,OP_PUSH_BOOL);op(c,1);text_value(c,indirect);
    p->pack=c->n;op(c,OP_AGG_PACK);op(c,AGG_RECORD);
    uint32_t source=1;for(unsigned i=0;i<3;i++)source+=f->bindings.layouts[i]<f->bindings.layouts[2];
    u32(c,source);u16(c,0);u16(c,2);
    integer(c);p->send=c->n;service(p,1,20);op(c,OP_POP);
    if(loop){op(c,OP_PUSH_BOOL);op(c,0);uint32_t back=branch(c,OP_JMP_TRUE,0);target(c,back,p->loop);}
    integer(c);p->receive=c->n;service(p,2,20);one(c,OP_STORE_LOCAL,3);
    uint32_t recv_error=branch(c,OP_FILE_RESULT_BRANCH,3);
    take(c,3,0);one(c,OP_AGG_GET,1);op(c,OP_POP);uint32_t join=branch(c,OP_JMP,0);
    target(c,recv_error,c->n);take(c,3,1);op(c,OP_POP);target(c,join,c->n);
    op(c,OP_REGION_END);one(c,OP_OWN_MOVE_LOCAL,6);integer(c);p->close=c->n;service(p,3,UINT16_MAX);op(c,OP_POP);
    integer(c);op(c,OP_RET);target(c,failed,c->n);take(c,1,1);op(c,OP_POP);integer(c);op(c,OP_RET);
    p->functions[0]=f->function;p->functions[0].code_length=c->n;p->params[0]=f->function_params;
    if(indirect){
        uint32_t start=c->n;p->literal=c->n;op(c,OP_PUSH_STR);u32(c,name(f,"ws://localhost/test"));op(c,OP_RET);
        p->functions[1]=(NvmFunctionEntry){.name_idx=name(f,"helper"),.result_count=1,.result_tag=TAG_STRING,.code_offset=start,.code_length=c->n-start};
        wr32(f->ownership+16,2);wr16(f->ownership+104,0);wr16(f->ownership+106,0);
        descriptor(f->ownership+108,TAG_STRING,0,NVM_V2_NO_INDEX);f->module.ownership_size=116;
    }
    f->module.functions=p->functions;f->module.function_count=indirect?2:1;f->module.function_param_types=p->params;
    f->module.code=c->bytes;f->module.code_size=c->n;
}
static void checked(bool indirect,bool loop,bool permute){
    Program p;build(&p,indirect,loop,permute);
    if(!indirect && !loop){
        NvmWebSocketBodyReport *r=NULL;OK(nvm_websocket_body_analyze(&p.f.module,&r));
        NvmWebSocketCodeFunction fn;CHECK(nvm_websocket_body_function(r,0,&fn));unsigned services=0;
        for(unsigned i=0;i<fn.instruction_count;i++){
            NvmWebSocketCodeInstruction in;NvmWebSocketBodyInstruction fact;
            CHECK(nvm_websocket_body_instruction(r,0,i,&in,&fact));
            if(in.decoded.opcode==OP_FILE_SERVICE){services++;CHECK(fact.reachable && fact.has_obligation);
                CHECK((fact.pending_checks & NVM_WEBSOCKET_FLOW_CHECK_TIMEOUT) && !fact.discharged_checks);}
        }
        CHECK(services==4);memset(&p,0,sizeof p);CHECK(nvm_websocket_body_function(r,0,&fn));nvm_websocket_body_free(r);
    }else if(!indirect){
        NvmWebSocketBodyReport *r=(void *)&p;CHECK(nvm_websocket_body_analyze(&p.f.module,&r)==NVM_WEBSOCKET_FLOW_UNRESOLVED && r==(void *)&p);
        NvmWebSocketCyclicReport *cyclic=NULL;OK(nvm_websocket_cyclic_analyze(&p.f.module,&cyclic));
        NvmWebSocketCyclicSummary summary;CHECK(nvm_websocket_cyclic_summary(cyclic,&summary) && !summary.runtime_admitted && summary.transfers);
        nvm_websocket_cyclic_free(cyclic);
    }else{
        NvmWebSocketIndirectFlow *r=NULL;OK(nvm_websocket_indirect_flow_analyze(&p.f.module,&r));
        NvmWebSocketIndirectFlowSummary summary;CHECK(nvm_websocket_indirect_flow_summary(r,&summary));
        CHECK(!summary.runtime_admitted && summary.targets.calls==2 && summary.candidate_applications);
        NvmWebSocketCodeFunction fn;CHECK(nvm_websocket_indirect_flow_function(r,0,&fn));
        unsigned services=0,calls=0;
        for(unsigned i=0;i<fn.instruction_count;i++){
            NvmWebSocketCodeInstruction in;CHECK(nvm_websocket_indirect_flow_instruction(r,0,i,&in));
            if(in.decoded.opcode==OP_CALL_INDIRECT){NvmWebSocketIndirectFlowCall call;
                CHECK(nvm_websocket_indirect_flow_call(r,0,i,0,&call));CHECK(call.candidates==2 && call.checked_candidates==2);calls++;}
            if(in.decoded.opcode==OP_FILE_SERVICE){NvmWebSocketCyclicVariant variant;
                CHECK(nvm_websocket_indirect_flow_variant(r,0,i,0,&variant));
                CHECK(variant.body.pending_checks & NVM_WEBSOCKET_FLOW_CHECK_TIMEOUT);services++;}
        }
        CHECK(services==4 && calls==2);memset(&p,0,sizeof p);CHECK(nvm_websocket_indirect_flow_summary(r,&summary));
        nvm_websocket_indirect_flow_free(r);
    }
#ifdef FLOW_INSTRUMENT
    CHECK(!live);
#endif
}
static void refusals(void){
    Program p;build(&p,true,true,false);
    uint32_t positions[]={p.send+5,p.close+5,p.pack+6,p.literal+1};
    for(unsigned i=0;i<4;i++){
        uint8_t saved=p.code.bytes[positions[i]];p.code.bytes[positions[i]]^=0x80;
        NvmWebSocketIndirectFlow *r=(void *)&p;CHECK(nvm_websocket_indirect_flow_analyze(&p.f.module,&r)!=NVM_WEBSOCKET_FLOW_OK && r==(void *)&p);
        p.code.bytes[positions[i]]=saved;
    }
    /* I reject an unmatched consuming move before a borrowed send. */
    p.code.bytes[p.loop+2]=OP_OWN_MOVE_LOCAL;
    NvmWebSocketIndirectFlow *r=(void *)&p;CHECK(nvm_websocket_indirect_flow_analyze(&p.f.module,&r)!=NVM_WEBSOCKET_FLOW_OK && r==(void *)&p);
#ifdef FLOW_INSTRUMENT
    build(&p,true,true,false);bool completed=false;
    for(int budget=0;budget<2048;budget++){
        allocation_budget=budget;r=(void *)&p;
        NvmWebSocketFlowStatus status=nvm_websocket_indirect_flow_analyze(&p.f.module,&r);
        allocation_budget=-1;
        if(status==NVM_WEBSOCKET_FLOW_OK){nvm_websocket_indirect_flow_free(r);completed=true;
            printf("I checked %d failed allocation prefixes.\n",budget);CHECK(!live);break;}
        CHECK(status==NVM_WEBSOCKET_FLOW_MEMORY && r==(void *)&p && !live);
    }
    CHECK(completed);
#endif
}

#ifdef WEBSOCKET_BODY_FIXTURE
#define main websocket_body_fixture_main
#endif
int main(void){
    for(unsigned permute=0;permute<2;permute++)for(unsigned indirect=0;indirect<2;indirect++)for(unsigned loop=0;loop<2;loop++)checked(indirect,loop,permute);
    refusals();printf("I passed %u WebSocket bytecode/body checks.\n",checks);return 0;
}

#ifdef WEBSOCKET_BODY_FIXTURE
#undef main
#endif
