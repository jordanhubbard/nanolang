#define WEBSOCKET_BODY_FIXTURE
#include "test_websocket_body.c"
#undef WEBSOCKET_BODY_FIXTURE
#include "../../src/nanoisa/websocket_codec.h"
#include "../../src/nanoisa/websocket_indirect_hosted.h"
#include "../../src/nanoisa/service_bindings_module.h"
static void roundtrip(bool permute) {
    Program p;build(&p,true,true,permute);p.f.module.header.flags=NVM_FLAG_HAS_MAIN;
    NvmV2Module wire={0};CHECK(nvm_websocket_from_module(&p.f.module,&wire)==NVM_V2_OK);
    for(uint32_t i=0;i<wire.functions.count;i++)wire.functions.items[i].max_stack=256;
    size_t size=0;NvmV2Result serialized=nvm_websocket_serialize(&wire,NULL,0,&size);
    if(serialized!=NVM_V2_OK)fprintf(stderr,"serialize status %d features %u\n",serialized,wire.extra_features);
    CHECK(serialized==NVM_V2_OK && size);
    uint8_t *bytes=malloc(size);CHECK(bytes);CHECK(nvm_websocket_serialize(&wire,bytes,size,&size)==NVM_V2_OK);
    uint8_t catalog=p.f.service[2];p.f.service[2]=1;
    size_t refused_size=77;NvmModule *refused=NULL;
    CHECK(nvm_websocket_serialize(&wire,NULL,0,&refused_size)!=NVM_V2_OK && refused_size==77);
    CHECK(nvm_websocket_to_module(&wire,&refused)!=NVM_V2_OK && !refused);
    p.f.service[2]=catalog;
    unsigned flag=8+p.f.bindings.layouts[0];p.f.ownership[flag]^=2;
    NvmV2Module bad={0};CHECK(nvm_websocket_from_module(&p.f.module,&bad)!=NVM_V2_OK);nvm_v2_module_free(&bad);
    p.f.ownership[flag]^=2;
    size_t saved_size=99;CHECK(nvm_v2_module_serialize(&wire,NULL,0,&saved_size)!=NVM_V2_OK && saved_size==99);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&wire,&copy)!=NVM_V2_OK && !copy);
    NvmV2Module decoded={0};CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)!=NVM_V2_OK);nvm_v2_module_free(&decoded);
    CHECK(nvm_websocket_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    CHECK(nvm_websocket_to_module(&decoded,&copy)==NVM_V2_OK && copy);
    CHECK(copy->service_size==p.f.module.service_size && !memcmp(copy->service_data,p.f.module.service_data,copy->service_size));
    CHECK(copy->code_size==p.code.n && !memcmp(copy->code,p.code.bytes,p.code.n));
    CHECK(nvm_service_execution_pending(copy) && nvm_service_bindings_validate(copy)!=NVM_V2_OK);
    nvm_module_free(copy);nvm_v2_module_free(&decoded);
    NvmWebSocketIndirectHostedPlan *plan=NULL;
    NvmWebSocketFlowStatus status=nvm_websocket_indirect_hosted_prepare(bytes,size,&plan);
    if(status!=NVM_WEBSOCKET_FLOW_OK)fprintf(stderr,"hosted status %d\n",status);
    OK(status);NvmWebSocketIndirectHostedStartup startup;
    CHECK(nvm_websocket_indirect_hosted_startup(plan,&startup) && !startup.runtime_admitted && startup.functions==2 && startup.frames==2);
    const uint8_t *literal=NULL;size_t length=0;
    uint32_t index=p.code.bytes[p.literal+1];
    CHECK(nvm_websocket_indirect_hosted_string(plan,index,&literal,&length) && length==19 && !memcmp(literal,"ws://localhost/test",19));
#ifdef FLOW_INSTRUMENT
    const NvmWebSocketCyclicReport *query=plan->query->ownership;
    bool probed=false;
    for(uint16_t i=0;i<query->plan->functions[0].instruction_count;i++) {
        const NvmWebSocketCodeInstruction *in=&query->plan->instructions[i];
        if(in->decoded.opcode!=OP_FILE_SERVICE)continue;
        NvmWebSocketCyclicVariant fact=query->sites[i].nodes[0]->fact;
        CHECK(file_ih_fact(plan,0,i,in,&fact));fact.body.pending_checks^=NVM_WEBSOCKET_FLOW_CHECK_TIMEOUT;
        CHECK(!file_ih_fact(plan,0,i,in,&fact));probed=true;
    }
    CHECK(probed);
#endif
    uint8_t *original=malloc(size);CHECK(original);memcpy(original,bytes,size);
    nvm_v2_module_free(&wire);memset(&p,0,sizeof p);memset(bytes,0,size);free(bytes);
    CHECK(nvm_websocket_indirect_hosted_string(plan,index,&literal,&length) && length==19 && !memcmp(literal,"ws://localhost/test",19));
    uint8_t *retained=malloc(size);CHECK(retained);CHECK(nvm_websocket_indirect_hosted_bytes(plan,0,retained,size) && !memcmp(retained,original,size));free(retained);
    nvm_websocket_indirect_hosted_free(plan);
#ifdef FLOW_INSTRUMENT
    bool completed=false;
    for(int budget=0;budget<2048;budget++) {
        allocation_budget=budget;plan=(void *)&p;
        NvmWebSocketFlowStatus got=nvm_websocket_indirect_hosted_prepare(original,size,&plan);
        allocation_budget=-1;
        if(got==NVM_WEBSOCKET_FLOW_OK){nvm_websocket_indirect_hosted_free(plan);completed=true;
            printf("I checked %d hosted allocation prefixes.\n",budget);CHECK(!live);break;}
        CHECK(got==NVM_WEBSOCKET_FLOW_MEMORY && plan==(void *)&p && !live);
    }
    CHECK(completed);
#endif
    /* I refuse truncation and corrupt containers without publishing a plan. */
    plan=(void *)&p;CHECK(nvm_websocket_indirect_hosted_prepare(original,size-1,&plan)!=NVM_WEBSOCKET_FLOW_OK && plan==(void *)&p);
    original[size-1]^=0x80;CHECK(nvm_websocket_indirect_hosted_prepare(original,size,&plan)!=NVM_WEBSOCKET_FLOW_OK && plan==(void *)&p);free(original);
#ifdef FLOW_INSTRUMENT
    CHECK(!live);
#endif
}
int main(void){roundtrip(false);roundtrip(true);printf("I passed %u private WebSocket codec/hosted checks.\n",checks);return 0;}
