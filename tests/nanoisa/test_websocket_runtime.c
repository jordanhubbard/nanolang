/* I exercise the private carrier; matched bytecode dispatch remains separate. */
#define WEBSOCKET_BODY_FIXTURE
#include "test_websocket_body.c"
#undef WEBSOCKET_BODY_FIXTURE
#include "../../src/nanoisa/websocket_codec.h"
#include "../../src/nanoisa/websocket_runtime.h"
#ifdef WS_RUNTIME_INSTRUMENT
static int budget=-1;
static unsigned runtime_live;
static void *ws_runtime_calloc(size_t n,size_t z) {
    if(!budget)return NULL;
    if(budget>0)budget--;
    void *p=calloc(n,z);if(p)runtime_live++;return p;
}
static void ws_runtime_free(void *p){if(p){CHECK(runtime_live);runtime_live--;}free(p);}
#define calloc ws_runtime_calloc
#define free ws_runtime_free
#include "../../src/nanoisa/websocket_runtime.c"
#undef calloc
#undef free
#endif
#define RT(x) CHECK((x)==NVM_WEBSOCKET_RUNTIME_OK)
static uint8_t *carrier_wire(bool permute,size_t *size,NvmWebSocketNominalBindings *bindings) {
    Program p;build(&p,false,false,permute);*bindings=p.f.bindings;
    /* My carrier owns a valid no-argument entry and exact unused service imports.
     * Trusted direct calls below do not claim to execute this entry's CFG. */
    p.code.n=0;integer(&p.code);op(&p.code,OP_RET);
    p.functions[0].arity=0;p.functions[0].code_length=p.code.n;wr16(p.f.ownership+22,0);
    p.f.module.code_size=p.code.n;p.f.module.header.flags=NVM_FLAG_HAS_MAIN;
    NvmV2Module wire={0};CHECK(nvm_websocket_from_module(&p.f.module,&wire)==NVM_V2_OK);
    wire.functions.items[0].max_stack=256;
    CHECK(nvm_websocket_serialize(&wire,NULL,0,size)==NVM_V2_OK);
    uint8_t *bytes=malloc(*size);CHECK(bytes);CHECK(nvm_websocket_serialize(&wire,bytes,*size,size)==NVM_V2_OK);
    nvm_v2_module_free(&wire);return bytes;
}
static NvmWebSocketRuntimeView view(NvmWebSocketRuntime *c,uint32_t root) {
    NvmWebSocketRuntimeView v;CHECK(nvm_websocket_runtime_view(c,root,&v));return v;
}
static NvmWebSocketRuntime *create(const uint8_t *bytes,size_t size,bool native,bool allow) {
    NvmWebSocketRuntime *c=NULL;RT(nvm_websocket_runtime_create(bytes,size,native?NVM_WEBSOCKET_RUNTIME_NATIVE:NVM_WEBSOCKET_RUNTIME_VM,&c));
    NlWsTransportPolicy policy={allow,true,getenv("NANOLANG_RESOLVER"),2000};
    RT(nvm_websocket_runtime_policy(c,&policy));memset(&policy,0,sizeof policy);
    RT(nvm_websocket_runtime_begin(c));return c;
}
static void call(NvmWebSocketRuntime *c,NvmWebSocketNominalBindings b,unsigned method,uint32_t reference,
                 uint32_t first,int64_t timeout,uint32_t output) {
    RT(nvm_websocket_runtime_scalar(c,1,TAG_INT,timeout));
    uint32_t inputs[]={first,1};
    RT(nvm_websocket_runtime_service(c,b.imports[method],reference,method==2?inputs+1:inputs,method==2?1:2,output));
    CHECK(!view(c,1).initialized);
}
static void finish(NvmWebSocketRuntime **c,bool complete) {
    if(complete){RT(nvm_websocket_runtime_scalar(*c,0,TAG_INT,42));RT(nvm_websocket_runtime_complete_root(*c,0));}
    NvmWebSocketRuntimeView out={.fields=99};NvmWebSocketRuntimeReport r=nvm_websocket_runtime_destroy(c,&out);
    CHECK(!*c && !r.cleanup.cleanup_failures);
    CHECK(complete?r.status==NVM_WEBSOCKET_RUNTIME_OK && out.values[0]==42:r.status!=NVM_WEBSOCKET_RUNTIME_OK && out.fields==99);
#ifdef WS_RUNTIME_INSTRUMENT
    CHECK(!runtime_live);
#endif
}
static void controls(const uint8_t *bytes,size_t size,NvmWebSocketNominalBindings b) {
    NvmWebSocketRuntime *c=NULL;RT(nvm_websocket_runtime_create(bytes,size,NVM_WEBSOCKET_RUNTIME_VM,&c));
    CHECK(nvm_websocket_runtime_begin(c)==NVM_WEBSOCKET_RUNTIME_INVALID);finish(&c,false);
    c=create(bytes,size,false,false);RT(nvm_websocket_runtime_string(c,0,"ws://127.0.0.1/test",19));
    call(c,b,0,UINT32_MAX,0,1000,2);NvmWebSocketFlowArm arm;
    RT(nvm_websocket_runtime_result_arm(c,2,&arm));CHECK(arm==NVM_WEBSOCKET_FLOW_ARM_ERROR);
    RT(nvm_websocket_runtime_take(c,2,arm,4));CHECK(view(c,4).values[0]==NL_WS_TRANSPORT_RIGHTS);
    RT(nvm_websocket_runtime_drop(c,4));finish(&c,true);
    c=create(bytes,size,false,true);RT(nvm_websocket_runtime_scalar(c,0,TAG_INT,1));
    CHECK(nvm_websocket_runtime_scalar(c,1,TAG_STRING,1)==NVM_WEBSOCKET_RUNTIME_TYPE);finish(&c,false);
#ifdef WS_RUNTIME_INSTRUMENT
    c=create(bytes,size,false,true);RT(nvm_websocket_runtime_string(c,0,"counted",7));
    RT(nvm_websocket_runtime_scalar(c,1,TAG_BOOL,1));uint32_t fields[]={1,0};
    RT(nvm_websocket_runtime_construct(c,2,0,fields,2,2));CHECK(frc_payload(c,&c->values[2]));
    c->values[2].view.values[1]=99;CHECK(!frc_payload(c,&c->values[2]));

    RT(nvm_websocket_runtime_scalar(c,1,TAG_INT,0));uint32_t send[]={2,1};
    /* I expose no reference here; validation must refuse before any host call. */
    CHECK(nvm_websocket_runtime_service(c,b.imports[1],0,send,2,4)!=NVM_WEBSOCKET_RUNTIME_OK);finish(&c,false);
    c=create(bytes,size,false,true);RT(nvm_websocket_runtime_string(c,0,"counted",7));
    RT(nvm_websocket_runtime_scalar(c,1,TAG_BOOL,1));c->values[0].view.values[0]=99;
    CHECK(nvm_websocket_runtime_construct(c,2,0,fields,2,2)==NVM_WEBSOCKET_RUNTIME_TYPE && !c->values[2].view.initialized);
    finish(&c,false);
    for(int n=0;n<32;n++) {
        budget=n;c=(void *)&b;NvmWebSocketRuntimeStatus s=nvm_websocket_runtime_create(bytes,size,NVM_WEBSOCKET_RUNTIME_VM,&c);budget=-1;
        if(s==NVM_WEBSOCKET_RUNTIME_OK){finish(&c,false);printf("I checked %d runtime allocation prefixes.\n",n);break;}
        CHECK(s==NVM_WEBSOCKET_RUNTIME_MEMORY && c==(void *)&b && !runtime_live);CHECK(n<31);
    }
#endif
}
int main(int argc,char **argv) {
    CHECK(argc==3);size_t size;NvmWebSocketNominalBindings b;
    bool native=strstr(argv[2],"native")!=NULL;
    uint8_t *bytes=carrier_wire(native,&size,&b);controls(bytes,size,b);
    NvmWebSocketRuntime *c=create(bytes,size,native,true);free(bytes);
    RT(nvm_websocket_runtime_string(c,0,argv[1],strlen(argv[1])));
    call(c,b,0,UINT32_MAX,0,2000,2);NvmWebSocketFlowArm arm;
    RT(nvm_websocket_runtime_result_arm(c,2,&arm));CHECK(arm==NVM_WEBSOCKET_FLOW_ARM_OK);
    if(!strcmp(argv[2],"unhandled")){finish(&c,false);return 0;}
    RT(nvm_websocket_runtime_take(c,2,arm,3));RT(nvm_websocket_runtime_region_begin(c));RT(nvm_websocket_runtime_borrow(c,3,0));
    if(!strcmp(argv[2],"borrowed")){finish(&c,false);return 0;}
    RT(nvm_websocket_runtime_string(c,0,"a\0b",3));RT(nvm_websocket_runtime_scalar(c,1,TAG_BOOL,1));
    uint32_t fields[]={1,0};RT(nvm_websocket_runtime_construct(c,2,0,fields,2,2));
    RT(nvm_websocket_runtime_copy(c,2,5));RT(nvm_websocket_runtime_project(c,5,1,5));
    char text[3];size_t length=0;CHECK(nvm_websocket_runtime_string_read(c,5,text,sizeof text,&length) && length==3 && !memcmp(text,"a\0b",3));
    RT(nvm_websocket_runtime_drop(c,5));
    call(c,b,1,0,2,-1,4);CHECK(view(c,4).arm==NVM_WEBSOCKET_FLOW_ARM_ERROR && view(c,4).values[0]==NL_WS_TRANSPORT_LIMIT);RT(nvm_websocket_runtime_drop(c,4));
    RT(nvm_websocket_runtime_string(c,0,"a\0b",3));RT(nvm_websocket_runtime_scalar(c,1,TAG_BOOL,1));
    RT(nvm_websocket_runtime_construct(c,2,0,fields,2,2));call(c,b,1,0,2,1000,4);
    CHECK(view(c,4).arm==NVM_WEBSOCKET_FLOW_ARM_OK && view(c,4).values[0]==3);RT(nvm_websocket_runtime_drop(c,4));
#ifdef WS_RUNTIME_INSTRUMENT
    if(!strcmp(argv[2],"allocation-receive") || !strcmp(argv[2],"budget-receive")) {
        RT(nvm_websocket_runtime_scalar(c,1,TAG_INT,1000));uint32_t input=1;
        bool allocation=!strcmp(argv[2],"allocation-receive");
        if(allocation)budget=0;else c->storage.allocation_bound=NVM_WEBSOCKET_RUNTIME_BYTES;
        NvmWebSocketRuntimeStatus refused=nvm_websocket_runtime_service(c,b.imports[2],0,&input,1,4);budget=-1;
        CHECK(refused==(allocation?NVM_WEBSOCKET_RUNTIME_MEMORY:NVM_WEBSOCKET_RUNTIME_LIMIT));
        CHECK(!c->values[4].view.initialized && !c->values[1].view.initialized && c->values[3].view.owning);
        finish(&c,false);return 0;
    }
#endif
    call(c,b,2,0,UINT32_MAX,1000,4);CHECK(view(c,4).arm==NVM_WEBSOCKET_FLOW_ARM_OK);
#ifdef WS_RUNTIME_INSTRUMENT
    CHECK(frc_payload(c,&c->values[4]));int64_t identity=c->values[4].view.values[1];
    c->values[4].view.values[1]=99;CHECK(!frc_payload(c,&c->values[4]));c->values[4].view.values[1]=identity;
#endif
    RT(nvm_websocket_runtime_take(c,4,NVM_WEBSOCKET_FLOW_ARM_OK,2));RT(nvm_websocket_runtime_project(c,2,1,2));
    CHECK(nvm_websocket_runtime_string_read(c,2,text,sizeof text,&length) && length==3 && !memcmp(text,"a\0b",3));RT(nvm_websocket_runtime_drop(c,2));
    RT(nvm_websocket_runtime_region_end(c));RT(nvm_websocket_runtime_move(c,3,0));
    bool invalid=!strcmp(argv[2],"invalid-close");call(c,b,3,UINT32_MAX,0,invalid?-1:1000,4);
    CHECK(!view(c,0).initialized);CHECK(view(c,4).arm==(invalid?NVM_WEBSOCKET_FLOW_ARM_ERROR:NVM_WEBSOCKET_FLOW_ARM_OK));
    if(invalid)CHECK(view(c,4).values[0]==NL_WS_TRANSPORT_LIMIT);
    RT(nvm_websocket_runtime_drop(c,4));finish(&c,true);
    printf("I passed %u WebSocket runtime checks.\n",checks);return 0;
}
