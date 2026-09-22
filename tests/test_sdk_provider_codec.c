/* I exercise raw transport ownership, not module or execution admission. */
#ifdef NDEBUG
#error "I require assertions in my raw SDK codec controls."
#endif
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
static int fail_allocation, allocation_calls;
static void *codec_malloc(size_t n) {
    allocation_calls++;
    if(fail_allocation)return NULL;
    return malloc(n);
}
static void *codec_calloc(size_t n,size_t w) {
    allocation_calls++;
    if(fail_allocation)return NULL;
    return calloc(n,w);
}
#define malloc codec_malloc
#define calloc codec_calloc
#include "../src/nanoisa/sdk_provider_codec.c"
#undef malloc
#undef calloc
static void refuse(const uint8_t *data,size_t bytes,NvmSdkResult expected) {
    NvmSdkProviderTransport *sentinel=(NvmSdkProviderTransport *)(uintptr_t)1;
    int before=allocation_calls;
    assert(nvm_sdk_provider_decode(data,bytes,NVM_SDK_PROVIDER_MAX_BYTES,&sentinel)==expected);
    assert(sentinel==(NvmSdkProviderTransport *)(uintptr_t)1);
    assert(allocation_calls==before);
}
static NvmPreparationBudget lifetime_budget(void) {
    return (NvmPreparationBudget){NVM_PREPARATION_MAX_BYTES,NVM_PREPARATION_MAX_STEPS};
}
static void lifetime_refuse(const uint8_t *data,size_t size,NvmSdkResult expected) {
    NvmPreparationBudget budget=lifetime_budget(),before=budget;
    NvmSdkProviderTransport *out=(void *)(uintptr_t)1;int calls=allocation_calls;
    assert(nvm_sdk_provider_lifetime_decode_budget(data,size,&budget,&out)==expected);
    assert(out==(void *)(uintptr_t)1 && budget.bytes==before.bytes && budget.steps==before.steps);
    assert(allocation_calls==calls);
}
static void lifetime_controls(void) {
    uint32_t refs[]={0,1},binding_policies[]={1,0};
    NvmSdkProviderRow provider={0,1,2,3,4,5};
    NvmSdkSignatureRow signature={0,0,1,1,1};
    NvmSdkBindingRow bindings[]={{NVM_SDK_BIND_IMPORT,0,UINT32_MAX,0,0},
                                {NVM_SDK_BIND_FUNCTION,0,UINT32_MAX,0,UINT32_MAX}};
    /* Policy0 is deliberately unused, but structurally checked. Policy1 selects
     * parameter node0/result node1; node2 supplies explicit child policy. */
    NvmSdkCallPolicy policies[]={{0,0,0,0,UINT32_MAX},{0,1,1,1,UINT32_MAX}};
    NvmSdkLifetimeNode nodes[]={{0,NVM_SDK_BORROW_CALL,UINT32_MAX,UINT32_MAX,2,1,0},
        {1,NVM_SDK_SNAPSHOT_RESULT,UINT32_MAX,7,0,0,0},
        {2,NVM_SDK_CALLBACK_RETAINED,UINT32_MAX,8,0,0,1}};
    NvmSdkLifetimeRows rows={{NULL,&provider,&signature,bindings,refs,0,1,1,2,2},binding_policies,policies,nodes,2,3};
    uint8_t *wire=NULL;size_t bytes=0;
    assert(nvm_sdk_provider_lifetime_encode(&rows,NVM_SDK_PROVIDER_MAX_BYTES,&wire,&bytes)==NVM_SDK_OK);
    assert(bytes==40+32+24+48+8+48+96 && get32(wire)==2 && get32(wire+4)==2 && get32(wire+28)==3);
    refuse(wire,bytes,NVM_SDK_INVALID); /* Every old raw entry retains revision1. */
    NvmPreparationBudget budget=lifetime_budget(),initial=budget;NvmSdkProviderTransport *p=NULL;
    assert(nvm_sdk_provider_lifetime_decode_budget(wire,bytes,&budget,&p)==NVM_SDK_OK);
    size_t charge=initial.bytes-budget.bytes;uint32_t steps=initial.steps-budget.steps;
    uint32_t counts[2],decl[5],policy=99;
    assert(nvm_sdk_provider_lifetime_counts(p,counts)&&counts[0]==2&&counts[1]==3);
    assert(nvm_sdk_provider_counts(p,decl)&&decl[1]==1&&decl[3]==2&&decl[4]==2);
    assert(nvm_sdk_provider_binding_policy(p,0,&policy)&&policy==1);
    assert(!nvm_sdk_provider_binding_policy(p,1,&policy)&&policy==1);
    NvmSdkCallPolicy call;assert(nvm_sdk_provider_call_policy(p,0,&call)&&!call.parameter_count);
    assert(nvm_sdk_provider_call_policy(p,1,&call)&&call.parameter_count==1&&call.result_first==1);
    NvmSdkLifetimeNode node;assert(nvm_sdk_provider_lifetime_node(p,2,&node)&&node.mode==NVM_SDK_CALLBACK_RETAINED&&node.callback_profile==1);
    node.type=987;assert(!nvm_sdk_provider_lifetime_node(p,3,&node)&&node.type==987);
    nvm_sdk_provider_transport_free(p);
    budget=(NvmPreparationBudget){charge,steps};p=NULL;
    assert(nvm_sdk_provider_lifetime_decode_budget(wire,bytes,&budget,&p)==NVM_SDK_OK&&!budget.bytes&&!budget.steps);
    nvm_sdk_provider_transport_free(p);
    for(unsigned dimension=0;dimension<2;dimension++) {
        budget=(NvmPreparationBudget){charge-(dimension==0),steps-(dimension==1)};NvmPreparationBudget before=budget;p=(void *)(uintptr_t)1;
        assert(nvm_sdk_provider_lifetime_decode_budget(wire,bytes,&budget,&p)==NVM_SDK_LIMIT);
        assert(p==(void *)(uintptr_t)1&&budget.bytes==before.bytes&&budget.steps==before.steps);
    }
    for(size_t n=0;n<bytes;n++)lifetime_refuse(wire,n,NVM_SDK_INVALID);
    uint8_t *bad=malloc(bytes+1);assert(bad);memcpy(bad,wire,bytes);bad[bytes]=0;
    lifetime_refuse(bad,bytes+1,NVM_SDK_INVALID);
    size_t policies_at=40+32+24+48+8,nodes_at=policies_at+48;
#define LIFE_BAD(at,value,result) do {memcpy(bad,wire,bytes);put32(bad+(at),(value));lifetime_refuse(bad,bytes,result);} while(0)
    LIFE_BAD(32,1,NVM_SDK_INVALID);LIFE_BAD(36,1,NVM_SDK_INVALID);
    LIFE_BAD(4,65537,NVM_SDK_LIMIT);LIFE_BAD(28,65537,NVM_SDK_LIMIT);
    LIFE_BAD(40+32+24+20,2,NVM_SDK_INVALID); /* out-of-range import policy */
    LIFE_BAD(40+32+24+24+20,1,NVM_SDK_INVALID); /* nonimport reserved */
    LIFE_BAD(policies_at+20,1,NVM_SDK_INVALID); /* unused policy reserved */
    LIFE_BAD(policies_at,4,NVM_SDK_INVALID); /* unused empty slice still bounded */
    LIFE_BAD(policies_at+24+4,65536,NVM_SDK_INVALID);
    LIFE_BAD(nodes_at+4,9,NVM_SDK_INVALID);LIFE_BAD(nodes_at+28,1,NVM_SDK_INVALID);
    LIFE_BAD(nodes_at+16,3,NVM_SDK_INVALID);LIFE_BAD(nodes_at+24,1,NVM_SDK_INVALID);
    LIFE_BAD(nodes_at+64+24,0,NVM_SDK_INVALID);LIFE_BAD(nodes_at+64+24,2,NVM_SDK_INVALID);LIFE_BAD(nodes_at+64+24,3,NVM_SDK_INVALID);
    LIFE_BAD(nodes_at+64+24,513,NVM_SDK_INVALID);LIFE_BAD(nodes_at+64+24,65537,NVM_SDK_INVALID);
#undef LIFE_BAD
    free(bad);
    fail_allocation=1;
    for(unsigned attempt=0;attempt<2;attempt++) {
        p=(void *)(uintptr_t)1;budget=lifetime_budget();NvmPreparationBudget before=budget;
        assert(nvm_sdk_provider_lifetime_decode_budget(wire,bytes,&budget,&p)==NVM_SDK_MEMORY);
        assert(p==(void *)(uintptr_t)1&&budget.bytes==before.bytes&&budget.steps==before.steps);
        uint8_t *out=(void *)(uintptr_t)1;size_t size=777;
        assert(nvm_sdk_provider_lifetime_encode(&rows,NVM_SDK_PROVIDER_MAX_BYTES,&out,&size)==NVM_SDK_MEMORY);
        assert(out==(void *)(uintptr_t)1&&size==777);
    }
    fail_allocation=0;budget=lifetime_budget();p=NULL;
    assert(nvm_sdk_provider_lifetime_decode_budget(wire,bytes,&budget,&p)==NVM_SDK_OK);
    memset(wire,0,bytes);free(wire);memset(nodes,0,sizeof nodes);
    assert(nvm_sdk_provider_lifetime_node(p,2,&node)&&node.hook_set==8&&node.callback_profile==1);
    nvm_sdk_provider_transport_free(p);
    NvmSdkLifetimeRows empty={0};wire=NULL;bytes=0;
    assert(nvm_sdk_provider_lifetime_encode(&empty,NVM_SDK_PROVIDER_MAX_BYTES,&wire,&bytes)==NVM_SDK_OK&&bytes==40);
    budget=lifetime_budget();p=NULL;assert(nvm_sdk_provider_lifetime_decode_budget(wire,bytes,&budget,&p)==NVM_SDK_OK);
    nvm_sdk_provider_transport_free(p);free(wire);
    puts("I passed private recursive lifetime wire controls without semantic admission.");
}

int main(void) {
    NvmSdkNominalRow nominal[]={
        {10,20,NVM_SDK_NOMINAL_RECORD,0,0,1},
        {11,20,NVM_SDK_NOMINAL_RECORD,1,1,1},
        {10,21,NVM_SDK_NOMINAL_OPAQUE,UINT32_MAX,2,0},
        {10,22,NVM_SDK_NOMINAL_ENUM,2,2,0}};
    NvmSdkProviderRow provider[]={{30,31,32,33,34,35}};
    NvmSdkSignatureRow signature[]={{2,0,2,2,1}};
    NvmSdkBindingRow binding[]={
        {NVM_SDK_BIND_IMPORT,0,UINT32_MAX,0,0},
        {NVM_SDK_BIND_FUNCTION,4,UINT32_MAX,0,UINT32_MAX},
        {NVM_SDK_BIND_FIELD,0,1,3,UINT32_MAX}};
    uint32_t refs[]={0,1,2};
    NvmSdkProviderRows rows={nominal,provider,signature,binding,refs,4,1,1,3,3};
    uint8_t *data=NULL;size_t size=0;
    assert(nvm_sdk_provider_encode(&rows,NVM_SDK_PROVIDER_MAX_BYTES,&data,&size)==NVM_SDK_OK);
    assert(size==32+4*32+32+24+3*24+3*4);
    assert(get32(data)==1 && get32(data+8)==4 && get32(data+24)==3);
    NvmSdkProviderTransport *plan=NULL;
    assert(nvm_sdk_provider_decode(data,size,NVM_SDK_PROVIDER_MAX_BYTES,&plan)==NVM_SDK_OK);
    /* Borrowed row mutation and released encoded storage cannot alter my copy. */
    memset(nominal,0,sizeof nominal);memset(data,0,size);free(data);data=NULL;
    uint32_t counts[5];assert(nvm_sdk_provider_counts(plan,counts));assert(counts[0]==4&&counts[3]==3);
    NvmSdkNominalRow a,b;assert(nvm_sdk_provider_nominal(plan,0,&a));assert(nvm_sdk_provider_nominal(plan,1,&b));
    assert(a.owner==10&&b.owner==11&&a.name==b.name&&a.layout!=b.layout);
    assert(nvm_sdk_provider_nominal(plan,3,&a)&&a.kind==NVM_SDK_NOMINAL_ENUM&&a.layout==2);
    a.owner=987;assert(!nvm_sdk_provider_nominal(plan,4,&a)&&a.owner==987);
    NvmSdkProviderRow p;assert(nvm_sdk_provider_requirement(plan,0,&p)&&p.generation_digest==34);
    NvmSdkSignatureRow s;assert(nvm_sdk_provider_signature(plan,0,&s)&&s.parameter_count==2&&s.result_count==1);
    NvmSdkBindingRow bind;assert(nvm_sdk_provider_binding(plan,2,&bind)&&bind.slot==1&&bind.detail==3);
    uint32_t ref=99;assert(nvm_sdk_provider_reference(plan,2,&ref)&&ref==2);
    assert(!nvm_sdk_provider_reference(plan,3,&ref)&&ref==2);
    /* Reconstruct the logical rows through public by-value accessors. */
    for(uint32_t i=0;i<4;i++)assert(nvm_sdk_provider_nominal(plan,i,&nominal[i]));
    assert(nvm_sdk_provider_encode(&rows,NVM_SDK_PROVIDER_MAX_BYTES,&data,&size)==NVM_SDK_OK);
    assert(size==plan->size&&!memcmp(data,plan->data,size));
    nvm_sdk_provider_transport_free(plan);plan=NULL;
    for(size_t n=0;n<size;n++)refuse(data,n,NVM_SDK_INVALID);
    uint8_t *bad=malloc(size+1);assert(bad);memcpy(bad,data,size);bad[size]=0;
    refuse(bad,size+1,NVM_SDK_INVALID);
    const size_t reserved[]={4,28,32+24,32+28,32+4*32+24,32+4*32+32+20,32+4*32+32+24+20};
    for(size_t i=0;i<sizeof reserved/sizeof *reserved;i++) {
        memcpy(bad,data,size);bad[reserved[i]]=1;refuse(bad,size,NVM_SDK_INVALID);
    }
    memcpy(bad,data,size);put32(bad+8,4097);refuse(bad,size,NVM_SDK_LIMIT);
    memcpy(bad,data,size);put32(bad+32+16,UINT32_MAX);refuse(bad,size,NVM_SDK_INVALID);
    free(bad);
    uint8_t *old=(uint8_t *)(uintptr_t)1;size_t old_size=777;
    assert(nvm_sdk_provider_encode(&rows,0,&old,&old_size)==NVM_SDK_LIMIT);
    assert(old==(uint8_t *)(uintptr_t)1&&old_size==777);
    plan=(NvmSdkProviderTransport *)(uintptr_t)1;
    assert(nvm_sdk_provider_decode(data,size,sizeof(NvmSdkProviderTransport)+size-1,&plan)==NVM_SDK_LIMIT);
    assert(plan==(NvmSdkProviderTransport *)(uintptr_t)1);
    plan=NULL;
    assert(nvm_sdk_provider_decode(data,size,sizeof(NvmSdkProviderTransport)+size,&plan)==NVM_SDK_OK);
    nvm_sdk_provider_transport_free(plan);
    fail_allocation=1;
    assert(nvm_sdk_provider_encode(&rows,NVM_SDK_PROVIDER_MAX_BYTES,&old,&old_size)==NVM_SDK_MEMORY);
    assert(old==(uint8_t *)(uintptr_t)1&&old_size==777);
    plan=(NvmSdkProviderTransport *)(uintptr_t)1;
    assert(nvm_sdk_provider_decode(data,size,NVM_SDK_PROVIDER_MAX_BYTES,&plan)==NVM_SDK_MEMORY);
    assert(plan==(NvmSdkProviderTransport *)(uintptr_t)1);
    fail_allocation=0;plan=NULL;
    assert(nvm_sdk_provider_decode(data,size,NVM_SDK_PROVIDER_MAX_BYTES,&plan)==NVM_SDK_OK);
    nvm_sdk_provider_transport_free(plan);free(data);
    NvmSdkProviderRows empty={0};data=NULL;size=0;
    assert(nvm_sdk_provider_encode(&empty,NVM_SDK_PROVIDER_MAX_BYTES,&data,&size)==NVM_SDK_OK&&size==32);
    plan=NULL;assert(nvm_sdk_provider_decode(data,size,NVM_SDK_PROVIDER_MAX_BYTES,&plan)==NVM_SDK_OK);
    nvm_sdk_provider_transport_free(plan);free(data);
    puts("I passed raw SDK transport ownership, refusal and allocation controls.");lifetime_controls();return 0;
}
