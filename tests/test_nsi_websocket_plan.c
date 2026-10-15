#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../src/nsi_websocket_plan.h"
#include "../src/nsi_cap.h"
static unsigned checks;
#define CHECK(x) do { checks++; if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);} } while(0)
#ifdef WEBSOCKET_PLAN_INSTRUMENT
static unsigned allocations;
static int allocation_fail;
static void *plan_malloc(size_t n) { allocations++;return allocation_fail ? NULL : malloc(n); }
#define malloc plan_malloc
#include "../src/nsi_websocket_plan.c"
#undef malloc
#endif
static NlWebSocketPlan *sentinel;
static void invalid(NlNsi *n) {
    NlWebSocketPlan *out=sentinel;
#ifdef WEBSOCKET_PLAN_INSTRUMENT
    unsigned old=allocations;
#endif
    CHECK(nl_websocket_plan_build(n,&out)==NL_WEBSOCKET_PLAN_INVALID);CHECK(out==sentinel);
#ifdef WEBSOCKET_PLAN_INSTRUMENT
    CHECK(allocations==old);
#endif
}
#define MUT(t,l,v) do { t save=(l);(l)=(v);invalid(n);(l)=save; } while(0)
static void strings(NlNsi *n,char **s) {
    char *save=*s;
    *s="wrong";invalid(n);
    if(save){*s=NULL;invalid(n);}
    *s=save;
}
int main(int argc,char **argv) {
    CHECK(argc==2);NlNsi *n=nl_nsi_load_path(argv[1]);CHECK(n!=NULL);
    CHECK(nl_websocket_plan_build(n,&sentinel)==NL_WEBSOCKET_PLAN_OK);CHECK(sentinel);
    CHECK(nl_websocket_plan_build(n,NULL)==NL_WEBSOCKET_PLAN_INVALID);invalid(NULL);
    MUT(int,n->version,1);strings(n,&n->iface.id);strings(n,&n->iface.name);
    MUT(size_t,n->method_count,0);MUT(size_t,n->method_count,SIZE_MAX);
    MUT(size_t,n->type_count,0);MUT(size_t,n->type_count,SIZE_MAX);
    MUT(size_t,n->error_count,0);MUT(size_t,n->error_count,SIZE_MAX);
    MUT(size_t,n->capability_count,0);MUT(size_t,n->capability_count,SIZE_MAX);
    MUT(NlNsiMethod *,n->methods,NULL);MUT(NlNsiType *,n->types,NULL);
    MUT(NlNsiError *,n->errors,NULL);MUT(NlNsiNamed *,n->capabilities,NULL);
    strings(n,&n->errors[0].id);strings(n,&n->errors[0].name);strings(n,&n->errors[0].version);
    for(size_t i=0;i<n->capability_count;i++) {
        strings(n,&n->capabilities[i].id);strings(n,&n->capabilities[i].name);
    }
    MUT(char *,n->capabilities[1].id,n->capabilities[0].id);
    { NlNsiNamed a=n->capabilities[0];n->capabilities[0]=n->capabilities[1];n->capabilities[1]=a;
      invalid(n);a=n->capabilities[0];n->capabilities[0]=n->capabilities[1];n->capabilities[1]=a; }
    for(size_t i=0;i<n->method_count;i++) {
        NlNsiMethod *m=&n->methods[i];strings(n,&m->id);strings(n,&m->name);
        MUT(int,m->idempotent,1);MUT(size_t,m->param_count,0);MUT(size_t,m->param_count,SIZE_MAX);
        MUT(NlNsiParam *,m->params,NULL);
        for(size_t j=0;j<m->param_count;j++) {
            NlNsiParam *p=&m->params[j];strings(n,&p->id);strings(n,&p->name);strings(n,&p->type_id);
            MUT(NlNsiDirection,p->direction,(NlNsiDirection)99);
            MUT(NlNsiOwnership,p->ownership,(NlNsiOwnership)99);
            MUT(NlNsiLifetime,p->lifetime,(NlNsiLifetime)99);
            MUT(NlNsiMutability,p->mutability,(NlNsiMutability)99);
            MUT(int,p->optional,1);MUT(NlNsiStreaming,p->streaming,NL_NSI_STREAM_IN);
            for(int mode=0;mode<=2;mode++) if(mode!=(int)p->ownership) { MUT(NlNsiOwnership,p->ownership,(NlNsiOwnership)mode); }
        }
    }
    for(size_t i=0;i<n->type_count;i++) {
        NlNsiType *t=&n->types[i];strings(n,&t->id);strings(n,&t->name);
        MUT(NlNsiTypeKind,t->kind,NL_NSI_TYPE_OPAQUE);MUT(size_t,t->member_count,SIZE_MAX);
        strings(n,&t->element_id);strings(n,&t->method_id);strings(n,&t->result_id);
        if(t->member_count) {
            MUT(size_t,t->member_count,0);MUT(NlNsiMember *,t->members,NULL);
            for(size_t j=0;j<t->member_count;j++) {
                strings(n,&t->members[j].id);strings(n,&t->members[j].name);strings(n,&t->members[j].type_id);
            }
        } else { NlNsiMember dummy={0};MUT(NlNsiMember *,t->members,&dummy); }
    }
    MUT(char *,n->methods[1].id,n->methods[0].id);
    MUT(char *,n->methods[1].name,n->methods[0].name);
    {NlNsiMethod tmp=n->methods[0];n->methods[0]=n->methods[1];n->methods[1]=tmp;invalid(n);tmp=n->methods[0];n->methods[0]=n->methods[1];n->methods[1]=tmp;}
#ifdef WEBSOCKET_PLAN_INSTRUMENT
    allocation_fail=1;NlWebSocketPlan *out=sentinel;
    CHECK(nl_websocket_plan_build(n,&out)==NL_WEBSOCKET_PLAN_MEMORY);CHECK(out==sentinel);
    allocation_fail=0;CHECK(nl_websocket_plan_build(n,&out)==NL_WEBSOCKET_PLAN_OK);nl_websocket_plan_free(out);
#endif
    MUT(char *,n->methods[0].params[0].type_id,"nsi:core/int");
    MUT(char *,n->types[2].members[1].type_id,"nsi:core/int");
    nl_nsi_free(n);
    CHECK(!strcmp(nl_websocket_plan_interface(sentinel),"nsi:nanolang/websocket"));
    CHECK(nl_websocket_plan_method_count(sentinel)==4 && nl_websocket_plan_type_count(sentinel)==7);
    CHECK(nl_websocket_plan_capability_count(sentinel)==2);
    CHECK(!strcmp(nl_websocket_plan_capability(sentinel,0)->id,"cap:nanolang/websocket.connect"));
    CHECK(!strcmp(nl_websocket_plan_capability(sentinel,1)->id,"cap:nanolang/net.lookup"));
    for(size_t i=0;i<4;i++) {
        const NlServicePlanMethod *m=nl_websocket_plan_method(sentinel,i);
        CHECK(m && m->abi_version==1 && m->id && m->name && m->binding_id && m->generated_name);
        CHECK(m->outcomes[1].owned_payload_type==NULL);
        CHECK(m->params[m->param_count-2].domain==NL_SERVICE_DOMAIN_TIMEOUT_MS);
        CHECK(!strcmp(m->params[m->param_count-2].type_id,"nsi:core/int"));
        if(!i) {
            CHECK(m->input_mode==NL_SERVICE_INPUT_NONE);
            CHECK(m->outcomes[0].input_state==NL_SERVICE_OWNER_NONE && m->outcomes[1].input_state==NL_SERVICE_OWNER_NONE);
            CHECK(!strcmp(m->outcomes[0].owned_payload_type,"nsi:nanolang/websocket#Connection"));
            CHECK(m->acquired_rights==(NL_CAP_READ|NL_CAP_WRITE|NL_CAP_TRANSFER));
            CHECK(!strcmp(m->params[0].type_id,"nsi:core/string"));
        } else {
            CHECK(m->outcomes[0].owned_payload_type==NULL && !m->acquired_rights);
            CHECK(m->outcomes[0].input_state==(i==3?NL_SERVICE_OWNER_CONSUMED:NL_SERVICE_OWNER_PRESERVED));
            CHECK(m->outcomes[1].input_state==m->outcomes[0].input_state);
            CHECK(m->input_mode==(i==3?NL_SERVICE_INPUT_CONSUME:NL_SERVICE_INPUT_EXCLUSIVE));
        }
        CHECK(m==nl_websocket_catalog_method(i));
    }
    CHECK(nl_websocket_plan_method(sentinel,1)->required_rights==NL_CAP_WRITE);
    CHECK(nl_websocket_plan_method(sentinel,2)->required_rights==NL_CAP_READ);
    CHECK(!nl_websocket_plan_method(sentinel,3)->required_rights);
    for(size_t i=0;i<7;i++) {
        const NlServicePlanType *t=nl_websocket_plan_type(sentinel,i);
        CHECK(t && t==nl_websocket_catalog_type(i));
        for(size_t j=0;j<t->member_count;j++)CHECK(t->members[j].id && t->members[j].name);
    }
    CHECK(nl_websocket_plan_type(sentinel,2)->member_count==2);
    CHECK(!strcmp(nl_websocket_plan_type(sentinel,2)->members[1].type_id,"nsi:core/string"));
    CHECK(!strcmp(nl_websocket_plan_type(sentinel,5)->members[0].type_id,"nsi:nanolang/websocket#Message"));
    CHECK(!nl_websocket_plan_type(sentinel,6)->members[0].type_id);
    CHECK(!nl_websocket_plan_method(sentinel,4) && !nl_websocket_plan_method(sentinel,SIZE_MAX));
    CHECK(!nl_websocket_plan_type(sentinel,7) && !nl_websocket_plan_type(sentinel,SIZE_MAX));
    CHECK(!nl_websocket_plan_capability(sentinel,2) && !nl_websocket_plan_capability(sentinel,SIZE_MAX));
    CHECK(!nl_websocket_plan_method_count(NULL) && !nl_websocket_plan_type_count(NULL) && !nl_websocket_plan_capability_count(NULL));
    CHECK(!nl_websocket_plan_interface(NULL) && !nl_websocket_plan_method(NULL,0) && !nl_websocket_plan_type(NULL,0) && !nl_websocket_plan_capability(NULL,0));
    CHECK(!strcmp(nl_websocket_catalog_interface(),"nsi:nanolang/websocket"));
    CHECK(!nl_websocket_catalog_method(4) && !nl_websocket_catalog_type(7) && !nl_websocket_catalog_capability(2));
    for(size_t i=0;i<2;i++)CHECK(nl_websocket_plan_capability(sentinel,i)==nl_websocket_catalog_capability(i));
    CHECK(nl_websocket_plan_storage_size()>0);
    const char *others[]={"tests/fixtures/nsi_file_plan.json","tests/fixtures/nsi_socket_plan.json","schema/nsi/modules/net.nsi.json"};
    for(unsigned i=0;i<3;i++) { NlNsi *other=nl_nsi_load_path(others[i]);CHECK(other);invalid(other);nl_nsi_free(other); }
    nl_websocket_plan_free(sentinel);nl_websocket_plan_free(NULL);
    printf("PASS %u checks; exact owned WebSocket catalog, no service execution\n",checks);return 0;
}
