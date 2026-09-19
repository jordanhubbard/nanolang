#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../src/nsi_file_plan.h"
#include "../src/nsi_cap.h"
static unsigned checks;
#define CHECK(x) do { checks++; if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);} } while(0)
#ifdef FILE_PLAN_INSTRUMENT
static unsigned allocations;
static int allocation_fail;
static void *plan_malloc(size_t n) { allocations++;return allocation_fail ? NULL : malloc(n); }
#define malloc plan_malloc
#include "../src/nsi_file_plan.c"
#undef malloc
#endif
static NlFilePlan *sentinel;
static void invalid(NlNsi *n) {
    NlFilePlan *out=sentinel;
#ifdef FILE_PLAN_INSTRUMENT
    unsigned old=allocations;
#endif
    CHECK(nl_file_plan_build(n,&out)==NL_FILE_PLAN_INVALID);CHECK(out==sentinel);
#ifdef FILE_PLAN_INSTRUMENT
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
    CHECK(nl_file_plan_build(n,&sentinel)==NL_FILE_PLAN_OK);CHECK(sentinel);
    CHECK(nl_file_plan_build(n,NULL)==NL_FILE_PLAN_INVALID);invalid(NULL);
    MUT(int,n->version,1);strings(n,&n->iface.id);strings(n,&n->iface.name);
    MUT(size_t,n->method_count,0);MUT(size_t,n->method_count,SIZE_MAX);
    MUT(size_t,n->type_count,0);MUT(size_t,n->type_count,SIZE_MAX);
    MUT(size_t,n->error_count,0);MUT(size_t,n->error_count,SIZE_MAX);
    MUT(size_t,n->capability_count,0);MUT(size_t,n->capability_count,SIZE_MAX);
    MUT(NlNsiMethod *,n->methods,NULL);MUT(NlNsiType *,n->types,NULL);
    MUT(NlNsiError *,n->errors,NULL);MUT(NlNsiNamed *,n->capabilities,NULL);
    strings(n,&n->errors[0].id);strings(n,&n->errors[0].name);strings(n,&n->errors[0].version);
    strings(n,&n->capabilities[0].id);strings(n,&n->capabilities[0].name);
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
#ifdef FILE_PLAN_INSTRUMENT
    allocation_fail=1;NlFilePlan *out=sentinel;
    CHECK(nl_file_plan_build(n,&out)==NL_FILE_PLAN_MEMORY);CHECK(out==sentinel);
    allocation_fail=0;CHECK(nl_file_plan_build(n,&out)==NL_FILE_PLAN_OK);nl_file_plan_free(out);
#endif
    nl_nsi_free(n); /* Every subsequent query is independent of source lifetime. */
    CHECK(!strcmp(nl_file_plan_interface(sentinel),"nsi:nanolang/filesystem"));
    CHECK(nl_file_plan_method_count(sentinel)==5);CHECK(nl_file_plan_type_count(sentinel)==8);
    for(size_t i=0;i<5;i++) {
        const NlFilePlanMethod *m=nl_file_plan_method(sentinel,i);
        CHECK(m && m->abi_version==1 && m->id && m->name && m->binding_id && m->generated_name);
        CHECK(m->outcomes[1].owned_payload_type==NULL);
        for(size_t j=0;j<m->param_count;j++) CHECK(m->params[j].id && m->params[j].type_id);
        if(i==0) { CHECK(m->input_mode==NL_FILE_INPUT_NONE);CHECK(!strcmp(m->outcomes[0].owned_payload_type,"nsi:nanolang/filesystem#File"));CHECK(m->acquired_rights==(NL_CAP_READ|NL_CAP_WRITE|NL_CAP_TRANSFER)); }
        else { CHECK(m->outcomes[0].owned_payload_type==NULL);CHECK(m->outcomes[0].input_state==(i==4?NL_FILE_OWNER_CONSUMED:NL_FILE_OWNER_PRESERVED));CHECK(m->outcomes[1].input_state==m->outcomes[0].input_state); }
    }
    CHECK(nl_file_plan_method(sentinel,1)->params[1].domain==NL_FILE_DOMAIN_BYTE_INT);
    CHECK(!strcmp(nl_file_plan_method(sentinel,1)->params[1].type_id,"nsi:core/int"));
    CHECK(nl_file_plan_type(sentinel,2)->members[0].domain==NL_FILE_DOMAIN_BYTE_INT);
    for(size_t i=0;i<8;i++){const NlFilePlanType *t=nl_file_plan_type(sentinel,i);CHECK(t && t->id && t->name);for(size_t j=0;j<t->member_count;j++)CHECK(t->members[j].id && t->members[j].name);}
    CHECK(nl_file_plan_method(sentinel,5)==NULL);CHECK(nl_file_plan_method(sentinel,SIZE_MAX)==NULL);
    CHECK(nl_file_plan_type(sentinel,8)==NULL);CHECK(nl_file_plan_type(sentinel,SIZE_MAX)==NULL);
    CHECK(nl_file_plan_method_count(NULL)==0);CHECK(nl_file_plan_type_count(NULL)==0);CHECK(nl_file_plan_interface(NULL)==NULL);
    CHECK(nl_file_plan_method(NULL,0)==NULL);CHECK(nl_file_plan_type(NULL,0)==NULL);
    nl_file_plan_free(sentinel);nl_file_plan_free(NULL);
    printf("PASS %u checks; exact private file plan, no service execution\n",checks);return 0;
}
