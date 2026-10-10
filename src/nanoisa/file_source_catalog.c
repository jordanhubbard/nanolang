#include "file_source_plan.h"
#include "service_source_catalog.h"
#include "../nsi_socket_plan.h"
#include "../nsi_file_catalog.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
int64_t nl_service_source_catalog_id(const char *id) {
 if(!id)return NL_SOURCE_CATALOG_NONE;
 if(!strcmp(id,nl_file_catalog_interface()))return NL_SOURCE_CATALOG_FILE;
 if(!strcmp(id,nl_socket_catalog_interface()))return NL_SOURCE_CATALOG_SOCKET;
 return NL_SOURCE_CATALOG_NONE;
}
int64_t nl_service_source_catalog_count(int64_t catalog,int64_t kind) {
 if(catalog==NL_SOURCE_CATALOG_FILE)return kind==1?8:kind==2?5:-1;
 if(catalog==NL_SOURCE_CATALOG_SOCKET)return kind==1?(int64_t)NL_SOCKET_PLAN_TYPES:kind==2?(int64_t)NL_SOCKET_PLAN_METHODS:-1;
 return -1;
}
static const char *source_interface(int64_t catalog) {
 return catalog==NL_SOURCE_CATALOG_FILE?nl_file_catalog_interface():
        catalog==NL_SOURCE_CATALOG_SOCKET?nl_socket_catalog_interface():"";
}
static const NlServicePlanType *source_type(int64_t catalog,size_t ordinal) {
 return catalog==NL_SOURCE_CATALOG_FILE?nl_file_catalog_type(ordinal):nl_socket_catalog_type(ordinal);
}
static const NlServicePlanMethod *source_method(int64_t catalog,size_t ordinal) {
 return catalog==NL_SOURCE_CATALOG_FILE?nl_file_catalog_method(ordinal):nl_socket_catalog_method(ordinal);
}
/* String fields: interface0; type0=id,1=name,2..4=member id/name/type;
 * method0..3=id/name/generated/binding,4..6=param id/name/type,7=owned outcome.
 * Number fields: type0=kind,1=count,2=member domain; method0..4=ABI/rights/
 * acquired/mode/param count,5..9=param direction/ownership/lifetime/mutability/
 * domain,10=outcome input state. All facts come from the existing catalog. */
const char *nl_service_source_catalog_string(int64_t catalog,int64_t k,int64_t i,int64_t f,int64_t j) {
 const char *s=NULL;
 if(k==0 && i==0 && f==0 && j==0)s=source_interface(catalog);
 if(k==1 && i>=0 && i<nl_service_source_catalog_count(catalog,1) && j>=0) {
  const NlServicePlanType *t=source_type(catalog,(size_t)i);
  if(f==0 && j==0)s=t->id;else if(f==1 && j==0)s=t->name;
  else if((uint64_t)j<t->member_count) {
   const NlServicePlanMember *m=&t->members[j];
   if(f==2)s=m->id;else if(f==3)s=m->name;else if(f==4)s=m->type_id;
  }
 }
 if(k==2 && i>=0 && i<nl_service_source_catalog_count(catalog,2) && j>=0) {
  const NlServicePlanMethod *m=source_method(catalog,(size_t)i);
  if(f==0 && j==0)s=m->id;else if(f==1 && j==0)s=m->name;else if(f==2 && j==0)s=m->generated_name;else if(f==3 && j==0)s=m->binding_id;
  else if(f==7 && j<2)s=m->outcomes[j].owned_payload_type;
  else if((uint64_t)j<m->param_count) {
   const NlServicePlanParam *p=&m->params[j];
   if(f==4)s=p->id;else if(f==5)s=p->name;else if(f==6)s=p->type_id;
  }
 }
 return s?s:"";
}
int64_t nl_service_source_catalog_number(int64_t catalog,int64_t k,int64_t i,int64_t f,int64_t j) {
 if(k==1 && i>=0 && i<nl_service_source_catalog_count(catalog,1) && j>=0) {
  const NlServicePlanType *t=source_type(catalog,(size_t)i);
  if(f==0 && j==0)return t->kind;
  if(f==1 && j==0)return (int64_t)t->member_count;
  if(f==2 && (uint64_t)j<t->member_count)return t->members[j].domain;
 }
 if(k==2 && i>=0 && i<nl_service_source_catalog_count(catalog,2) && j>=0) {
  const NlServicePlanMethod *m=source_method(catalog,(size_t)i);
  if(f==0 && j==0)return m->abi_version;
  if(f==1 && j==0)return m->required_rights;
  if(f==2 && j==0)return m->acquired_rights;
  if(f==3 && j==0)return m->input_mode;
  if(f==4 && j==0)return (int64_t)m->param_count;
  if(f==10 && j<2)return m->outcomes[j].input_state;
  if((uint64_t)j<m->param_count) {
   const NlServicePlanParam *p=&m->params[j];
   if(f==5)return p->direction;
   if(f==6)return p->ownership;
   if(f==7)return p->lifetime;
   if(f==8)return p->mutability;
   if(f==9)return p->domain;
  }
 }
 return -1;
}
typedef struct { char bytes[32768];size_t used;bool ok; } CatalogText;
static void catalog_text(CatalogText *b,const char *s) {
 size_t n=strlen(s);if(!b->ok || n>=sizeof b->bytes-b->used){b->ok=false;return;}
 memcpy(b->bytes+b->used,s,n);b->used+=n;b->bytes[b->used]=0;
}
static void catalog_number(CatalogText *b,int64_t n) {
 char text[32];int rc=snprintf(text,sizeof text,"#%" PRId64 ";",n);
 if(rc<0 || (size_t)rc>=sizeof text){b->ok=false;return;}catalog_text(b,text);
}
static void catalog_string(CatalogText *b,const char *s) {
 char text[32];int rc=snprintf(text,sizeof text,"%zu:",strlen(s));
 if(rc<0 || (size_t)rc>=sizeof text){b->ok=false;return;}
 catalog_text(b,text);catalog_text(b,s);catalog_text(b,";");
}
bool nl_service_source_catalog_view(int64_t catalog,char *out,size_t cap,size_t *needed) {
 if(!needed || (!out && cap) || nl_service_source_catalog_count(catalog,1)<0)return false;
 CatalogText b={{0},0,true};catalog_text(&b,catalog==NL_SOURCE_CATALOG_FILE?"file-source-catalog1;":"socket-source-catalog1;");
 catalog_string(&b,source_interface(catalog));catalog_number(&b,nl_service_source_catalog_count(catalog,1));catalog_number(&b,nl_service_source_catalog_count(catalog,2));
 for(int64_t i=0;i<nl_service_source_catalog_count(catalog,1);i++) {
  for(int64_t f=0;f<2;f++)catalog_string(&b,nl_service_source_catalog_string(catalog,1,i,f,0));
  for(int64_t f=0;f<2;f++)catalog_number(&b,nl_service_source_catalog_number(catalog,1,i,f,0));
  int64_t count=nl_service_source_catalog_number(catalog,1,i,1,0);
  for(int64_t j=0;j<count;j++) {
   for(int64_t f=2;f<5;f++)catalog_string(&b,nl_service_source_catalog_string(catalog,1,i,f,j));
   catalog_number(&b,nl_service_source_catalog_number(catalog,1,i,2,j));
  }
 }
 for(int64_t i=0;i<nl_service_source_catalog_count(catalog,2);i++) {
  for(int64_t f=0;f<4;f++)catalog_string(&b,nl_service_source_catalog_string(catalog,2,i,f,0));
  for(int64_t f=0;f<5;f++)catalog_number(&b,nl_service_source_catalog_number(catalog,2,i,f,0));
  int64_t count=nl_service_source_catalog_number(catalog,2,i,4,0);
  for(int64_t j=0;j<count;j++) {
   for(int64_t f=4;f<7;f++)catalog_string(&b,nl_service_source_catalog_string(catalog,2,i,f,j));
   for(int64_t f=5;f<10;f++)catalog_number(&b,nl_service_source_catalog_number(catalog,2,i,f,j));
  }
  for(int64_t j=0;j<2;j++) {
   catalog_number(&b,nl_service_source_catalog_number(catalog,2,i,10,j));
   catalog_string(&b,nl_service_source_catalog_string(catalog,2,i,7,j));
  }
 }
 if(!b.ok || (out && cap<=b.used))return false;
 if(out)memcpy(out,b.bytes,b.used+1);
 *needed=b.used+1;return true;
}

/* I preserve the original File-only entry points and exact view bytes. */
const char *nl_file_source_catalog_string(int64_t k,int64_t i,int64_t f,int64_t j) {
 return nl_service_source_catalog_string(NL_SOURCE_CATALOG_FILE,k,i,f,j);
}
int64_t nl_file_source_catalog_number(int64_t k,int64_t i,int64_t f,int64_t j) {
 return nl_service_source_catalog_number(NL_SOURCE_CATALOG_FILE,k,i,f,j);
}
bool nl_file_source_catalog_view(char *out,size_t cap,size_t *needed) {
 return nl_service_source_catalog_view(NL_SOURCE_CATALOG_FILE,out,cap,needed);
}
