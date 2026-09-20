#include "file_source_plan.h"
#include "../nsi_file_catalog.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
/* String fields: interface0; type0=id,1=name,2..4=member id/name/type;
 * method0..3=id/name/generated/binding,4..6=param id/name/type,7=owned outcome.
 * Number fields: type0=kind,1=count,2=member domain; method0..4=ABI/rights/
 * acquired/mode/param count,5..9=param direction/ownership/lifetime/mutability/
 * domain,10=outcome input state. All facts come from the existing catalog. */
const char *nl_file_source_catalog_string(int64_t k,int64_t i,int64_t f,int64_t j) {
 const char *s=NULL;
 if(k==0 && i==0 && f==0 && j==0)s=nl_file_catalog_interface();
 if(k==1 && i>=0 && i<8 && j>=0) {
  const NlFilePlanType *t=nl_file_catalog_type((size_t)i);
  if(f==0 && j==0)s=t->id;else if(f==1 && j==0)s=t->name;
  else if((uint64_t)j<t->member_count) {
   const NlFilePlanMember *m=&t->members[j];
   if(f==2)s=m->id;else if(f==3)s=m->name;else if(f==4)s=m->type_id;
  }
 }
 if(k==2 && i>=0 && i<5 && j>=0) {
  const NlFilePlanMethod *m=nl_file_catalog_method((size_t)i);
  if(f==0 && j==0)s=m->id;else if(f==1 && j==0)s=m->name;else if(f==2 && j==0)s=m->generated_name;else if(f==3 && j==0)s=m->binding_id;
  else if(f==7 && j<2)s=m->outcomes[j].owned_payload_type;
  else if((uint64_t)j<m->param_count) {
   const NlFilePlanParam *p=&m->params[j];
   if(f==4)s=p->id;else if(f==5)s=p->name;else if(f==6)s=p->type_id;
  }
 }
 return s?s:"";
}
int64_t nl_file_source_catalog_number(int64_t k,int64_t i,int64_t f,int64_t j) {
 if(k==1 && i>=0 && i<8 && j>=0) {
  const NlFilePlanType *t=nl_file_catalog_type((size_t)i);
  if(f==0 && j==0)return t->kind;
  if(f==1 && j==0)return (int64_t)t->member_count;
  if(f==2 && (uint64_t)j<t->member_count)return t->members[j].domain;
 }
 if(k==2 && i>=0 && i<5 && j>=0) {
  const NlFilePlanMethod *m=nl_file_catalog_method((size_t)i);
  if(f==0 && j==0)return m->abi_version;
  if(f==1 && j==0)return m->required_rights;
  if(f==2 && j==0)return m->acquired_rights;
  if(f==3 && j==0)return m->input_mode;
  if(f==4 && j==0)return (int64_t)m->param_count;
  if(f==10 && j<2)return m->outcomes[j].input_state;
  if((uint64_t)j<m->param_count) {
   const NlFilePlanParam *p=&m->params[j];
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
typedef char CatalogNumberText[32];
size_t nl_file_source_catalog_buffer_bytes(void) { return sizeof(CatalogText)+sizeof(CatalogNumberText); }
static void catalog_text(CatalogText *b,const char *s) {
 size_t n=strlen(s);if(!b->ok || n>=sizeof b->bytes-b->used){b->ok=false;return;}
 memcpy(b->bytes+b->used,s,n);b->used+=n;b->bytes[b->used]=0;
}
static void catalog_number(CatalogText *b,int64_t n) {
 CatalogNumberText text;int rc=snprintf(text,sizeof text,"#%" PRId64 ";",n);
 if(rc<0 || (size_t)rc>=sizeof text){b->ok=false;return;}catalog_text(b,text);
}
static void catalog_string(CatalogText *b,const char *s) {
 CatalogNumberText text;int rc=snprintf(text,sizeof text,"%zu:",strlen(s));
 if(rc<0 || (size_t)rc>=sizeof text){b->ok=false;return;}
 catalog_text(b,text);catalog_text(b,s);catalog_text(b,";");
}
bool nl_file_source_catalog_view(char *out,size_t cap,size_t *needed) {
 if(!needed || (!out && cap))return false;
 CatalogText b={{0},0,true};catalog_text(&b,"file-source-catalog1;");
 catalog_string(&b,nl_file_catalog_interface());catalog_number(&b,8);catalog_number(&b,5);
 for(int64_t i=0;i<8;i++) {
  for(int64_t f=0;f<2;f++)catalog_string(&b,nl_file_source_catalog_string(1,i,f,0));
  for(int64_t f=0;f<2;f++)catalog_number(&b,nl_file_source_catalog_number(1,i,f,0));
  int64_t count=nl_file_source_catalog_number(1,i,1,0);
  for(int64_t j=0;j<count;j++) {
   for(int64_t f=2;f<5;f++)catalog_string(&b,nl_file_source_catalog_string(1,i,f,j));
   catalog_number(&b,nl_file_source_catalog_number(1,i,2,j));
  }
 }
 for(int64_t i=0;i<5;i++) {
  for(int64_t f=0;f<4;f++)catalog_string(&b,nl_file_source_catalog_string(2,i,f,0));
  for(int64_t f=0;f<5;f++)catalog_number(&b,nl_file_source_catalog_number(2,i,f,0));
  int64_t count=nl_file_source_catalog_number(2,i,4,0);
  for(int64_t j=0;j<count;j++) {
   for(int64_t f=4;f<7;f++)catalog_string(&b,nl_file_source_catalog_string(2,i,f,j));
   for(int64_t f=5;f<10;f++)catalog_number(&b,nl_file_source_catalog_number(2,i,f,j));
  }
  for(int64_t j=0;j<2;j++) {
   catalog_number(&b,nl_file_source_catalog_number(2,i,10,j));
   catalog_string(&b,nl_file_source_catalog_string(2,i,7,j));
  }
 }
 if(!b.ok || (out && cap<=b.used))return false;
 if(out)memcpy(out,b.bytes,b.used+1);
 *needed=b.used+1;return true;
}
