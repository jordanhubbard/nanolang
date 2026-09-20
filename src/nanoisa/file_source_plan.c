#include "file_source_plan.h"
#include "../nsi_file_catalog.h"
#include <stdlib.h>
#include <string.h>
struct NlFileSourcePlan { size_t count,bytes;NlFileSourceRow rows[]; };
static bool text_ok(NlFileSourceText s,size_t maximum,bool name) {
 if(!s.data || !s.size || s.size>maximum || memchr(s.data,0,s.size))return false;
 if(!name)return true;
 for(size_t i=0;i<s.size;i++) {
  unsigned char c=(unsigned char)s.data[i];bool letter=(c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_';
  if(!letter && !(i && c>='0'&&c<='9'))return false;
 }
 return true;
}
static bool text_equal(NlFileSourceText a,NlFileSourceText b) {
 return a.size==b.size && (!a.size || !memcmp(a.data,b.data,a.size));
}
static bool text_literal(NlFileSourceText a,const char *b) {
 NlFileSourceText v={b,strlen(b)};return text_equal(a,v);
}
static bool add_text(size_t *n,NlFileSourceText s) {
 if(s.size>=NL_FILE_SOURCE_TEXT_BUDGET-*n)return false;
 *n+=s.size+1;return true;
}
static bool id_ok(uint32_t id){return id && id<=INT32_MAX;}
static NlFileSourceText copy_text(char **p,NlFileSourceText s) {
 NlFileSourceText r={*p,s.size};memcpy(*p,s.data,s.size);(*p)[s.size]=0;*p+=s.size+1;return r;
}
NlFileSourceStatus nl_file_source_plan_build(const NlFileSourceRequest *requests,size_t nr,
 const NlFileSourceAlias *aliases,size_t na,const NlFileSourceOrdinary *ordinary,size_t no,NlFileSourcePlan **out) {
 if(!out || !requests || !nr || (na&&!aliases) || (no&&!ordinary))return NL_FILE_SOURCE_INVALID;
 if(nr>NL_FILE_SOURCE_REQUESTS || na>NL_FILE_SOURCE_ALIASES || no>NL_FILE_SOURCE_ORDINARY)return NL_FILE_SOURCE_LIMIT;
 char catalog[32768];size_t needed=0;
 if(!nl_file_source_catalog_view(catalog,sizeof catalog,&needed))return NL_FILE_SOURCE_UNRESOLVED;
 /* Bounded temporary rows borrow input until validation finishes. */
 NlFileSourceRow rows[NL_FILE_SOURCE_REQUESTS*NL_FILE_SOURCE_BINDINGS+NL_FILE_SOURCE_ALIASES];
 NlFileSourceOrdinary space[NL_FILE_SOURCE_REQUESTS*NL_FILE_SOURCE_BINDINGS+NL_FILE_SOURCE_ALIASES+NL_FILE_SOURCE_ORDINARY];
 size_t count=0,ns=0,budget=0,owned_text=0;
 for(size_t r=0;r<nr;r++) {
  const NlFileSourceRequest *q=&requests[r];
  if(!text_ok(q->module,4096,false) || !text_ok(q->interface_id,256,false) ||
     !text_ok(q->catalog_view,32767,false) || !id_ok(q->line) || !id_ok(q->column) ||
     !q->bindings || q->binding_count!=13)return NL_FILE_SOURCE_INVALID;
  if(q->catalog_version!=1 || !text_literal(q->interface_id,nl_file_catalog_interface()) ||
     q->catalog_view.size!=needed-1 || memcmp(q->catalog_view.data,catalog,needed-1))return NL_FILE_SOURCE_UNRESOLVED;
  if(!add_text(&budget,q->module)||!add_text(&budget,q->interface_id)||!add_text(&budget,q->catalog_view))return NL_FILE_SOURCE_LIMIT;
  for(size_t j=0;j<r;j++)if(text_equal(q->module,requests[j].module))return NL_FILE_SOURCE_INVALID;
  uint16_t seen=0;
  for(size_t j=0;j<13;j++) {
   const NlFileSourceBinding *b=&q->bindings[j];
   if(!id_ok(b->id)||b->kind>1||b->ordinal>=(b->kind?5u:8u)||!text_ok(b->name,128,true))return NL_FILE_SOURCE_INVALID;
   uint32_t index=b->ordinal+(b->kind?8u:0u);uint16_t bit=(uint16_t)(1u<<index);
   if((seen&bit)||!text_literal(b->name,nl_file_source_catalog_string(b->kind+1,b->ordinal,1,0)))return NL_FILE_SOURCE_INVALID;
   seen|=bit;if(!add_text(&budget,b->name))return NL_FILE_SOURCE_LIMIT;
   uint32_t category=b->kind?5u:b->ordinal==0?1u:b->ordinal==3?2u:b->ordinal<3?3u:4u;
   rows[count++]=(NlFileSourceRow){q->module,b->name,b->id,b->id,(uint32_t)r,b->kind,b->ordinal,category,
    b->kind?(uint32_t)nl_file_source_catalog_number(2,b->ordinal,3,0):NL_FILE_SOURCE_NO_INDEX,
    b->kind?3u+b->ordinal:NL_FILE_SOURCE_NO_INDEX,NL_FILE_SOURCE_NO_INDEX,NL_FILE_SOURCE_NO_INDEX,q->line,q->column};
   space[ns++]=(NlFileSourceOrdinary){q->module,b->name,b->id};
  }
 }
 size_t base_count=count;
 for(size_t a=0;a<na;a++) {
  const NlFileSourceAlias *v=&aliases[a];
  if(!id_ok(v->id)||!id_ok(v->target)||!text_ok(v->module,4096,false)||!text_ok(v->name,128,true))return NL_FILE_SOURCE_INVALID;
  if(!add_text(&budget,v->module)||!add_text(&budget,v->name))return NL_FILE_SOURCE_LIMIT;
  size_t t=0;while(t<base_count && rows[t].id!=v->target)t++;
  if(t==base_count)return NL_FILE_SOURCE_UNRESOLVED;
  NlFileSourceRow row=rows[t];row.module=v->module;row.name=v->name;row.id=v->id;row.target=v->target;
  rows[count++]=row;space[ns++]=(NlFileSourceOrdinary){v->module,v->name,v->id};
 }
 for(size_t i=0;i<no;i++) {
  const NlFileSourceOrdinary *v=&ordinary[i];
  if(!id_ok(v->id)||!text_ok(v->module,4096,false)||!text_ok(v->name,128,true))return NL_FILE_SOURCE_INVALID;
  if(!add_text(&budget,v->module)||!add_text(&budget,v->name))return NL_FILE_SOURCE_LIMIT;
  space[ns++]=*v;
 }
 for(size_t i=0;i<ns;i++)for(size_t j=0;j<i;j++)
  if(space[i].id==space[j].id || (text_equal(space[i].module,space[j].module)&&text_equal(space[i].name,space[j].name)))return NL_FILE_SOURCE_INVALID;
 for(size_t i=0;i<count;i++) {
  if(!add_text(&budget,rows[i].module)||!add_text(&budget,rows[i].name)||
     !add_text(&owned_text,rows[i].module)||!add_text(&owned_text,rows[i].name))return NL_FILE_SOURCE_LIMIT;
 }
 if(count>(SIZE_MAX-sizeof(NlFileSourcePlan))/sizeof(NlFileSourceRow))return NL_FILE_SOURCE_LIMIT;
 size_t bytes=sizeof(NlFileSourcePlan)+count*sizeof(NlFileSourceRow);
 if(owned_text>SIZE_MAX-bytes)return NL_FILE_SOURCE_LIMIT;
 bytes+=owned_text;NlFileSourcePlan *p=malloc(bytes);if(!p)return NL_FILE_SOURCE_MEMORY;
 p->count=count;p->bytes=bytes;char *cursor=(char *)(p->rows+count);
 for(size_t i=0;i<count;i++){p->rows[i]=rows[i];p->rows[i].module=copy_text(&cursor,rows[i].module);p->rows[i].name=copy_text(&cursor,rows[i].name);}
 *out=p;return NL_FILE_SOURCE_OK;
}
void nl_file_source_plan_free(NlFileSourcePlan *p){free(p);}
size_t nl_file_source_plan_count(const NlFileSourcePlan *p){return p?p->count:0;}
size_t nl_file_source_plan_bytes(const NlFileSourcePlan *p){return p?p->bytes:0;}
bool nl_file_source_plan_row(const NlFileSourcePlan *p,size_t i,NlFileSourceRow *out) {
 if(!p || !out || i>=p->count)return false;
 *out=p->rows[i];return true;
}
