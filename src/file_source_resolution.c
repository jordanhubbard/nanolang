#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include "file_source_resolution.h"
#include "nanolang.h"
#include "file_source_input.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#define FR_MODULES 5000u
#define FR_LINES 50000u
#define FR_FILE_LINES 20000u
#define FR_ROWS (NL_FILE_SOURCE_REQUESTS*NL_FILE_SOURCE_BINDINGS+NL_FILE_SOURCE_ALIASES+NL_FILE_SOURCE_ORDINARY)
#define FR_INPUT_BYTES (UINT64_C(64)*1024*1024)
typedef struct {
 char *path,*source;size_t size,next;ASTNode *ast;uint32_t ordinal;
 unsigned state;
} ResolutionModule;
typedef struct {
 uint32_t owner,target_module;ASTNode *node;
} ResolutionImport;
typedef struct {
 NlFileVisibility value;uint32_t owner,target_owner;char namespace_key[4097];char *owned_qualifier;
} ResolutionName;
struct NlFileResolution {
 NlFileResolutionReport report;
 ResolutionModule modules[FR_MODULES];
 ResolutionImport imports[FR_LINES];size_t import_count;
 ResolutionName names[FR_ROWS];size_t name_count,base_count;
 uint32_t sorted[FR_MODULES],postorder[FR_MODULES];size_t finished;
 NlFileCompanionRequest requests[NL_FILE_SOURCE_REQUESTS];
 NlFileSourceRequest plan_requests[NL_FILE_SOURCE_REQUESTS];
 NlFileSourceBinding bindings[NL_FILE_SOURCE_REQUESTS][NL_FILE_SOURCE_BINDINGS];
 NlFileSourceAlias aliases[NL_FILE_SOURCE_ALIASES];
 NlFileSourceOrdinary ordinary[NL_FILE_SOURCE_ORDINARY];
 size_t alias_count,ordinary_count;
 NlFileCompanionSet *snapshots;NlFileSourcePlan *plan;
 uint64_t max_files,max_lines,max_file_lines,lines;
};
static NlFileSourceText fr_text(const char *s){return (NlFileSourceText){s,strlen(s)};}
static bool fr_work(NlFileResolution *r,uint64_t n) {
 if(n>NL_FILE_COMPANION_WORK-r->report.work){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
 r->report.work+=n;return true;
}
static uint64_t fr_limit(const char *name,uint64_t cap) {
 const char *s=getenv(name);if(!s||!*s)return cap;
 uint64_t n=0;for(;*s;s++) {
  if(*s<'0'||*s>'9'||n>(UINT64_MAX-(unsigned)(*s-'0'))/10)return cap;
  n=n*10+(unsigned)(*s-'0');
 }
 return n&&n<cap?n:cap;
}
static bool fr_source(NlFileResolution *r,ResolutionModule *m) {
 if(r->report.source_bytes>=FR_INPUT_BYTES){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
 NlFileCompanionReport read=nl_file_source_input_read(m->path,(size_t)(FR_INPUT_BYTES-r->report.source_bytes),&m->source,&m->size);
 if(read.status!=NL_FILE_COMPANION_OK) {
  r->report.companion=read;
  r->report.status=read.status==NL_FILE_COMPANION_LIMIT?NL_FILE_RESOLUTION_LIMIT:
   read.status==NL_FILE_COMPANION_MEMORY?NL_FILE_RESOLUTION_MEMORY:
   read.status==NL_FILE_COMPANION_INVALID?NL_FILE_RESOLUTION_INVALID:NL_FILE_RESOLUTION_IO;return false;
 }
 r->report.input_bytes+=m->size+1;r->report.source_bytes+=m->size+1;uint64_t lines=1;
 for(size_t i=0;i<m->size;i++)if(m->source[i]=='\n')lines++;
 if(lines>r->max_file_lines||lines>r->max_lines-r->lines){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
 r->lines+=lines;
 if(!fr_work(r,read.work_reserved+m->size+1))return false;
 int count=0;Token *tokens=tokenize(m->source,&count);
 if(!tokens){r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}
 m->ast=parse_program(tokens,count);free_tokens(tokens,count);
 if(!m->ast){r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}
 return true;
}
static int fr_module(NlFileResolution *r,const char *path) {
 char *canonical=realpath(path,NULL);
 if(!canonical){r->report.status=NL_FILE_RESOLUTION_IO;return -1;}
 size_t length=strlen(canonical);
 if(length>NL_FILE_COMPANION_PATH){free(canonical);r->report.status=NL_FILE_RESOLUTION_LIMIT;return -1;}
 for(size_t i=0;i<r->report.modules;i++) {
  if(!fr_work(r,length+1)){free(canonical);return -1;}
  if(!strcmp(canonical,r->modules[i].path)){free(canonical);return (int)i;}
 }
 if(r->report.modules>=r->max_files) {
  free(canonical);r->report.status=NL_FILE_RESOLUTION_LIMIT;return -1;
 }
 size_t i=r->report.modules++;ResolutionModule *m=&r->modules[i];m->path=canonical;
 r->report.input_bytes+=length+1;
 if(!fr_source(r,m))return -1;
 return (int)i;
}
static bool fr_graph(NlFileResolution *r,const char *path) {
 int root=fr_module(r,path);if(root<0)return false;
 uint32_t stack[FR_MODULES];size_t depth=1;stack[0]=(uint32_t)root;r->modules[root].state=1;
 while(depth) {
  uint32_t owner=stack[depth-1];ResolutionModule *m=&r->modules[owner];
  if(m->next==(size_t)m->ast->as.program.count) {
   m->state=2;r->postorder[r->finished++]=owner;depth--;continue;
  }
  ASTNode *node=m->ast->as.program.items[m->next++];
  if(!fr_work(r,1))return false;
  if(node->type!=AST_IMPORT)continue;
  if(r->import_count==FR_LINES){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
  const char *selected=resolve_module_path(node->as.import_stmt.module_path,m->path);
  if(!selected){r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}
  /* Package extraction is effectful and does not provide retained source facts. */
  if(strstr(selected,".nano.tar.zst")){free((char *)selected);r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}
  int target=fr_module(r,selected);free((char *)selected);if(target<0)return false;
  r->imports[r->import_count++]=(ResolutionImport){owner,(uint32_t)target,node};
  if(r->modules[target].state==1){r->report.status=NL_FILE_RESOLUTION_INVALID;return false;}
  if(!r->modules[target].state){r->modules[target].state=1;stack[depth++]=(uint32_t)target;}
 }
 for(size_t i=0;i<r->report.modules;i++)r->sorted[i]=(uint32_t)i;
 for(size_t i=1;i<r->report.modules;i++) {
  uint32_t v=r->sorted[i];size_t j=i;
  while(j) {
   if(!fr_work(r,strlen(r->modules[v].path)+strlen(r->modules[r->sorted[j-1]].path)+1))return false;
   if(strcmp(r->modules[r->sorted[j-1]].path,r->modules[v].path)<=0)break;
   r->sorted[j]=r->sorted[j-1];j--;
  }
  r->sorted[j]=v;
 }
 for(size_t i=0;i<r->report.modules;i++)r->modules[r->sorted[i]].ordinal=(uint32_t)i+1;
 return true;
}
static bool fr_identifier(const char *s) {
 size_t n=strlen(s);if(!n||n>128)return false;
 for(size_t i=0;i<n;i++) {
  unsigned char c=(unsigned char)s[i];
  if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_'||(i&&c>='0'&&c<='9')))return false;
 }
 return true;
}
static bool fr_qualifier(const char *s) {
 size_t start=0,n=strlen(s);if(n>4096)return false;
 for(size_t i=0;i<=n;i++)if(i==n||s[i]=='.') {
  size_t width=i-start;if(!width||width>128)return false;
  for(size_t j=start;j<i;j++) {
   unsigned char c=(unsigned char)s[j];
   if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_'||(j>start&&c>='0'&&c<='9')))return false;
  }
  start=i+1;
 }
 return true;
}
static bool fr_join_qualifier(NlFileResolution *r,char out[4097],const char *a,const char *b) {
 size_t an=strlen(a),bn=strlen(b),dot=an&&bn?1:0;
 if(an>4096||bn>4096-an||dot>4096-an-bn){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
 if(!fr_work(r,an+bn+dot+1))return false;
 memcpy(out,a,an);if(dot)out[an]='.';memcpy(out+an+dot,b,bn+1);return true;
}
static bool fr_name(NlFileResolution *r,uint32_t owner,const char *qualifier,const char *name,
                    bool exported,bool service,uint32_t kind,uint32_t ordinal,const ResolutionName *target) {
 if(!fr_identifier(name)||(*qualifier&&!fr_qualifier(qualifier))) {
  r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;
 }
 for(size_t i=0;i<r->name_count;i++) {
  ResolutionName *old=&r->names[i];if(!fr_work(r,strlen(name)+strlen(qualifier)+2))return false;
  if(old->owner==owner&&!strcmp(old->value.qualifier.data,qualifier)&&!strcmp(old->value.name.data,name)) {
   if(target&&old->value.target==target->value.target&&old->value.exported==exported)return true;
   r->report.status=NL_FILE_RESOLUTION_INVALID;return false;
  }
 }
 if(r->name_count==FR_ROWS){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
 ResolutionName *n=&r->names[r->name_count];memset(n,0,sizeof *n);
 n->owner=owner;n->target_owner=target?target->target_owner:owner;
 uint32_t id=(uint32_t)r->name_count+1;
 NlFileSourceText origin=fr_text(r->modules[owner].path);
 n->value=(NlFileVisibility){origin,fr_text(qualifier),fr_text(name),
  target?target->value.target_origin:origin,target?target->value.target_name:fr_text(name),
  id,target?target->value.target:id,kind,ordinal,exported,service};
 int used=*qualifier?snprintf(n->namespace_key,sizeof n->namespace_key,"N%zu:%s%zu:%s",origin.size,origin.data,strlen(qualifier),qualifier):
  snprintf(n->namespace_key,sizeof n->namespace_key,"%s",origin.data);
 if(used<0||(size_t)used>=sizeof n->namespace_key){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
 if(!fr_work(r,(uint64_t)used+sizeof *n+strlen(qualifier)+1))return false;
 n->owned_qualifier=strdup(qualifier);
 if(!n->owned_qualifier){r->report.status=NL_FILE_RESOLUTION_MEMORY;return false;}
 n->value.qualifier.data=n->owned_qualifier;
 r->report.input_bytes+=strlen(qualifier)+1;
 r->name_count++;return true;
}
static bool fr_locals(NlFileResolution *r) {
 for(size_t sorted=0;sorted<r->report.modules;sorted++) {
  uint32_t owner=r->sorted[sorted];ResolutionModule *m=&r->modules[owner];
  if(m->ast->as.program.count>4096){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
  for(int i=0;i<m->ast->as.program.count;i++) {
   ASTNode *node=m->ast->as.program.items[i];const char *name=NULL;bool exported=false;uint32_t kind=0;
   switch(node->type) {
    case AST_FUNCTION:if(node->as.function.is_anonymous){r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}name=node->as.function.name;exported=node->as.function.is_pub;kind=2;break;
    case AST_STRUCT_DEF:name=node->as.struct_def.name;exported=node->as.struct_def.is_pub;kind=3;break;
    case AST_ENUM_DEF:name=node->as.enum_def.name;exported=node->as.enum_def.is_pub;kind=4;break;
    case AST_UNION_DEF:if(node->as.union_def.generic_param_count){r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}name=node->as.union_def.name;exported=node->as.union_def.is_pub;kind=5;break;
    case AST_OPAQUE_TYPE:name=node->as.opaque_type.name;kind=6;break;
    case AST_LET:if(node->as.let.is_destructure||node->as.let.is_destructure_projection){r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}name=node->as.let.name;exported=!node->as.let.is_mut;kind=7;break;
    case AST_SERVICE_DECL: {
     if(r->report.services==NL_FILE_SOURCE_REQUESTS){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
     size_t q=r->report.services++;
     node->as.service_decl.origin_index=m->ordinal-1;
     r->requests[q]=(NlFileCompanionRequest){fr_text(m->path),
      {node->as.service_decl.document_path,(size_t)node->as.service_decl.path_bytes},
      {node->as.service_decl.interface_id,(size_t)node->as.service_decl.interface_bytes},
      (uint32_t)node->as.service_decl.catalog_version,(uint32_t)node->line,(uint32_t)node->column};
     for(uint32_t b=0;b<13;b++) {
      uint32_t type=b<8?0:1,ordinal=b<8?b:b-8;
      const char *binding=nl_file_source_catalog_string(type+1,ordinal,1,0);
      if(!fr_name(r,owner,"",binding,true,true,type,ordinal,NULL))return false;
      r->bindings[q][b]=(NlFileSourceBinding){r->names[r->name_count-1].value.id,type,ordinal,fr_text(binding)};
     }
     break;
    }
    case AST_IMPORT:case AST_MODULE_DECL:case AST_SHADOW:break;
    default:r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;
   }
   if(name&&!fr_name(r,owner,"",name,exported,false,kind,0,NULL))return false;
  }
 }
 r->base_count=r->name_count;return true;
}
static bool fr_metadata(NlFileResolution *r) {
 for(size_t i=0;i<r->report.modules;i++) {
  const char *path=r->modules[i].path;size_t size=strlen(path);
  int64_t status=nl_file_source_metadata(path,(int64_t)size);
  if(status) {r->report.status=status<0?NL_FILE_RESOLUTION_IO:NL_FILE_RESOLUTION_UNRESOLVED;return false;}
  if(!fr_work(r,size+12))return false;
 }
 return true;
}
static bool fr_import_names(NlFileResolution *r) {
 for(size_t order=0;order<r->finished;order++) {
  uint32_t owner=r->postorder[order];
  for(size_t e=0;e<r->import_count;e++) {
   if(!fr_work(r,1))return false;
   ResolutionImport *edge=&r->imports[e];if(edge->owner!=owner)continue;
   ASTNode *imp=edge->node;const char *prefix=imp->as.import_stmt.module_alias;
   if(!prefix)prefix="";
   if(*prefix&&!fr_name(r,owner,"",prefix,imp->as.import_stmt.is_pub_use,false,8,0,NULL))return false;
   size_t available=r->name_count;
   if(imp->as.import_stmt.is_selective&&!imp->as.import_stmt.is_wildcard) {
    for(int j=0;j<imp->as.import_stmt.import_symbol_count;j++) {
     const char *original=imp->as.import_stmt.import_symbols[j];
     const char *alias=imp->as.import_stmt.import_aliases?imp->as.import_stmt.import_aliases[j]:NULL;
     if(!alias||!*alias)alias=original;
     const ResolutionName *target=NULL;
     for(size_t n=0;n<available;n++) {
      ResolutionName *v=&r->names[n];if(!fr_work(r,strlen(original)+1))return false;
      if(v->owner==edge->target_module&&v->value.exported&&!v->value.qualifier.size&&!strcmp(v->value.name.data,original)) {
       if(target){r->report.status=NL_FILE_RESOLUTION_INVALID;return false;}target=v;
      }
     }
     if(!target){r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}
     if(!fr_name(r,owner,prefix,alias,imp->as.import_stmt.is_pub_use,target->value.service,target->value.kind,target->value.ordinal,target))return false;
     if(target->value.kind==8) {
      size_t original_size=strlen(original);
      for(size_t n=0;n<available;n++) {
       const ResolutionName *v=&r->names[n];
       if(!fr_work(r,original_size+v->value.qualifier.size+2))return false;
       if(v->owner!=edge->target_module||!v->value.exported||v->value.qualifier.size<original_size||
          strncmp(v->value.qualifier.data,original,original_size)||
          (v->value.qualifier.data[original_size]&&v->value.qualifier.data[original_size]!='.'))continue;
       const char *suffix=v->value.qualifier.data+original_size;if(*suffix=='.')suffix++;
       char renamed[4097],qualified[4097];
       if(!fr_join_qualifier(r,renamed,alias,suffix)||!fr_join_qualifier(r,qualified,prefix,renamed))return false;
       if(!fr_name(r,owner,qualified,v->value.name.data,imp->as.import_stmt.is_pub_use,v->value.service,v->value.kind,v->value.ordinal,v))return false;
      }
     }
    }
   } else for(size_t n=0;n<available;n++) {
    const ResolutionName *v=&r->names[n];
    if(!fr_work(r,1))return false;
    if(v->owner!=edge->target_module||!v->value.exported)continue;
    char qualifier[4097];if(!fr_join_qualifier(r,qualifier,prefix,v->value.qualifier.data))return false;
    if(!fr_name(r,owner,qualifier,v->value.name.data,imp->as.import_stmt.is_pub_use,v->value.service,v->value.kind,v->value.ordinal,v))return false;
   }
  }
 }
 return true;
}
static bool fr_sort_aliases(NlFileResolution *r) {
 for(size_t i=r->base_count;i<r->name_count;i++) {
  /* I reserve all comparisons before moving owning pointers. */
  if(!fr_work(r,(uint64_t)(i-r->base_count+1)*16384))return false;
  ResolutionName value=r->names[i];size_t j=i;
  while(j>r->base_count) {
   ResolutionName *old=&r->names[j-1];
   int cmp=strcmp(old->namespace_key,value.namespace_key);
   if(cmp<0||(cmp==0&&strcmp(old->value.name.data,value.value.name.data)<=0))break;
   r->names[j]=*old;j--;
  }
  r->names[j]=value;
 }
 for(size_t i=r->base_count;i<r->name_count;i++)r->names[i].value.id=(uint32_t)i+1;
 /* Namespace aliases are created after original declarations, so sorting also
  * remaps their original target IDs by the retained origin/name tuple. */
 for(size_t i=r->base_count;i<r->name_count;i++)if(r->names[i].value.kind==8) {
  ResolutionName *v=&r->names[i];bool found=false;
  for(size_t j=r->base_count;j<r->name_count;j++) {
   ResolutionName *candidate=&r->names[j];if(!fr_work(r,8192))return false;
   if(candidate->value.kind==8&&!candidate->value.qualifier.size&&
      !strcmp(candidate->value.origin.data,v->value.target_origin.data)&&
      !strcmp(candidate->value.name.data,v->value.target_name.data)) {
    v->value.target=candidate->value.id;found=true;break;
   }
  }
  if(!found){r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}
 }
 return true;
}
static bool fr_plan(NlFileResolution *r) {
 r->report.companion=nl_file_companions_prepare(r->requests,r->report.services,&r->snapshots);
 if(r->report.companion.status!=NL_FILE_COMPANION_OK) {
  r->report.status=r->report.companion.status==NL_FILE_COMPANION_MEMORY?NL_FILE_RESOLUTION_MEMORY:
   r->report.companion.status==NL_FILE_COMPANION_LIMIT?NL_FILE_RESOLUTION_LIMIT:
   r->report.companion.status==NL_FILE_COMPANION_IO?NL_FILE_RESOLUTION_IO:NL_FILE_RESOLUTION_UNRESOLVED;
  return false;
 }
 for(size_t q=0;q<r->report.services;q++) {
  NlFileCompanionView v;if(!nl_file_companions_view(r->snapshots,q,&v)){r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;return false;}
  r->plan_requests[q]=(NlFileSourceRequest){v.request.module_path,v.request.interface_id,v.catalog_view,
   v.request.catalog_version,v.request.line,v.request.column,r->bindings[q],13};
 }
 for(size_t i=0;i<r->name_count;i++) {
  ResolutionName *n=&r->names[i];NlFileSourceText key=fr_text(n->namespace_key);
  if(!n->value.service) {
   if(r->ordinary_count==NL_FILE_SOURCE_ORDINARY){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
   r->ordinary[r->ordinary_count++]=(NlFileSourceOrdinary){key,n->value.name,n->value.id};
  } else if(i>=r->base_count) {
   if(r->alias_count==NL_FILE_SOURCE_ALIASES){r->report.status=NL_FILE_RESOLUTION_LIMIT;return false;}
   r->aliases[r->alias_count++]=(NlFileSourceAlias){key,n->value.name,n->value.id,n->value.target};
  }
 }
 NlFileSourceStatus status=nl_file_source_plan_build(r->plan_requests,r->report.services,r->aliases,r->alias_count,
  r->ordinary,r->ordinary_count,&r->plan);
 if(status!=NL_FILE_SOURCE_OK){r->report.status=status==NL_FILE_SOURCE_MEMORY?NL_FILE_RESOLUTION_MEMORY:
  status==NL_FILE_SOURCE_LIMIT?NL_FILE_RESOLUTION_LIMIT:NL_FILE_RESOLUTION_UNRESOLVED;return false;}
 r->report.plan_bytes=nl_file_source_plan_bytes(r->plan);
 r->report.ordinary=r->ordinary_count;r->report.visibility=r->name_count;return true;
}
void nl_file_source_resolution_free(NlFileResolution *r) {
 if(!r)return;
 for(size_t i=0;i<r->name_count;i++)free(r->names[i].owned_qualifier);
 nl_file_source_plan_free(r->plan);nl_file_companions_free(r->snapshots);
 for(size_t i=0;i<r->report.modules;i++){free_ast(r->modules[i].ast);free(r->modules[i].source);free(r->modules[i].path);}
 free(r);
}
NlFileResolutionReport nl_file_source_resolve(const char *path,NlFileResolution **out) {
 NlFileResolutionReport report={0};report.status=NL_FILE_RESOLUTION_INVALID;
 if(!path||!*path||!out)return report;
 NlFileResolution *r=calloc(1,sizeof *r);if(!r){report.status=NL_FILE_RESOLUTION_MEMORY;return report;}
 r->report.status=NL_FILE_RESOLUTION_UNRESOLVED;
 r->max_files=fr_limit("NANO_IMPORT_MAX_FILES",FR_MODULES);
 r->max_lines=fr_limit("NANO_IMPORT_MAX_LINES",FR_LINES);
 r->max_file_lines=fr_limit("NANO_IMPORT_MAX_LINES_PER_FILE",FR_FILE_LINES);
 r->report.input_bytes=sizeof *r;r->report.work=sizeof *r;
 if(fr_graph(r,path)) {
  bool services=false;
  for(size_t i=0;i<r->report.modules;i++)services|=ast_has_service_declaration(r->modules[i].ast);
  if(!services){r->report.status=NL_FILE_RESOLUTION_NONE;goto done;}
  if(fr_locals(r)&&fr_metadata(r)&&fr_import_names(r)&&fr_sort_aliases(r)&&fr_plan(r)) {
   r->report.status=NL_FILE_RESOLUTION_PREPARED;*out=r;return r->report;
  }
 }
 done:report=r->report;nl_file_source_resolution_free(r);return report;
}
size_t nl_file_source_resolution_origins(const NlFileResolution *r){return r?r->report.modules:0;}
bool nl_file_source_resolution_origin(const NlFileResolution *r,size_t i,NlFileSourceText *path,NlFileSourceText *source) {
 if(!r||i>=r->report.modules||!path||!source)return false;
 const ResolutionModule *m=&r->modules[r->sorted[i]];
 *path=fr_text(m->path);*source=(NlFileSourceText){m->source,m->size};return true;
}
size_t nl_file_source_resolution_visibility(const NlFileResolution *r){return r?r->name_count:0;}
bool nl_file_source_resolution_row(const NlFileResolution *r,size_t i,NlFileVisibility *out) {
 if(!r||i>=r->name_count||!out)return false;
 *out=r->names[i].value;return true;
}

bool nl_file_source_resolution_snapshot(const NlFileResolution *r,size_t i,NlFileCompanionView *out) {
 return r&&nl_file_companions_view(r->snapshots,i,out);
}

size_t nl_file_source_resolution_plan_rows(const NlFileResolution *r) {
 return r?nl_file_source_plan_count(r->plan):0;
}
bool nl_file_source_resolution_plan_row(const NlFileResolution *r,size_t i,NlFileSourceRow *out) {
 return r&&nl_file_source_plan_row(r->plan,i,out);
}
