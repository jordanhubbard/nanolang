/* I reconstruct normal rich tables without running their section decoders. */
#include "../src/nanoisa/service_bindings_module.h"
#include "../src/nanoisa/service_codec_internal.h"
#include "../src/nanoisa/service_websocket_nominal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {const uint8_t *data;size_t size,pos;bool ok;} Cursor;
static const uint8_t *take(Cursor *c,size_t n){
 if(!c->ok || n>c->size-c->pos){c->ok=false;return NULL;}
 const uint8_t *p=c->data+c->pos;c->pos+=n;return p;
}
static uint32_t u32(Cursor *c){const uint8_t *p=take(c,4);return p?(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24:0;}
static uint64_t u64(Cursor *c){uint64_t low=u32(c),high=u32(c);return low|(high<<32);}
static NvmV2Result websocket(const NvmModule *m){
 NvmWebSocketNominalPlan *p=NULL;NvmWebSocketNominalStatus r=nvm_websocket_nominal_plan(m,&p);nvm_websocket_nominal_plan_free(p);
 if(r==NVM_WEBSOCKET_NOMINAL_DESCRIBED)return NVM_V2_OK;
 if(r==NVM_WEBSOCKET_NOMINAL_MEMORY)return NVM_V2_ERR_TRUNCATED;
 return r==NVM_WEBSOCKET_NOMINAL_LIMIT?NVM_V2_ERR_INDEX_RANGE:NVM_V2_ERR_SECTION_TYPE;
}
static void analyze(const uint8_t *bytes,size_t size,bool private_websocket){
 Cursor c={bytes,size,0,true};NvmV2Module m={0};
 m.extra_features=u32(&c);m.links.count=u32(&c);m.callbacks.count=u32(&c);
 uint32_t nc=u32(&c),ns=u32(&c),nl=u32(&c),nf=u32(&c),ni=u32(&c),code=u32(&c),service=u32(&c),own=u32(&c);
 if(!c.ok || (uint64_t)nc+ns+nf+ni>size/4){puts("99");return;}
 m.constants.items=calloc(nc?nc:1,sizeof *m.constants.items);m.signatures.items=calloc(ns?ns:1,sizeof *m.signatures.items);
 m.functions.items=calloc(nf?nf:1,sizeof *m.functions.items);m.imports.items=calloc(ni?ni:1,sizeof *m.imports.items);
 if(!m.constants.items || !m.signatures.items || !m.functions.items || !m.imports.items)c.ok=false;
 m.constants.count=nc;m.signatures.count=ns;m.functions.count=nf;m.imports.count=ni;
 for(uint32_t i=0;c.ok && i<nc;i++){
  uint32_t tag=u32(&c),n=u32(&c);if(tag>UINT8_MAX){c.ok=false;break;}
  m.constants.items[i]=(NvmV2Constant){(uint8_t)tag,n,take(&c,n)};
 }
 for(uint32_t i=0;c.ok && i<ns;i++){
  uint32_t params=u32(&c),results=u32(&c),pn=u32(&c),rn=u32(&c);
  if(params>UINT16_MAX || results>UINT16_MAX || (pn && pn!=params) || (rn && rn!=results)){c.ok=false;break;}
  const uint8_t *p=take(&c,pn),*r=take(&c,rn);
  m.signatures.items[i]=(NvmV2Signature){(uint16_t)params,(uint16_t)results,pn?p:NULL,rn?r:NULL};
 }
 for(uint32_t i=0;c.ok && i<nf;i++){
  uint32_t sig=u32(&c),locals=u32(&c);uint64_t start=u64(&c),length=u64(&c);
  if(locals>UINT16_MAX){c.ok=false;break;}
  m.functions.items[i]=(NvmV2Function){.signature_idx=sig,.local_count=(uint16_t)locals,.code_offset=start,.code_length=length};
 }
 for(uint32_t i=0;c.ok && i<ni;i++){
  uint32_t mod=u32(&c),sym=u32(&c),sig=u32(&c),kind=u32(&c);
  if(kind>UINT8_MAX){c.ok=false;break;}
  m.imports.items[i]=(NvmV2Import){.module_name_idx=mod,.symbol_name_idx=sym,.signature_idx=sig,.kind=(uint8_t)kind};
 }
 const uint8_t *layouts=take(&c,nl);Cursor l={layouts,nl,0,c.ok};uint32_t count=u32(&l);
 if(!l.ok || count>(nl-4)/8)l.ok=false;
 if(l.ok){m.layouts.items=calloc(count?count:1,sizeof *m.layouts.items);if(!m.layouts.items)l.ok=false;else m.layouts.count=count;}
 for(uint32_t i=0;l.ok && i<count;i++){
  const uint8_t *header=take(&l,4);if(!header || header[1]){l.ok=false;break;}
  uint16_t fields=(uint16_t)(header[2]|(uint16_t)header[3]<<8);uint32_t name=u32(&l);
  NvmV2Layout *row=&m.layouts.items[i];row->kind=header[0];row->name_idx=name;row->field_count=fields;
  row->fields=calloc(fields?fields:1,sizeof *row->fields);if(!row->fields){l.ok=false;break;}
  for(uint16_t j=0;l.ok && j<fields;j++){
   const uint8_t *field=take(&l,4);if(!field || field[1] || field[2] || field[3]){l.ok=false;break;}
   uint32_t nested=u32(&l),field_name=u32(&l);row->fields[j]=(NvmV2LayoutField){field[0],nested,field_name};
  }
 }
 if(!l.ok || l.pos!=l.size)c.ok=false;
 m.code=take(&c,code);m.code_size=code;
 m.service_data=take(&c,service);m.service_size=service;if(!service)m.service_data=NULL;
 m.ownership_data=take(&c,own);m.ownership_size=own;if(!own)m.ownership_data=NULL;
 if(c.ok && c.pos==c.size){
  NvmV2Result r=private_websocket?nvm_private_nominal_wire(&m,4,7,websocket):nvm_v2_service_bindings_validate(&m);
  printf("%u|%u,%u\n",(unsigned)r,nvm_v2_service_bindings_present(&m)?1u:0u,nvm_v2_file_instructions_present(&m)?1u:0u);
 }else puts("99");
 if(m.layouts.items)for(uint32_t i=0;i<m.layouts.count;i++)free(m.layouts.items[i].fields);
 free(m.layouts.items);free(m.constants.items);free(m.signatures.items);free(m.functions.items);free(m.imports.items);
}
int main(int argc,char **argv){
 if(argc<2)return 2;
 bool websocket_route=!strcmp(argv[1],"1");
 for(int i=2;i<argc;i++){
  FILE *f=fopen(argv[i],"rb");if(!f || fseek(f,0,SEEK_END))return 2;long size=ftell(f);
  if(size<0 || fseek(f,0,SEEK_SET))return 2;
  uint8_t *bytes=malloc(size?(size_t)size:1);if(!bytes)return 2;
  if(fread(bytes,1,(size_t)size,f)!=(size_t)size || fclose(f))return 2;
  analyze(bytes,(size_t)size,websocket_route);free(bytes);
 }
 return 0;
}
