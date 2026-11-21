/* I expose actual nominal plans/getters through normal immutable carriers. */
#include "../src/nanoisa/service_file_nominal.h"
#include "../src/nanoisa/service_socket_nominal.h"
#include "../src/nanoisa/service_websocket_nominal.h"
#include "../src/nanoisa/services_nominal.h"
#include "../src/nanoisa/nvm_v2_sections.h"
#include "../src/nanoisa/websocket_codec.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {bool found;uint32_t global,instance,catalog,ordinal,source,kind,flags,category;} Row;
static Row missing(void){return (Row){false,UINT32_MAX,UINT32_MAX,0,UINT32_MAX,UINT32_MAX,0,0,0};}
#define SINGLE_ROW(v,c) ((Row){true,(v).global_index,(v).catalog_ordinal==UINT32_MAX?UINT32_MAX:0,(v).catalog_ordinal==UINT32_MAX?0:c,(v).catalog_ordinal,(v).source_ordinal,(v).layout_kind,(v).ownership_flags,(uint32_t)(v).category})
#define SINGLE_CASE(n,T,call) case n:{T v,prior;memset(&v,0xa5,sizeof v);memcpy(&prior,&v,sizeof v);if(!(call)){assert(!memcmp(&v,&prior,sizeof v));return missing();}return SINGLE_ROW(v,n-1);}
static Row get_layout(unsigned profile,const void *plan,uint32_t index){
 switch(profile){
 SINGLE_CASE(2,NvmFileNominalLayout,nvm_file_nominal_layout(plan,index,&v))
 SINGLE_CASE(3,NvmSocketNominalLayout,nvm_socket_nominal_layout(plan,index,&v))
 SINGLE_CASE(4,NvmWebSocketNominalLayout,nvm_websocket_nominal_layout(plan,index,&v))
 default:{NvmMultiNominalLayout v,prior;memset(&v,0xa5,sizeof v);memcpy(&prior,&v,sizeof v);
 if(!nvm_multi_nominal_layout(plan,index,&v)){assert(!memcmp(&v,&prior,sizeof v));return missing();}
 NvmServicesNominalLayout unified;assert(nvm_services_nominal_layout(plan,index,&unified));
 return (Row){true,v.global_index,v.instance,v.catalog,v.catalog_ordinal,v.source_ordinal,v.layout_kind,v.ownership_flags,(uint32_t)unified.category};}
 }
}
static Row get_type(unsigned profile,const void *plan,uint32_t instance,uint32_t ordinal){
 if(profile!=5 && instance)return missing();
 switch(profile){
 SINGLE_CASE(2,NvmFileNominalLayout,nvm_file_nominal_type(plan,ordinal,&v))
 SINGLE_CASE(3,NvmSocketNominalLayout,nvm_socket_nominal_type(plan,ordinal,&v))
 SINGLE_CASE(4,NvmWebSocketNominalLayout,nvm_websocket_nominal_type(plan,ordinal,&v))
 default:{NvmMultiNominalLayout v,prior;memset(&v,0xa5,sizeof v);memcpy(&prior,&v,sizeof v);
 if(!nvm_multi_nominal_type(plan,instance,ordinal,&v)){assert(!memcmp(&v,&prior,sizeof v));return missing();}
 return get_layout(profile,plan,v.global_index);}
 }
}
static Row get_source(unsigned profile,const void *plan,uint8_t kind,uint32_t ordinal){
 switch(profile){
 SINGLE_CASE(2,NvmFileNominalLayout,nvm_file_nominal_source(plan,kind,ordinal,&v))
 SINGLE_CASE(3,NvmSocketNominalLayout,nvm_socket_nominal_source(plan,kind,ordinal,&v))
 SINGLE_CASE(4,NvmWebSocketNominalLayout,nvm_websocket_nominal_source(plan,kind,ordinal,&v))
 default:{NvmServicesNominalLayout v,prior;memset(&v,0xa5,sizeof v);memcpy(&prior,&v,sizeof v);
 if(!nvm_services_nominal_source(plan,kind,ordinal,&v)){assert(!memcmp(&v,&prior,sizeof v));return missing();}
 return get_layout(profile,plan,v.global_index);}
 }
}
static int64_t get_import(unsigned profile,const void *plan,uint32_t instance,uint32_t ordinal){
 if(profile!=5 && instance)return -1;
 uint32_t value=123456;bool ok=false;
 switch(profile){case 2:ok=nvm_file_nominal_import(plan,ordinal,&value);break;
 case 3:ok=nvm_socket_nominal_import(plan,ordinal,&value);break;
 case 4:ok=nvm_websocket_nominal_import(plan,ordinal,&value);break;
 default:ok=nvm_multi_nominal_import(plan,instance,ordinal,&value);break;}
 if(!ok){assert(value==123456);return -1;}return value;
}
#define BUILD_CASE(n,T,fn) case n:{T *p=(T *)&sentinel;status=fn(m,&p);if(status)assert(p==(T *)&sentinel);else *out=p;break;}
static unsigned build(unsigned profile,const NvmModule *m,void **out){
 unsigned status=1;static int sentinel;
 switch(profile){BUILD_CASE(2,NvmFileNominalPlan,nvm_file_nominal_plan)
 BUILD_CASE(3,NvmSocketNominalPlan,nvm_socket_nominal_plan)
 BUILD_CASE(4,NvmWebSocketNominalPlan,nvm_websocket_nominal_plan)
 BUILD_CASE(5,NvmMultiNominalPlan,nvm_multi_nominal_plan)}return status;
}
static uint32_t layout_count(unsigned profile,const void *p){switch(profile){case 2:return nvm_file_nominal_layout_count(p);case 3:return nvm_socket_nominal_layout_count(p);case 4:return nvm_websocket_nominal_layout_count(p);default:return nvm_multi_nominal_layout_count(p);}}
static void release(unsigned profile,void *p){switch(profile){case 2:nvm_file_nominal_plan_free(p);break;case 3:nvm_socket_nominal_plan_free(p);break;case 4:nvm_websocket_nominal_plan_free(p);break;default:nvm_multi_nominal_plan_free(p);}}
static void row_print(Row r){printf("%u,%u,%u,%u,%u,%u,%u,%u,%u,",(unsigned)r.found,r.global,r.instance,r.catalog,r.ordinal,r.source,r.kind,r.flags,r.category);}
static void report(unsigned profile,const NvmModule *m){
 void *plan=NULL;unsigned status=build(profile,m,&plan);if(status){assert(!plan);printf("%u\n",status);return;}
 uint32_t count=layout_count(profile,plan),instances=profile==5?nvm_multi_nominal_instance_count(plan):1;
 printf("0|%u,%u,",count,instances);
 for(uint32_t i=0;i<=count;i++)row_print(get_layout(profile,plan,i));
 for(uint32_t i=0;i<=instances;i++){
  Row first=get_type(profile,plan,i,0);unsigned types=first.found?nvm_multi_nominal_catalog_types(first.catalog):0;
  unsigned methods=first.found?nvm_multi_nominal_catalog_methods(first.catalog):0;
  for(unsigned j=0;j<=types;j++)row_print(get_type(profile,plan,i,j));
  for(unsigned j=0;j<=methods;j++)printf("%lld,",(long long)get_import(profile,plan,i,j));
 }
 for(uint8_t kind=0;kind<4;kind++){
  uint32_t end=kind==0?m->struct_count:kind==2?m->union_count:0;
  for(uint32_t i=0;i<=end;i++)row_print(get_source(profile,plan,kind,i));
 }
 putchar('\n');release(profile,plan);
}
static uint32_t read32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static bool take(size_t *cursor, size_t length, size_t size, size_t *start) {
    if (length > size - *cursor) return false;
    *start = *cursor; *cursor += length;
    return true;
}
static void analyze_model(const unsigned char *bytes,size_t size,unsigned profile,const uint8_t *service,uint32_t service_size,NvmImportEntry *imports,uint8_t **import_params,uint32_t import_count,uint32_t refs,uint32_t callbacks) {
    if (size < 28) { puts("99"); return; }
    uint32_t layout_size = read32(bytes), own_size = read32(bytes + 4);
    uint32_t strings = read32(bytes + 8), functions = read32(bytes + 12);
    size_t cursor = 28;
    if (strings > (size - cursor) / 4) { puts("99"); return; }
    NvmModule model = {0};
    model.struct_count = read32(bytes + 16); model.enum_count = read32(bytes + 20);
    model.union_count = read32(bytes + 24);
    char **names = calloc(strings ? strings : 1, sizeof(*names));
    uint32_t *lengths = calloc(strings ? strings : 1, sizeof(*lengths));
    NvmFunctionEntry *entries = NULL; uint8_t **params = NULL;
    bool ok = names && lengths;
    for (uint32_t i = 0; ok && i < strings; ++i) {
        size_t at;
        if (!take(&cursor, 4, size, &at)) { ok = false; break; }
        uint32_t length = read32(bytes + at);
        if (!take(&cursor, length, size, &at)) { ok = false; break; }
        names[i] = malloc((size_t)length + 1);
        if (!names[i]) { ok = false; break; }
        memcpy(names[i], bytes + at, length); names[i][length] = 0; lengths[i] = length;
    }
    if (ok && functions > (size - cursor) / 32) ok = false;
    if (ok) {
        entries = calloc(functions ? functions : 1, sizeof(*entries));
        params = calloc(functions ? functions : 1, sizeof(*params));
        if (!entries || !params) ok = false;
    }
    for (uint32_t i = 0; ok && i < functions; ++i) {
        size_t at;
        if (!take(&cursor, 32, size, &at)) { ok = false; break; }
        const unsigned char *p = bytes + at;
        uint32_t arity = read32(p + 8), locals = read32(p + 12), upvalues = read32(p + 16);
        uint32_t results = read32(p + 20), result_tag = read32(p + 24), types = read32(p + 28);
        if (arity > UINT16_MAX || locals > UINT16_MAX || upvalues > UINT16_MAX || results > UINT8_MAX ||
            result_tag > UINT8_MAX || (types && types != arity) || !take(&cursor, types, size, &at)) { ok = false; break; }
        entries[i].code_offset = read32(p); entries[i].code_length = read32(p + 4);
        entries[i].arity = (uint16_t)arity; entries[i].local_count = (uint16_t)locals;
        entries[i].upvalue_count = (uint16_t)upvalues; entries[i].result_count = (uint8_t)results;
        entries[i].result_tag = (uint8_t)result_tag;
        params[i] = types ? (uint8_t *)(bytes + at) : NULL;
    }
    size_t layout_at = 0, own_at = 0;
    if (ok && (!take(&cursor, layout_size, size, &layout_at) ||
               !take(&cursor, own_size, size, &own_at) || size - cursor > UINT32_MAX)) ok = false;
    if (ok) {
        model.strings = names; model.string_lengths = lengths; model.string_count = strings;
        model.functions = entries; model.function_param_types = params; model.function_count = functions;
        model.layout_data = layout_size ? (uint8_t *)(bytes + layout_at) : NULL; model.layout_size = layout_size;
        model.ownership_data = own_size ? (uint8_t *)(bytes + own_at) : NULL; model.ownership_size = own_size;
        model.code_size = (uint32_t)(size - cursor); model.code = model.code_size ? (uint8_t *)(bytes + cursor) : NULL;
        model.service_data=(uint8_t *)service;model.service_size=service_size;
        model.imports=imports;model.import_param_types=import_params;model.import_count=import_count;
        model.module_ref_count=refs;model.callback_contract_count=callbacks;
        report(profile,&model);
    } else puts("99");
    if (names) for (uint32_t i = 0; i < strings; ++i) free(names[i]);
    free(names); free(lengths); free(entries); free(params);
}
static void analyze(const uint8_t *bytes,size_t size,unsigned profile){
 if(size<16){puts("99");return;}size_t cursor=16;
 uint32_t service_size=read32(bytes),count=read32(bytes+4),refs=read32(bytes+8),callbacks=read32(bytes+12);
 if(count>(size-cursor)/24){puts("99");return;}
 NvmImportEntry *imports=calloc(count?count:1,sizeof *imports);uint8_t **params=calloc(count?count:1,sizeof *params);
 bool ok=imports && params;
 for(uint32_t i=0;ok && i<count;i++){
  size_t at;if(!take(&cursor,24,size,&at)){ok=false;break;}
  const uint8_t *p=bytes+at;uint32_t kind=read32(p+8),arity=read32(p+12),result=read32(p+16),types=read32(p+20);
  if(kind>UINT8_MAX || arity>UINT16_MAX || result>UINT8_MAX || (types && types!=arity) || !take(&cursor,types,size,&at)){ok=false;break;}
  imports[i].module_name_idx=read32(p);imports[i].function_name_idx=read32(p+4);imports[i].kind=(uint8_t)kind;
  imports[i].param_count=(uint16_t)arity;imports[i].return_type=(uint8_t)result;params[i]=types?(uint8_t *)(bytes+at):NULL;
 }
 size_t start=0;if(ok)ok=take(&cursor,service_size,size,&start);
 if(ok)analyze_model(bytes+cursor,size-cursor,profile,bytes+start,service_size,imports,params,count,refs,callbacks);else puts("99");
 free(imports);free(params);
}
static void write32(FILE *f,uint32_t n) {
    unsigned char b[4]={(unsigned char)n,(unsigned char)(n>>8),(unsigned char)(n>>16),(unsigned char)(n>>24)};
    assert(fwrite(b,1,4,f)==4);
}
static int extract_module(unsigned profile,const char *input,const char *output) {
    FILE *f=fopen(input,"rb");if(!f || fseek(f,0,SEEK_END))return 2;
    long size=ftell(f);if(size<0 || fseek(f,0,SEEK_SET))return 2;
    unsigned char *bytes=malloc(size?(size_t)size:1);if(!bytes)return 2;
    if(fread(bytes,1,(size_t)size,f)!=(size_t)size || fclose(f))return 2;
    NvmV2Module module={0};NvmModule *m=NULL;
    NvmV2Result status=profile==4?nvm_websocket_deserialize(bytes,(size_t)size,&module):nvm_v2_module_deserialize(bytes,(size_t)size,&module);
    if(status==NVM_V2_OK)status=profile==4?nvm_websocket_to_module(&module,&m):nvm_v2_to_nvm_module(&module,&m);
    if(status!=NVM_V2_OK){fprintf(stderr,"I could not decode the actual producer module: %u\n",(unsigned)status);return 8;}
    f=fopen(output,"wb");if(!f)return 2;
    write32(f,m->service_size);write32(f,m->import_count);
    write32(f,m->module_ref_count);write32(f,m->callback_contract_count);
    for(uint32_t i=0;i<m->import_count;i++) {
        const NvmImportEntry *p=&m->imports[i];
        uint32_t n=m->import_param_types && m->import_param_types[i]?p->param_count:0;
        write32(f,p->module_name_idx);write32(f,p->function_name_idx);write32(f,p->kind);
        write32(f,p->param_count);write32(f,p->return_type);write32(f,n);
        if(n)assert(fwrite(m->import_param_types[i],1,n,f)==n);
    }
    if(m->service_size)assert(fwrite(m->service_data,1,m->service_size,f)==m->service_size);
    write32(f,m->layout_size);write32(f,m->ownership_size);write32(f,m->string_count);write32(f,m->function_count);
    write32(f,m->struct_count);write32(f,m->enum_count);write32(f,m->union_count);
    for(uint32_t i=0;i<m->string_count;i++) {
        uint32_t n=m->string_lengths?m->string_lengths[i]:(uint32_t)strlen(m->strings[i]);
        write32(f,n);assert(fwrite(m->strings[i],1,n,f)==n);
    }
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *p=&m->functions[i];
        uint32_t n=m->function_param_types && m->function_param_types[i]?p->arity:0;
        write32(f,p->code_offset);write32(f,p->code_length);write32(f,p->arity);write32(f,p->local_count);
        write32(f,p->upvalue_count);write32(f,p->result_count);write32(f,p->result_tag);write32(f,n);
        if(n)assert(fwrite(m->function_param_types[i],1,n,f)==n);
    }
    if(m->layout_size)assert(fwrite(m->layout_data,1,m->layout_size,f)==m->layout_size);
    if(m->ownership_size)assert(fwrite(m->ownership_data,1,m->ownership_size,f)==m->ownership_size);
    if(m->code_size)assert(fwrite(m->code,1,m->code_size,f)==m->code_size);
    assert(!fclose(f));nvm_module_free(m);nvm_v2_module_free(&module);free(bytes);return 0;
}

int main(int argc,char **argv){
 if(argc==5 && !strcmp(argv[1],"--extract"))return extract_module((unsigned)strtoul(argv[2],NULL,10),argv[3],argv[4]);
 if(argc<2){return 2;}
 unsigned profile=(unsigned)strtoul(argv[1],NULL,10);
 for(int i=2;i<argc;i++){
  FILE *f=fopen(argv[i],"rb");if(!f || fseek(f,0,SEEK_END))return 2;long n=ftell(f);
  if(n<0 || fseek(f,0,SEEK_SET)){return 2;}
  uint8_t *bytes=malloc(n?(size_t)n:1);if(!bytes)return 2;
  if(fread(bytes,1,(size_t)n,f)!=(size_t)n || fclose(f))return 2;
  analyze(bytes,(size_t)n,profile);free(bytes);
 }
 return 0;
}
