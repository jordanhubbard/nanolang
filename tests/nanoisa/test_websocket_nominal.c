#include "../../src/nanoisa/service_websocket_nominal.h"
#include "../../src/nanoisa/service_file_nominal.h"
#include "../../src/nanoisa/service_socket_nominal.h"
#include "../../src/nanoisa/nvm_v2_sections.h"
#include "../../src/nanoisa/isa.h"
#include "../../src/nsi_websocket_plan.h"
#ifdef NOMINAL_PUBLIC_TEST
#include "../../src/nanoisa/service_bindings_module.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(x) do{checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
#ifdef NOMINAL_INSTRUMENT
static bool fail_allocation;static unsigned allocations,live;
static void *nominal_malloc(size_t n){allocations++;if(fail_allocation)return NULL;void *p=malloc(n);if(p)live++;return p;}
static void nominal_free(void *p){if(p){CHECK(live);live--;}free(p);}
#define malloc nominal_malloc
#define free nominal_free
#include "../../src/nanoisa/service_websocket_nominal_plan.c"
#undef malloc
#undef free
#endif
#include "websocket_nominal_fixture.h"
static void reject(Fixture *f) {
    NvmWebSocketNominalPlan *saved=(void *)f,*p=saved;
#ifdef NOMINAL_INSTRUMENT
    unsigned before=allocations;
#endif
    CHECK(nvm_websocket_nominal_plan(&f->module,&p)!=NVM_WEBSOCKET_NOMINAL_DESCRIBED && p==saved);
#ifdef NOMINAL_INSTRUMENT
    CHECK(allocations==before);
#endif
}
static void query(bool permute) {
    Fixture f;fixture(&f,permute);NvmWebSocketNominalPlan *p=NULL;
    CHECK(nvm_websocket_nominal_plan(&f.module,&p)==NVM_WEBSOCKET_NOMINAL_DESCRIBED);
#ifdef NOMINAL_PUBLIC_TEST
    CHECK(nvm_service_bindings_present(&f.module));
    CHECK(nvm_service_bindings_validate(&f.module)==NVM_V2_ERR_SECTION_TYPE);
    CHECK(nvm_service_execution_pending(&f.module));
#endif
    NvmWebSocketNominalLayout row;
    CHECK(nvm_websocket_nominal_layout_count(p)==8);
    CHECK(nvm_websocket_nominal_layout(p,0,&row) && row.category==NVM_WEBSOCKET_CATEGORY_UNKNOWN);
    for(unsigned i=0;i<7;i++) {
        CHECK(nvm_websocket_nominal_type(p,i,&row) && row.global_index==f.bindings.layouts[i]);
        CHECK(row.ownership_flags==((i==0 || i==3)?3:1));
        NvmWebSocketNominalLayout other;
        CHECK(nvm_websocket_nominal_source(p,row.layout_kind,row.source_ordinal,&other) && row.global_index==other.global_index && row.catalog_ordinal==other.catalog_ordinal);
    }
    for(unsigned i=0;i<4;i++){uint32_t index;CHECK(nvm_websocket_nominal_import(p,i,&index) && index==f.bindings.imports[i]);}
    memset(&f,0,sizeof f);CHECK(nvm_websocket_nominal_type(p,2,&row) && row.category==NVM_WEBSOCKET_CATEGORY_RECORD);
    NvmWebSocketNominalLayout saved=row;
    CHECK(!nvm_websocket_nominal_type(p,7,&row) && !memcmp(&row,&saved,sizeof row));
    CHECK(!nvm_websocket_nominal_layout(NULL,0,&row) && !nvm_websocket_nominal_layout(p,8,&row));
    CHECK(!nvm_websocket_nominal_source(p,NVM_V2_LAYOUT_UNION,99,&row));
    uint32_t index=99;CHECK(!nvm_websocket_nominal_import(p,4,&index) && index==99);
    nvm_websocket_nominal_plan_free(p);
}
static void refusals(void) {
    Fixture f;fixture(&f,false);
#define MUTATE(type,place,value) do{type saved=(place);(place)=(value);reject(&f);(place)=saved;}while(0)
    MUTATE(uint32_t,f.module.import_count,5);
    MUTATE(uint8_t,f.params[0][0],TAG_INT);
    MUTATE(uint8_t,f.params[1][1],TAG_STRING);
    MUTATE(uint8_t,f.imports[2].kind,NVM_IMPORT_FFI);
    MUTATE(uint32_t,f.imports[0].function_name_idx,f.imports[1].function_name_idx);
    size_t field=f.offsets[2]+8+12;
    MUTATE(uint8_t,f.layouts[field],TAG_INT);
    MUTATE(uint8_t,f.layouts[field+1],1);
    uint8_t copy[1024];memcpy(copy,f.layouts,sizeof copy);
    wr32(f.layouts+field+4,f.bindings.layouts[0]);reject(&f);memcpy(f.layouts,copy,sizeof copy);
    wr32(f.layouts+f.offsets[2]+4,0);reject(&f);memcpy(f.layouts,copy,sizeof copy);
    wr32(f.layouts+f.offsets[5]+8+4,f.bindings.layouts[5]);reject(&f);memcpy(f.layouts,copy,sizeof copy);
    wr32(f.layouts+f.offsets[5]+8+4,f.bindings.layouts[1]);reject(&f);memcpy(f.layouts,copy,sizeof copy);
    MUTATE(uint8_t,f.ownership[8+f.bindings.layouts[0]],1);
    MUTATE(uint8_t,f.ownership[8+f.bindings.layouts[3]],1);
    MUTATE(uint8_t,f.ownership[8+f.bindings.layouts[2]],3);
    MUTATE(uint8_t,f.ownership[33],1);
    MUTATE(uint8_t,f.ownership[32],TAG_INT);
    MUTATE(uint32_t,f.module.struct_count,3);
    for(uint32_t i=0;i<f.module.layout_size;i++) { uint32_t size=f.module.layout_size;f.module.layout_size=i;reject(&f);f.module.layout_size=size; }
    for(uint32_t i=0;i<64;i++) {f.module.ownership_size=i;reject(&f);}f.module.ownership_size=64;
#ifdef NOMINAL_INSTRUMENT
    NvmWebSocketNominalPlan *saved=(void *)&f,*p=saved;fail_allocation=true;
    CHECK(nvm_websocket_nominal_plan(&f.module,&p)==NVM_WEBSOCKET_NOMINAL_MEMORY && p==saved && !live);
    fail_allocation=false;
#endif
#undef MUTATE
}
static void raw(void) {
    NvmWebSocketNominalBindings b={{0,1,2,3},{0,1,2,3,4,5,6}},got;
    uint8_t bytes[112];memset(bytes,0xa5,sizeof bytes);size_t size=99;
    CHECK(nvm_websocket_nominal_encode(&b,bytes,sizeof bytes,&size)==NVM_SERVICE_OK && size==104);
    const uint8_t header[]={2,0,3,0,4,0,0,0,7,0,0,0,0,0,0,0};CHECK(!memcmp(bytes,header,16));
    for(unsigned i=104;i<112;i++)CHECK(bytes[i]==0xa5);
    CHECK(nvm_websocket_nominal_decode(bytes,104,&got)==NVM_SERVICE_OK && !memcmp(&b,&got,sizeof b));
    CHECK(nvm_websocket_nominal_encode(&b,NULL,0,&size)==NVM_SERVICE_OK && size==104);
    for(size_t i=0;i<104;i++) {
        got=b;CHECK(nvm_websocket_nominal_decode(bytes,i,&got)==NVM_SERVICE_SIZE && !memcmp(&b,&got,sizeof b));
        uint8_t out[104];memset(out,0xa5,sizeof out);size=99;
        CHECK(nvm_websocket_nominal_encode(&b,out,i,&size)==NVM_SERVICE_SIZE && size==99);
        for(unsigned j=0;j<104;j++)CHECK(out[j]==0xa5);
        if(i<16 || (i-16)%8<4) {
            bytes[i]^=0x40;got=b;CHECK(nvm_websocket_nominal_decode(bytes,104,&got)!=NVM_SERVICE_OK && !memcmp(&b,&got,sizeof b));bytes[i]^=0x40;
        }
    }
    NvmFileNominalBindings file;NvmSocketNominalBindings socket;
    CHECK(nvm_file_nominal_decode(bytes,104,&file)!=NVM_SERVICE_OK);
    CHECK(nvm_socket_nominal_decode(bytes,104,&socket)!=NVM_SERVICE_OK);
    for(unsigned group=0;group<2;group++)for(unsigned i=0;i<(group?7:4);i++) {
        NvmWebSocketNominalBindings bad=b;uint32_t *v=group?bad.layouts:bad.imports;v[i]=UINT32_MAX;
        CHECK(nvm_websocket_nominal_check(&bad)==NVM_SERVICE_INDEX);
        for(unsigned j=0;j<i;j++){bad=b;v=group?bad.layouts:bad.imports;v[i]=v[j];CHECK(nvm_websocket_nominal_check(&bad)==NVM_SERVICE_INDEX);}
    }
    union {NvmWebSocketNominalBindings bindings;uint8_t data[104];} overlap;overlap.bindings=b;
    CHECK(nvm_websocket_nominal_encode(&overlap.bindings,overlap.data,104,&size)==NVM_SERVICE_OK);
    CHECK(nvm_websocket_nominal_decode(overlap.data,104,&overlap.bindings)==NVM_SERVICE_OK && !memcmp(&overlap.bindings,&b,sizeof b));
    CHECK(nvm_websocket_nominal_decode(NULL,104,&got)==NVM_SERVICE_ARGUMENT);
    CHECK(nvm_websocket_nominal_encode(&b,bytes,104,NULL)==NVM_SERVICE_ARGUMENT);
    size=99;CHECK(!nvm_websocket_nominal_storage_bound(UINT32_MAX,&size) && size==99);
}
int main(void) {
    raw();query(false);query(true);refusals();nvm_websocket_nominal_plan_free(NULL);
#ifdef NOMINAL_INSTRUMENT
    CHECK(!live);
#endif
    printf("PASS %u checks; private counted-string WebSocket nominal map, no execution admission\n",checks);return 0;
}
