/* I test private complete storage transport, not module admission. */
#ifdef NDEBUG
#error "I require assertions in my SDK module snapshot controls."
#endif
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
static size_t calls,fail_at;
static int persistent;
static int fail_now(void) { ++calls;return fail_at&&(calls==fail_at||(persistent&&calls>=fail_at)); }
static void *snapshot_malloc(size_t n) {return fail_now()?NULL:malloc(n);}
static void *snapshot_calloc(size_t n,size_t w) {return fail_now()?NULL:calloc(n,w);}
#define malloc snapshot_malloc
#define calloc snapshot_calloc
#include "../src/nanoisa/sdk_signature_snapshot.c"
#include "../src/nanoisa/sdk_module_snapshot.c"
#undef malloc
#undef calloc
static void check_copy(const NvmV2Module *m) {
    const uint8_t expected[]={7,0,8,255};
    assert(m->isa_version==1&&m->entry_point==0&&m->has_debug&&m->extra_features==17);
    assert(m->metadata.count==1&&m->metadata.items[0].value_idx==9);
    assert(m->constants.count==1&&m->constants.items[0].length==4);
    assert(!memcmp(m->constants.items[0].payload,expected,4));
    assert(m->layouts.count==1&&m->layouts.items[0].field_count==1);
    assert(m->layouts.items[0].fields[0].type_tag==TAG_INT&&m->layouts.items[0].fields[0].name_idx==4);
    assert(m->functions.count==1&&m->functions.items[0].signature_idx==1);
    assert(m->globals.count==1&&m->globals.items[0].name_idx==3);
    assert(m->imports.count==1&&m->imports.items[0].signature_idx==0);
    assert(m->callbacks.count==1&&m->callbacks.items[0].signature_idx==NVM_V2_NO_INDEX);
    assert(m->links.count==1&&m->links.items[0].signature_idx==1);
    assert(m->debug.count==1&&m->debug.items[0].source_line==73);
    assert(m->signatures.count==3&&m->signatures.items[2].result_tags[0]==TAG_STRING);
    assert(m->signatures.items[0].param_tags[0]==TAG_INT&&m->signatures.items[1].param_tags[0]==TAG_INT);
    assert(m->code_size==4&&!memcmp(m->code,expected,4));
#define CHECK_BYTES(name) assert(m->name##_size==4&&!memcmp(m->name##_data,expected,4))
    CHECK_BYTES(capture);CHECK_BYTES(ownership);CHECK_BYTES(passive);CHECK_BYTES(service);
#undef CHECK_BYTES
}
int main(void) {
    uint8_t bytes[]={7,0,8,255},tags[]={TAG_INT,TAG_BOOL,TAG_STRING};
    NvmV2MetadataEntry metadata[]={{2,9}};
    NvmV2Constant constants[]={{TAG_STRING,4,bytes}};
    NvmV2Signature signatures[]={{1,1,tags,tags+1},{1,1,tags,tags+1},{0,1,NULL,tags+2}};
    NvmV2LayoutField fields[]={{TAG_INT,NVM_V2_NO_INDEX,4}};
    NvmV2Layout layouts[]={{NVM_V2_LAYOUT_STRUCT,1,2,fields}};
    NvmV2Function functions[]={{.signature_idx=1}};
    NvmV2Global globals[]={{.name_idx=3}};
    NvmV2Import imports[]={{.signature_idx=0}};
    NvmV2Callback callbacks[]={{.signature_idx=NVM_V2_NO_INDEX}};
    NvmV2Link links[]={{.signature_idx=1}};
    NvmV2DebugEntry debug[]={{.source_line=73}};
    NvmV2Module source={0};source.isa_version=1;source.entry_point=0;source.extra_features=17;source.has_debug=true;
#define SET_TABLE(name,Type) source.name=(Type){name,sizeof name/sizeof *name}
    SET_TABLE(metadata,NvmV2Metadata);SET_TABLE(constants,NvmV2Constants);
    SET_TABLE(signatures,NvmV2Signatures);SET_TABLE(layouts,NvmV2Layouts);
    SET_TABLE(functions,NvmV2Functions);SET_TABLE(globals,NvmV2Globals);
    SET_TABLE(imports,NvmV2Imports);SET_TABLE(callbacks,NvmV2Callbacks);
    SET_TABLE(links,NvmV2Links);SET_TABLE(debug,NvmV2Debug);
#undef SET_TABLE
    source.code=bytes;source.code_size=4;
#define SET_BYTES(name) source.name##_data=bytes;source.name##_size=4
    SET_BYTES(capture);SET_BYTES(ownership);SET_BYTES(passive);SET_BYTES(service);
#undef SET_BYTES
    NvmSdkModuleSnapshot *p=NULL;calls=0;
    assert(nvm_sdk_module_snapshot_prepare(&source,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_OK);
    size_t sites=calls,budget=nvm_sdk_module_snapshot_bytes(p);assert(sites>10);check_copy(nvm_sdk_module_snapshot_view(p));
    nvm_sdk_module_snapshot_free(p);
    for(int mode=0;mode<2;++mode)for(size_t at=1;at<=sites;++at) {
        NvmV2Module original=source;calls=0;fail_at=at;persistent=mode;p=(void *)(uintptr_t)1;
        assert(nvm_sdk_module_snapshot_prepare(&source,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_MEMORY);
        assert(p==(void *)(uintptr_t)1&&!memcmp(&source,&original,sizeof source));
        fail_at=0;calls=0;p=NULL;
        assert(nvm_sdk_module_snapshot_prepare(&source,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_OK);
        assert(calls==sites);check_copy(nvm_sdk_module_snapshot_view(p));nvm_sdk_module_snapshot_free(p);
    }
    calls=0;p=(void *)(uintptr_t)1;
    assert(nvm_sdk_module_snapshot_prepare(&source,budget-1,&p)==NVM_SDK_LIMIT&&!calls&&p==(void *)(uintptr_t)1);
    source.code_size=UINT64_MAX;
    assert(nvm_sdk_module_snapshot_prepare(&source,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_LIMIT&&!calls&&p==(void *)(uintptr_t)1);
    source.code_size=4;source.layouts.items[0].fields=NULL;
    assert(nvm_sdk_module_snapshot_prepare(&source,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_INVALID&&!calls&&p==(void *)(uintptr_t)1);
    source.layouts.items[0].fields=fields;source.constants.count=UINT32_MAX;
    assert(nvm_sdk_module_snapshot_prepare(&source,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_LIMIT&&!calls&&p==(void *)(uintptr_t)1);
    source.constants.count=1;
    size_t measured=99;uint32_t work=97;
    assert(nvm_sdk_signature_snapshot_measure(&source,NVM_SDK_GENERATION_MAX_BYTES,0,&measured,&work)==NVM_SDK_LIMIT);
    assert(measured==99&&work==97&&!calls);
    p=NULL;assert(nvm_sdk_module_snapshot_prepare(&source,budget,&p)==NVM_SDK_OK);
    NvmSdkModuleSnapshot *next=NULL;
    assert(nvm_sdk_module_snapshot_prepare(nvm_sdk_module_snapshot_view(p),budget,&next)==NVM_SDK_OK);
    memset(bytes,0,sizeof bytes);memset(tags,0,sizeof tags);memset(fields,0,sizeof fields);
    memset(metadata,0,sizeof metadata);memset(constants,0,sizeof constants);memset(signatures,0,sizeof signatures);
    memset(layouts,0,sizeof layouts);memset(functions,0,sizeof functions);memset(globals,0,sizeof globals);
    memset(imports,0,sizeof imports);memset(callbacks,0,sizeof callbacks);memset(links,0,sizeof links);memset(debug,0,sizeof debug);
    memset(&source,0,sizeof source);check_copy(nvm_sdk_module_snapshot_view(p));
    nvm_sdk_module_snapshot_free(p);check_copy(nvm_sdk_module_snapshot_view(next));nvm_sdk_module_snapshot_free(next);
    source.has_debug=true;p=NULL;
    assert(nvm_sdk_module_snapshot_prepare(&source,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_OK);
    assert(nvm_sdk_module_snapshot_view(p)->has_debug&&!nvm_sdk_module_snapshot_view(p)->debug.count);
    nvm_sdk_module_snapshot_free(p);
    puts("I retained complete independent module storage and transactional allocation recovery.");
    return 0;
}
