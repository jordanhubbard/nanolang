/* I observe the real shared query APIs, including unchanged failure buffers. */
#include "../src/nanoisa/ownership_contracts.h"
#include "nvm2c_affine_model_reader.h"
#include <assert.h>
#include <stdio.h>

static void number(uint32_t n) { printf("%u,", n); }
static void unchanged(const void *data, size_t size) {
    const uint8_t *bytes=data;
    for(size_t i=0;i<size;i++)assert(bytes[i]==0xa5);
}
static void value(NvmOwnershipEffectValue v) { number(v.tag);number(v.layout); }
static bool observe(const uint8_t *bytes,size_t size) {
    if(size<20)return false;
    uint32_t op=read32(bytes),a=read32(bytes+4),b=read32(bytes+8),flags=read32(bytes+12),length=read32(bytes+16);
    if(op>5 || flags>63 || length!=size-20 || (op==0 && b>65535) || (op==5 && b>65535))return false;
    NvmModule model={0};if(!load_model(bytes+20,length,&model))return false;
    uint8_t dummy=0;NvmFunctionEntry *functions=model.functions;
    if(!(flags&2))model.ownership_data=NULL;
    else if(!model.ownership_size)model.ownership_data=&dummy;
    if(!(flags&4))model.layout_data=NULL;
    else if(!model.layout_size)model.layout_data=&dummy;
    const NvmModule *module=(flags&1)?&model:NULL;
    bool output=(flags&8)!=0,count_output=(flags&16)!=0;
    NvmV2Result result=NVM_V2_OK;
    uint16_t fields[33],count=0xa5a5;
    uint32_t global_count=0xa5a5a5a5,element=0xa5a5a5a5;
    NvmOwnershipGlobal globals[257];NvmOwnershipEffect effect;
    NvmOwnershipEffectTypes types;NvmUnionVariantFact variant;
    memset(fields,0xa5,sizeof fields);memset(globals,0xa5,sizeof globals);
    memset(&effect,0xa5,sizeof effect);memset(&types,0xa5,sizeof types);memset(&variant,0xa5,sizeof variant);
    switch(op) {
    case 0:result=nvm_ownership_path(module,a,output?fields:NULL,(uint16_t)b,count_output?&count:NULL);break;
    case 1:result=nvm_ownership_globals(module,output?globals:NULL,a,count_output?&global_count:NULL);break;
    case 2:result=nvm_ownership_effect(module,a,output?&effect:NULL);break;
    case 3:result=nvm_ownership_effect_types(module,output?&types:NULL);break;
    case 4:
        if(!(flags&32))model.functions=NULL;
        result=nvm_ownership_array_constructor(module,a,b,output?&element:NULL);
        model.functions=functions;break;
    case 5:result=nvm_ownership_union_variant(module,a,(uint16_t)b,output?&variant:NULL);break;
    }
    if(result!=NVM_V2_OK) {
        unchanged(fields,sizeof fields);unchanged(globals,sizeof globals);
        unchanged(&effect,sizeof effect);unchanged(&types,sizeof types);unchanged(&variant,sizeof variant);
        assert(count==0xa5a5 && global_count==0xa5a5a5a5 && element==0xa5a5a5a5);
    }
    number(result);
    if(result==NVM_V2_OK) switch(op) {
    case 0:
        assert(count<=32);number(count);
        for(uint16_t i=0;i<count;i++)number(fields[i]);
        unchanged(fields+count,sizeof fields-count*sizeof(*fields));break;
    case 1:
        assert(global_count<=256);number(global_count);
        if(output)for(uint32_t i=0;i<global_count;i++){number(globals[i].tag);number(globals[i].mutable);number(globals[i].layout);}
        unchanged(globals+(output?global_count:0),sizeof globals-(output?global_count:0)*sizeof(*globals));break;
    case 2:
        assert(effect.parameter_count<=256);number(effect.key_idx);value(effect.result);number(effect.parameter_count);
        for(uint16_t i=0;i<effect.parameter_count;i++){value(effect.parameters[i]);}
        break;
    case 3:
        assert(types.count<=256);number(types.count);
        for(uint32_t i=0;i<types.count;i++){value(types.items[i]);}
        break;
    case 4:number(element);break;
    case 5:number(variant.layout);number(variant.name_idx);number(variant.field_offset);number(variant.field_count);break;
    }
    free_model(&model);return true;
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *file=fopen(argv[i],"rb");if(!file){puts("D");continue;}
        if(fseek(file,0,SEEK_END)){fclose(file);puts("D");continue;}
        long size=ftell(file);if(size<0 || fseek(file,0,SEEK_SET)){fclose(file);puts("D");continue;}
        uint8_t *bytes=malloc((size_t)size+1);if(!bytes){fclose(file);return 2;}
        bool loaded=fread(bytes,1,(size_t)size,file)==(size_t)size;fclose(file);
        if(!loaded || !observe(bytes,(size_t)size))printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
