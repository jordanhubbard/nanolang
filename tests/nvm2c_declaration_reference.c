/* I expose the production shared declaration validator, not an imitation. */
#include "../src/nanoisa/ownership_contracts.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/nanoisa/ordinary_array_authority.h"
#include "../src/nanoisa/ownership_declaration_projection.h"
#include "../src/nanoisa/ownership_map_declarations.h"
#include "../src/nanoisa/record_map_structure_private.h"
#include "../src/nanoisa/ownership_layouts_private.h"
#include <stdint.h>
static int selected_profile;
static void need(bool value) {if(!value)exit(7);}
static void number(uint32_t value) {printf("%u,",(unsigned)value);}
static void declaration_report(NvmModule *m) {
    int profile=selected_profile;unsigned status;
    NvmOrdinaryArrayAuthority *ordinary=(void *)(uintptr_t)1;
    NvmOwnershipDeclarationPlan *base=(void *)(uintptr_t)1;
    NvmMapDeclarationPlan *map=(void *)(uintptr_t)1;
    NvmOrdinaryArrayCounts ordinary_counts={0};NvmDeclarationCounts counts={0};
    NvmMapDeclarationCounts map_counts={0};
    if(profile==1) {
        status=nvm_describe_ordinary_array_authority(m,&ordinary).status;
        if(status!=NVM_OAA_DESCRIBED){need(ordinary==(void *)(uintptr_t)1);printf("%u\n",status);return;}
        need(nvm_ordinary_array_authority_counts(ordinary,&ordinary_counts));
        counts.layouts=ordinary_counts.layouts;counts.types=ordinary_counts.types;counts.bindings=ordinary_counts.bindings;
    } else if(profile==2) {
        status=nvm_prepare_ownership_declarations(m,&base).status;
        if(status!=NVM_DECL_PREPARED){need(base==(void *)(uintptr_t)1);printf("%u\n",status);return;}
        need(nvm_ownership_declarations_counts(base,&counts));
    } else {
        NvmMapDeclarationPlan *public_map=(void *)(uintptr_t)1;
        unsigned public_status=nvm_prepare_map_declarations(m,&public_map).status;
        base=(void *)(uintptr_t)2;
        status=nvm_prepare_map_declarations_split(m,&map,&base).status;
        need(public_status==status);
        if(status!=NVM_MAP_DESCRIBED) {
            need(map==(void *)(uintptr_t)1 && base==(void *)(uintptr_t)2 && public_map==(void *)(uintptr_t)1);
            printf("%u\n",status);return;
        }
        need(nvm_ownership_declarations_counts(base,&counts));
        need(nvm_map_declarations_counts(map,&map_counts));
        NvmMapDeclarationCounts other;need(nvm_map_declarations_counts(public_map,&other));
        need(other.scalars==map_counts.scalars && other.pairs==map_counts.pairs && other.bindings==map_counts.bindings);
        nvm_map_declarations_free(public_map);
    }
    NvmV2Layouts layouts={0};
    if(profile==1)need(nvm_ownership_layouts_private_decode(m->layout_data,m->layout_size,&layouts)==NVM_V2_OK);
    printf("%u|",status);
    number(counts.layouts);number(counts.types);number(counts.bindings);number(counts.unions);number(counts.variants);
    number(map_counts.scalars);number(map_counts.pairs);number(map_counts.bindings);
    for(uint32_t i=0;i<counts.layouts;i++) {
        NvmDeclarationLayout row;
        if(profile==1)row=(NvmDeclarationLayout){layouts.items[i].kind,m->ownership_data[8+i],layouts.items[i].field_count,layouts.items[i].name_idx};
        else need(nvm_ownership_declarations_layout(base,i,&row));
        number(row.kind);number(row.flags);number(row.fields);number(row.name);
        for(uint32_t f=0;f<row.fields;f++) {
            NvmV2LayoutField field;
            if(profile==1)field=layouts.items[i].fields[f];else need(nvm_ownership_declarations_field(base,i,(uint16_t)f,&field));
            number(field.type_tag);number(field.nested_idx);number(field.name_idx);
        }
    }
    for(uint32_t i=0;i<counts.types;i++) {
        NvmOrdinaryArrayType row;
        need(profile==1?nvm_ordinary_array_authority_type(ordinary,i,&row):nvm_ownership_declarations_type(base,i,&row));
        number(row.tag);number(row.referent);
    }
    for(uint32_t i=0;i<counts.bindings;i++) {
        NvmOrdinaryArrayBinding row;
        need(profile==1?nvm_ordinary_array_authority_binding(ordinary,i,&row):nvm_ownership_declarations_binding(base,i,&row));
        number(row.layout);number(row.field);number(row.element_type);
    }
    for(uint32_t ordinal=0;ordinal<counts.unions;ordinal++) {
        uint16_t n=0;NvmUnionVariantFact row;
        while(n<256 && nvm_ownership_declarations_variant(base,ordinal,n,&row))n++;
        number(n);
        for(uint16_t v=0;v<n;v++) {
            need(nvm_ownership_declarations_variant(base,ordinal,v,&row));
            number(row.layout);number(row.name_idx);number(row.field_offset);number(row.field_count);
        }
        row=(NvmUnionVariantFact){99,98,97,96};
        need(!nvm_ownership_declarations_variant(base,ordinal,n,&row));
        need(row.layout==99 && row.name_idx==98 && row.field_offset==97 && row.field_count==96);
    }
    for(uint32_t i=0;i<map_counts.scalars;i++){uint8_t tag;need(nvm_map_declarations_scalar(map,i,&tag));number(tag);}
    for(uint32_t i=0;i<map_counts.pairs;i++){NvmMapDeclarationPair row;need(nvm_map_declarations_pair(map,i,&row));number(row.key_type);number(row.value_type);}
    for(uint32_t i=0;i<map_counts.bindings;i++){NvmMapDeclarationBinding row;need(nvm_map_declarations_binding(map,i,&row));number(row.layout);number(row.field);number(row.pair);}
    if(profile==3) {
        for(uint32_t i=0;i<counts.layouts;i++) {
            NvmDeclarationLayout layout;need(nvm_ownership_declarations_layout(base,i,&layout));
            for(uint32_t f=0;f<=(uint32_t)layout.fields;f++) {
                if(f>UINT16_MAX){number(0);number(0);number(0);continue;}
                NvmMapDeclarationPair row={0};bool found=nvm_map_declarations_field(map,i,(uint16_t)f,&row);
                number(found);number(row.key_type);number(row.value_type);
            }
        }
        NvmMapDeclarationPair row={99,98};need(!nvm_map_declarations_field(map,counts.layouts,0,&row));need(row.key_type==99 && row.value_type==98);
    }
    putchar('\n');
    if(profile==1){nvm_v2_layouts_free(&layouts);nvm_ordinary_array_authority_free(ordinary);}
    else {nvm_ownership_declarations_free(base);if(profile==3)nvm_map_declarations_free(map);}
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
static void analyze(const unsigned char *bytes, size_t size) {
    if(size<16){puts("99:0");return;}
    uint32_t imports=read32(bytes),modules=read32(bytes+4),service=read32(bytes+8),passive=read32(bytes+12);
    bytes+=16;size-=16;
    if (size < 28) { puts("99:0"); return; }
    uint32_t layout_size = read32(bytes), own_size = read32(bytes + 4);
    uint32_t strings = read32(bytes + 8), functions = read32(bytes + 12);
    size_t cursor = 28;
    if (strings > (size - cursor) / 4) { puts("99:0"); return; }
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
        uint8_t marker=0;
        model.import_count=imports;model.module_ref_count=modules;
        model.service_data=service?&marker:NULL;model.service_size=service?1:0;
        model.passive_data=passive?&marker:NULL;model.passive_size=passive?1:0;
        declaration_report(&model);
    } else puts("99:0");
    if (names) for (uint32_t i = 0; i < strings; ++i) free(names[i]);
    free(names); free(lengths); free(entries); free(params);
}
static void write32(FILE *f,uint32_t n) {
    unsigned char b[4]={(unsigned char)n,(unsigned char)(n>>8),(unsigned char)(n>>16),(unsigned char)(n>>24)};
    need(fwrite(b,1,4,f)==4);
}
static int extract_module(const char *input,const char *output) {
    FILE *f=fopen(input,"rb");if(!f || fseek(f,0,SEEK_END))return 2;
    long size=ftell(f);if(size<0 || fseek(f,0,SEEK_SET))return 2;
    unsigned char *bytes=malloc(size?(size_t)size:1);if(!bytes)return 2;
    if(fread(bytes,1,(size_t)size,f)!=(size_t)size || fclose(f))return 2;
    NvmV2Module module={0};NvmModule *m=NULL;
    NvmV2Result status=nvm_v2_module_deserialize(bytes,(size_t)size,&module);
    if(status==NVM_V2_OK)status=nvm_v2_to_nvm_module(&module,&m);
    if(status!=NVM_V2_OK){fprintf(stderr,"I could not decode the actual producer module: %u\n",(unsigned)status);return 8;}
    f=fopen(output,"wb");if(!f)return 2;
    write32(f,m->import_count);write32(f,m->module_ref_count);
    write32(f,m->service_size || m->service_data);write32(f,m->passive_size || m->passive_data);
    write32(f,m->layout_size);write32(f,m->ownership_size);write32(f,m->string_count);write32(f,m->function_count);
    write32(f,m->struct_count);write32(f,m->enum_count);write32(f,m->union_count);
    for(uint32_t i=0;i<m->string_count;i++) {
        uint32_t n=m->string_lengths?m->string_lengths[i]:(uint32_t)strlen(m->strings[i]);
        write32(f,n);need(fwrite(m->strings[i],1,n,f)==n);
    }
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *p=&m->functions[i];
        uint32_t n=m->function_param_types && m->function_param_types[i]?p->arity:0;
        write32(f,p->code_offset);write32(f,p->code_length);write32(f,p->arity);write32(f,p->local_count);
        write32(f,p->upvalue_count);write32(f,p->result_count);write32(f,p->result_tag);write32(f,n);
        if(n)need(fwrite(m->function_param_types[i],1,n,f)==n);
    }
    if(m->layout_size)need(fwrite(m->layout_data,1,m->layout_size,f)==m->layout_size);
    if(m->ownership_size)need(fwrite(m->ownership_data,1,m->ownership_size,f)==m->ownership_size);
    if(m->code_size)need(fwrite(m->code,1,m->code_size,f)==m->code_size);
    need(!fclose(f));nvm_module_free(m);nvm_v2_module_free(&module);free(bytes);return 0;
}

int main(int argc, char **argv) {
    if(argc==4 && !strcmp(argv[1],"--extract"))return extract_module(argv[2],argv[3]);
    if(argc<2)return 2;
    selected_profile=atoi(argv[1]);
    if(selected_profile<1 || selected_profile>3)return 2;
    for (int i = 2; i < argc; ++i) {
        FILE *input = fopen(argv[i], "rb");
        if (!input || fseek(input, 0, SEEK_END)) return 2;
        long size = ftell(input);
        if (size < 0 || fseek(input, 0, SEEK_SET)) return 2;
        unsigned char *bytes = malloc(size ? (size_t)size : 1);
        if (!bytes) return 2;
        if (fread(bytes, 1, (size_t)size, input) != (size_t)size || fclose(input)) return 2;
        analyze(bytes, (size_t)size); free(bytes);
    }
    return 0;
}
