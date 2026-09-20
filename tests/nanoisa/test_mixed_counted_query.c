/* I reuse reviewed wire construction, then compare explicit target constants. */
#define main mc_original_query_main
#include "test_record_array_origins.c"
#undef main
#include "mixed_counted_catalog.h"
static void retain_section(const char *directory,unsigned tag,const char *suffix,const void *bytes,size_t length){
    char path[4096];int n=snprintf(path,sizeof path,"%s/catalog-%u-%s.bin",directory,tag,suffix);CHECK(n>0&&(size_t)n<sizeof path);
    FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(bytes,1,length,f)==length);CHECK(!fclose(f));
}
int main(int argc,char **argv){
    CHECK(argc==2);
    for(uint8_t tag=TAG_INT;tag<=TAG_STRING;tag++){
        Input c;basic(&c,tag,true);
        /* I append one ordinary nested record; the fourth flag uses old padding. */
        c.layouts[0]=4;c.ownership[4]=4;c.ownership[11]=NVM_LAYOUT_COMPLETE;c.m.struct_count=3;
        b8(c.layouts,&c.l,NVM_V2_LAYOUT_STRUCT);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,1);b32(c.layouts,&c.l,NO);
        b8(c.layouts,&c.l,TAG_STRUCT);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,0);b32(c.layouts,&c.l,1);b32(c.layouts,&c.l,NO);
        c.m.layout_size=(uint32_t)c.l;
        retain_section(argv[1],tag,"code",c.code,c.n);
        retain_section(argv[1],tag,"layouts",c.layouts,c.l);
        retain_section(argv[1],tag,"ownership",c.ownership,c.o);
        NvmRecordArrayOrigins *p=query(&c,NVM_ARRAY_ELIGIBLE);
        memset(&c,0,sizeof c); /* Every accessor below uses retained copied facts. */
        NvmDeclarationCounts count;CHECK(nvm_record_array_declaration_counts(p,&count));
        CHECK(count.layouts==4&&count.bindings==2&&count.unions==1);
        for(unsigned ordinal=0;ordinal<3;ordinal++){
            NvmDeclarationLayout layout;NvmV2LayoutField field;
            CHECK(nvm_record_array_declaration_layout(p,ordinal+1,&layout));
            CHECK(layout.kind==NVM_V2_LAYOUT_STRUCT&&layout.fields==mc_records[ordinal].field_count);
            CHECK(mc_records[ordinal].global_layout_index==ordinal+1);
            CHECK(nvm_record_array_declaration_field(p,ordinal+1,0,&field));
            CHECK(field.type_tag==(ordinal==2?TAG_STRUCT:TAG_ARRAY));
            CHECK(field.nested_idx==(ordinal==2?1:NO));
        }
        for(unsigned binding=0;binding<2;binding++){
            NvmOrdinaryArrayBinding b;NvmOrdinaryArrayType t;
            CHECK(nvm_record_array_declaration_binding(p,binding,&b));
            CHECK(b.layout==binding+1&&b.field==0&&b.element_type==0);
            CHECK(nvm_record_array_declaration_type(p,b.element_type,&t)&&t.tag==tag);
        }
        NvmRecordHeapOrigin origin;uint16_t mask=0;
        CHECK(nvm_record_array_origin(p,0,&origin)&&origin.kind==NVM_HEAP_ORIGIN_ARRAY&&origin.declared_tag==tag);
        CHECK(nvm_record_array_required_elements(p,0,&mask)&&mask==MASK(tag));
        CHECK(nvm_record_array_origin(p,1,&origin)&&origin.record_ordinal==0&&origin.layout_index==1);
        CHECK(mc_literals[0].length==4&&mc_literals[0].data[1]==0&&mc_literals[0].data[3]==0xa9);
        printf("I matched tag %u: records {ordinal0,global1,fields1},{ordinal1,global2,fields1},{ordinal2,global3,fields1}; nested referent1; two ARRAY bindings field0/type0; origin0 mask%u.\n",tag,(unsigned)mask);
        nvm_record_array_origins_free(p);
    }
    puts("I matched five-tag copied query facts to exact counted target descriptors; no admission.");return 0;
}
