/* I reuse the original ordinary controls without changing their assertions. */
#define main ordinary_controls_main
#include "test_ordinary_array_authority.c"
#undef main
#include "../../src/nanoisa/ownership_declaration_projection.h"
#include "../../src/nanoisa/reference_places.h"
#include "../../src/nanoisa/ownership_layouts_private.h"

typedef struct {
    NvmModule m;NvmFunctionEntry function;uint8_t param,*params[1];
    uint8_t l[4096],o[32768];size_t ln,on,field[5][2],row[5];
    size_t result,local,path,ext[2],variant[2],type,bindings;
} Mixed;
static void descriptor_bytes(Mixed *c,uint8_t tag,uint8_t mode,uint32_t layout){
    u8(c->o,&c->on,tag);u8(c->o,&c->on,mode);u16(c->o,&c->on,0);u32(c->o,&c->on,layout);
}
static void mixed_make(Mixed *c,bool forward){
    memset(c,0,sizeof *c);u32(c->l,&c->ln,5);
    for(unsigned i=0;i<5;i++){
        c->row[i]=c->ln;bool un=i==0||i==4;
        u8(c->l,&c->ln,un?NVM_V2_LAYOUT_UNION:NVM_V2_LAYOUT_STRUCT);u8(c->l,&c->ln,0);u16(c->l,&c->ln,2);u32(c->l,&c->ln,i);
        for(unsigned f=0;f<2;f++){
            c->field[i][f]=c->ln;
            uint8_t tag=un?(f?TAG_STRING:TAG_INT):(i==1&&f==1?(forward?TAG_STRUCT:TAG_INT):TAG_ARRAY);
            u8(c->l,&c->ln,tag);u8(c->l,&c->ln,0);u16(c->l,&c->ln,0);
            u32(c->l,&c->ln,tag==TAG_STRUCT?2:UINT32_MAX);u32(c->l,&c->ln,10+i*2+f);
        }
    }
    u32(c->o,&c->on,3);u32(c->o,&c->on,5);
    for(unsigned i=0;i<5;i++)u8(c->o,&c->on,i==0||i==4?0:NVM_LAYOUT_COMPLETE);
    while(c->on%4)u8(c->o,&c->on,0);
    u32(c->o,&c->on,1);u16(c->o,&c->on,1);u16(c->o,&c->on,1);
    c->result=c->on;descriptor_bytes(c,TAG_INT,0,UINT32_MAX);
    c->local=c->on;descriptor_bytes(c,TAG_UNION,0,4);
    u32(c->o,&c->on,12);c->path=c->on;u32(c->o,&c->on,1);u16(c->o,&c->on,1);u16(c->o,&c->on,0);u16(c->o,&c->on,0);u16(c->o,&c->on,0);
    u32(c->o,&c->on,2);c->ext[0]=c->on;
    u16(c->o,&c->on,1);u16(c->o,&c->on,1);u32(c->o,&c->on,52);u32(c->o,&c->on,2);
    for(unsigned i=0;i<2;i++){
        u32(c->o,&c->on,i?4:0);u16(c->o,&c->on,2);u16(c->o,&c->on,0);c->variant[i]=c->on;
        for(unsigned v=0;v<2;v++){u32(c->o,&c->on,20+v);u16(c->o,&c->on,(uint16_t)v);u16(c->o,&c->on,1);}
    }
    c->ext[1]=c->on;u16(c->o,&c->on,2);u16(c->o,&c->on,1);u32(c->o,&c->on,108);
    u32(c->o,&c->on,5);c->type=c->on;
    const uint8_t tags[]={TAG_INT,TAG_U8,TAG_FLOAT,TAG_BOOL,TAG_STRING};
    for(unsigned i=0;i<5;i++){u8(c->o,&c->on,tags[i]);u8(c->o,&c->on,0);u16(c->o,&c->on,0);u32(c->o,&c->on,UINT32_MAX);}
    u32(c->o,&c->on,5);c->bindings=c->on;
    for(unsigned i=0;i<5;i++){u32(c->o,&c->on,i?2+(i-1)/2:1);u16(c->o,&c->on,i?(i-1)%2:0);u16(c->o,&c->on,0);u32(c->o,&c->on,i);}
    c->function.local_count=1;c->function.arity=1;c->function.result_count=1;c->function.result_tag=TAG_INT;
    c->param=TAG_UNION;c->params[0]=&c->param;c->m.functions=&c->function;c->m.function_param_types=c->params;c->m.function_count=1;
    c->m.layout_data=c->l;c->m.layout_size=(uint32_t)c->ln;c->m.ownership_data=c->o;c->m.ownership_size=(uint32_t)c->on;
    c->m.struct_count=3;c->m.union_count=2;c->m.string_count=32;
}
static void mixed_refuse(Mixed *c,NvmDeclarationStatus expected){
    NvmOwnershipDeclarationPlan *p=(void *)(uintptr_t)1;NvmDeclarationResult r=nvm_prepare_ownership_declarations(&c->m,&p);
    if(r.status!=expected)fprintf(stderr,"I expected declaration status %u, got %u\n",(unsigned)expected,(unsigned)r.status);
    CHECK(r.status==expected);CHECK(p==(void *)(uintptr_t)1);
}
static void facts(NvmOwnershipDeclarationPlan *p,bool forward){
    NvmDeclarationCounts n;CHECK(nvm_ownership_declarations_counts(p,&n));CHECK(n.layouts==5&&n.types==5&&n.bindings==5&&n.unions==2&&n.variants==4);
    for(unsigned i=0;i<5;i++){
        NvmDeclarationLayout l;CHECK(nvm_ownership_declarations_layout(p,i,&l));bool un=i==0||i==4;
        CHECK(l.kind==(un?NVM_V2_LAYOUT_UNION:NVM_V2_LAYOUT_STRUCT)&&l.name==i&&l.fields==2&&l.flags==(un?0:NVM_LAYOUT_COMPLETE));
        for(unsigned f=0;f<2;f++){
            NvmV2LayoutField field;CHECK(nvm_ownership_declarations_field(p,i,(uint16_t)f,&field));
            uint8_t tag=un?(f?TAG_STRING:TAG_INT):(i==1&&f==1?(forward?TAG_STRUCT:TAG_INT):TAG_ARRAY);
            CHECK(field.type_tag==tag&&field.nested_idx==(tag==TAG_STRUCT?2:UINT32_MAX)&&field.name_idx==10+i*2+f);
        }
        NvmOrdinaryArrayType t;const uint8_t tags[]={TAG_INT,TAG_U8,TAG_FLOAT,TAG_BOOL,TAG_STRING};
        CHECK(nvm_ownership_declarations_type(p,i,&t));CHECK(t.tag==tags[i]&&t.referent==UINT32_MAX);
        NvmOrdinaryArrayBinding b;CHECK(nvm_ownership_declarations_binding(p,i,&b));CHECK(b.layout==(i?2+(i-1)/2:1)&&b.field==(i?(i-1)%2:0)&&b.element_type==i);
    }
    for(unsigned u=0;u<2;u++)for(unsigned v=0;v<2;v++){
        NvmUnionVariantFact f;CHECK(nvm_ownership_declarations_variant(p,u,(uint16_t)v,&f));
        CHECK(f.layout==(u?4:0)&&f.name_idx==20+v&&f.field_offset==v&&f.field_count==1);
    }
}
#define UNCHANGED(Type,call) do { Type out,old;memset(&out,0xa5,sizeof out);memcpy(&old,&out,sizeof out);CHECK(!(call));CHECK(!memcmp(&out,&old,sizeof out)); } while(0)
static void getter_errors(NvmOwnershipDeclarationPlan *p){
    UNCHANGED(NvmDeclarationCounts,nvm_ownership_declarations_counts(NULL,&out));
    UNCHANGED(NvmDeclarationLayout,nvm_ownership_declarations_layout(p,5,&out));
    UNCHANGED(NvmDeclarationLayout,nvm_ownership_declarations_layout(NULL,0,&out));
    UNCHANGED(NvmV2LayoutField,nvm_ownership_declarations_field(p,5,0,&out));
    UNCHANGED(NvmV2LayoutField,nvm_ownership_declarations_field(p,0,2,&out));
    UNCHANGED(NvmV2LayoutField,nvm_ownership_declarations_field(NULL,0,0,&out));
    UNCHANGED(NvmOrdinaryArrayType,nvm_ownership_declarations_type(p,5,&out));
    UNCHANGED(NvmOrdinaryArrayType,nvm_ownership_declarations_type(NULL,0,&out));
    UNCHANGED(NvmOrdinaryArrayBinding,nvm_ownership_declarations_binding(p,5,&out));
    UNCHANGED(NvmOrdinaryArrayBinding,nvm_ownership_declarations_binding(NULL,0,&out));
    UNCHANGED(NvmUnionVariantFact,nvm_ownership_declarations_variant(p,2,0,&out));
    UNCHANGED(NvmUnionVariantFact,nvm_ownership_declarations_variant(p,0,2,&out));
    UNCHANGED(NvmUnionVariantFact,nvm_ownership_declarations_variant(NULL,0,0,&out));
    CHECK(!nvm_ownership_declarations_counts(p,NULL));CHECK(!nvm_ownership_declarations_layout(p,0,NULL));CHECK(!nvm_ownership_declarations_field(p,0,0,NULL));
    CHECK(!nvm_ownership_declarations_type(p,0,NULL));CHECK(!nvm_ownership_declarations_binding(p,0,NULL));CHECK(!nvm_ownership_declarations_variant(p,0,0,NULL));
}
static void positive(bool forward){
    Mixed *c=malloc(sizeof *c);CHECK(c);mixed_make(c,forward);NvmOwnershipDeclarationPlan *p=NULL;
    CHECK(nvm_prepare_ownership_declarations(&c->m,&p).status==NVM_DECL_PREPARED);facts(p,forward);getter_errors(p);
    NvmOrdinaryArrayAuthority *old=(void *)(uintptr_t)1;
    CHECK(nvm_describe_ordinary_array_authority(&c->m,&old).status==(forward?NVM_OAA_INVALID:NVM_OAA_UNKNOWN));CHECK(old==(void *)(uintptr_t)1);
    bool needs=true;CHECK(nvm_ownership_contracts_validate(&c->m,&needs)!=NVM_V2_OK);CHECK(!needs);
    UNCHANGED(NvmUnionVariantFact,nvm_ownership_union_variant(&c->m,0,0,&out)==NVM_V2_OK);
    memset(c,0,sizeof *c);free(c);facts(p,forward);getter_errors(p);nvm_ownership_declarations_free(p);
}
static void malformed(void){
    Mixed c;
#define BAD(change,status) do {mixed_make(&c,false);change;mixed_refuse(&c,status);} while(0)
    BAD(c.o[c.ext[0]+2]=2,NVM_DECL_INVALID);BAD(c.o[c.ext[1]+2]=2,NVM_DECL_INVALID);
    BAD(c.o[c.ext[1]]=1,NVM_DECL_INVALID);BAD(c.o[c.ext[0]]=2,NVM_DECL_INVALID);
    BAD(patch(c.o,c.ext[0]+4,51),NVM_DECL_INVALID);BAD(patch(c.o,c.ext[1]+4,107),NVM_DECL_INVALID);
    BAD(patch(c.o,c.variant[1]-8,0),NVM_DECL_INVALID);BAD(c.o[c.variant[0]+6]=2,NVM_DECL_INVALID);
    BAD(patch(c.o,c.variant[0]+8,20),NVM_DECL_INVALID);BAD(patch(c.o,c.variant[0],32),NVM_DECL_INVALID);
    BAD(c.o[c.type+1]=1,NVM_DECL_INVALID);BAD(patch(c.o,c.type+4,0),NVM_DECL_INVALID);
    BAD(patch(c.o,c.bindings+12,1),NVM_DECL_INVALID);BAD(patch(c.o,c.bindings,0),NVM_DECL_INVALID);
    BAD(patch(c.o,c.bindings+8,5),NVM_DECL_INVALID);BAD(c.o[c.bindings+4]=1,NVM_DECL_INVALID);
    BAD(c.o[c.bindings+6]=1,NVM_DECL_INVALID);BAD(c.o[c.result+2]=1,NVM_DECL_INVALID);
    BAD(c.o[c.result]=TAG_BOOL,NVM_DECL_INVALID);BAD(patch(c.o,c.local+4,1),NVM_DECL_INVALID);
    BAD(c.function.local_count=2,NVM_DECL_INVALID);BAD(c.param=TAG_INT,NVM_DECL_INVALID);
    BAD(c.o[c.path+6]=1,NVM_DECL_INVALID);BAD(c.o[c.path+10]=1,NVM_DECL_INVALID);
    BAD(c.o[c.path+4]=0,NVM_DECL_INVALID);BAD(c.o[8]=NVM_LAYOUT_COMPLETE,NVM_DECL_INVALID);
    BAD(c.m.struct_count=2,NVM_DECL_INVALID);BAD(c.m.union_count=1,NVM_DECL_INVALID);
    BAD(c.l[c.field[0][0]]=TAG_ARRAY,NVM_DECL_INVALID);BAD(c.l[c.row[4]+1]=1,NVM_DECL_INVALID);
    BAD(c.m.import_count=1,NVM_DECL_UNKNOWN);BAD(c.m.module_ref_count=1,NVM_DECL_UNKNOWN);
    BAD(c.m.service_data=(void *)(uintptr_t)1,NVM_DECL_UNKNOWN);BAD(c.m.service_size=1,NVM_DECL_UNKNOWN);
    BAD(c.m.passive_data=(void *)(uintptr_t)1,NVM_DECL_UNKNOWN);BAD(c.m.passive_size=1,NVM_DECL_UNKNOWN);
    BAD(c.m.import_count=1;c.o[c.type+1]=1,NVM_DECL_INVALID);
    BAD(c.m.service_size=1;c.o[c.variant[0]+6]=2,NVM_DECL_INVALID);
    BAD(c.o[c.type]=TAG_ARRAY;patch(c.o,c.type+4,1),NVM_DECL_UNKNOWN);
    BAD(c.o[c.type]=TAG_STRUCT;patch(c.o,c.type+4,2),NVM_DECL_UNKNOWN);
    BAD(c.o[c.type]=TAG_ARRAY;patch(c.o,c.type+4,0),NVM_DECL_INVALID);
    BAD(c.m.layout_size=16777217,NVM_DECL_LIMIT);BAD(c.m.ownership_size=16777217,NVM_DECL_LIMIT);
    BAD(patch(c.l,0,257),NVM_DECL_LIMIT);BAD(patch(c.o,c.type-4,4097),NVM_DECL_LIMIT);
    BAD(patch(c.o,c.bindings-4,65537),NVM_DECL_LIMIT);
    BAD(c.m.ownership_data=NULL;c.m.ownership_size=0,NVM_DECL_UNKNOWN);
    BAD(c.m.layout_data=NULL;c.m.layout_size=0,NVM_DECL_UNKNOWN);
    BAD(c.m.layout_data=NULL,NVM_DECL_INVALID);BAD(c.m.ownership_data=NULL,NVM_DECL_INVALID);
    mixed_make(&c,true);patch(c.l,c.field[1][1]+4,4);mixed_refuse(&c,NVM_DECL_INVALID);
    mixed_make(&c,true);c.l[c.field[2][0]]=TAG_STRUCT;patch(c.l,c.field[2][0]+4,1);mixed_refuse(&c,NVM_DECL_INVALID);
    mixed_make(&c,false);size_t n=c.on;for(size_t i=1;i<n;i++){mixed_make(&c,false);c.m.ownership_size=(uint32_t)i;mixed_refuse(&c,NVM_DECL_INVALID);}
    mixed_make(&c,false);n=c.ln;for(size_t i=1;i<n;i++){mixed_make(&c,false);c.m.layout_size=(uint32_t)i;mixed_refuse(&c,NVM_DECL_INVALID);}
    BAD(c.m.ownership_size++,NVM_DECL_INVALID);BAD(c.m.layout_size++,NVM_DECL_INVALID);
#undef BAD
}
static void borrowed_suffix(void){
    Mixed c;mixed_make(&c,false);
    /* I retain an unsupported STRING-bearing borrowed resource beside arrays. */
    c.o[8+3]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;
    c.l[c.field[3][0]]=TAG_STRING;c.l[c.field[3][1]]=TAG_INT;
    c.param=TAG_STRUCT;c.o[c.local]=TAG_STRUCT;c.o[c.local+1]=NVM_REFERENCE_SHARED;patch(c.o,c.local+4,3);
    patch(c.o,c.bindings-4,3);patch(c.o,c.ext[1]+4,84);c.m.ownership_size-=24;
    mixed_refuse(&c,NVM_DECL_UNKNOWN);
    NvmOrdinaryArrayAuthority *p=(void *)(uintptr_t)1;CHECK(nvm_describe_ordinary_array_authority(&c.m,&p).status==NVM_OAA_UNKNOWN);CHECK(p==(void *)(uintptr_t)1);
    c.o[c.type+1]=1;mixed_refuse(&c,NVM_DECL_INVALID);
    CHECK(nvm_describe_ordinary_array_authority(&c.m,&p).status==NVM_OAA_UNKNOWN);CHECK(p==(void *)(uintptr_t)1);
}
static void union_only(void){
    Mixed c;mixed_make(&c,false);
    for(unsigned i=1;i<4;i++)for(unsigned f=0;f<2;f++)c.l[c.field[i][f]]=TAG_INT;
    patch(c.o,c.ext[0]-4,1);c.m.ownership_size=(uint32_t)c.ext[1];
    bool needs=false;CHECK(nvm_ownership_contracts_validate(&c.m,&needs)==NVM_V2_OK&&needs);
    NvmOwnershipDeclarationPlan *p=NULL;CHECK(nvm_prepare_ownership_declarations(&c.m,&p).status==NVM_DECL_PREPARED);
    NvmDeclarationCounts n;CHECK(nvm_ownership_declarations_counts(p,&n)&&n.unions==2&&n.variants==4&&!n.types&&!n.bindings);
    for(unsigned u=0;u<2;u++)for(unsigned v=0;v<2;v++){
        NvmUnionVariantFact a,b;CHECK(nvm_ownership_union_variant(&c.m,u,(uint16_t)v,&a)==NVM_V2_OK);CHECK(nvm_ownership_declarations_variant(p,u,(uint16_t)v,&b));
        CHECK(a.layout==b.layout&&a.name_idx==b.name_idx&&a.field_offset==b.field_offset&&a.field_count==b.field_count);
    }
    nvm_ownership_declarations_free(p);
    patch(c.o,0,2);c.m.ownership_size=(uint32_t)(c.path-4);patch(c.o,c.path-4,0);c.m.ownership_size+=4;
    c.param=TAG_INT;c.o[c.local]=TAG_INT;patch(c.o,c.local+4,UINT32_MAX);
    mixed_refuse(&c,NVM_DECL_UNKNOWN); /* valid legacy declarations lack variant facts */
}
static void union_budget(unsigned count){
    Mixed c;memset(&c,0,sizeof c);u32(c.l,&c.ln,count);
    for(unsigned i=0;i<count;i++){u8(c.l,&c.ln,NVM_V2_LAYOUT_UNION);u8(c.l,&c.ln,0);u16(c.l,&c.ln,0);u32(c.l,&c.ln,i);}
    u32(c.o,&c.on,3);u32(c.o,&c.on,count);for(unsigned i=0;i<count;i++)u8(c.o,&c.on,0);
    while(c.on%4)u8(c.o,&c.on,0);
    u32(c.o,&c.on,0);u32(c.o,&c.on,4);u32(c.o,&c.on,0);u32(c.o,&c.on,1);
    u16(c.o,&c.on,1);u16(c.o,&c.on,1);u32(c.o,&c.on,4+(8+8*256)*count);u32(c.o,&c.on,count);size_t first=c.on;
    for(unsigned i=0;i<count;i++){u32(c.o,&c.on,i);u16(c.o,&c.on,256);u16(c.o,&c.on,0);for(unsigned v=0;v<256;v++){u32(c.o,&c.on,count+v);u16(c.o,&c.on,0);u16(c.o,&c.on,0);}}
    c.m.layout_data=c.l;c.m.layout_size=(uint32_t)c.ln;c.m.ownership_data=c.o;c.m.ownership_size=(uint32_t)c.on;c.m.union_count=count;c.m.string_count=count+256;
    if(count==8){mixed_refuse(&c,NVM_DECL_LIMIT);return;}
    NvmOwnershipDeclarationPlan *p=NULL;CHECK(nvm_prepare_ownership_declarations(&c.m,&p).status==NVM_DECL_PREPARED);
    NvmDeclarationCounts n;CHECK(nvm_ownership_declarations_counts(p,&n)&&n.unions==7&&n.variants==7*256);nvm_ownership_declarations_free(p);
    c.o[first+4]=1;mixed_refuse(&c,NVM_DECL_INVALID); /* 257 variants */
}
static void old_profiles(void){
    Case c;make_case(&c,TAG_FLOAT,true);NvmOwnershipDeclarationPlan *p=NULL;
    CHECK(nvm_prepare_ownership_declarations(&c.module,&p).status==NVM_DECL_PREPARED);
    NvmDeclarationCounts n;CHECK(nvm_ownership_declarations_counts(p,&n)&&n.layouts==2&&n.types==1&&n.bindings==1&&!n.unions&&!n.variants);
    NvmOrdinaryArrayType t;CHECK(nvm_ownership_declarations_type(p,0,&t)&&t.tag==TAG_FLOAT);nvm_ownership_declarations_free(p);
    for(unsigned version=1;version<=2;version++){
        make_case(&c,TAG_INT,false);c.layouts[12]=TAG_INT;patch(c.ownership,0,version);
        c.module.ownership_size=version==1?16:20;patch(c.ownership,16,0);p=NULL;
        CHECK(nvm_prepare_ownership_declarations(&c.module,&p).status==NVM_DECL_PREPARED);
        CHECK(nvm_ownership_declarations_counts(p,&n)&&n.layouts==1&&!n.types&&!n.bindings&&!n.unions&&!n.variants);nvm_ownership_declarations_free(p);
    }
}
static void mixed_maximum(void){
    const uint32_t layouts=256,fields=65536,types=4096;
    size_t lb=4u+8u*layouts+12u*fields,ob=8u+layouts+4u+8u+4u+8u+36u+8u+8u+8u*types+12u*fields;
    uint8_t *l=calloc(lb+12,1),*o=calloc(ob,1);CHECK(l&&o);size_t n=0;
    u32(l,&n,layouts);
    for(unsigned i=0;i<layouts;i++){
        unsigned count=i==2?65535:i==3?1:0;
        u8(l,&n,i<2?NVM_V2_LAYOUT_UNION:NVM_V2_LAYOUT_STRUCT);u8(l,&n,0);u16(l,&n,(uint16_t)count);u32(l,&n,0);
        for(unsigned j=0;j<count;j++){u8(l,&n,TAG_ARRAY);u8(l,&n,0);u16(l,&n,0);u32(l,&n,UINT32_MAX);u32(l,&n,1);}
    }CHECK(n==lb);n=0;u32(o,&n,3);u32(o,&n,layouts);
    for(unsigned i=0;i<layouts;i++)u8(o,&n,i<2?0:NVM_LAYOUT_COMPLETE);
    u32(o,&n,0);u32(o,&n,4);u32(o,&n,0);u32(o,&n,2);
    u16(o,&n,1);u16(o,&n,1);u32(o,&n,36);u32(o,&n,2);
    for(unsigned i=0;i<2;i++){u32(o,&n,i);u16(o,&n,1);u16(o,&n,0);u32(o,&n,1);u16(o,&n,0);u16(o,&n,0);}
    u16(o,&n,2);u16(o,&n,1);u32(o,&n,8u+8u*types+12u*fields);size_t tc=n;u32(o,&n,types);
    for(unsigned i=0;i<types;i++){u8(o,&n,TAG_STRING);u8(o,&n,0);u16(o,&n,0);u32(o,&n,UINT32_MAX);}
    size_t bc=n;u32(o,&n,fields);
    for(unsigned i=0;i<fields;i++){u32(o,&n,i==65535?3:2);u16(o,&n,i==65535?0:(uint16_t)i);u16(o,&n,0);u32(o,&n,types-1);}
    CHECK(n==ob);Mixed c;memset(&c,0,sizeof c);c.m.layout_data=l;c.m.layout_size=(uint32_t)lb;c.m.ownership_data=o;c.m.ownership_size=(uint32_t)ob;c.m.struct_count=254;c.m.union_count=2;c.m.string_count=2;
    NvmOwnershipDeclarationPlan *p=NULL;CHECK(nvm_prepare_ownership_declarations(&c.m,&p).status==NVM_DECL_PREPARED);
    NvmDeclarationCounts counts;CHECK(nvm_ownership_declarations_counts(p,&counts));CHECK(counts.layouts==layouts&&counts.types==types&&counts.bindings==fields&&counts.unions==2&&counts.variants==2);
    patch(o,tc,types+1);mixed_refuse(&c,NVM_DECL_LIMIT);patch(o,tc,types);
    patch(o,bc,fields+1);mixed_refuse(&c,NVM_DECL_LIMIT);patch(o,bc,fields);
    patch(l,0,layouts+1);mixed_refuse(&c,NVM_DECL_LIMIT);patch(l,0,layouts);
    size_t fifth=4u+8u*4u+12u*fields;l[fifth+2]=1;c.m.layout_size+=12;mixed_refuse(&c,NVM_DECL_LIMIT);
    free(l);free(o);
    CHECK(nvm_ownership_declarations_counts(p,&counts)&&counts.layouts==layouts&&counts.bindings==fields);
    NvmOrdinaryArrayBinding b;CHECK(nvm_ownership_declarations_binding(p,fields-1,&b));CHECK(b.layout==3&&!b.field&&b.element_type==types-1);
    NvmOrdinaryArrayType t;CHECK(nvm_ownership_declarations_type(p,types-1,&t));CHECK(t.tag==TAG_STRING&&t.referent==UINT32_MAX);
    NvmUnionVariantFact v;CHECK(nvm_ownership_declarations_variant(p,1,0,&v));CHECK(v.layout==1&&v.name_idx==1&&!v.field_count);
    nvm_ownership_declarations_free(p);
}
/* I compose all preparation owners under one transaction and one allowance. */
#ifdef OAA_INSTRUMENT
static void *sdk_test_malloc(size_t bytes) { return oaa_test_calloc(bytes,1); }
#define malloc sdk_test_malloc
#define calloc oaa_test_calloc
#define free oaa_test_free
#endif
#include "../../src/nanoisa/sdk_signature_snapshot.c"
#include "../../src/nanoisa/sdk_module_snapshot.c"
#include "../../src/nanoisa/sdk_provider_codec.c"
#include "../../src/nanoisa/sdk_provider_validation.c"
#ifdef OAA_INSTRUMENT
#undef malloc
#undef calloc
#undef free
#endif
static bool budget_chain(const NvmV2Module *source,NvmPreparationBudget *budget) {
    NvmPreparationBudget remaining=*budget;
    NvmSdkModuleSnapshot *module=NULL;NvmOwnershipDeclarationPlan *declarations=NULL;
    NvmSdkProviderTransport *provider=NULL;
    uint8_t wire[32]={NVM_SDK_PROVIDER_REVISION};
    bool ok=nvm_sdk_module_snapshot_prepare_budget(source,&remaining,&module)==NVM_SDK_OK;
    if(ok)ok=nvm_prepare_ownership_declarations_v2_budget(nvm_sdk_module_snapshot_view(module),&remaining,&declarations).status==NVM_DECL_PREPARED;
    if(ok)ok=nvm_sdk_provider_decode_budget(wire,sizeof wire,&remaining,&provider)==NVM_SDK_OK;
    if(ok){facts(declarations,true);*budget=remaining;}
    nvm_sdk_provider_transport_free(provider);nvm_ownership_declarations_free(declarations);
    nvm_sdk_module_snapshot_free(module);return ok;
}
static void combined_budget_controls(const NvmV2Module *m) {
    NvmPreparationBudget full={NVM_PREPARATION_MAX_BYTES,NVM_PREPARATION_MAX_STEPS},left=full;
    CHECK(budget_chain(m,&left));
    NvmPreparationBudget exact={full.bytes-left.bytes,full.steps-left.steps},trial=exact;
    CHECK(exact.bytes&&exact.steps);CHECK(budget_chain(m,&trial));CHECK(!trial.bytes&&!trial.steps);
    trial=exact;--trial.bytes;NvmPreparationBudget before=trial;
    CHECK(!budget_chain(m,&trial));CHECK(trial.bytes==before.bytes&&trial.steps==before.steps);
    trial=exact;--trial.steps;before=trial;
    CHECK(!budget_chain(m,&trial));CHECK(trial.bytes==before.bytes&&trial.steps==before.steps);
    NvmSdkModuleSnapshot *sentinel=(void *)(uintptr_t)1;trial=(NvmPreparationBudget){0,0};before=trial;
    CHECK(nvm_sdk_module_snapshot_prepare_budget(m,&trial,&sentinel)==NVM_SDK_LIMIT);
    CHECK(sentinel==(void *)(uintptr_t)1&&trial.bytes==before.bytes&&trial.steps==before.steps);
    NvmOwnershipDeclarationPlan *decl=(void *)(uintptr_t)1;
    CHECK(nvm_prepare_ownership_declarations_v2_budget(m,&trial,&decl).status==NVM_DECL_LIMIT);
    CHECK(decl==(void *)(uintptr_t)1&&!trial.bytes&&!trial.steps);
#ifdef OAA_INSTRUMENT
    size_t baseline=live;calls=0;trial=full;CHECK(budget_chain(m,&trial));size_t measured=calls;
    CHECK(live==baseline&&measured>20);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<measured;i++) {
        calls=0;fail_at=i;persistent=mode!=0;trial=full;
        CHECK(!budget_chain(m,&trial));CHECK(live==baseline);
        CHECK(trial.bytes==full.bytes&&trial.steps==full.steps);
        fail_at=SIZE_MAX;persistent=false;trial=full;
        CHECK(budget_chain(m,&trial));CHECK(live==baseline);
    }
#endif
}
static void retained_v2(void) {
    Mixed c;mixed_make(&c,true);
    NvmV2Module m={0};
    CHECK(nvm_ownership_mixed_layouts_private_decode(c.l,c.ln,&m.layouts)==NVM_V2_OK);
    uint8_t parameter=TAG_UNION,result=TAG_INT,wrong=TAG_BOOL;
    NvmV2Signature signatures[]={{1,1,&wrong,&wrong},{1,1,&parameter,&result},{1,1,&parameter,&result},{0,1,NULL,&wrong}};
    NvmV2Function function={.signature_idx=2,.local_count=1};
    NvmV2Constant constants[32];memset(constants,0,sizeof constants);
    for(unsigned i=0;i<32;i++)constants[i].tag=TAG_STRING;
    m.signatures=(NvmV2Signatures){signatures,4};m.functions=(NvmV2Functions){&function,1};
    m.constants=(NvmV2Constants){constants,32};m.ownership_data=c.o;m.ownership_size=(uint32_t)c.on;
    NvmOwnershipDeclarationPlan *p=NULL;
    CHECK(nvm_prepare_ownership_declarations_v2(&m,&p).status==NVM_DECL_PREPARED);facts(p,true);nvm_ownership_declarations_free(p);
#define V2_BAD(change,undo,expected) do {change;p=(void *)(uintptr_t)1;CHECK(nvm_prepare_ownership_declarations_v2(&m,&p).status==expected);CHECK(p==(void *)(uintptr_t)1);undo;} while(0)
    V2_BAD(function.signature_idx=0,function.signature_idx=2,NVM_DECL_INVALID);
    V2_BAD(function.signature_idx=4,function.signature_idx=2,NVM_DECL_INVALID);
    V2_BAD(signatures[2].param_tags=NULL,signatures[2].param_tags=&parameter,NVM_DECL_INVALID);
    V2_BAD(constants[0].tag=TAG_INT,constants[0].tag=TAG_STRING,NVM_DECL_INVALID);
    V2_BAD(constants[10].tag=TAG_BOOL,constants[10].tag=TAG_STRING,NVM_DECL_INVALID);
    V2_BAD(constants[20].tag=TAG_INT,constants[20].tag=TAG_STRING,NVM_DECL_INVALID);
    V2_BAD(c.o[c.ext[1]+2]=2,c.o[c.ext[1]+2]=1,NVM_DECL_INVALID);
    V2_BAD(m.layouts.count=257,m.layouts.count=5,NVM_DECL_LIMIT);
    V2_BAD(m.functions.count=UINT32_MAX,m.functions.count=1,NVM_DECL_LIMIT);
#undef V2_BAD
#ifdef OAA_INSTRUMENT
    size_t baseline=live;calls=0;p=NULL;
    CHECK(nvm_prepare_ownership_declarations_v2(&m,&p).status==NVM_DECL_PREPARED);
    size_t measured=calls;nvm_ownership_declarations_free(p);CHECK(live==baseline&&measured>=12);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<measured;i++) {
        calls=0;fail_at=i;persistent=mode!=0;p=(void *)(uintptr_t)1;
        CHECK(nvm_prepare_ownership_declarations_v2(&m,&p).status==NVM_DECL_MEMORY);
        CHECK(p==(void *)(uintptr_t)1&&live==baseline);
        fail_at=SIZE_MAX;persistent=false;p=NULL;
        CHECK(nvm_prepare_ownership_declarations_v2(&m,&p).status==NVM_DECL_PREPARED);
        facts(p,true);nvm_ownership_declarations_free(p);CHECK(live==baseline);
    }
#endif
    combined_budget_controls(&m);
    p=NULL;CHECK(nvm_prepare_ownership_declarations_v2(&m,&p).status==NVM_DECL_PREPARED);
    memset(c.o,0,c.on);memset(signatures,0,sizeof signatures);memset(constants,0,sizeof constants);
    nvm_v2_layouts_free(&m.layouts);facts(p,true);getter_errors(p);nvm_ownership_declarations_free(p);
}
typedef struct {
    NvmV2Module m;NvmV2Layout layouts[5];NvmV2LayoutField fields[5][4];
    NvmV2Signature signatures[3];NvmV2Function function;NvmV2Constant names[32];
    uint8_t parameter,result,wrong,wire[2048];size_t n,extension,types,bindings,result_at;
} Typed;
static void typed_make(Typed *c) {
    memset(c,0,sizeof *c);
    const uint8_t kinds[]={NVM_V2_LAYOUT_STRUCT,NVM_V2_LAYOUT_TUPLE,NVM_V2_LAYOUT_UNION,NVM_V2_LAYOUT_ENUM,NVM_V2_LAYOUT_STRUCT};
    const uint16_t counts[]={4,2,2,0,1};
    for(unsigned i=0;i<5;i++)c->layouts[i]=(NvmV2Layout){kinds[i],counts[i],i,c->fields[i]};
    c->fields[0][0]=(NvmV2LayoutField){TAG_TUPLE,1,10};
    c->fields[0][1]=(NvmV2LayoutField){TAG_ARRAY,UINT32_MAX,11};
    c->fields[0][2]=(NvmV2LayoutField){TAG_FUNCTION,UINT32_MAX,12};
    c->fields[0][3]=(NvmV2LayoutField){TAG_OPAQUE,UINT32_MAX,13};
    c->fields[1][0]=(NvmV2LayoutField){TAG_UNION,2,14};
    c->fields[1][1]=(NvmV2LayoutField){TAG_ENUM,3,15};
    c->fields[2][0]=(NvmV2LayoutField){TAG_STRUCT,4,16};
    c->fields[2][1]=(NvmV2LayoutField){TAG_STRING,UINT32_MAX,17};
    c->fields[4][0]=(NvmV2LayoutField){TAG_INT,UINT32_MAX,18};
    c->parameter=TAG_ENUM;c->result=TAG_TUPLE;c->wrong=TAG_BOOL;
    c->signatures[0]=(NvmV2Signature){1,1,&c->wrong,&c->wrong};
    c->signatures[1]=(NvmV2Signature){1,1,&c->parameter,&c->result};
    c->signatures[2]=c->signatures[1];c->function.signature_idx=2;c->function.local_count=1;
    for(unsigned i=0;i<32;i++)c->names[i].tag=TAG_STRING;
    c->m.layouts=(NvmV2Layouts){c->layouts,5};c->m.signatures=(NvmV2Signatures){c->signatures,3};
    c->m.functions=(NvmV2Functions){&c->function,1};c->m.constants=(NvmV2Constants){c->names,32};
    u32(c->wire,&c->n,3);u32(c->wire,&c->n,5);
    for(unsigned i=0;i<5;i++)u8(c->wire,&c->n,i==2?0:NVM_LAYOUT_COMPLETE);
    while(c->n%4)u8(c->wire,&c->n,0);
    u32(c->wire,&c->n,1);u16(c->wire,&c->n,1);u16(c->wire,&c->n,1);
    c->result_at=c->n;
    u8(c->wire,&c->n,TAG_TUPLE);u8(c->wire,&c->n,0);u16(c->wire,&c->n,0);u32(c->wire,&c->n,1);
    u8(c->wire,&c->n,TAG_ENUM);u8(c->wire,&c->n,0);u16(c->wire,&c->n,0);u32(c->wire,&c->n,3);
    u32(c->wire,&c->n,4);u32(c->wire,&c->n,0);u32(c->wire,&c->n,2);
    u16(c->wire,&c->n,1);u16(c->wire,&c->n,1);u32(c->wire,&c->n,28);
    u32(c->wire,&c->n,1);u32(c->wire,&c->n,2);u16(c->wire,&c->n,2);u16(c->wire,&c->n,0);
    for(unsigned v=0;v<2;v++){u32(c->wire,&c->n,20+v);u16(c->wire,&c->n,(uint16_t)v);u16(c->wire,&c->n,1);}
    c->extension=c->n;u16(c->wire,&c->n,2);u16(c->wire,&c->n,2);u32(c->wire,&c->n,84);
    u32(c->wire,&c->n,8);c->types=c->n;
    const uint8_t tags[]={TAG_INT,TAG_STRUCT,TAG_TUPLE,TAG_UNION,TAG_ENUM,TAG_OPAQUE,TAG_FUNCTION,TAG_ARRAY};
    const uint32_t refs[]={UINT32_MAX,4,1,2,3,65535,65534,2};
    for(unsigned i=0;i<8;i++){u8(c->wire,&c->n,tags[i]);u8(c->wire,&c->n,0);u16(c->wire,&c->n,0);u32(c->wire,&c->n,refs[i]);}
    u32(c->wire,&c->n,1);c->bindings=c->n;
    u32(c->wire,&c->n,0);u16(c->wire,&c->n,1);u16(c->wire,&c->n,0);u32(c->wire,&c->n,7);
    c->m.ownership_data=c->wire;c->m.ownership_size=(uint32_t)c->n;
}
static NvmPreparationBudget typed_budget(void){return (NvmPreparationBudget){NVM_PREPARATION_MAX_BYTES,NVM_PREPARATION_MAX_STEPS};}
static void typed_facts(NvmOwnershipDeclarationPlan *p) {
    NvmDeclarationCounts counts;CHECK(nvm_ownership_declarations_counts(p,&counts));
    CHECK(counts.layouts==5&&counts.types==8&&counts.bindings==1&&counts.unions==1&&counts.variants==2);
    CHECK(nvm_ownership_declarations_foreign_unresolved(p));
    NvmOrdinaryArrayType t;CHECK(nvm_ownership_declarations_type(p,5,&t));CHECK(t.tag==TAG_OPAQUE&&t.referent==65535);
    CHECK(nvm_ownership_declarations_type(p,6,&t));CHECK(t.tag==TAG_FUNCTION&&t.referent==65534);
    CHECK(nvm_ownership_declarations_type(p,4,&t));CHECK(t.tag==TAG_ENUM&&t.referent==3);
    NvmUnionVariantFact v;CHECK(nvm_ownership_declarations_variant(p,0,0,&v));CHECK(v.layout==2&&v.field_count==1&&v.field_offset==0);
    NvmV2LayoutField f;CHECK(nvm_ownership_declarations_field(p,1,0,&f));CHECK(f.type_tag==TAG_UNION&&f.nested_idx==2);
}
static void typed_profile_controls(void) {
    Typed c;typed_make(&c);NvmOwnershipDeclarationPlan *p=NULL;NvmPreparationBudget b=typed_budget();
    CHECK(nvm_prepare_ownership_declarations_typed_v2(&c.m,&b,&p).status==NVM_DECL_PREPARED);typed_facts(p);nvm_ownership_declarations_free(p);
    p=(void *)(uintptr_t)1;CHECK(nvm_prepare_ownership_declarations_v2(&c.m,&p).status==NVM_DECL_INVALID);CHECK(p==(void *)(uintptr_t)1);
#define TYPED_BAD(change) do {typed_make(&c);change;b=typed_budget();NvmPreparationBudget before=b;p=(void *)(uintptr_t)1;CHECK(nvm_prepare_ownership_declarations_typed_v2(&c.m,&b,&p).status==NVM_DECL_INVALID);CHECK(p==(void *)(uintptr_t)1&&b.bytes==before.bytes&&b.steps==before.steps);} while(0)
    TYPED_BAD(c.fields[0][0].nested_idx=3);
    TYPED_BAD((c.fields[4][0]=(NvmV2LayoutField){TAG_STRUCT,0,18}));
    TYPED_BAD(c.fields[0][2].nested_idx=4);
    TYPED_BAD(c.fields[0][3].nested_idx=4);
    TYPED_BAD(c.fields[1][1].nested_idx=UINT32_MAX);
    TYPED_BAD(c.wire[c.extension+2]=3);
    TYPED_BAD(c.wire[c.extension+2]=1);
    TYPED_BAD(c.wire[c.types+1]=1);
    TYPED_BAD(patch(c.wire,c.types+4,0));
    TYPED_BAD(patch(c.wire,c.types+8+4,3));
    TYPED_BAD(patch(c.wire,c.types+5*8+4,65536));
    TYPED_BAD(patch(c.wire,c.types+6*8+4,UINT32_MAX));
    TYPED_BAD(patch(c.wire,c.types+7*8+4,7));
    TYPED_BAD(patch(c.wire,c.bindings+8,8));
    TYPED_BAD(c.names[20].tag=TAG_INT);
    TYPED_BAD(c.function.signature_idx=0);
    TYPED_BAD(patch(c.wire,c.result_at+4,UINT32_MAX));
#undef TYPED_BAD
    for(unsigned depth=64;depth<=65;depth++) {
        typed_make(&c);c.n=c.types-4;u32(c.wire,&c.n,depth);
        for(unsigned i=0;i<depth;i++) {
            u8(c.wire,&c.n,i+1==depth?TAG_STRING:TAG_ARRAY);u8(c.wire,&c.n,0);u16(c.wire,&c.n,0);
            u32(c.wire,&c.n,i+1==depth?UINT32_MAX:i+1);
        }
        u32(c.wire,&c.n,1);u32(c.wire,&c.n,0);u16(c.wire,&c.n,1);u16(c.wire,&c.n,0);u32(c.wire,&c.n,0);
        patch(c.wire,c.extension+4,20+depth*8);c.m.ownership_size=(uint32_t)c.n;
        b=typed_budget();NvmPreparationBudget before=b;p=(void *)(uintptr_t)1;
        CHECK(nvm_prepare_ownership_declarations_typed_v2(&c.m,&b,&p).status==(depth==64?NVM_DECL_PREPARED:NVM_DECL_LIMIT));
        if(depth==64)nvm_ownership_declarations_free(p);
        else CHECK(p==(void *)(uintptr_t)1&&b.bytes==before.bytes&&b.steps==before.steps);
    }
    /* I validate prior-only tables too; no forward edge triggers this check. */
    NvmV2LayoutField leaf={TAG_INT,0,0};NvmV2Layout prior[2]={{NVM_V2_LAYOUT_STRUCT,0,0,NULL},{NVM_V2_LAYOUT_STRUCT,1,1,&leaf}};
    NvmV2Layouts ls={prior,2},out={(void *)(uintptr_t)1,7};uint8_t wire[64];
    size_t bytes=nvm_v2_layouts_encoded_size(&ls);CHECK(nvm_v2_layouts_encode(&ls,wire,bytes)==NVM_V2_OK);
    CHECK(nvm_ownership_typed_layouts_private_decode_detailed(wire,bytes,&out,NULL)!=NVM_V2_OK);CHECK(out.items==(void *)(uintptr_t)1&&out.count==7);
#ifdef OAA_INSTRUMENT
    typed_make(&c);size_t baseline=live;calls=0;b=typed_budget();p=NULL;
    CHECK(nvm_prepare_ownership_declarations_typed_v2(&c.m,&b,&p).status==NVM_DECL_PREPARED);
    size_t measured=calls;nvm_ownership_declarations_free(p);CHECK(live==baseline);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<measured;i++) {
        b=typed_budget();NvmPreparationBudget before=b;calls=0;fail_at=i;persistent=mode!=0;p=(void *)(uintptr_t)1;
        CHECK(nvm_prepare_ownership_declarations_typed_v2(&c.m,&b,&p).status==NVM_DECL_MEMORY);
        CHECK(p==(void *)(uintptr_t)1&&live==baseline&&b.bytes==before.bytes&&b.steps==before.steps);
        fail_at=SIZE_MAX;persistent=false;b=typed_budget();p=NULL;
        CHECK(nvm_prepare_ownership_declarations_typed_v2(&c.m,&b,&p).status==NVM_DECL_PREPARED);typed_facts(p);nvm_ownership_declarations_free(p);CHECK(live==baseline);
    }
#endif
    typed_make(&c);b=typed_budget();p=NULL;CHECK(nvm_prepare_ownership_declarations_typed_v2(&c.m,&b,&p).status==NVM_DECL_PREPARED);
    memset(&c,0,sizeof c);typed_facts(p);nvm_ownership_declarations_free(p);
}

typedef struct {
    Typed typed;char names[32][24];NvmSdkNominalRow nominal[6];
    NvmSdkSignatureRow signature;NvmSdkBindingRow bindings[3];uint32_t refs[3],policies[3];
    NvmSdkLifetimeRows rows;
} DescriptionCase;
static void description_case(DescriptionCase *c) {
    memset(c,0,sizeof *c);typed_make(&c->typed);
    for(unsigned i=0;i<32;i++) {
        int n=snprintf(c->names[i],sizeof c->names[i],"name%u",i);CHECK(n>0&&(size_t)n<sizeof c->names[i]);
        c->typed.names[i].payload=(uint8_t *)c->names[i];c->typed.names[i].length=(uint32_t)n;
    }
    c->nominal[0]=(NvmSdkNominalRow){30,0,NVM_SDK_NOMINAL_RECORD,0,0,0};
    c->nominal[1]=(NvmSdkNominalRow){30,2,NVM_SDK_NOMINAL_UNION,2,0,0};
    c->nominal[2]=(NvmSdkNominalRow){30,3,NVM_SDK_NOMINAL_ENUM,3,0,0};
    c->nominal[3]=(NvmSdkNominalRow){31,0,NVM_SDK_NOMINAL_RECORD,4,0,0};
    c->nominal[4]=(NvmSdkNominalRow){30,9,NVM_SDK_NOMINAL_OPAQUE,UINT32_MAX,0,0};
    c->nominal[5]=c->nominal[0];
    patch(c->typed.wire,c->typed.types+5*8+4,4);patch(c->typed.wire,c->typed.types+6*8+4,0);
    c->signature=(NvmSdkSignatureRow){2,0,1,1,1};c->refs[0]=4;c->refs[1]=2;
    c->bindings[0]=(NvmSdkBindingRow){NVM_SDK_BIND_FUNCTION,0,UINT32_MAX,0,UINT32_MAX};
    c->bindings[1]=(NvmSdkBindingRow){NVM_SDK_BIND_FIELD,0,2,6,UINT32_MAX};
    c->bindings[2]=(NvmSdkBindingRow){NVM_SDK_BIND_FIELD,0,3,5,UINT32_MAX};
    c->rows.declarations=(NvmSdkProviderRows){c->nominal,NULL,&c->signature,c->bindings,c->refs,6,0,1,3,2};
    c->rows.binding_policies=c->policies;
}
static void description_wire_free(void *p) {
#ifdef OAA_INSTRUMENT
    oaa_test_free(p);
#else
    free(p);
#endif
}
static void description_controls(void) {
    DescriptionCase c;description_case(&c);uint8_t *wire=NULL;size_t bytes=0;
    CHECK(nvm_sdk_provider_lifetime_encode(&c.rows,NVM_SDK_PROVIDER_MAX_BYTES,&wire,&bytes)==NVM_SDK_OK);
    NvmPreparationBudget budget=typed_budget(),initial=budget;NvmSdkDescription *p=NULL;
    CHECK(nvm_sdk_description_prepare(&c.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK);
    const NvmV2Module *owned=nvm_sdk_description_module(p);CHECK(owned&&owned!=&c.typed.m&&owned->functions.items[0].signature_idx==2);
    CHECK(owned->constants.items[0].payload!=(uint8_t *)c.names[0]&&owned->constants.items[0].length==5);
    size_t charge=initial.bytes-budget.bytes;uint32_t work=initial.steps-budget.steps;nvm_sdk_description_free(p);
    budget=(NvmPreparationBudget){charge,work};p=NULL;
    CHECK(nvm_sdk_description_prepare(&c.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK&&!budget.bytes&&!budget.steps);nvm_sdk_description_free(p);
    for(unsigned dimension=0;dimension<2;dimension++) {
        budget=(NvmPreparationBudget){charge-(dimension==0),work-(dimension==1)};NvmPreparationBudget before=budget;p=(void *)(uintptr_t)1;
        CHECK(nvm_sdk_description_prepare(&c.typed.m,wire,bytes,&budget,&p)==NVM_SDK_LIMIT);
        CHECK(p==(void *)(uintptr_t)1&&budget.bytes==before.bytes&&budget.steps==before.steps);
    }
#ifdef OAA_INSTRUMENT
    size_t baseline=live;calls=0;budget=typed_budget();p=NULL;
    CHECK(nvm_sdk_description_prepare(&c.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK);
    size_t measured=calls;nvm_sdk_description_free(p);CHECK(live==baseline);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<measured;i++) {
        calls=0;fail_at=i;persistent=mode!=0;budget=typed_budget();NvmPreparationBudget before=budget;p=(void *)(uintptr_t)1;
        CHECK(nvm_sdk_description_prepare(&c.typed.m,wire,bytes,&budget,&p)==NVM_SDK_MEMORY);
        CHECK(p==(void *)(uintptr_t)1&&live==baseline&&budget.bytes==before.bytes&&budget.steps==before.steps);
        fail_at=SIZE_MAX;persistent=false;budget=typed_budget();p=NULL;
        CHECK(nvm_sdk_description_prepare(&c.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK);nvm_sdk_description_free(p);CHECK(live==baseline);
    }
#endif
    description_wire_free(wire);
#define DESC_BAD(change) do {description_case(&c);change;wire=NULL;bytes=0;CHECK(nvm_sdk_provider_lifetime_encode(&c.rows,NVM_SDK_PROVIDER_MAX_BYTES,&wire,&bytes)==NVM_SDK_OK);budget=typed_budget();NvmPreparationBudget before=budget;p=(void *)(uintptr_t)1;CHECK(nvm_sdk_description_prepare(&c.typed.m,wire,bytes,&budget,&p)==NVM_SDK_INVALID);CHECK(p==(void *)(uintptr_t)1&&budget.bytes==before.bytes&&budget.steps==before.steps);description_wire_free(wire);} while(0)
    DESC_BAD(c.nominal[5].layout=4);
    DESC_BAD(c.nominal[5].owner=31);
    DESC_BAD(c.nominal[3].owner=30);
    DESC_BAD(c.nominal[2].kind=NVM_SDK_NOMINAL_RECORD);
    DESC_BAD(c.nominal[4].name=UINT32_MAX);
    DESC_BAD(c.signature.coarse_signature=1);
    DESC_BAD(c.bindings[2]=c.bindings[1]);
    DESC_BAD(c.bindings[1].detail=5);
    DESC_BAD(c.refs[0]=5);
    DESC_BAD(c.typed.names[30].tag=TAG_INT);
    DESC_BAD(c.names[0][2]=0);
    DESC_BAD(c.names[16][2]=0);
    /* Generic-key cycles without a reference edge refuse before equality. */
    DESC_BAD(c.nominal[3].argument_count=1;c.nominal[3].argument_first=2;c.refs[2]=1;c.rows.declarations.reference_count=3);
#undef DESC_BAD
    /* The same generic reference is permitted through an ARRAY edge. */
    description_case(&c);c.nominal[3].argument_count=1;c.nominal[3].argument_first=2;
    c.refs[2]=7;c.rows.declarations.reference_count=3;
    wire=NULL;bytes=0;CHECK(nvm_sdk_provider_lifetime_encode(&c.rows,NVM_SDK_PROVIDER_MAX_BYTES,&wire,&bytes)==NVM_SDK_OK);
    budget=typed_budget();p=NULL;CHECK(nvm_sdk_description_prepare(&c.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK);
    nvm_sdk_description_free(p);description_wire_free(wire);
    description_case(&c);wire=NULL;bytes=0;CHECK(nvm_sdk_provider_lifetime_encode(&c.rows,NVM_SDK_PROVIDER_MAX_BYTES,&wire,&bytes)==NVM_SDK_OK);
    budget=typed_budget();p=NULL;CHECK(nvm_sdk_description_prepare(&c.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK);
    memset(&c,0,sizeof c);description_wire_free(wire);owned=nvm_sdk_description_module(p);
    CHECK(owned&&owned->functions.items[0].signature_idx==2&&owned->constants.items[0].length==5&&!memcmp(owned->constants.items[0].payload,"name0",5));
    nvm_sdk_description_free(p);
}

typedef struct {
    DescriptionCase base;NvmV2Signature coarse[4];uint8_t string_tag;
    NvmSdkSignatureRow exact[2];NvmSdkBindingRow bindings[4];uint32_t refs[3],selectors[4];
    NvmV2Import import;NvmSdkProviderRow provider;NvmSdkCallPolicy policies[2];
    NvmSdkLifetimeNode nodes[2];
} PolicyCase;
static void policy_case(PolicyCase *c) {
    memset(c,0,sizeof *c);description_case(&c->base);DescriptionCase *b=&c->base;
    memcpy(c->coarse,b->typed.signatures,sizeof b->typed.signatures);c->string_tag=TAG_STRING;
    c->coarse[3]=(NvmV2Signature){1,1,&c->string_tag,&c->string_tag};
    b->typed.m.signatures=(NvmV2Signatures){c->coarse,4};
    b->typed.wire[b->typed.types+7*8]=TAG_STRING;patch(b->typed.wire,b->typed.types+7*8+4,UINT32_MAX);
    c->exact[0]=b->signature;c->exact[1]=(NvmSdkSignatureRow){3,2,1,2,1};
    c->refs[0]=b->refs[0];c->refs[1]=b->refs[1];c->refs[2]=7;
    memcpy(c->bindings,b->bindings,sizeof b->bindings);
    c->bindings[3]=(NvmSdkBindingRow){NVM_SDK_BIND_IMPORT,0,UINT32_MAX,1,0};c->selectors[3]=1;
    c->import=(NvmV2Import){22,23,3,NVM_V2_IMPORT_ARTIFACT};b->typed.m.imports=(NvmV2Imports){&c->import,1};
    c->provider=(NvmSdkProviderRow){22,24,25,26,27,28};
    c->nodes[0]=(NvmSdkLifetimeNode){7,NVM_SDK_BORROW_CALL,UINT32_MAX,UINT32_MAX,0,0,0};
    c->nodes[1]=(NvmSdkLifetimeNode){7,NVM_SDK_SNAPSHOT_RESULT,UINT32_MAX,UINT32_MAX,0,0,0};
    c->policies[0]=(NvmSdkCallPolicy){0,1,1,1,UINT32_MAX};c->policies[1]=c->policies[0];
    b->rows.declarations=(NvmSdkProviderRows){b->nominal,&c->provider,c->exact,c->bindings,c->refs,6,1,2,4,3};
    b->rows.binding_policies=c->selectors;b->rows.policies=c->policies;b->rows.nodes=c->nodes;
    b->rows.policy_count=2;b->rows.node_count=2;
}
static void description_policy_controls(void) {
    PolicyCase c;policy_case(&c);uint8_t *wire=NULL;size_t bytes=0;NvmSdkDescription *p=NULL;
    CHECK(nvm_sdk_provider_lifetime_encode(&c.base.rows,NVM_SDK_PROVIDER_MAX_BYTES,&wire,&bytes)==NVM_SDK_OK);
    NvmPreparationBudget budget=typed_budget();
    CHECK(nvm_sdk_description_prepare(&c.base.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK);
    const NvmV2Module *owned=nvm_sdk_description_module(p);
    CHECK(owned&&owned->imports.count==1&&owned->imports.items[0].signature_idx==3&&owned->signatures.count==4);
    nvm_sdk_description_free(p);
#ifdef OAA_INSTRUMENT
    size_t baseline=live;calls=0;budget=typed_budget();p=NULL;
    CHECK(nvm_sdk_description_prepare(&c.base.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK);
    size_t measured=calls;nvm_sdk_description_free(p);CHECK(live==baseline);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<measured;i++) {
        calls=0;fail_at=i;persistent=mode!=0;budget=typed_budget();NvmPreparationBudget before=budget;p=(void *)(uintptr_t)1;
        CHECK(nvm_sdk_description_prepare(&c.base.typed.m,wire,bytes,&budget,&p)==NVM_SDK_MEMORY);
        CHECK(p==(void *)(uintptr_t)1&&live==baseline&&budget.bytes==before.bytes&&budget.steps==before.steps);
        fail_at=SIZE_MAX;persistent=false;budget=typed_budget();p=NULL;
        CHECK(nvm_sdk_description_prepare(&c.base.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK);nvm_sdk_description_free(p);CHECK(live==baseline);
    }
#endif
    description_wire_free(wire);
#define POLICY_BAD(change) do {policy_case(&c);change;wire=NULL;bytes=0;CHECK(nvm_sdk_provider_lifetime_encode(&c.base.rows,NVM_SDK_PROVIDER_MAX_BYTES,&wire,&bytes)==NVM_SDK_OK);budget=typed_budget();NvmPreparationBudget before=budget;p=(void *)(uintptr_t)1;CHECK(nvm_sdk_description_prepare(&c.base.typed.m,wire,bytes,&budget,&p)==NVM_SDK_INVALID);CHECK(p==(void *)(uintptr_t)1&&budget.bytes==before.bytes&&budget.steps==before.steps);description_wire_free(wire);} while(0)
    POLICY_BAD(c.nodes[0].mode=NVM_SDK_SNAPSHOT_RESULT);
    POLICY_BAD(c.nodes[1].mode=NVM_SDK_BORROW_CALL);
    POLICY_BAD(c.nodes[1].mode=NVM_SDK_BORROW_ARGUMENT_RESULT;c.nodes[1].owner_argument=1);
    POLICY_BAD(c.nodes[0].owner_argument=0);
    POLICY_BAD(c.nodes[0].hook_set=29);
    POLICY_BAD(c.nodes[1].type=0);
    /* Policy zero is unused by the import, but its own call context still matters. */
    POLICY_BAD(c.policies[0].parameter_first=1);
    POLICY_BAD(c.policies[0].result_first=0);
    POLICY_BAD(c.policies[1].execution=0);
    POLICY_BAD(c.import.signature_idx=2);
    POLICY_BAD(c.provider.target=UINT32_MAX);
#undef POLICY_BAD
    policy_case(&c);c.nodes[1].mode=NVM_SDK_BORROW_ARGUMENT_RESULT;c.nodes[1].owner_argument=0;
    wire=NULL;bytes=0;CHECK(nvm_sdk_provider_lifetime_encode(&c.base.rows,NVM_SDK_PROVIDER_MAX_BYTES,&wire,&bytes)==NVM_SDK_OK);
    budget=typed_budget();p=NULL;CHECK(nvm_sdk_description_prepare(&c.base.typed.m,wire,bytes,&budget,&p)==NVM_SDK_OK);
    nvm_sdk_description_free(p);description_wire_free(wire);
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);CHECK(ordinary_controls_main()==0);
    puts("I begin complete mixed declaration controls");positive(false);positive(true);malformed();borrowed_suffix();union_only();union_budget(7);union_budget(8);mixed_maximum();old_profiles();
    NvmOwnershipDeclarationPlan *sentinel=(void *)(uintptr_t)1;CHECK(nvm_prepare_ownership_declarations(NULL,&sentinel).status==NVM_DECL_INVALID);CHECK(sentinel==(void *)(uintptr_t)1);
    Mixed c;mixed_make(&c,true);CHECK(nvm_prepare_ownership_declarations(&c.m,NULL).status==NVM_DECL_INVALID);nvm_ownership_declarations_free(NULL);
#ifdef OAA_INSTRUMENT
    CHECK(!live);calls=0;NvmOwnershipDeclarationPlan *p=NULL;CHECK(nvm_prepare_ownership_declarations(&c.m,&p).status==NVM_DECL_PREPARED);
    size_t measured=calls;nvm_ownership_declarations_free(p);CHECK(!live&&measured>=11);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<measured;i++){
        mixed_make(&c,true);calls=0;fail_at=i;persistent=mode!=0;p=(void *)(uintptr_t)1;
        NvmDeclarationStatus status=nvm_prepare_ownership_declarations(&c.m,&p).status;
        CHECK(status==NVM_DECL_MEMORY||status==NVM_DECL_UNKNOWN);CHECK(p==(void *)(uintptr_t)1);CHECK(!live);
        fail_at=SIZE_MAX;persistent=false;calls=0;p=NULL;CHECK(nvm_prepare_ownership_declarations(&c.m,&p).status==NVM_DECL_PREPARED);facts(p,true);nvm_ownership_declarations_free(p);CHECK(!live);
    }
    calls=0;union_budget(8);CHECK(!calls&&!live);
    printf("I covered %zu mixed allocation positions in both failure modes; two query TUs instrumented\n",measured);
#endif
    retained_v2();typed_profile_controls();description_controls();description_policy_controls();
    printf("PASS %u complete mixed declaration checks; no execution authority\n",checks);return 0;
}
