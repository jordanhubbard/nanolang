/* I exercise the real publisher with explicit checked-AST corruption controls. */
#include "../../src/nanovirt/codegen.c"
#include "../../modules/nanoisa/nanoisa.h"
#include "../../src/nanoisa/disassembler.h"
int g_argc=0;char **g_argv=NULL;
static unsigned checks;
#define CHECK(c) do {checks++;if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(1);}}while(0)
static void exact_types(void) {
    TypeInfo key={.base_type=TYPE_STRING},value={.base_type=TYPE_INT};
    TypeInfo *args[]={&key,&value};
    TypeInfo map={.base_type=TYPE_HASHMAP,.generic_name="HashMap",.type_params=args,.type_param_count=2};
    uint8_t tag=255;CHECK(ordinary_map_pair(&map,&tag)&&tag==TAG_INT);CHECK(!ordinary_map_pair(NULL,&tag)&&tag==TAG_INT);
    for(unsigned i=0;i<2;i++){TypeInfo *saved=args[i];args[i]=NULL;CHECK(!ordinary_map_pair(&map,&tag));args[i]=saved;}
    map.type_param_count=1;CHECK(!ordinary_map_pair(&map,&tag));map.type_param_count=2;
    map.generic_name="Other";CHECK(!ordinary_map_pair(&map,&tag));map.generic_name="HashMap";
    key.base_type=TYPE_INT;CHECK(!ordinary_map_pair(&map,&tag));key.base_type=TYPE_STRING;
    value.element_type=&key;CHECK(!ordinary_map_pair(&map,&tag));value.element_type=NULL;
    value.service_ordinal=1;CHECK(!ordinary_map_pair(&map,&tag));value.service_ordinal=0;
    value.type_var_count=1;CHECK(!ordinary_map_pair(&map,&tag));value.type_var_count=0;
    map.fn_sig=(void *)(uintptr_t)1;CHECK(!ordinary_map_pair(&map,&tag));map.fn_sig=NULL;
    CHECK(tag==TAG_INT);value.base_type=TYPE_STRING;
    CHECK(ordinary_map_pair(&map,&tag)&&tag==TAG_STRING);
    value.base_type=TYPE_BOOL;CHECK(!ordinary_map_pair(&map,&tag)&&tag==TAG_STRING);
    value.base_type=TYPE_INT;CHECK(ordinary_map_pair(&map,&tag)&&tag==TAG_INT);
}
/* I mutate only publication declarations after independently valid SS lowering.
 * Restored publication must reproduce the original bytes, proving this test CG
 * carries the actual declaration identities and copied local-slot facts. */
static void publisher_arity_control(ASTNode *program,ASTNode *record,int field,NvmModule *module) {
    CHECK(nvm_verify(module).ok);CHECK(module->ownership_size&&module->layout_size);
    uint8_t *ownership=module->ownership_data,*layout=module->layout_data;
    uint32_t ownership_size=module->ownership_size,layout_size=module->layout_size;
    CG *cg=calloc(1,sizeof *cg);CHECK(cg);cg->module=module;
    for(int i=0;i<program->as.program.count;i++) {
        ASTNode *node=program->as.program.items[i];if(node->type!=AST_STRUCT_DEF)continue;
        CHECK(cg->struct_count<MAX_STRUCT_DEFS);
        CgStructDef *sd=&cg->structs[cg->struct_count];
        sd->name=node->as.struct_def.name;sd->declaration=node;
        sd->field_names=node->as.struct_def.field_names;sd->field_type_names=node->as.struct_def.field_type_names;
        sd->field_count=node->as.struct_def.field_count;sd->def_idx=cg->struct_count++;
    }
    CgAuthoritySlot *facts=NULL;cg->authority_slots=&facts;
    uint8_t header[8];ordinary_word(header,3,4);ordinary_word(header+4,cg->struct_count,4);
    CHECK(ownership_size>=8&&!memcmp(ownership,header,8));
    size_t at=8+((cg->struct_count+3u)&~3u);
    ordinary_word(header,module->function_count,4);
    CHECK(at+4<=ownership_size&&!memcmp(ownership+at,header,4));at+=4;
    for(uint32_t f=0;f<module->function_count;f++) {
        CHECK(at+12<=ownership_size);at+=12;
        for(uint16_t local=0;local<module->functions[f].local_count;local++) {
            CHECK(at+8<=ownership_size);CgAuthoritySlot *fact=malloc(sizeof *fact);CHECK(fact);
            *fact=(CgAuthoritySlot){f,local,ownership[at],facts};facts=fact;at+=8;
        }
    }
    module->ownership_data=NULL;module->ownership_size=0;module->layout_data=NULL;module->layout_size=0;
    CHECK(nvm_verify(module).ok);
    TypeInfo *type=record->as.struct_def.field_type_info[field];CHECK(type&&type->type_param_count==2);
    type->type_param_count=1;publish_ordinary_authority(cg,program,NULL);
    CHECK(!cg->had_error&&!module->ownership_data&&!module->layout_data);
    CHECK(!module->ownership_size&&!module->layout_size&&nvm_verify(module).ok);
    type->type_param_count=2;publish_ordinary_authority(cg,program,NULL);
    CHECK(!cg->had_error&&nvm_verify(module).ok);
    CHECK(module->ownership_size==ownership_size&&module->layout_size==layout_size);
    CHECK(!memcmp(module->ownership_data,ownership,ownership_size));
    CHECK(!memcmp(module->layout_data,layout,layout_size));
    free(ownership);free(layout);
    while(facts){CgAuthoritySlot *next=facts->next;free(facts);facts=next;}free(cg);
}
static void compile_case(const char *source,unsigned mutation,bool published,bool shadows) {
    int count=0;Token *tokens=tokenize(source,&count);CHECK(tokens);
    ASTNode *program=parse_program(tokens,count);CHECK(program);
    Environment *env=create_environment();CHECK(env);env->suppress_shadow_warnings=true;
    CHECK(type_check(program,env));
    ASTNode *record=NULL;
    for(int i=0;i<program->as.program.count;i++)if(program->as.program.items[i]->type==AST_STRUCT_DEF){record=program->as.program.items[i];break;}
    CHECK(record);int field=-1;
    for(int i=0;i<record->as.struct_def.field_count;i++)if(record->as.struct_def.field_types[i]==TYPE_HASHMAP){field=i;break;}
    TypeInfo *saved=NULL;Type saved_coarse=TYPE_VOID;int saved_count=0;
    if(mutation&&mutation!=4){CHECK(field>=0);saved=record->as.struct_def.field_type_info[field];CHECK(saved);saved_coarse=record->as.struct_def.field_types[field];saved_count=saved->type_param_count;
        if(mutation==1)record->as.struct_def.field_type_info[field]=NULL;
        if(mutation==2)record->as.struct_def.field_types[field]=TYPE_INT;
        if(mutation==3)saved->type_param_count=1;
    }
    CodegenResult r=shadows?codegen_compile_shadows(program,env,NULL,NULL):codegen_compile(program,env,NULL,NULL);
    if(!r.ok)fprintf(stderr,"codegen: %s\n",r.error_msg);
    CHECK(r.ok&&r.module);CHECK((r.module->ownership_size!=0)==published);CHECK((r.module->layout_size!=0)==published);
    CHECK(nvm_verify(r.module).ok);
    if(published) {
        CHECK(nvm_record_map_publication(r.module).status==NVM_ARRAY_ELIGIBLE);
        char *text=disasm_module_styled(r.module,DISASM_STYLE_CANONICAL);CHECK(text);
        CHECK(nl_nanoisa_array_publication(text)==1);free(text);
        size_t last=r.module->ownership_size-1;uint8_t saved_byte=r.module->ownership_data[last];
        r.module->ownership_data[last]=255;
        text=disasm_module_styled(r.module,DISASM_STYLE_CANONICAL);CHECK(text);
        CHECK(nl_nanoisa_array_publication(text)==-1);free(text);
        r.module->ownership_data[last]=saved_byte;
        CHECK(nvm_verify(r.module).ok);
    }
    if(mutation==4)publisher_arity_control(program,record,field,r.module);
    nvm_module_free(r.module);
    if(mutation&&mutation!=4){record->as.struct_def.field_type_info[field]=saved;record->as.struct_def.field_types[field]=saved_coarse;saved->type_param_count=saved_count;}
    free_environment(env);free_ast(program);free_tokens(tokens,count);
}
int main(int argc,char **argv) {
    CHECK(argc==2||(argc==3&&!strcmp(argv[2],"--string")));exact_types();
    FILE *f=fopen(argv[1],"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long size=ftell(f);CHECK(size>0);rewind(f);
    char *text=malloc((size_t)size+1);CHECK(text);CHECK(fread(text,1,(size_t)size,f)==(size_t)size);text[size]=0;CHECK(!fclose(f));
    compile_case(text,0,true,false);compile_case(text,0,true,true);
    for(unsigned i=1;i<=(argc==3?2u:3u);i++)compile_case(text,i,false,false);
    if(argc==3)compile_case(text,4,true,false);
    free(text);
    compile_case("struct Plain { value: int } fn main()->int { let m: HashMap<string,int> = (map_new) let p: Plain = Plain { value: 1 } (map_put m \"x\" p.value) return 0 }",0,false,false);
    compile_case("struct Box { values: array<int>, map: HashMap<string,int> } fn main()->int { let m: HashMap<string,int> = (map_new) let b: Box = Box { values: [1], map: m } (map_put m \"x\" 1) (map_remove m \"missing\") assert (== (str_trim \" x \" ) \"x\") assert (== (at b.values 0) 1) return 0 }",0,false,false);
    compile_case("struct Plain { value: int } fn main()->int { let m: HashMap<string,string> = (map_new) let p: Plain = Plain { value: 1 } (map_put m \"x\" \"value\") return 0 }",0,false,false);
    compile_case("struct Both { ints: HashMap<string,int>, texts: HashMap<string,string> } fn main()->int { let b: Both = Both { ints: (map_new), texts: (map_new) } (map_set b.ints \"x\" 1) (map_set b.texts \"x\" \"v\") return 0 }",0,false,false);
    compile_case("struct Both { ints: HashMap<string,int>, texts: HashMap<string,string>, values: array<int> } fn main()->int { let b: Both = Both { ints: (map_new), texts: (map_new), values: [1] } (map_set b.ints \"x\" 1) (map_set b.texts \"x\" \"v\") assert (== (at b.values 0) 1) return 0 }",0,false,false);
    printf("PASS %u paired C publication declaration/ordinary/shadow/omission checks\n",checks);return 0;
}
