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
    CHECK(ordinary_map_pair(&map));CHECK(!ordinary_map_pair(NULL));
    for(unsigned i=0;i<2;i++){TypeInfo *saved=args[i];args[i]=NULL;CHECK(!ordinary_map_pair(&map));args[i]=saved;}
    map.type_param_count=1;CHECK(!ordinary_map_pair(&map));map.type_param_count=2;
    map.generic_name="Other";CHECK(!ordinary_map_pair(&map));map.generic_name="HashMap";
    key.base_type=TYPE_INT;CHECK(!ordinary_map_pair(&map));key.base_type=TYPE_STRING;
    value.element_type=&key;CHECK(!ordinary_map_pair(&map));value.element_type=NULL;
    value.service_ordinal=1;CHECK(!ordinary_map_pair(&map));value.service_ordinal=0;
    value.type_var_count=1;CHECK(!ordinary_map_pair(&map));value.type_var_count=0;
    map.fn_sig=(void *)(uintptr_t)1;CHECK(!ordinary_map_pair(&map));map.fn_sig=NULL;
    CHECK(ordinary_map_pair(&map));
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
    if(mutation){CHECK(field>=0);saved=record->as.struct_def.field_type_info[field];CHECK(saved);saved_coarse=record->as.struct_def.field_types[field];saved_count=saved->type_param_count;
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
    nvm_module_free(r.module);
    if(mutation){record->as.struct_def.field_type_info[field]=saved;record->as.struct_def.field_types[field]=saved_coarse;saved->type_param_count=saved_count;}
    free_environment(env);free_ast(program);free_tokens(tokens,count);
}
int main(int argc,char **argv) {
    CHECK(argc==2);exact_types();
    FILE *f=fopen(argv[1],"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long size=ftell(f);CHECK(size>0);rewind(f);
    char *text=malloc((size_t)size+1);CHECK(text);CHECK(fread(text,1,(size_t)size,f)==(size_t)size);text[size]=0;CHECK(!fclose(f));
    compile_case(text,0,true,false);compile_case(text,0,true,true);
    for(unsigned i=1;i<=3;i++)compile_case(text,i,false,false);
    free(text);
    compile_case("struct Plain { value: int } fn main()->int { let m: HashMap<string,int> = (map_new) let p: Plain = Plain { value: 1 } (map_put m \"x\" p.value) return 0 }",0,false,false);
    compile_case("struct Box { values: array<int>, map: HashMap<string,int> } fn main()->int { let m: HashMap<string,int> = (map_new) let b: Box = Box { values: [1], map: m } (map_put m \"x\" 1) (map_remove m \"missing\") assert (== (str_trim \" x \" ) \"x\") assert (== (at b.values 0) 1) return 0 }",0,false,false);
    printf("PASS %u paired C publication declaration/ordinary/shadow/omission checks\n",checks);return 0;
}
