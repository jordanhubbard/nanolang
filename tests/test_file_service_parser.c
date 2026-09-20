/* I exercise actual parser ownership, not service execution. */
#include "nanolang.h"
#include "emit_typed_ast.h"
#include "reflection.h"
#include "type_infer.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc; char **g_argv; char g_project_root[4096]=".";
const char *get_project_root(void) { return g_project_root; }
static void *live[8192], *items_pointer;
static size_t live_count, attempt, fail_at, hits;
static bool fail_items, persistent;
static bool rejected(void) { attempt++; if (fail_at && (attempt==fail_at || (persistent && attempt>fail_at))) { hits++; return true; } return false; }
static void remember(void *p) { if(p) { assert(live_count<8192);live[live_count++]=p; } }
static void forget(void *p) { for(size_t i=0;i<live_count;i++)if(live[i]==p){live[i]=live[--live_count];return;} }
static void *pmalloc(size_t n) { if(rejected())return NULL;void*p=malloc(n);remember(p);if(attempt==2)items_pointer=p;return p; }
static void *pcalloc(size_t n,size_t z) { if(rejected())return NULL;void*p=calloc(n,z);remember(p);return p; }
static void pfree(void *p) { forget(p);free(p); }
static void *prealloc(void *p,size_t n) {
    if(rejected())return NULL;
    if(fail_items && p==items_pointer){hits++;return NULL;}
    /* I remove a tracked address before realloc and restore it on failure. */
    bool tracked=false;for(size_t i=0;i<live_count;i++)if(live[i]==p){tracked=true;break;}
    forget(p);void*q=realloc(p,n);if(q)remember(q);else if(tracked)remember(p);return q;
}
static char *pstrdup(const char *s) { size_t n=strlen(s)+1;char*p=pmalloc(n);if(p)memcpy(p,s,n);return p; }
#define malloc pmalloc
#define calloc pcalloc
#define realloc prealloc
#define strdup pstrdup
#define free pfree
#include "../src/parser.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free
static const char *declaration="service \"nsi:nanolang/filesystem\" catalog 1 from \"interface.nsi.json\"";
static ASTNode *parse(const char *source) { int n=0;Token*t=tokenize(source,&n);assert(t);ASTNode*p=parse_program(t,n);free_tokens(t,n);return p; }
static void reset(void) { assert(live_count==0);attempt=hits=fail_at=0;items_pointer=NULL;fail_items=persistent=false; }
static void retained(ASTNode*p,int count) {
    assert(p&&p->type==AST_PROGRAM&&p->as.program.count==count);
    ASTNode*n=p->as.program.items[0];assert(n->type==AST_SERVICE_DECL);
    assert(!strcmp(n->as.service_decl.interface_id,"nsi:nanolang/filesystem"));
    assert(n->as.service_decl.interface_bytes==23&&n->as.service_decl.path_bytes==18);
    assert(!strcmp(n->as.service_decl.document_path,"interface.nsi.json"));
    assert(n->as.service_decl.catalog_version==1&&n->as.service_decl.origin_index==-1);
    assert(ast_has_service_declaration(p));
}
static void grammar(void) {
    const char *bad[]={"service","pub service \"nsi:nanolang/filesystem\" catalog 1 from \"x\"",
      "service \"nsi:nanolang/filesystem\" catalog 01 from \"x\"",
      "service \"nsi:nanolang/filesystem\" catalog 1.0 from \"x\"",
      "service \"nsi:nanolang/filesystem\" catalog 2 from \"x\"",
      "service \"nsi:nanolang/filesystem\" catalog 1 from \"\"",
      "service \"nsi:nanolang/filesystem\" catalog 1 from \"/x\"",
      "service \"nsi:nanolang/filesystem\" catalog 1 from \"a\\0suffix\"",
      "service \"nsi:nanolang/filesystem\" catalog 1 from \"x\" extra",
      "service \"nsi:nanolang/filesystem\"\ncatalog 1 from \"x\""};
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);i++){reset();assert(!parse(bad[i]));assert(!live_count);}
    reset();ASTNode*p=parse(declaration);retained(p,1);free_ast(p);assert(!live_count);
    char text[1024];snprintf(text,sizeof(text),"%s # comment\nfn service() -> int { return 7 }\nshadow service { assert true }\n",declaration);
    reset();p=parse(text);retained(p,3);free_ast(p);assert(!live_count);
    int n;Token*t=tokenize(declaration,&n);assert(t&&n==7);
    int64_t original=t[5].value_bytes;
    const int64_t invalid[]={-1,0,17,19,1048577};
    for(size_t i=0;i<sizeof(invalid)/sizeof(*invalid);i++){reset();t[5].value_bytes=invalid[i];assert(!parse_program(t,n));assert(!live_count);}
    t[5].value_bytes=original;
    const unsigned char invalid_utf8[][5]={{0xc0,0x80,0},{0xed,0xa0,0x80,0},{0xf4,0x90,0x80,0x80,0},{0xe2,0x82,0},{0x80,0}};
    const char *saved=t[5].value;
    for(size_t i=0;i<sizeof(invalid_utf8)/sizeof(*invalid_utf8);i++){
        t[5].value=(char*)invalid_utf8[i];t[5].value_bytes=(int64_t)strlen(t[5].value);reset();assert(!parse_program(t,n));assert(!live_count);
    }
    t[5].value="é";t[5].value_bytes=2;reset();p=parse_program(t,n);assert(p);free_ast(p);assert(!live_count);
    t[5].value=saved;t[5].value_bytes=original;free_tokens(t,n);
}
static void allocations(void) {
    char text[4096]="";for(int i=0;i<33;i++){strcat(text,declaration);strcat(text,"\n");}
    reset();ASTNode*p=parse(text);assert(p);size_t total=attempt;free_ast(p);assert(!live_count);
    for(int mode=0;mode<2;mode++)for(size_t i=1;i<=total;i++){
        reset();persistent=mode!=0;fail_at=i;p=parse(text);assert(!p&&hits>=1&&!live_count);
        fail_at=0;p=parse(text);assert(p);free_ast(p);assert(!live_count);
        printf("allocation:%s:%zu:refused:recovered\n",mode?"persistent":"transient",i);
    }
    reset();p=parse(declaration);assert(p);ASTNode*source=p->as.program.items[0];size_t base=live_count;
    for(size_t i=1;i<=3;i++){attempt=hits=0;fail_at=i;assert(!clone_ast_node(source));assert(hits==1&&live_count==base);}
    fail_at=0;ASTNode*copy=clone_ast_node(source);assert(copy&&copy->as.service_decl.interface_id!=source->as.service_decl.interface_id);free_ast(p);assert(!strcmp(copy->as.service_decl.document_path,"interface.nsi.json"));assert(copy->as.service_decl.interface_bytes==23&&copy->as.service_decl.path_bytes==18&&copy->as.service_decl.catalog_version==1&&copy->as.service_decl.origin_index==-1);free_ast(copy);assert(!live_count);
    strcpy(text,declaration);strcat(text,"\nfn f0() -> int { let callback = fn() -> int { return 1 } return (callback) }\n");
    for(int i=1;i<15;i++){char row[80];snprintf(row,sizeof(row),"fn f%d() -> int { return %d }\n",i,i);strcat(text,row);}
    reset();p=parse(text);assert(p&&p->as.program.count==17);free_ast(p);assert(!live_count);
    reset();fail_items=true;p=parse(text);assert(!p&&hits==1&&!live_count);fail_items=false;
}
int main(int argc,char **argv) {
    assert(argc==2);grammar();allocations();reset();
    FILE*f=fopen(argv[1],"rb");assert(f);assert(!fseek(f,0,SEEK_END));long size=ftell(f);assert(size>0&&size<1048576);rewind(f);
    char*s=malloc((size_t)size+1);assert(s&&fread(s,1,(size_t)size,f)==(size_t)size);s[size]=0;assert(!fclose(f));
    ASTNode*p=parse(s);free(s);retained(p,6);Environment*env=create_environment();assert(env);
    assert(!type_check(p,env));assert(!type_check_module(p,env));assert(!process_imports(p,env,NULL,"binding.nano"));
    assert(!type_check_root_shadows(p,env));assert(!run_shadow_tests(p,env,false));assert(!transpile_to_c(p,env,"binding.nano"));
    assert(!hm_infer_program(p,"binding.nano"));HMInferResult inference=hm_infer_program_for_lsp(p,"binding.nano");assert(!inference.ok&&!inference.ctx&&!inference.env);hm_infer_result_free(&inference);
    emit_typed_ast_json("binding.nano",p,env); /* I require no stdout beyond exact final/counter rows. */
    char output[4096];assert(snprintf(output,sizeof(output),"%s.refusal",argv[1])<(int)sizeof(output));
    f=fopen(output,"wb");assert(f&&fwrite("unchanged",1,9,f)==9&&!fclose(f));
    assert(!emit_module_reflection(output,p,env,"binding"));f=fopen(output,"rb");assert(f);char bytes[10]={0};assert(fread(bytes,1,10,f)==9&&!memcmp(bytes,"unchanged",9));assert(!fclose(f));
    free_environment(env);free_ast(p);assert(!live_count);puts("publisher:1:5:retained:consumers-refused");return 0;
}
