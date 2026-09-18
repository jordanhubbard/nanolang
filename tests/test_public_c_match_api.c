/* I retain exact union identity refusal and recover in the same process. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/c_backend.c"
int *passive_binding_order(const ASTNode *block){(void)block;assert(!"I reached unrelated passive code.");return NULL;}
static void save(const char *path){FILE *f=fopen(path,"w");assert(f);assert(fputs("previous",f)>=0);assert(fclose(f)==0);}
static void retained(const char *path){FILE *f=fopen(path,"r");char text[16]={0};assert(f);assert(fread(text,1,sizeof text,f)==8);assert(strcmp(text,"previous")==0);assert(fclose(f)==0);}
int main(int argc,char **argv){
 assert(argc==2);
 char *variants[]={"Item"};int counts[]={1};Type field_types[]={TYPE_INT};Type *types[]={field_types};
 char *field_names[]={"value"};char **names[]={field_names};
 ASTNode declaration={.type=AST_UNION_DEF};declaration.as.union_def.name="Choice";
 declaration.as.union_def.variant_count=1;declaration.as.union_def.variant_names=variants;
 declaration.as.union_def.variant_field_counts=counts;declaration.as.union_def.variant_field_types=types;
 declaration.as.union_def.variant_field_names=names;
 ASTNode zero={.type=AST_NUMBER};ASTNode *values[]={&zero};
 ASTNode construct={.type=AST_UNION_CONSTRUCT};construct.as.union_construct.union_name="Choice";
 construct.as.union_construct.variant_name="Item";construct.as.union_construct.field_count=1;
 construct.as.union_construct.field_names=field_names;construct.as.union_construct.field_values=values;
 ASTNode result={.type=AST_RETURN};result.as.return_stmt.value=&zero;
 ASTNode *bodies[]={&result};char *bindings[]={"p"};char *patterns[]={"Item"};
 ASTNode match={.type=AST_MATCH};match.as.match_expr.expr=&construct;
 match.as.match_expr.arm_count=1;match.as.match_expr.union_type_name="Choice";
 match.as.match_expr.pattern_variants=patterns;match.as.match_expr.pattern_bindings=bindings;match.as.match_expr.arm_bodies=bodies;
 ASTNode *statements[]={&match,&result};ASTNode body={.type=AST_BLOCK};body.as.block.count=2;body.as.block.statements=statements;
 ASTNode function={.type=AST_FUNCTION};function.as.function.name="main";function.as.function.return_type=TYPE_INT;function.as.function.body=&body;
 ASTNode *items[]={&declaration,&function};ASTNode root={.type=AST_PROGRAM};root.as.program.count=2;root.as.program.items=items;
 CBOptions options={0};CBCtx context={0};context.root=&root;int d=-1,v=-1;
 assert(ctx_union_variant(&context,"Choice","Item",&d,&v)==&declaration&&d==0&&v==0);
 assert(ctx_union_variant(&context,"Other","Item",&d,&v)==NULL);
 assert(ctx_union_variant(&context,"Choice","Other",&d,&v)==NULL);
 for(int mode=0;mode<3;++mode){
  match.as.match_expr.union_type_name=mode==0?"Other":"Choice";
  patterns[0]=mode==1?"Other":mode==2?"_":"Item";
  save(argv[1]);assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)!=0);retained(argv[1]);
  FILE *stream=tmpfile();assert(stream);assert(fputs("previous",stream)>=0);
  assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&options)!=0);assert(ftell(stream)==8);assert(fclose(stream)==0);
  context.out=tmpfile();assert(context.out);context.error=NULL;
  assert(emit_stmt(&context,&match)!=0);assert(strstr(context.error,"exact declared"));
  const char *first=context.error;ctx_error(&context,"I retain the first diagnostic.");assert(context.error==first);
  assert(fclose(context.out)==0);
 }
 match.as.match_expr.union_type_name="Choice";patterns[0]="Item";
 assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)==0);
 FILE *f=fopen(argv[1],"r");assert(f);char text[32768]={0};assert(fread(text,1,sizeof text-1,f)>0);assert(fclose(f)==0);
 assert(strstr(text,"nano_cb_0_payload_0_0 p = nano_cb_0_match_value.as.Item"));
 assert(!strstr(text,"__typeof__"));
 return 0;
}
