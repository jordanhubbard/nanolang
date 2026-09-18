/* I retain publication and exact nominal identity across refused API calls. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/c_backend.c"
int *passive_binding_order(const ASTNode *block){(void)block;assert(!"I reached unrelated passive code.");return NULL;}
static void retained(const char *path){FILE *f=fopen(path,"r");char text[16]={0};assert(f);assert(fread(text,1,sizeof text,f)==8);assert(strcmp(text,"previous")==0);assert(fclose(f)==0);}
int main(int argc,char **argv){
 assert(argc==2);
 char *variants[]={"Item"};int counts[]={1};Type field_types[]={TYPE_INT};Type *types[]={field_types};
 char *names[]={"value"};char **fields[]={names};ASTNode decl={.type=AST_UNION_DEF};
 decl.as.union_def.name="Choice";decl.as.union_def.variant_count=1;decl.as.union_def.variant_names=variants;
 decl.as.union_def.variant_field_counts=counts;decl.as.union_def.variant_field_types=types;decl.as.union_def.variant_field_names=fields;
 ASTNode other=decl;other.as.union_def.name="Other";
 ASTNode zero={.type=AST_NUMBER};ASTNode boolean={.type=AST_BOOL};ASTNode *values[]={&zero};
 ASTNode construct={.type=AST_UNION_CONSTRUCT};construct.as.union_construct.union_name="Choice";
 construct.as.union_construct.variant_name="Item";construct.as.union_construct.field_count=1;
 construct.as.union_construct.field_names=names;construct.as.union_construct.field_values=values;
 ASTNode ret={.type=AST_RETURN};ret.as.return_stmt.value=&construct;
 ASTNode *returns[]={&ret};ASTNode body={.type=AST_BLOCK};body.as.block.count=1;body.as.block.statements=returns;
 ASTNode fn={.type=AST_FUNCTION};fn.as.function.name="produce";fn.as.function.return_type=TYPE_UNION;
 fn.as.function.return_struct_type_name="Choice";fn.as.function.body=&body;
 ASTNode mainret={.type=AST_RETURN};mainret.as.return_stmt.value=&zero;
 ASTNode call={.type=AST_CALL};call.as.call.name="produce";
 ASTNode local={.type=AST_LET};local.as.let.name="result";local.as.let.var_type=TYPE_UNION;local.as.let.type_name="Choice";local.as.let.value=&call;
 ASTNode *statements[]={&local,&mainret};ASTNode mainbody={.type=AST_BLOCK};mainbody.as.block.count=2;mainbody.as.block.statements=statements;
 ASTNode mainfn={.type=AST_FUNCTION};mainfn.as.function.name="main";mainfn.as.function.return_type=TYPE_INT;mainfn.as.function.body=&mainbody;
 ASTNode *items[]={&decl,&other,&fn,&mainfn};ASTNode root={.type=AST_PROGRAM};root.as.program.count=4;root.as.program.items=items;
 CBOptions options={0};
 for(int mode=0;mode<10;++mode){
  fn.as.function.return_struct_type_name="Choice";fn.as.function.is_extern=false;
  construct.as.union_construct.union_name="Choice";construct.as.union_construct.variant_name="Item";
  decl.as.union_def.generic_param_count=0;decl.as.union_def.is_extern=false;other.as.union_def.name="Other";
  local.as.let.type_name="Choice";values[0]=&zero;call.as.call.func_expr=NULL;
  if(mode==0)fn.as.function.return_struct_type_name=NULL;
  if(mode==1)fn.as.function.return_struct_type_name="Unknown";
  if(mode==2)construct.as.union_construct.union_name="Other";
  if(mode==3)construct.as.union_construct.variant_name="Missing";
  if(mode==4)decl.as.union_def.generic_param_count=1;
  if(mode==5)fn.as.function.is_extern=true;
  if(mode==6)decl.as.union_def.is_extern=true;
  if(mode==7)local.as.let.type_name="Other";
  if(mode==8)values[0]=&boolean;
  if(mode==9)other.as.union_def.name="Choice";
  FILE *f=fopen(argv[1],"w");assert(f);assert(fputs("previous",f)>=0);assert(fclose(f)==0);
  assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)!=0);retained(argv[1]);
  FILE *stream=tmpfile();assert(stream);assert(fputs("previous",stream)>=0);
  assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&options)!=0);assert(ftell(stream)==8);assert(fclose(stream)==0);
 }
 other.as.union_def.name="Other";
 CBCtx ctx={0};ctx.root=&root;ctx_add_sym(&ctx,"produce",TYPE_FUNCTION);
 assert(!ctx_union_value(&ctx,&call));const char *first=ctx.error;assert(first);
 ctx_error(&ctx,"I retain my first error.");assert(ctx.error==first);
 ctx=(CBCtx){0};ctx.root=&root;call.as.call.func_expr=&zero;
 assert(!ctx_union_value(&ctx,&call));assert(ctx.error);call.as.call.func_expr=NULL;
 assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)==0);
 FILE *f=fopen(argv[1],"r");assert(f);char text[32768]={0};assert(fread(text,1,sizeof text-1,f)>0);assert(fclose(f)==0);
 assert(strstr(text,"NanoUnion_Choice produce(void);"));assert(strstr(text,"NanoUnion_Choice produce(void) {"));
 return 0;
}
