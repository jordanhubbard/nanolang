/* I preserve binding identity and publication when string length refuses. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/c_backend.c"
int *passive_binding_order(const ASTNode *block){(void)block;assert(!"I reached an unrelated passive fixture.");return NULL;}
static void expression(CBCtx *context,ASTNode *node,char *text,size_t size){
 context->out=tmpfile();assert(context->out);assert(emit_expr(context,node)==0);
 assert(fflush(context->out)==0);rewind(context->out);size_t n=fread(text,1,size-1,context->out);text[n]=0;
 assert(fclose(context->out)==0);
}
static void callable_refusal(CBCtx *context,ASTNode *node){
 context->out=tmpfile();assert(context->out);context->error=NULL;
 assert(emit_expr(context,node)!=0);
 assert(strstr(context->error,"first-class callable values or indirect calls"));
 assert(ftell(context->out)==0);assert(fclose(context->out)==0);context->error=NULL;
}
int main(int argc,char **argv){
 assert(argc==2);
 ASTNode string={.type=AST_STRING};string.as.string_val="ordinary";
 ASTNode *args[]={&string};ASTNode call={.type=AST_CALL};
 call.as.call.name="str_length";call.as.call.args=args;call.as.call.arg_count=1;
 CBCtx context={0};strcpy(context.prefix,"nano_fixture_");char text[4096];
 expression(&context,&call,text,sizeof text);assert(strstr(text,"(int64_t)strlen(")==text);
 assert(infer_expr_type(&context,&call)==TYPE_INT);
 ctx_push_scope(&context);ctx_add_sym(&context,"str_length",TYPE_FUNCTION);
 callable_refusal(&context,&call);
 assert(infer_expr_type(&context,&call)==TYPE_UNKNOWN);ctx_pop_scope(&context);
 FunctionSignature signature={0};signature.return_type=TYPE_INT;call.as.call.checked_signature=&signature;
 expression(&context,&call,text,sizeof text);assert(strstr(text,"str_length(")==text);
 call.as.call.checked_signature=NULL;
 ASTNode callee={.type=AST_IDENTIFIER};callee.as.identifier="callback";call.as.call.func_expr=&callee;
 callable_refusal(&context,&call);
 assert(infer_expr_type(&context,&call)==TYPE_UNKNOWN);call.as.call.func_expr=NULL;
 ASTNode declared={.type=AST_FUNCTION};declared.as.function.name="str_length";declared.as.function.return_type=TYPE_INT;
 ASTNode *declarations[]={&declared};ASTNode declaration_root={.type=AST_PROGRAM};declaration_root.as.program.items=declarations;declaration_root.as.program.count=1;
 context.root=&declaration_root;expression(&context,&call,text,sizeof text);assert(strstr(text,"str_length(")==text);context.root=NULL;
 ASTNode wrong_float={.type=AST_FLOAT},wrong_bool={.type=AST_BOOL},wrong_int={.type=AST_NUMBER};
 ASTNode *wrong[]={&wrong_float,&wrong_bool,&wrong_int};
 for(size_t i=0;i<sizeof wrong/sizeof wrong[0];++i){
  args[0]=wrong[i];assert(infer_expr_type(&context,&call)==TYPE_UNKNOWN);
  context.out=tmpfile();assert(context.out);context.error=NULL;
  assert(emit_expr(&context,&call)!=0);assert(strstr(context.error,"exact STRING"));
  const char *first=context.error;ctx_error(&context,"I retain my first diagnostic.");assert(context.error==first);
  assert(fclose(context.out)==0);
 }
 args[0]=&string;call.as.call.arg_count=0;context.out=tmpfile();assert(context.out);context.error=NULL;
 assert(emit_expr(&context,&call)!=0);assert(fclose(context.out)==0);call.as.call.arg_count=1;
 ASTNode unknown={.type=AST_IDENTIFIER};unknown.as.identifier="unresolved";
 args[0]=&unknown;call.as.call.name="str_length";
 ASTNode local={.type=AST_LET};local.as.let.name="result";local.as.let.var_type=TYPE_INT;local.as.let.value=&call;
 ASTNode zero={.type=AST_NUMBER};ASTNode result={.type=AST_RETURN};result.as.return_stmt.value=&zero;
 ASTNode *statements[]={&local,&result};ASTNode block={.type=AST_BLOCK};block.as.block.statements=statements;block.as.block.count=2;
 ASTNode main_function={.type=AST_FUNCTION};main_function.as.function.name="main";main_function.as.function.return_type=TYPE_INT;main_function.as.function.body=&block;
 ASTNode *items[]={&main_function};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=1;
 FILE *file=fopen(argv[1],"w");assert(file);assert(fputs("previous",file)>=0);assert(fclose(file)==0);
 CBOptions options={0};assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)!=0);
 file=fopen(argv[1],"r");assert(file);memset(text,0,sizeof text);assert(fread(text,1,sizeof text,file)==8);assert(strcmp(text,"previous")==0);assert(fclose(file)==0);
 FILE *stream=tmpfile();assert(stream);assert(fputs("sentinel",stream)>=0);
 assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&options)!=0);
 assert(ftell(stream)==8);assert(fclose(stream)==0);
 args[0]=&string;assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)==0);
 return 0;
}
