/* I preserve binding identity and publication when scalar formatting refuses. */
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
 ASTNode floating={.type=AST_FLOAT};floating.as.float_val=1.0;
 ASTNode *args[]={&floating};ASTNode call={.type=AST_CALL};
 call.as.call.name="float_to_string";call.as.call.args=args;call.as.call.arg_count=1;
 CBCtx context={0};strcpy(context.prefix,"nano_fixture_");char text[4096];
 expression(&context,&call,text,sizeof text);assert(strstr(text,"nano_fixture_float_text_new(")==text);
 assert(infer_expr_type(&context,&call)==TYPE_STRING);
 ctx_push_scope(&context);ctx_add_sym(&context,"float_to_string",TYPE_FUNCTION);
 callable_refusal(&context,&call);
 assert(infer_expr_type(&context,&call)==TYPE_UNKNOWN);ctx_pop_scope(&context);
 FunctionSignature signature={0};signature.return_type=TYPE_STRING;call.as.call.checked_signature=&signature;
 expression(&context,&call,text,sizeof text);assert(strstr(text,"float_to_string(")==text);
 call.as.call.checked_signature=NULL;
 ASTNode callee={.type=AST_IDENTIFIER};callee.as.identifier="callback";call.as.call.func_expr=&callee;
 callable_refusal(&context,&call);
 assert(infer_expr_type(&context,&call)==TYPE_UNKNOWN);call.as.call.func_expr=NULL;
 ASTNode declared={.type=AST_FUNCTION};declared.as.function.name="float_to_string";declared.as.function.return_type=TYPE_STRING;
 ASTNode *declarations[]={&declared};ASTNode declaration_root={.type=AST_PROGRAM};declaration_root.as.program.items=declarations;declaration_root.as.program.count=1;
 context.root=&declaration_root;expression(&context,&call,text,sizeof text);assert(strstr(text,"float_to_string(")==text);context.root=NULL;
 call.as.call.name="print";expression(&context,&call,text,sizeof text);assert(strstr(text,"nano_fixture_public_float_print(")==text);
 ctx_push_scope(&context);ctx_add_sym(&context,"print",TYPE_FUNCTION);callable_refusal(&context,&call);ctx_pop_scope(&context);
 call.as.call.func_expr=&callee;callable_refusal(&context,&call);call.as.call.func_expr=NULL;
 ASTNode unknown={.type=AST_IDENTIFIER};unknown.as.identifier="unresolved";
 args[0]=&unknown;call.as.call.name="float_to_string";
 ASTNode local={.type=AST_LET};local.as.let.name="result";local.as.let.var_type=TYPE_STRING;local.as.let.value=&call;
 ASTNode zero={.type=AST_NUMBER};ASTNode result={.type=AST_RETURN};result.as.return_stmt.value=&zero;
 ASTNode *statements[]={&local,&result};ASTNode block={.type=AST_BLOCK};block.as.block.statements=statements;block.as.block.count=2;
 ASTNode main_function={.type=AST_FUNCTION};main_function.as.function.name="main";main_function.as.function.return_type=TYPE_INT;main_function.as.function.body=&block;
 ASTNode *items[]={&main_function};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=1;
 FILE *file=fopen(argv[1],"w");assert(file);assert(fputs("previous",file)>=0);assert(fclose(file)==0);
 CBOptions options={0};assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)!=0);
 file=fopen(argv[1],"r");assert(file);memset(text,0,sizeof text);assert(fread(text,1,sizeof text,file)==8);assert(strcmp(text,"previous")==0);assert(fclose(file)==0);
 args[0]=&floating;assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)==0);
 return 0;
}
