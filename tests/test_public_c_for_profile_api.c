/* I refuse FOR without mutating compiler bindings or either published output. */
#define main public_c_options_fixture_main
#include "test_public_c_profile_options_api.c"
#undef main
int main(int argc,char **argv){
 assert(argc==2);CBOptions opts={0};
 ASTNode zero={.type=AST_NUMBER},ret={.type=AST_RETURN};ret.as.return_stmt.value=&zero;
 ASTNode *statements[]={&ret};ASTNode body={.type=AST_BLOCK};body.as.block.statements=statements;body.as.block.count=1;
 ASTNode fn={.type=AST_FUNCTION};fn.as.function.name="main";fn.as.function.return_type=TYPE_INT;fn.as.function.body=&body;
 ASTNode *items[]={&fn,NULL};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=1;
 ASTNode count={.type=AST_NUMBER};count.as.number=3;
 ASTNode loop={.type=AST_FOR};loop.as.for_stmt.var_name="value";loop.as.for_stmt.range_expr=&count;loop.as.for_stmt.body=&body;
 CBCtx context={0};context.root=&root;context.out=tmpfile();assert(context.out);
 ctx_push_scope(&context);ctx_add_sym(&context,"value",TYPE_STRING);
 int symbols=context.sym_count,depth=context.scope_depth;
 assert(emit_stmt(&context,&loop)!=0);assert(strstr(context.error,"for-loop lowering"));assert(ftell(context.out)==0);
 assert(context.sym_count==symbols && context.scope_depth==depth && ctx_lookup_type(&context,"value")==TYPE_STRING);
 const char *first=context.error;ctx_error(&context,"I preserve the first refusal.");assert(context.error==first);assert(!fclose(context.out));ctx_pop_scope(&context);
 /* I exercise top-level, statement and value publication boundaries. */
 items[1]=&loop;root.as.program.count=2;refuse(&root,argv[1],&opts);root.as.program.count=1;
 statements[0]=&loop;refuse(&root,argv[1],&opts);statements[0]=&ret;
 ret.as.return_stmt.value=&loop;refuse(&root,argv[1],&opts);ret.as.return_stmt.value=&zero;
 assert(c_backend_emit(&root,argv[1],"ordinary.nano",&opts)==0);
 FILE *stream=tmpfile();assert(stream);assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&opts)==0);assert(!fclose(stream));return 0;
}
