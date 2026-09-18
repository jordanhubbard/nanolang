/* I retain direct signatures while refusing unsupported boolean conversions. */
#define main public_c_options_fixture_main
#include "test_public_c_profile_options_api.c"
#undef main
int main(int argc,char **argv){
 assert(argc==3);CBOptions opts={0};
 ASTNode yes={.type=AST_BOOL};yes.as.bool_val=true;
 ASTNode zero={.type=AST_NUMBER},number={.type=AST_NUMBER};number.as.number=7;
 ASTNode text={.type=AST_STRING};text.as.string_val="declared";
 ASTNode *args[]={&yes,&yes};ASTNode call={.type=AST_CALL};call.as.call.name="bool_to_string";call.as.call.args=args;call.as.call.arg_count=1;
 ASTNode local={.type=AST_LET};local.as.let.name="result";local.as.let.var_type=TYPE_STRING;local.as.let.value=&call;
 ASTNode reference={.type=AST_IDENTIFIER};reference.as.identifier="result";
 ASTNode *compare_args[]={&reference,&text};ASTNode compare={.type=AST_PREFIX_OP};compare.as.prefix_op.op=TOKEN_EQ;compare.as.prefix_op.args=compare_args;compare.as.prefix_op.arg_count=2;
 ASTNode check={.type=AST_ASSERT};check.as.assert.condition=&compare;
 ASTNode ret={.type=AST_RETURN};ret.as.return_stmt.value=&zero;
 ASTNode *statements[]={&local,&check,&ret};ASTNode body={.type=AST_BLOCK};body.as.block.statements=statements;body.as.block.count=3;
 ASTNode main_fn={.type=AST_FUNCTION};main_fn.as.function.name="main";main_fn.as.function.return_type=TYPE_INT;main_fn.as.function.body=&body;
 ASTNode helper_ret={.type=AST_RETURN};helper_ret.as.return_stmt.value=&text;
 ASTNode *helper_statements[]={&helper_ret};ASTNode helper_body={.type=AST_BLOCK};helper_body.as.block.statements=helper_statements;helper_body.as.block.count=1;
 Parameter parameter={.name="value",.type=TYPE_INT};ASTNode helper={.type=AST_FUNCTION};helper.as.function.name="ordinary";helper.as.function.return_type=TYPE_STRING;helper.as.function.params=&parameter;helper.as.function.param_count=1;helper.as.function.body=&helper_body;
 ASTNode *items[]={&main_fn,&helper};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=2;
 ASTNode real={.type=AST_FLOAT},unknown={.type=AST_IDENTIFIER};unknown.as.identifier="unresolved";
 ASTNode *bad[]={&number,&real,&text,&unknown};CBCtx context={0};context.root=&root;strcpy(context.prefix,"nano_fixture_");
 for(size_t i=0;i<sizeof bad/sizeof bad[0];i++){
  args[0]=bad[i];assert(infer_expr_type(&context,&call)==TYPE_UNKNOWN);context.out=tmpfile();assert(context.out);context.error=NULL;
  assert(emit_expr(&context,&call)!=0);assert(strstr(context.error,"one exact BOOL"));assert(ftell(context.out)==0);assert(!fclose(context.out));
  refuse(&root,argv[1],&opts);
 }
 args[0]=&yes;
 for(int count=0;count<=2;count+=2){call.as.call.arg_count=count;assert(infer_expr_type(&context,&call)==TYPE_UNKNOWN);refuse(&root,argv[1],&opts);}call.as.call.arg_count=1;
 ctx_push_scope(&context);ctx_add_sym(&context,"bool_to_string",TYPE_FUNCTION);context.out=tmpfile();context.error=NULL;assert(context.out);assert(emit_expr(&context,&call)!=0);assert(strstr(context.error,"first-class callable"));assert(!fclose(context.out));ctx_pop_scope(&context);
 ASTNode callee={.type=AST_IDENTIFIER};callee.as.identifier="callback";call.as.call.func_expr=&callee;refuse(&root,argv[1],&opts);call.as.call.func_expr=NULL;
 Type parameter_types[]={TYPE_INT};FunctionSignature signature={.param_count=1,.param_types=parameter_types,.return_type=TYPE_STRING};
 ASTNode qualified={.type=AST_MODULE_QUALIFIED_CALL};qualified.as.module_qualified_call.module_alias="mod";qualified.as.module_qualified_call.function_name="bool_to_string";qualified.as.module_qualified_call.args=args;qualified.as.module_qualified_call.arg_count=1;
 if(!strcmp(argv[2],"builtin")){text.as.string_val="true";}
 else{args[0]=&number;helper.as.function.name="bool_to_string";
  if(!strcmp(argv[2],"signature"))call.as.call.checked_signature=&signature;
  if(!strcmp(argv[2],"qualified")){helper.as.function.name="mod_bool_to_string";local.as.let.value=&qualified;}
 }
 assert(c_backend_emit(&root,argv[1],"ordinary.nano",&opts)==0);
 FILE *stream=tmpfile();assert(stream);assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&opts)==0);assert(!fclose(stream));return 0;
}
