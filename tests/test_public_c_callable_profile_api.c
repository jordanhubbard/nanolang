/* I keep callable value refusal separate from descriptive direct signatures. */
#define main public_c_options_fixture_main
#include "test_public_c_profile_options_api.c"
#undef main
int main(int argc,char **argv){
 assert(argc==3);
 ASTNode zero={.type=AST_NUMBER};ASTNode ret={.type=AST_RETURN};ret.as.return_stmt.value=&zero;
 ASTNode *statements[]={&ret,&ret};ASTNode body={.type=AST_BLOCK};body.as.block.statements=statements;body.as.block.count=1;
 ASTNode fn={.type=AST_FUNCTION};fn.as.function.name="main";fn.as.function.return_type=TYPE_INT;fn.as.function.body=&body;
 ASTNode helper_ret={.type=AST_RETURN};helper_ret.as.return_stmt.value=&zero;ASTNode *helper_items[]={&helper_ret};ASTNode helper_body={.type=AST_BLOCK};helper_body.as.block.statements=helper_items;helper_body.as.block.count=1;
 ASTNode helper=fn;helper.as.function.name="target";helper.as.function.body=&helper_body;
 ASTNode *items[]={&fn,&helper,NULL};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=2;CBOptions opts={0};
 FunctionSignature scalar={.return_type=TYPE_INT};
 ASTNode reference={.type=AST_IDENTIFIER};reference.as.identifier="target";
 ASTNode local={.type=AST_LET};local.as.let.name="callback";local.as.let.var_type=TYPE_FUNCTION;local.as.let.fn_sig=&scalar;local.as.let.value=&reference;
 statements[0]=&local;refuse(&root,argv[1],&opts);statements[0]=&ret;
 items[2]=&local;root.as.program.count=3;refuse(&root,argv[1],&opts);root.as.program.count=2;
 helper.as.function.return_type=TYPE_FUNCTION;helper.as.function.return_fn_sig=&scalar;refuse(&root,argv[1],&opts);helper.as.function.return_type=TYPE_INT;helper.as.function.return_fn_sig=NULL;
 Parameter parameter={.name="callback",.type=TYPE_FUNCTION,.fn_sig=&scalar};helper.as.function.params=&parameter;helper.as.function.param_count=1;refuse(&root,argv[1],&opts);helper.as.function.is_extern=true;refuse(&root,argv[1],&opts);helper.as.function.is_extern=false;helper.as.function.param_count=0;
 TypeInfo callable={.base_type=TYPE_FUNCTION,.fn_sig=&scalar};TypeInfo *params[]={&callable};TypeInfo shell={.base_type=TYPE_STRUCT,.type_param_count=1,.type_params=params};helper.as.function.return_type_info=&shell;refuse(&root,argv[1],&opts);helper.as.function.return_type_info=NULL;
 Type field_types[]={TYPE_FUNCTION};char *fields[]={"callback"};ASTNode record={.type=AST_STRUCT_DEF};record.as.struct_def.name="Holder";record.as.struct_def.field_count=1;record.as.struct_def.field_names=fields;record.as.struct_def.field_types=field_types;items[2]=&record;root.as.program.count=3;refuse(&root,argv[1],&opts);
 int counts[]={1};Type *types[]={field_types};char **names[]={fields};char *variants[]={"Some"};ASTNode un={.type=AST_UNION_DEF};un.as.union_def.name="Choice";un.as.union_def.variant_count=1;un.as.union_def.variant_field_counts=counts;un.as.union_def.variant_field_types=types;un.as.union_def.variant_field_names=names;un.as.union_def.variant_names=variants;items[2]=&un;refuse(&root,argv[1],&opts);root.as.program.count=2;
 ret.as.return_stmt.value=&reference;refuse(&root,argv[1],&opts);
 ASTNode call={.type=AST_CALL};call.as.call.name="target";call.as.call.func_expr=&reference;call.as.call.checked_signature=&scalar;ret.as.return_stmt.value=&call;refuse(&root,argv[1],&opts);call.as.call.func_expr=NULL;
 FunctionSignature nested={.return_type=TYPE_FUNCTION,.return_fn_sig=&scalar};FunctionSignature outer={.return_type=TYPE_INT,.return_fn_sig=&nested};call.as.call.checked_signature=&outer;refuse(&root,argv[1],&opts);call.as.call.checked_signature=&scalar;
 local.as.let.name="target";local.as.let.var_type=TYPE_INT;local.as.let.fn_sig=NULL;local.as.let.value=&zero;statements[0]=&local;statements[1]=&ret;body.as.block.count=2;refuse(&root,argv[1],&opts);body.as.block.count=1;statements[0]=&ret;
 ASTNode qualified={.type=AST_MODULE_QUALIFIED_CALL};qualified.as.module_qualified_call.module_alias="mod";qualified.as.module_qualified_call.function_name="target";ret.as.return_stmt.value=&qualified;
 /* Bare target is not the emitted mod_target declaration. */
 refuse(&root,argv[1],&opts);helper.as.function.name="mod_target";
 local.as.let.name="mod_target";statements[0]=&local;body.as.block.count=2;refuse(&root,argv[1],&opts);body.as.block.count=1;statements[0]=&ret;
 if(!strcmp(argv[2],"direct")){helper.as.function.name="target";ret.as.return_stmt.value=&call;}
 if(!strcmp(argv[2],"scalar")){helper.as.function.name="target";local.as.let.name="target";reference.as.identifier="target";ret.as.return_stmt.value=&reference;statements[0]=&local;statements[1]=&ret;body.as.block.count=2;}
 assert(c_backend_emit(&root,argv[1],"ordinary.nano",&opts)==0);
 FILE *stream=tmpfile();assert(stream);assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&opts)==0);assert(!fclose(stream));return 0;
}
