/* I reuse the independently qualified publication sentinel checks. */
#define main public_c_options_fixture_main
#include "test_public_c_profile_options_api.c"
#undef main
int main(int argc,char **argv){
 assert(argc==3);
 ASTNode zero={.type=AST_NUMBER};ASTNode ret={.type=AST_RETURN};ret.as.return_stmt.value=&zero;
 ASTNode *statements[]={&ret};ASTNode body={.type=AST_BLOCK};body.as.block.statements=statements;body.as.block.count=1;
 ASTNode fn={.type=AST_FUNCTION};fn.as.function.name="main";fn.as.function.return_type=TYPE_INT;fn.as.function.body=&body;
 ASTNode helper=fn;helper.as.function.name="array_length";
 ASTNode *items[]={&fn,&helper,NULL};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=2;
 CBOptions opts={0};
 ASTNode array={.type=AST_ARRAY_LITERAL};ret.as.return_stmt.value=&array;refuse(&root,argv[1],&opts);
 ASTNode *elements[]={&zero};array.as.array_literal.elements=elements;array.as.array_literal.element_count=1;array.as.array_literal.element_type=TYPE_INT;refuse(&root,argv[1],&opts);ret.as.return_stmt.value=&zero;
 helper.as.function.return_type=TYPE_ARRAY;refuse(&root,argv[1],&opts);helper.as.function.return_type=TYPE_INT;
 Parameter parameter={.name="values",.type=TYPE_ARRAY};helper.as.function.params=&parameter;helper.as.function.param_count=1;refuse(&root,argv[1],&opts);
 helper.as.function.is_extern=true;refuse(&root,argv[1],&opts);helper.as.function.is_extern=false;helper.as.function.param_count=0;
 ASTNode local={.type=AST_LET};local.as.let.name="values";local.as.let.var_type=TYPE_ARRAY;local.as.let.value=&array;
 statements[0]=&local;refuse(&root,argv[1],&opts);statements[0]=&ret;
 items[2]=&local;root.as.program.count=3;refuse(&root,argv[1],&opts);root.as.program.count=2;
 TypeInfo inner={.base_type=TYPE_ARRAY};TypeInfo *params[]={&inner};TypeInfo outer={.base_type=TYPE_STRUCT,.type_params=params,.type_param_count=1};
 helper.as.function.return_type_info=&outer;refuse(&root,argv[1],&opts);helper.as.function.return_type_info=NULL;
 Type args[]={TYPE_ARRAY};FunctionSignature sig={.param_types=args,.param_count=1,.return_type=TYPE_INT};
 parameter.type=TYPE_FUNCTION;parameter.fn_sig=&sig;helper.as.function.param_count=1;refuse(&root,argv[1],&opts);helper.as.function.param_count=0;
 ASTNode call={.type=AST_CALL};call.as.call.name="array_length";call.as.call.checked_signature=&sig;ret.as.return_stmt.value=&call;refuse(&root,argv[1],&opts);
 sig.param_count=0;sig.return_type=TYPE_ARRAY;refuse(&root,argv[1],&opts);call.as.call.checked_signature=NULL;ret.as.return_stmt.value=&zero;
 Type fields[]={TYPE_ARRAY};char *names[]={"values"};ASTNode record={.type=AST_STRUCT_DEF};record.as.struct_def.name="Holder";record.as.struct_def.field_count=1;record.as.struct_def.field_types=fields;record.as.struct_def.field_names=names;
 items[2]=&record;root.as.program.count=3;refuse(&root,argv[1],&opts);
 int counts[]={1};Type *variant_types[]={fields};char **variant_names[]={names};char *variants[]={"Some"};ASTNode un={.type=AST_UNION_DEF};un.as.union_def.name="Choice";un.as.union_def.variant_count=1;un.as.union_def.variant_field_counts=counts;un.as.union_def.variant_field_types=variant_types;un.as.union_def.variant_field_names=variant_names;un.as.union_def.variant_names=variants;
 items[2]=&un;refuse(&root,argv[1],&opts);root.as.program.count=2;
 const char *builtins[]={"array_new","array_length","array_get","array_set","array_slice","map","filter","reduce","str_split","str_join","map_keys","file_read_bytes","bytes_from_string"};
 items[1]=NULL;root.as.program.count=1;ret.as.return_stmt.value=&call;
 for(size_t i=0;i<sizeof builtins/sizeof builtins[0];i++){call.as.call.name=(char*)builtins[i];refuse(&root,argv[1],&opts);}
 /* Local value facts cannot silently become integer arguments/operations. */
 CBCtx c={0};c.root=&root;c.out=tmpfile();assert(c.out);ctx_add_sym(&c,"values",TYPE_ARRAY);
 ASTNode id={.type=AST_IDENTIFIER};id.as.identifier="values";assert(emit_expr(&c,&id)!=0);assert(c.error);assert(!fclose(c.out));
 /* Exact emitted qualified declaration identity wins over builtin spelling. */
 helper.as.function.name="mod_array_length";items[1]=&helper;root.as.program.count=2;
 ASTNode qualified={.type=AST_MODULE_QUALIFIED_CALL};qualified.as.module_qualified_call.module_alias="mod";qualified.as.module_qualified_call.function_name="array_length";
 ret.as.return_stmt.value=&qualified;helper.as.function.body=NULL;
 ASTNode helper_ret={.type=AST_RETURN};helper_ret.as.return_stmt.value=&zero;ASTNode *helper_items[]={&helper_ret};ASTNode helper_body={.type=AST_BLOCK};helper_body.as.block.statements=helper_items;helper_body.as.block.count=1;helper.as.function.body=&helper_body;
 if(!strcmp(argv[2],"direct")){helper.as.function.name="array_length";call.as.call.name="array_length";ret.as.return_stmt.value=&call;}
 assert(c_backend_emit(&root,argv[1],"ordinary.nano",&opts)==0);
 FILE *stream=tmpfile();assert(stream);assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&opts)==0);assert(!fclose(stream));
 return 0;
}
