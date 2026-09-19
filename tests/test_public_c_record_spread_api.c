/* I refuse reached spread without publishing either output and then recover. */
#define main public_c_options_fixture_main
#include "test_public_c_profile_options_api.c"
#undef main
int main(int argc,char **argv){
 assert(argc==2);CBOptions opts={0};
 ASTNode zero={.type=AST_NUMBER},ret={.type=AST_RETURN};ret.as.return_stmt.value=&zero;
 ASTNode *statements[]={&ret};ASTNode body={.type=AST_BLOCK};body.as.block.statements=statements;body.as.block.count=1;
 ASTNode fn={.type=AST_FUNCTION};fn.as.function.name="main";fn.as.function.return_type=TYPE_INT;fn.as.function.body=&body;
 Type types[]={TYPE_INT};char *names[]={"value"};
 ASTNode record={.type=AST_STRUCT_DEF};record.as.struct_def.name="Record";record.as.struct_def.field_count=1;record.as.struct_def.field_names=names;record.as.struct_def.field_types=types;
 ASTNode *values[]={&zero};ASTNode base={.type=AST_STRUCT_LITERAL};base.as.struct_literal.struct_name="Record";base.as.struct_literal.field_names=names;base.as.struct_literal.field_values=values;base.as.struct_literal.field_count=1;
 ASTNode spread=base;spread.as.struct_literal.spread_source=&base;
 ASTNode *items[]={&record,&fn};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=2;
 ASTNode local={.type=AST_LET};local.as.let.name="copy";local.as.let.var_type=TYPE_STRUCT;local.as.let.type_name="Record";local.as.let.value=&spread;
 ASTNode *block_values[]={&base};ASTNode lifted_base={.type=AST_BLOCK};lifted_base.as.block.statements=block_values;lifted_base.as.block.count=1;
 ASTNode *override_values[]={&zero};ASTNode lifted_override={.type=AST_BLOCK};lifted_override.as.block.statements=override_values;lifted_override.as.block.count=1;
 ASTNode *nested_values[]={&spread};ASTNode nested=base;nested.as.struct_literal.field_values=nested_values;
 ASTNode *shapes[]={&spread,&nested};
 for(size_t i=0;i<2;i++){statements[0]=shapes[i];refuse(&root,argv[1],&opts);}
 statements[0]=&local;refuse(&root,argv[1],&opts);
 spread.as.struct_literal.spread_source=&lifted_base;refuse(&root,argv[1],&opts);
 spread.as.struct_literal.spread_source=&base;values[0]=&lifted_override;refuse(&root,argv[1],&opts);values[0]=&zero;
 CBCtx context={0};context.root=&root;context.out=tmpfile();assert(context.out);
 (void)cb_info(&context,&spread);assert(context.error && strstr(context.error,"record spread"));assert(ftell(context.out)==0);assert(!fclose(context.out));
 statements[0]=&ret;
 assert(c_backend_emit(&root,argv[1],"ordinary.nano",&opts)==0);
 FILE *stream=tmpfile();assert(stream);assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&opts)==0);assert(!fclose(stream));return 0;
}
