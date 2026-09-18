/* I refuse unsupported type facts before either publication API changes bytes. */
#define main public_c_options_fixture_main
#include "test_public_c_profile_options_api.c"
#undef main
int main(int argc,char **argv){
 assert(argc==3);CBOptions opts={0};
 ASTNode zero={.type=AST_NUMBER},ret={.type=AST_RETURN};ret.as.return_stmt.value=&zero;
 ASTNode *statements[]={&ret,&ret};ASTNode body={.type=AST_BLOCK};body.as.block.statements=statements;body.as.block.count=1;
 ASTNode main_fn={.type=AST_FUNCTION};main_fn.as.function.name="main";main_fn.as.function.return_type=TYPE_INT;main_fn.as.function.body=&body;
 ASTNode helper=main_fn;helper.as.function.name="helper";
 ASTNode *items[]={&main_fn,&helper,NULL,NULL};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=2;
 ASTNode local={.type=AST_LET};local.as.let.name="value";local.as.let.value=&zero;
 Parameter param={.name="argument",.type=TYPE_INT};
 Type fields[]={TYPE_INT};char *names[]={"field"};char *owners[]={NULL};
 ASTNode record={.type=AST_STRUCT_DEF};record.as.struct_def.name="Record";record.as.struct_def.field_count=1;record.as.struct_def.field_names=names;record.as.struct_def.field_types=fields;record.as.struct_def.field_type_names=owners;
 int counts[]={1};Type *variant_types[]={fields};char **variant_fields[]={names};char *variants[]={"Some"};
 ASTNode un={.type=AST_UNION_DEF};un.as.union_def.name="Choice";un.as.union_def.variant_count=1;un.as.union_def.variant_names=variants;un.as.union_def.variant_field_counts=counts;un.as.union_def.variant_field_types=variant_types;un.as.union_def.variant_field_names=variant_fields;
 /* I enumerate every currently declared kind; nominal and scalar positives follow. */
 for(int value=TYPE_INT;value<=TYPE_BORROW_MUT;value++){
  Type type=(Type)value;
  if(type==TYPE_INT||type==TYPE_U8||type==TYPE_FLOAT||type==TYPE_BOOL||type==TYPE_STRING||type==TYPE_ENUM||type==TYPE_STRUCT||type==TYPE_UNION)continue;
  local.as.let.var_type=type;statements[0]=&local;refuse(&root,argv[1],&opts);statements[0]=&ret;
  items[2]=&local;root.as.program.count=3;refuse(&root,argv[1],&opts);root.as.program.count=2;
  param.type=type;helper.as.function.params=&param;helper.as.function.param_count=1;refuse(&root,argv[1],&opts);helper.as.function.is_extern=true;refuse(&root,argv[1],&opts);helper.as.function.is_extern=false;helper.as.function.param_count=0;
  if(type!=TYPE_VOID){helper.as.function.return_type=type;refuse(&root,argv[1],&opts);helper.as.function.return_type=TYPE_INT;}
  fields[0]=type;items[2]=&record;root.as.program.count=3;refuse(&root,argv[1],&opts);items[2]=&un;refuse(&root,argv[1],&opts);root.as.program.count=2;
  if(type!=TYPE_VOID){TypeInfo inner={.base_type=type};TypeInfo *parts[]={&inner};TypeInfo outer={.base_type=TYPE_INT,.type_params=parts,.type_param_count=1};helper.as.function.return_type_info=&outer;refuse(&root,argv[1],&opts);helper.as.function.return_type_info=NULL;}
 }
 fields[0]=TYPE_INT;
 /* Missing/wrong nominal owner and declaration order never fabricate int64 storage. */
 for(int value=TYPE_STRUCT;value<=TYPE_UNION;value++){
  if(value==TYPE_ENUM)continue;
  local.as.let.var_type=(Type)value;local.as.let.type_name=NULL;statements[0]=&local;refuse(&root,argv[1],&opts);local.as.let.type_name="Missing";refuse(&root,argv[1],&opts);statements[0]=&ret;
  helper.as.function.return_type=(Type)value;helper.as.function.return_struct_type_name="Missing";refuse(&root,argv[1],&opts);helper.as.function.return_type=TYPE_INT;helper.as.function.return_struct_type_name=NULL;
 }
 Type earlier_fields[]={TYPE_INT};char *earlier_names[]={"number"};
 ASTNode earlier=record;earlier.as.struct_def.name="Earlier";earlier.as.struct_def.field_types=earlier_fields;earlier.as.struct_def.field_names=earlier_names;earlier.as.struct_def.field_type_names=NULL;
 fields[0]=TYPE_STRUCT;owners[0]="Earlier";
 items[0]=&record;items[1]=&earlier;items[2]=&main_fn;root.as.program.count=3;refuse(&root,argv[1],&opts);
 owners[0]="Record";refuse(&root,argv[1],&opts);
 owners[0]="Earlier";earlier_fields[0]=TYPE_STRUCT;char *cycle[]={"Record"};earlier.as.struct_def.field_type_names=cycle;refuse(&root,argv[1],&opts);earlier_fields[0]=TYPE_INT;earlier.as.struct_def.field_type_names=NULL;
 items[0]=&earlier;items[1]=&record;
 earlier.as.struct_def.is_extern=true;refuse(&root,argv[1],&opts);earlier.as.struct_def.is_extern=false;
 earlier.as.struct_def.is_resource=true;refuse(&root,argv[1],&opts);earlier.as.struct_def.is_resource=false;
 earlier.as.struct_def.field_count=0;refuse(&root,argv[1],&opts);earlier.as.struct_def.field_count=1;
 /* Recovery emits a prior-order complete record graph, or existing scalar carriers. */
 if(strcmp(argv[2],"record")){
  Type type=!strcmp(argv[2],"u8")?TYPE_U8:!strcmp(argv[2],"enum")?TYPE_ENUM:TYPE_INT;
  ASTNode argument={.type=AST_IDENTIFIER};argument.as.identifier="argument";
  ASTNode relay_return={.type=AST_RETURN};relay_return.as.return_stmt.value=&argument;
  ASTNode *relay_statements[]={&relay_return};ASTNode relay_body={.type=AST_BLOCK};relay_body.as.block.statements=relay_statements;relay_body.as.block.count=1;
  helper.as.function.body=&relay_body;helper.as.function.return_type=type;helper.as.function.params=&param;helper.as.function.param_count=1;param.type=type;
  items[0]=&main_fn;items[1]=&helper;root.as.program.count=2;
  assert(c_backend_emit(&root,argv[1],"profile.nano",&opts)==0);
  FILE *stream=tmpfile();assert(stream);assert(c_backend_emit_fp(&root,stream,"profile.nano",&opts)==0);assert(!fclose(stream));
 }else{
  assert(c_backend_emit(&root,argv[1],"profile.nano",&opts)==0);
  FILE *stream=tmpfile();assert(stream);assert(c_backend_emit_fp(&root,stream,"profile.nano",&opts)==0);assert(!fclose(stream));
 }
 return 0;
}
