/* I qualify hosted/library options and semantic refusal before publication. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/c_backend.c"
int *passive_binding_order(const ASTNode *block){(void)block;assert(!"I reached unrelated passive code.");return NULL;}
static void refuse(ASTNode *root,const char *path,const CBOptions *opts){
 FILE *f=fopen(path,"w");assert(f);assert(fputs("previous",f)>=0);assert(!fclose(f));
 assert(c_backend_emit(root,path,"profile.nano",opts)!=0);
 f=fopen(path,"r");char bytes[16]={0};assert(f);assert(fread(bytes,1,sizeof bytes,f)==8);assert(!strcmp(bytes,"previous"));assert(!fclose(f));
 f=tmpfile();assert(f);assert(fputs("previous",f)>=0);assert(c_backend_emit_fp(root,f,"profile.nano",opts)!=0);assert(ftell(f)==8);rewind(f);memset(bytes,0,sizeof bytes);assert(fread(bytes,1,sizeof bytes,f)==8);assert(!strcmp(bytes,"previous"));assert(!fclose(f));
}
int main(int argc,char **argv){
 assert(argc==3);int mode=atoi(argv[2]);
 ASTNode number={.type=AST_NUMBER};number.as.number=42;
 ASTNode ret={.type=AST_RETURN};ret.as.return_stmt.value=&number;
 ASTNode *body_items[]={&ret};ASTNode body={.type=AST_BLOCK};body.as.block.statements=body_items;body.as.block.count=1;
 ASTNode main_fn={.type=AST_FUNCTION};main_fn.as.function.name="main";main_fn.as.function.return_type=TYPE_INT;main_fn.as.function.body=&body;
 ASTNode call={.type=AST_CALL};call.as.call.name="main";
 ASTNode probe_ret={.type=AST_RETURN};probe_ret.as.return_stmt.value=&call;
 ASTNode *probe_items[]={&probe_ret};ASTNode probe_body={.type=AST_BLOCK};probe_body.as.block.statements=probe_items;probe_body.as.block.count=1;
 ASTNode probe={.type=AST_FUNCTION};probe.as.function.name="probe";probe.as.function.return_type=TYPE_INT;probe.as.function.body=&probe_body;
 ASTNode *items[]={&main_fn,&probe,NULL};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=2;
 CBOptions opts={0};
 /* Every refused API invocation must recover without retained option/error state. */
 opts.no_stdlib=true;refuse(&root,argv[1],&opts);opts.no_stdlib=false;
 ASTNode global={.type=AST_LET};global.as.let.name="value";global.as.let.var_type=TYPE_INT;global.as.let.value=&number;
 items[2]=&global;root.as.program.count=3;opts.no_main=true;refuse(&root,argv[1],&opts);opts.no_main=false;root.as.program.count=2;
 ASTNodeType kinds[]={AST_TUPLE_LITERAL,AST_TUPLE_INDEX,AST_EFFECT_DECL,AST_HANDLE_EXPR,AST_EFFECT_HANDLER,AST_EFFECT_OP,AST_ASYNC_FN,AST_AWAIT,AST_TRY_OP};
 for(size_t i=0;i<sizeof kinds/sizeof kinds[0];i++){
  ASTNode unsupported={.type=kinds[i]};
  items[2]=&unsupported;root.as.program.count=3;refuse(&root,argv[1],&opts);root.as.program.count=2;
  body_items[0]=&unsupported;refuse(&root,argv[1],&opts);body_items[0]=&ret;
  ret.as.return_stmt.value=&unsupported;refuse(&root,argv[1],&opts);ret.as.return_stmt.value=&number;
 }
 main_fn.as.function.is_anonymous=true;refuse(&root,argv[1],&opts);main_fn.as.function.is_anonymous=false;
 ASTNode reference={.type=AST_IDENTIFIER};reference.as.identifier="ordinary";reference.lambda_definition=&main_fn;
 ret.as.return_stmt.value=&reference;refuse(&root,argv[1],&opts);ret.as.return_stmt.value=&number;
 opts.no_main=mode>=3;opts.static_strings=mode==2 || mode==6;opts.verbose=mode==1;
 if(mode==4){items[0]=&probe;root.as.program.count=1;probe_ret.as.return_stmt.value=&number;}
 if(mode==5)probe_ret.as.return_stmt.value=&number;
 ASTNode literal={.type=AST_STRING};literal.as.string_val="stable";
 ASTNode text_return={.type=AST_RETURN};text_return.as.return_stmt.value=&literal;
 ASTNode *text_items[]={&text_return};ASTNode text_body={.type=AST_BLOCK};text_body.as.block.statements=text_items;text_body.as.block.count=1;
 ASTNode text_function={.type=AST_FUNCTION};text_function.as.function.name="literal_text";text_function.as.function.return_type=TYPE_STRING;text_function.as.function.body=&text_body;
 items[root.as.program.count++]=&text_function;
 assert(c_backend_emit(&root,argv[1],"profile.nano",mode==0?NULL:&opts)==0);
 FILE *out=tmpfile();assert(out);assert(c_backend_emit_fp(&root,out,"profile.nano",mode==0?NULL:&opts)==0);assert(ftell(out)>0);assert(!fclose(out));
 return 0;
}
