/* I refuse unresolved/mixed string equality before publication, then recover. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/c_backend.c"
int *passive_binding_order(const ASTNode *block){(void)block;assert(!"I reached an unrelated passive fixture.");return NULL;}
static void save(const char *path){FILE *f=fopen(path,"w");assert(f);assert(fputs("previous",f)>=0);assert(fclose(f)==0);}
static void retained(const char *path){FILE *f=fopen(path,"r");char text[16]={0};assert(f);assert(fread(text,1,sizeof text,f)==8);assert(strcmp(text,"previous")==0);assert(fclose(f)==0);}
int main(int argc,char **argv){
 assert(argc==2);
 ASTNode string={.type=AST_STRING};string.as.string_val="exact";
 ASTNode unknown={.type=AST_IDENTIFIER};unknown.as.identifier="unknown";
 ASTNode integer={.type=AST_NUMBER};integer.as.number=1;
 ASTNode *args[]={&string,&unknown};ASTNode comparison={.type=AST_PREFIX_OP};comparison.as.prefix_op.op=TOKEN_EQ;comparison.as.prefix_op.args=args;comparison.as.prefix_op.arg_count=2;
 ASTNode assertion={.type=AST_ASSERT};assertion.as.assert.condition=&comparison;
 ASTNode zero={.type=AST_NUMBER};ASTNode result={.type=AST_RETURN};result.as.return_stmt.value=&zero;
 ASTNode *statements[]={&assertion,&result};ASTNode block={.type=AST_BLOCK};block.as.block.statements=statements;block.as.block.count=2;
 ASTNode function={.type=AST_FUNCTION};function.as.function.name="main";function.as.function.return_type=TYPE_INT;function.as.function.body=&block;
 ASTNode *items[]={&function};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=1;CBOptions options={0};
 for(int op=0;op<2;++op){comparison.as.prefix_op.op=op?TOKEN_NE:TOKEN_EQ;
  for(int side=0;side<2;++side){for(int kind=0;kind<2;++kind){
   args[side]=&string;args[1-side]=kind?&integer:&unknown;
   save(argv[1]);assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)!=0);retained(argv[1]);
   FILE *stream=tmpfile();assert(stream);assert(fputs("previous",stream)>=0);
   assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&options)!=0);assert(ftell(stream)==8);assert(fclose(stream)==0);
  }}
 }
 args[0]=&string;args[1]=&string;comparison.as.prefix_op.op=TOKEN_EQ;
 assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)==0);
 FILE *f=fopen(argv[1],"r");assert(f);char text[32768]={0};assert(fread(text,1,sizeof text-1,f)>0);assert(fclose(f)==0);
 assert(strstr(text,"const char *nano_cb_0_sl[1], *nano_cb_0_sr[1]"));
 return 0;
}
