/* I distinguish continuing values from enclosing exits before publishing C. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/c_backend.c"
int *passive_binding_order(const ASTNode *block){(void)block;assert(!"I reached unrelated passive code.");return NULL;}
static void retained(const char *path){FILE *f=fopen(path,"r");char text[16]={0};assert(f);assert(fread(text,1,sizeof text,f)==8);assert(!strcmp(text,"previous"));assert(!fclose(f));}
int main(int argc,char **argv){
 assert(argc==2);
 ASTNode zero={.type=AST_NUMBER}, one={.type=AST_NUMBER};one.as.number=1;
 ASTNode boolean={.type=AST_BOOL};boolean.as.bool_val=true;
 ASTNode *oneitems[]={&one};ASTNode oneblock={.type=AST_BLOCK};oneblock.as.block.count=1;oneblock.as.block.statements=oneitems;
 ASTNode *boolitems[]={&boolean};ASTNode boolblock={.type=AST_BLOCK};boolblock.as.block.count=1;boolblock.as.block.statements=boolitems;
 ASTNode early={.type=AST_RETURN};early.as.return_stmt.value=&zero;
 ASTNode *exititems[]={&early};ASTNode exitblock={.type=AST_BLOCK};exitblock.as.block.count=1;exitblock.as.block.statements=exititems;
 ASTNode absent={.type=AST_LET};absent.as.let.name="absent";absent.as.let.var_type=TYPE_INT;absent.as.let.value=&exitblock;
 ASTNode use={.type=AST_IDENTIFIER};use.as.identifier="absent";
 ASTNode *nesteditems[]={&absent,&use};ASTNode nested={.type=AST_BLOCK};nested.as.block.count=2;nested.as.block.statements=nesteditems;
 ASTNode local={.type=AST_LET};local.as.let.name="result";local.as.let.var_type=TYPE_INT;local.as.let.value=&nested;
 ASTNode final={.type=AST_RETURN};final.as.return_stmt.value=&zero;
 ASTNode *statements[]={&local,&final};ASTNode body={.type=AST_BLOCK};body.as.block.count=2;body.as.block.statements=statements;
 ASTNode function={.type=AST_FUNCTION};function.as.function.name="main";function.as.function.return_type=TYPE_INT;function.as.function.body=&body;
 ASTNode *items[]={&function};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=1;
 ASTNode empty={.type=AST_BLOCK};ASTNode unknown={.type=AST_IDENTIFIER};unknown.as.identifier="unresolved";
 ASTNode *unknownitems[]={&unknown};ASTNode unknownblock={.type=AST_BLOCK};unknownblock.as.block.count=1;unknownblock.as.block.statements=unknownitems;
 ASTNode branch={.type=AST_IF};branch.as.if_stmt.condition=&boolean;branch.as.if_stmt.then_branch=&oneblock;branch.as.if_stmt.else_branch=&boolblock;
 ASTNode loop={.type=AST_WHILE};loop.as.while_stmt.condition=&boolblock;loop.as.while_stmt.body=&empty;
 CBOptions options={0};
 for(int mode=0;mode<4;mode++){
  statements[0]=&local;local.as.let.value=&empty;
  if(mode==1)local.as.let.value=&unknownblock;
  if(mode==2)local.as.let.value=&branch;
  if(mode==3)statements[0]=&loop;
  FILE *f=fopen(argv[1],"w");assert(f);assert(fputs("previous",f)>=0);assert(!fclose(f));
  assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)!=0);retained(argv[1]);
  FILE *stream=tmpfile();assert(stream);assert(fputs("previous",stream)>=0);
  assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&options)!=0);assert(ftell(stream)==8);assert(!fclose(stream));
  CBCtx c={0};c.root=&root;c.out=tmpfile();assert(c.out);
  assert(emit_stmt(&c,statements[0])!=0);assert(c.error);assert(!c.lift_temps);
  const char *first=c.error;ctx_error(&c,"I retain my first diagnostic.");assert(first==c.error);assert(!fclose(c.out));
 }
 statements[0]=&local;local.as.let.value=&nested;
 assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)==0);
 FILE *f=fopen(argv[1],"r");assert(f);char text[32768]={0};assert(fread(text,1,sizeof text-1,f)>0);assert(!fclose(f));
 assert(!strstr(text,"absent"));assert(!strstr(text,"({"));
 /* A control-only selected branch does not erase the other continuing branch. */
 branch.as.if_stmt.then_branch=&exitblock;branch.as.if_stmt.else_branch=&oneblock;
 CBCtx c={0};c.root=&root;c.out=tmpfile();assert(c.out);
 CBValueInfo info=cb_info(&c,&branch);assert(info.continues && info.type==TYPE_INT && !c.error);
 assert(!fclose(c.out));
 return 0;
}
