/* I test literal ownership and every explicit allocation refusal in staging. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static size_t allocations,fail_at,live;
static void *fixture_malloc(size_t size){++allocations;if(fail_at==allocations)return NULL;void *p=malloc(size);if(p)++live;return p;}
static void fixture_free(void *p){if(p){assert(live);--live;}free(p);}
#define malloc fixture_malloc
#define free fixture_free
#include "../src/c_backend.c"
#undef free
#undef malloc
int *passive_binding_order(const ASTNode *block){(void)block;assert(!"I reached an unrelated passive fixture.");return NULL;}
static void previous(const char *path){FILE *f=fopen(path,"w");assert(f);assert(fputs("previous",f)>=0);assert(fclose(f)==0);}
static void retained(const char *path){FILE *f=fopen(path,"r");char text[16]={0};assert(f);assert(fread(text,1,sizeof text,f)==8);assert(strcmp(text,"previous")==0);assert(fclose(f)==0);}
int main(int argc,char **argv){
 assert(argc==2);
 char *decoded=nl_decode_string_literal("tail\\");assert(decoded&&strcmp(decoded,"tail\\")==0);fixture_free(decoded);
 decoded=nl_decode_string_literal("a\\0b");const char bytes[]={'a',0,'b',0};assert(decoded&&memcmp(decoded,bytes,sizeof bytes)==0);fixture_free(decoded);
 decoded=nl_decode_string_literal("\\q\\x41");assert(decoded&&strcmp(decoded,"\\q\\x41")==0);fixture_free(decoded);assert(live==0);
 allocations=0;fail_at=1;assert(nl_decode_string_literal("ordinary")==NULL);fail_at=0;assert(live==0);
 ASTNode string={.type=AST_STRING};string.as.string_val="tab\\t8";
 ASTNode local={.type=AST_LET};local.as.let.name="value";local.as.let.var_type=TYPE_STRING;local.as.let.value=&string;
 ASTNode zero={.type=AST_NUMBER};ASTNode result={.type=AST_RETURN};result.as.return_stmt.value=&zero;
 ASTNode *statements[]={&local,&result};ASTNode block={.type=AST_BLOCK};block.as.block.statements=statements;block.as.block.count=2;
 ASTNode function={.type=AST_FUNCTION};function.as.function.name="main";function.as.function.return_type=TYPE_INT;function.as.function.body=&block;
 ASTNode *items[]={&function};ASTNode root={.type=AST_PROGRAM};root.as.program.items=items;root.as.program.count=1;CBOptions options={0};
 allocations=0;assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)==0);size_t count=allocations;assert(count>=4&&live==0);
 for(size_t i=1;i<=count;++i){previous(argv[1]);allocations=0;fail_at=i;assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)!=0);fail_at=0;assert(live==0);retained(argv[1]);assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)==0);assert(live==0);}
 FILE *stream=tmpfile();assert(stream);assert(fputs("previous",stream)>=0);allocations=0;fail_at=1;
 assert(c_backend_emit_fp(&root,stream,"ordinary.nano",&options)!=0);fail_at=0;assert(ftell(stream)==8&&live==0);assert(fclose(stream)==0);
 CBCtx context={0};context.out=tmpfile();assert(context.out);allocations=0;fail_at=1;
 assert(emit_expr(&context,&string)!=0);assert(strcmp(context.error,"I could not decode my C string literal.")==0);ctx_error(&context,"later");assert(strcmp(context.error,"I could not decode my C string literal.")==0);fail_at=0;assert(fclose(context.out)==0&&live==0);
 string.as.string_val=NULL;assert(c_backend_emit(&root,argv[1],"ordinary.nano",&options)==0);assert(live==0);
 return 0;
}
