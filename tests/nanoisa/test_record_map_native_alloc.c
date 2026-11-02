#define MAP_ORIGIN_MAIN native_retained_origin_controls
#include "test_record_map_origins.c"
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#include "../../src/nanoisa/nvm2c.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
int g_argc;char **g_argv;
int main(int argc,char **argv) {
    (void)argv;setvbuf(stdout,NULL,_IONBF,0);Input c;init(&c,TAG_INT,false);
    map_new(&c,TAG_STRING,TAG_INT);scalar(&c,TAG_STRING);op(&c,OP_PUSH_I64);
    for(unsigned i=0;i<8;i++)b8(c.code,&c.n,i?0:7);
    op(&c,OP_HM_SET);op(&c,OP_AGG_PACK);b8(c.code,&c.n,AGG_RECORD);b32(c.code,&c.n,0);b16(c.code,&c.n,0);b16(c.code,&c.n,1);
    op(&c,OP_POP);map_finish(&c);
    Input saved=c;char error[512]={0};ra_calls=0;
    char *prior=nvm2c_emit_record_map_private(&c.m,error,sizeof error);
    CHECK(prior);size_t calls=ra_calls,live=ra_live,bytes=ra_bytes;
    printf("I reached %zu compiler allocation positions, retained %zu objects/%zu bytes.\n",calls,live,bytes);
    CHECK(calls&&live&&bytes);
    if(argc>1)for(unsigned persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<calls;fail++) {
        ra_calls=0;ra_fail=fail;ra_persistent=(int)persistent;error[0]=0;
        char *output=nvm2c_emit_record_map_private(&c.m,error,sizeof error);
        ra_fail=SIZE_MAX;ra_persistent=0;
        if(output){CHECK(!strcmp(output,prior));ra_test_free(output);}else CHECK(error[0]);
        CHECK(!memcmp(&c,&saved,sizeof c));CHECK(ra_live==live&&ra_bytes==bytes);
        output=nvm2c_emit_record_map_private(&c.m,error,sizeof error);CHECK(output&&!strcmp(output,prior));ra_test_free(output);
        CHECK(ra_live==live&&ra_bytes==bytes);
    }
    ra_test_free(prior);CHECK(!ra_live&&!ra_bytes);
    printf("PASS %u compiler allocation checks; %zu positions %s.\n",checks,calls,argc>1?"transient/persistent plus retry":"measured only");return 0;
}
