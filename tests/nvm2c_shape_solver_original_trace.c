/* I observe API calls in my unchanged original tests; I implement no graph operation. */
#include "../src/nanoisa/nvm2c_shape.c"
#include <limits.h>

static FILE *trace_file;
static NvmShapeGraph *trace_graph;
static uint32_t trace_cases, trace_ops;
static long trace_header;
static void trace_word(uint32_t v) {
    unsigned char bytes[4]={(unsigned char)v,(unsigned char)(v>>8),(unsigned char)(v>>16),(unsigned char)(v>>24)};
    if(fwrite(bytes,1,4,trace_file)!=4)abort();
}
static void trace_op(NvmShapeGraph *g,uint32_t op,uint32_t a,uint32_t b,uint32_t c,uint64_t expected) {
    if(!trace_graph){trace_graph=g;trace_header=ftell(trace_file);if(trace_header<0)abort();trace_word(1);trace_word(0);trace_ops=0;}
    if(trace_graph!=g || expected>UINT32_MAX || trace_ops==UINT32_MAX)abort();
    trace_word(op);trace_word(a);trace_word(b);trace_word(c);trace_word((uint32_t)expected);trace_word(1);++trace_ops;
}
static void trace_state(NvmShapeGraph *g) {
    trace_op(g,13,0,0,0,!g->error);trace_op(g,22,0,0,0,g->count);
    trace_op(g,23,0,0,0,g->conversion_count);trace_op(g,24,0,0,0,g->array_alias_count);
    trace_op(g,25,0,0,0,g->array_write_count);trace_op(g,26,0,0,0,g->array_read_count);
    trace_op(g,21,0,0,0,0);
}
static NvmShapeId trace_new(NvmShapeGraph *g,NvmShapeKind k){NvmShapeId r=nvm_shape_new(g,k);trace_op(g,0,(uint32_t)k,0,0,r);return r;}
static NvmShapeId trace_root(NvmShapeGraph *g,NvmShapeId a){NvmShapeId r=nvm_shape_root(g,a);trace_op(g,1,a,0,0,r);return r;}
static NvmShapeKind trace_kind(NvmShapeGraph *g,NvmShapeId a){NvmShapeKind r=nvm_shape_kind(g,a);trace_op(g,2,a,0,0,r);return r;}
static NvmShapeId trace_child(NvmShapeGraph *g,NvmShapeId a,uint32_t b){NvmShapeId r=nvm_shape_child(g,a,b);trace_op(g,3,a,b,0,r);return r;}
static NvmShapeId trace_lookup(NvmShapeGraph *g,NvmShapeId a,uint32_t b){NvmShapeId r=nvm_shape_lookup(g,a,b);trace_op(g,4,a,b,0,r);return r;}
static int trace_unify(NvmShapeGraph *g,NvmShapeId a,NvmShapeId b){int r=nvm_shape_unify(g,a,b);trace_op(g,5,a,b,0,r);return r;}
static int trace_tags_add(NvmShapeGraph *g,NvmShapeId a,uint16_t b){int r=nvm_shape_tags_add(g,a,b);trace_op(g,6,a,b,0,r);return r;}
static uint16_t trace_tags(NvmShapeGraph *g,NvmShapeId a){uint16_t r=nvm_shape_tags(g,a);trace_op(g,7,a,0,0,r);return r;}
static int trace_function_add(NvmShapeGraph *g,NvmShapeId a,uint32_t b){int r=nvm_shape_function_add(g,a,b);trace_op(g,8,a,b,0,r);return r;}
static size_t trace_function_count(NvmShapeGraph *g,NvmShapeId a){size_t r=nvm_shape_function_count(g,a);trace_op(g,9,a,0,0,r);return r;}
static int trace_function_target(NvmShapeGraph *g,NvmShapeId a,size_t b,uint32_t *target){
    int r=nvm_shape_function_target(g,a,b,target);
    if(b<=UINT32_MAX)trace_op(g,10,a,(uint32_t)b,target!=NULL,r);
    else {if(!target)abort();trace_op(g,12,a,(uint32_t)b,(uint32_t)((uint64_t)b>>32),r);}
    if(r)trace_op(g,27,a,(uint32_t)b,0,*target);
    return r;
}
static int trace_convert(NvmShapeGraph *g,NvmShapeId a,NvmShapeId b){int r=nvm_shape_convert(g,a,b);trace_op(g,14,a,b,0,r);return r;}
static int trace_checked(NvmShapeGraph *g,NvmShapeId a,NvmShapeId b){int r=nvm_shape_checked_payload(g,a,b);trace_op(g,15,a,b,0,r);return r;}
static int trace_alias(NvmShapeGraph *g,NvmShapeId a,NvmShapeId b){int r=nvm_shape_alias_view(g,a,b);trace_op(g,16,a,b,0,r);return r;}
static int trace_array_alias(NvmShapeGraph *g,NvmShapeId a,NvmShapeId b){int r=nvm_shape_array_alias(g,a,b);trace_op(g,17,a,b,0,r);return r;}
static int trace_write(NvmShapeGraph *g,NvmShapeId a,NvmShapeId b){int r=nvm_shape_array_write(g,a,b);trace_op(g,18,a,b,0,r);return r;}
static int trace_read(NvmShapeGraph *g,NvmShapeId a,NvmShapeId b){int r=nvm_shape_array_read(g,a,b);trace_op(g,19,a,b,0,r);return r;}
static int trace_solve(NvmShapeGraph *g){int r=nvm_shape_solve_conversions(g);trace_op(g,20,0,0,0,r);trace_state(g);return r;}
static void trace_destroy(NvmShapeGraph *g){
    trace_state(g);nvm_shape_destroy(g);trace_op(g,11,0,0,0,0);trace_state(g);
    long end=ftell(trace_file);if(end<0 || fseek(trace_file,trace_header+4,SEEK_SET))abort();trace_word(trace_ops);
    if(fseek(trace_file,end,SEEK_SET)){abort();}
    ++trace_cases;trace_graph=NULL;
}
#define nvm_shape_new trace_new
#define nvm_shape_root trace_root
#define nvm_shape_kind trace_kind
#define nvm_shape_child trace_child
#define nvm_shape_lookup trace_lookup
#define nvm_shape_unify trace_unify
#define nvm_shape_tags_add trace_tags_add
#define nvm_shape_tags trace_tags
#define nvm_shape_function_add trace_function_add
#define nvm_shape_function_count trace_function_count
#define nvm_shape_function_target trace_function_target
#define nvm_shape_convert trace_convert
#define nvm_shape_checked_payload trace_checked
#define nvm_shape_alias_view trace_alias
#define nvm_shape_array_alias trace_array_alias
#define nvm_shape_array_write trace_write
#define nvm_shape_array_read trace_read
#define nvm_shape_solve_conversions trace_solve
#define nvm_shape_destroy trace_destroy
#define main original_main
#include "nanoisa/test_nvm2c_shape.c"
#undef main
int main(int argc,char **argv){
    if(argc!=2)return 2;
    trace_file=fopen(argv[1],"wb+");if(!trace_file)return 2;trace_word(0);
    int result=original_main();
    if(trace_graph || fseek(trace_file,0,SEEK_SET)){abort();}
    trace_word(trace_cases);
    if(fclose(trace_file))return 2;
    fprintf(stderr,"I recorded %u original graph programs.\n",trace_cases);
    return result;
}
