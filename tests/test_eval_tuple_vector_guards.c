/* I distinguish a deliberate invariant refusal from a crash or child access. */
#include <stdbool.h>
#include <stdlib.h>
static bool deny_tuple_malloc;
static size_t tuple_allocation_failures,tuple_malloc_calls;
void *nano_tuple_test_malloc(size_t bytes) {
    tuple_malloc_calls++;
    if (deny_tuple_malloc) { deny_tuple_malloc=false;tuple_allocation_failures++;return NULL; }
    return malloc(bytes);
}
#define main original_eval_suite_main
#include "test_eval.c"
#undef main
#include "../src/eval/carrier_runtime.h"
static Environment *refusal_env;
static int refusal_marker,refusal_case;
static void confirm_refusal(void) {
    ASSERT_EQ(env_carrier_stats(refusal_env).registered,0);
    ASSERT_EQ(tuple_malloc_calls,refusal_case==3 ? 1 : 0);
    ASSERT_EQ(tuple_allocation_failures,refusal_case==3 ? 1 : 0);
    ASSERT(!deny_tuple_malloc);
    ASSERT_EQ(write(refusal_marker,"R",1),1);
}
static void refuse(int mode,int which) {
    int marker[2];ASSERT_EQ(pipe(marker),0);pid_t child=fork();ASSERT(child>=0);
    if(!child) {
        close(marker[0]);refusal_marker=marker[1];refusal_case=which;
        tuple_malloc_calls=0;tuple_allocation_failures=0;deny_tuple_malloc=false;
        refusal_env=create_environment();ASSERT(refusal_env);
        if(mode)ASSERT(env_carrier_enable(refusal_env));
        ASSERT_EQ(atexit(confirm_refusal),0);
        ASTNode tuple={0},first={0};tuple.type=AST_TUPLE_LITERAL;first.type=AST_NUMBER;first.as.number=7;
        ASTNode *children[]={&first,NULL};
        if(which==0){tuple.as.tuple_literal.element_count=-1;tuple.as.tuple_literal.elements=(ASTNode **)(uintptr_t)1;}
        if(which==1){tuple.as.tuple_literal.element_count=1;tuple.as.tuple_literal.elements=NULL;}
        if(which==2){tuple.as.tuple_literal.element_count=2;tuple.as.tuple_literal.elements=children;}
        if(which==3){tuple.as.tuple_literal.element_count=1;tuple.as.tuple_literal.elements=children;deny_tuple_malloc=true;}
        (void)repl_eval_node(&tuple,refusal_env);_exit(99);
    }
    close(marker[1]);int status=0;ASSERT_EQ(waitpid(child,&status,0),child);
    ASSERT(WIFEXITED(status));ASSERT_EQ(WEXITSTATUS(status),1);
    char seen=0;ASSERT_EQ(read(marker[0],&seen,1),1);ASSERT_EQ(seen,'R');close(marker[0]);
}
int main(void) {
    for (int mode=0;mode<2;mode++) {
        for(int which=0;which<4;which++)refuse(mode,which);
        Environment *env=create_environment();ASSERT(env);if(mode)ASSERT(env_carrier_enable(env));
        ASTNode tuple={0},first={0},second={0};tuple.type=AST_TUPLE_LITERAL;
        first.type=AST_NUMBER;first.as.number=7;second.type=AST_NUMBER;second.as.number=11;
        ASTNode *children[]={&first,&second};tuple.as.tuple_literal.elements=children;tuple.as.tuple_literal.element_count=2;
        Value value=repl_eval_node(&tuple,env);ASSERT_EQ(value.type,VAL_TUPLE);
        ASSERT_EQ(value.as.tuple_val->element_count,2);ASSERT_EQ(value.as.tuple_val->elements[0].as.int_val,7);ASSERT_EQ(value.as.tuple_val->elements[1].as.int_val,11);
        if(!mode){free(value.as.tuple_val->elements);free(value.as.tuple_val);}
        tuple.as.tuple_literal.element_count=0;tuple.as.tuple_literal.elements=NULL;
        value=repl_eval_node(&tuple,env);ASSERT_EQ(value.type,VAL_TUPLE);ASSERT_EQ(value.as.tuple_val->element_count,0);
        if(!mode){free(value.as.tuple_val->elements);free(value.as.tuple_val);}
        free_environment(env);ASSERT_EQ(env_carrier_last_retired_stats().live,0);
    }
    puts("I verify eight deliberate tuple refusals before allocation/ownership, preserve valid/empty ordinary/private tuples, and retire successful ownership.");return 0;
}
