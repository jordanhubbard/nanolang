/* I preserve borrowed callers when either library scratch allocation fails. */
#include "nanolang.h"
#include "eval/eval_string.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc;
char **g_argv;
static bool fail_next;
static unsigned failures;
void *nano_string_buffer_malloc(size_t size) {
    if (fail_next) { fail_next=false;failures++;return NULL; }
    return malloc(size);
}
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail STRING buffer check %s at line %d.\n",#c,__LINE__);exit(1); } } while(0)
int main(void) {
    char left[]="left",right[]="right";
    Value args[3]={create_void(),create_void(),create_void()};
    args[0].type=VAL_STRING;args[0].as.string_val=left;
    args[1].type=VAL_STRING;args[1].as.string_val=right;
    fail_next=true;Value value=builtin_str_concat(args);CHECK(value.type==VAL_VOID);CHECK(!fail_next);
    CHECK(args[0].as.string_val==left && args[1].as.string_val==right);
    value=builtin_str_concat(args);CHECK(value.type==VAL_STRING);CHECK(strcmp(value.as.string_val,"leftright")==0);gc_release(value.as.string_val);
    args[1]=create_int(1);args[2]=create_int(2);
    fail_next=true;value=builtin_str_substring(args);CHECK(value.type==VAL_STRING);CHECK(strcmp(value.as.string_val,"")==0);CHECK(!fail_next);gc_release(value.as.string_val);
    value=builtin_str_substring(args);CHECK(value.type==VAL_STRING);CHECK(strcmp(value.as.string_val,"ef")==0);gc_release(value.as.string_val);
    CHECK(strcmp(left,"left")==0 && strcmp(right,"right")==0);CHECK(failures==2);
    puts("I preserve borrowed caller buffers across two scratch allocation failures and both successful recovery calls.");
    return 0;
}
