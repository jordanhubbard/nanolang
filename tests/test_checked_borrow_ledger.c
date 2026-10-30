/* I exercise the actual scoped provider, including injected allocation failures. */
#include "nanolang.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned alloc_calls,fail_at,denied,live;
static bool deny(void){alloc_calls++;if(fail_at && alloc_calls==fail_at){denied++;return true;}return false;}
static void *probe_malloc(size_t n){if(deny())return NULL;void *p=malloc(n);if(p)live++;return p;}
static void *probe_calloc(size_t n,size_t z){if(deny())return NULL;void *p=calloc(n,z);if(p)live++;return p;}
static void *probe_realloc(void *p,size_t n){if(deny())return NULL;bool fresh=!p;void *q=realloc(p,n);if(q && fresh)live++;return q;}
static void probe_free(void *p){if(p){assert(live);live--;free(p);}}
#define malloc probe_malloc
#define calloc probe_calloc
#define realloc probe_realloc
#define free probe_free
#include "../src/checked_borrow_facts.c"
#undef malloc
#undef calloc
#undef realloc
#undef free

typedef struct {Environment env;ASTNode fn,call,arg,field,root;ASTNode *args[1],*place[1];} Fixture;
static void init(Fixture *f) {
    memset(f,0,sizeof *f);f->env.current_file="module.nano";
    f->fn.type=AST_FUNCTION;f->fn.as.function.affine_check_epoch=1;
    f->fn.as.function.affine_completed_epoch=1;f->fn.as.function.checked_affine=1;
    f->root.type=AST_IDENTIFIER;f->root.as.identifier="owner";
    f->field.line=3;f->field.column=4;f->field.type=AST_FIELD_ACCESS;f->field.as.field_access.object=&f->root;f->field.as.field_access.field_name="leaf";
    f->place[0]=&f->field;f->arg.type=AST_CALL;f->arg.line=2;f->arg.column=8;
    f->arg.as.call.args=f->place;f->arg.as.call.arg_count=1;f->arg.as.call.borrow_mode=2;
    f->args[0]=&f->arg;f->call.type=AST_CALL;f->call.as.call.name="helper";f->call.line=2;f->call.column=2;
    f->call.as.call.args=f->args;f->call.as.call.arg_count=1;
}
static bool capture(Fixture *f){return checked_borrow_capture(&f->env,&f->fn,&f->call,&f->arg,0,"owner",".leaf");}
static const CheckedBorrowEvent *lookup(Fixture *f){return checked_borrow_lookup(&f->env,&f->fn,&f->call,&f->arg,0);}
static void seal(Fixture *f){checked_borrow_finish_capture(&f->env);assert(checked_borrow_seal(&f->env));}
static void success(void) {
    Fixture f;init(&f);assert(checked_borrow_begin(&f.env));assert(!checked_borrow_begin(&f.env));
    assert(!checked_borrow_seal(&f.env));assert(capture(&f));assert(!lookup(&f));seal(&f);
    const CheckedBorrowEvent *e=lookup(&f);assert(e && !strcmp(e->projection,".leaf"));
    f.field.as.field_access.field_name="changed";assert(!lookup(&f));f.field.as.field_access.field_name="leaf";
    f.call.as.call.name="other";assert(!lookup(&f));f.call.as.call.name="helper";
    f.fn.as.function.checked_affine=0;assert(!lookup(&f));f.fn.as.function.checked_affine=1;
    f.fn.as.function.affine_check_epoch=2;assert(!lookup(&f));f.fn.as.function.affine_check_epoch=1;
    ASTNode cloned=f.arg;f.args[0]=&cloned;assert(!lookup(&f));f.args[0]=&f.arg;assert(lookup(&f));
    checked_borrow_end(&f.env);checked_borrow_end(&f.env);assert(!f.env.checked_borrow_facts && !live);
}
static void invalid_and_limits(void) {
    Fixture f;init(&f);assert(checked_borrow_begin(&f.env));
    f.env.current_file="bad\xff";assert(capture(&f));f.env.current_file="module.nano";
    f.arg.column=0;assert(capture(&f));f.arg.column=8;seal(&f);assert(!lookup(&f));checked_borrow_end(&f.env);
    assert(checked_borrow_begin(&f.env));assert(capture(&f));checked_borrow_function_begin(&f.env,&f.fn);seal(&f);assert(!lookup(&f));checked_borrow_end(&f.env);
    assert(checked_borrow_begin(&f.env));assert(capture(&f));
    char *huge=malloc(BORROW_WORK_LIMIT+2);assert(huge);memset(huge,'x',BORROW_WORK_LIMIT+1);huge[BORROW_WORK_LIMIT+1]=0;
    f.env.current_file=huge;assert(capture(&f));free(huge);seal(&f);assert(!lookup(&f));checked_borrow_end(&f.env);assert(!live);
}
static void allocation_failures(void) {
    alloc_calls=0;success();unsigned count=alloc_calls;assert(count==3);
    for(unsigned i=1;i<=count;i++) {
        Fixture f;init(&f);alloc_calls=denied=0;fail_at=i;
        bool began=checked_borrow_begin(&f.env);
        if(began){assert(!capture(&f));checked_borrow_finish_capture(&f.env);assert(!checked_borrow_seal(&f.env));}
        assert(denied==1);checked_borrow_end(&f.env);assert(!live);
        fail_at=0;success(); /* independent recovery after each real denial */
    }
}
static bool capture_read(Fixture *f,unsigned category) {
    return checked_field_read_capture(&f->env,&f->fn,&f->field,&f->root,category,"owner",".leaf");
}
static const CheckedFieldReadEvent *lookup_read(Fixture *f) {
    return checked_field_read_lookup(&f->env,&f->fn,&f->field);
}
static void paired_read_success(void) {
    Fixture f;init(&f);assert(checked_borrow_begin(&f.env));assert(capture(&f));assert(capture_read(&f,2));
    assert(!lookup_read(&f));seal(&f);assert(lookup(&f));
    const CheckedFieldReadEvent *read=lookup_read(&f);assert(read && read->category==2 && !strcmp(read->projection,".leaf"));
    ASTNode clone=f.root;f.field.as.field_access.object=&clone;assert(!lookup_read(&f));f.field.as.field_access.object=&f.root;
    f.root.as.identifier="other";assert(!lookup_read(&f));f.root.as.identifier="owner";
    f.field.as.field_access.field_name="wrong";assert(!lookup_read(&f));f.field.as.field_access.field_name="leaf";
    f.field.column++;assert(!lookup_read(&f));f.field.column--;
    f.fn.as.function.affine_check_epoch++;assert(!lookup_read(&f));f.fn.as.function.affine_check_epoch--;
    assert(lookup_read(&f));checked_borrow_end(&f.env);assert(!live);
}
static bool read_growth(void) {
    Fixture f;init(&f);ASTNode fields[17];
    bool ok=checked_borrow_begin(&f.env) && capture(&f);
    for(unsigned i=0;i<17 && ok;i++) {
        fields[i]=f.field;fields[i].line=100+(int)i;
        ok=checked_field_read_capture(&f.env,&f.fn,&fields[i],&f.root,1,"owner",".leaf");
    }
    checked_borrow_finish_capture(&f.env);
    if(ok) {
        assert(checked_borrow_seal(&f.env));
        for(unsigned i=0;i<17;i++)assert(checked_field_read_lookup(&f.env,&f.fn,&fields[i]));
        assert(lookup(&f));
    } else assert(!checked_borrow_seal(&f.env));
    checked_borrow_end(&f.env);assert(!live);return ok;
}
static void field_read_controls(void) {
    alloc_calls=0;paired_read_success();unsigned allocations=alloc_calls;assert(allocations==5);
    for(unsigned failure=1;failure<=allocations;failure++) {
        Fixture f;init(&f);alloc_calls=denied=0;fail_at=failure;
        bool accepted=checked_borrow_begin(&f.env) && capture(&f) && capture_read(&f,1);
        assert(!accepted && denied==1);checked_borrow_finish_capture(&f.env);
        assert(!checked_borrow_seal(&f.env));checked_borrow_end(&f.env);assert(!live);
        fail_at=0;paired_read_success();
    }
    for(unsigned category=1;category<=3;category++) {
        Fixture f;init(&f);assert(checked_borrow_begin(&f.env));assert(capture_read(&f,category));seal(&f);
        assert(lookup_read(&f)->category==category);checked_borrow_end(&f.env);
    }
    for(unsigned reverse=0;reverse<2;reverse++) {
        Fixture f;init(&f);assert(checked_borrow_begin(&f.env));assert(capture(&f));assert(capture_read(&f,1));
        f.env.checked_borrow_facts->total_events=BORROW_EVENT_LIMIT;
        assert(reverse?capture(&f):capture_read(&f,3));seal(&f);assert(!lookup(&f) && !lookup_read(&f));
        checked_borrow_end(&f.env);assert(!live);
    }
    Fixture f;init(&f);assert(checked_borrow_begin(&f.env));assert(capture(&f));assert(capture_read(&f,1));
    checked_borrow_function_begin(&f.env,&f.fn);seal(&f);assert(!lookup(&f) && !lookup_read(&f));
    checked_borrow_end(&f.env);assert(!live);
    alloc_calls=0;assert(read_growth());unsigned growth_calls=alloc_calls;assert(growth_calls>20);
    for(unsigned failure=1;failure<=growth_calls;failure++) {
        alloc_calls=denied=0;fail_at=failure;assert(!read_growth() && denied==1);
        fail_at=0;assert(read_growth());
    }
    printf("checked field reads: %u growth allocation prefixes and independent recovery passed\n",growth_calls);
    puts("checked field reads: exact topology/category, shared limits, allocation failures and recovery passed");
}
static ASTNode write_node(Fixture *f) {
    ASTNode node={0};node.type=AST_SET;node.line=27;node.column=3;
    node.as.set.name="owner";node.as.set.field_name="leaf";node.as.set.value=&f->root;return node;
}
static bool capture_write(Fixture *f,ASTNode *node) {
    return checked_field_write_capture(&f->env,&f->fn,node,"owner",".leaf");
}
static void field_write_controls(void) {
    Fixture f;init(&f);ASTNode node=write_node(&f);
    assert(checked_borrow_begin(&f.env) && capture(&f) && capture_read(&f,3) && capture_write(&f,&node));
    assert(capture_write(&f,&node));assert(f.env.checked_borrow_facts->write_count==1);
    assert(!checked_field_write_lookup(&f.env,&f.fn,&node));seal(&f);
    const CheckedFieldWriteEvent *event=checked_field_write_lookup(&f.env,&f.fn,&node);assert(event && event->rhs==&f.root);
    ASTNode clone=node;assert(!checked_field_write_lookup(&f.env,&f.fn,&clone));
    f.root.type=AST_FIELD_ACCESS;assert(!checked_field_write_lookup(&f.env,&f.fn,&node));f.root.type=AST_IDENTIFIER;
    node.as.set.value=&f.field;assert(!checked_field_write_lookup(&f.env,&f.fn,&node));node.as.set.value=&f.root;
    node.as.set.name="other";assert(!checked_field_write_lookup(&f.env,&f.fn,&node));node.as.set.name="owner";
    node.as.set.field_name="wrong";assert(!checked_field_write_lookup(&f.env,&f.fn,&node));node.as.set.field_name="leaf";
    node.column++;assert(!checked_field_write_lookup(&f.env,&f.fn,&node));node.column--;
    f.fn.as.function.affine_check_epoch++;assert(!checked_field_write_lookup(&f.env,&f.fn,&node));f.fn.as.function.affine_check_epoch--;
    assert(checked_field_write_lookup(&f.env,&f.fn,&node));checked_borrow_end(&f.env);assert(!live);
    for(unsigned trigger=0;trigger<3;trigger++) {
        init(&f);node=write_node(&f);assert(checked_borrow_begin(&f.env) && capture(&f) && capture_read(&f,3) && capture_write(&f,&node));
        f.env.checked_borrow_facts->total_events=BORROW_EVENT_LIMIT;
        assert(trigger==0?capture(&f):trigger==1?capture_read(&f,3):capture_write(&f,&node));seal(&f);
        assert(!lookup(&f) && !lookup_read(&f) && !checked_field_write_lookup(&f.env,&f.fn,&node));
        checked_borrow_end(&f.env);assert(!live);
    }
    init(&f);node=write_node(&f);assert(checked_borrow_begin(&f.env) && capture_write(&f,&node));
    checked_borrow_function_begin(&f.env,&f.fn);seal(&f);assert(!checked_field_write_lookup(&f.env,&f.fn,&node));
    checked_borrow_end(&f.env);assert(!live);
    for(unsigned failure=0;failure<=24;failure++) {
        init(&f);ASTNode nodes[17];alloc_calls=denied=0;fail_at=failure;
        bool ok=checked_borrow_begin(&f.env) && capture(&f) && capture_read(&f,3);
        for(unsigned i=0;i<17 && ok;i++){nodes[i]=write_node(&f);nodes[i].line+=i;ok=capture_write(&f,&nodes[i]);}
        if(failure)assert(!ok && denied==1);else assert(ok && alloc_calls==24);
        checked_borrow_finish_capture(&f.env);
        if(ok){assert(checked_borrow_seal(&f.env));for(unsigned i=0;i<17;i++)assert(checked_field_write_lookup(&f.env,&f.fn,&nodes[i]));}
        else assert(!checked_borrow_seal(&f.env));
        checked_borrow_end(&f.env);assert(!live);fail_at=0;success();
    }
    puts("checked field writes: topology, all-family limits, 24 reached allocation prefixes and recovery passed");
}
#ifdef CHECKED_BORROW_ENV_LINK
int g_argc;
char **g_argv;
static void actual_environment_teardown(void) {
    for(unsigned failure=0;failure<=7;failure++) {
        Fixture f;init(&f);ASTNode assignment=write_node(&f);
        Environment *env=create_environment();assert(env);
        env->current_file="module.nano";
        alloc_calls=denied=0;fail_at=failure;
        bool began=checked_borrow_begin(env);
        bool captured=began && checked_borrow_capture(env,&f.fn,&f.call,&f.arg,0,"owner",".leaf") &&
            checked_field_read_capture(env,&f.fn,&f.field,&f.root,1,"owner",".leaf") &&
            checked_field_write_capture(env,&f.fn,&assignment,"owner",".leaf");
        if(failure)assert(!captured && denied==1);else assert(captured);
        /* Teardown owns snapshots, never the expired AST that produced them. */
        memset(&f,0,sizeof f);
        free_environment(env);
        assert(!live);
        fail_at=0;success();
    }
    puts("actual Environment teardown: active scope, failures and recovery passed");
}
#endif
int main(void){
    success();invalid_and_limits();allocation_failures();field_read_controls();field_write_controls();
#ifdef CHECKED_BORROW_ENV_LINK
    actual_environment_teardown();
#endif
    puts("checked borrow ledger: lifecycle, invalidation, limits and allocation recovery passed");return 0;
}
