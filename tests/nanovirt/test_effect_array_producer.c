/* I inspect unpublished producer output while preserving executable refusal. */
#include "nanolang.h"
#include "nanovirt/codegen.h"
#include "nanoisa/ownership_contracts.h"
#include "nanoisa/affine_bytecode.h"
#include "nanoisa/affine_state.h"
#include "nanoisa/verifier.h"
#include "nanovm/vm_decode.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc;
char **g_argv;
static unsigned checks,seen,minimum_sites;
static bool expected_analysis;
static size_t allocation_attempts,fail_at;
void *nanovirt_test_contract_realloc(void *pointer,size_t size) {
    allocation_attempts++;
    return fail_at && allocation_attempts==fail_at?NULL:realloc(pointer,size);
}
#ifdef NANOVIRT_TEST_CONTRACT_LSAN_SCOPE
extern void __lsan_disable(void);
extern void __lsan_enable(void);
#define FRONTEND_BEGIN() __lsan_disable()
#define FRONTEND_END() __lsan_enable()
#else
#define FRONTEND_BEGIN() ((void)0)
#define FRONTEND_END() ((void)0)
#endif
#define CHECK(c) do { checks++;assert(c); } while(0)
NvmVerifyResult effect_array_producer_verify(const NvmModule *m) {
    NvmOwnershipEffectTypes types;
    NvmV2Result contract=nvm_ownership_effect_types(m,&types);
    if(contract!=NVM_V2_OK || !types.count) {
        NvmVerifyResult why=nvm_verify(m);
        fprintf(stderr,"contract %d bytes %u: %s\n",contract,m->ownership_size,why.error_msg);
    }
    CHECK(contract==NVM_V2_OK && types.count);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    bool analyzed=true;unsigned sites=0;
    for(uint32_t f=0;f<m->function_count;f++) {
        NvmAffineAnalysis result=nvm_affine_analyze_function(m,f);
        if(expected_analysis && !result.ok)fprintf(stderr,"function %u: %s at %u\n",f,result.message,result.byte_offset);
        analyzed &= result.ok;
        VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];CHECK(vm_decode_function(m,f,&decoded,error));
        for(uint32_t i=0;i<decoded.instruction_count;i++) {
            VmDecodedInstruction *in=&decoded.instructions[i];uint32_t element=UINT32_MAX;
            CHECK(nvm_ownership_array_constructor(m,f,in->byte_offset,&element)==NVM_V2_OK);
            if(element!=NVM_V2_NO_INDEX) {
                CHECK(element<types.count);CHECK(in->instruction.opcode==OP_ARR_LITERAL && !in->instruction.operands[1].u16);
                CHECK(in->instruction.operands[0].u8==types.items[element].tag);sites++;
            }
        }
        vm_decoded_function_free(&decoded);
    }
    CHECK(analyzed==expected_analysis);CHECK(sites>=minimum_sites);seen++;
    NvmVerifyResult result=nvm_verify(m);CHECK(!result.ok);
    CHECK(strstr(result.error_msg,"matched collection runtime cleanup"));return result;
}
static void run_source(const char *declarations,const char *body,bool accepted,unsigned sites,bool frontend_accepted) {
    char source[16384];int size=snprintf(source,sizeof source,
        "resource struct Handle{value:int}\nfn consume(h:Handle)->int{let Handle{value}=h return value}\nshadow consume{assert true}\n%s\nfn main()->int{%s}\nshadow main{assert true}\n",declarations,body);
    CHECK(size>0 && (size_t)size<sizeof source);
    FRONTEND_BEGIN();
    int count=0;Token *tokens=tokenize(source,&count);CHECK(tokens);
    ASTNode *program=parse_program(tokens,count);CHECK(program);
    Environment *env=create_environment();CHECK(env);env->suppress_shadow_warnings=true;
    fprintf(stderr,"case %s\n",body);
    bool checked=type_check(program,env);CHECK(checked==frontend_accepted);FRONTEND_END();
    if(!checked){free_environment(env);free_ast(program);free_tokens(tokens,count);return;}
    for(unsigned shadows=0;shadows<2;shadows++) {
        expected_analysis=accepted;minimum_sites=sites;seen=0;allocation_attempts=0;fail_at=0;
        CodegenResult result=(shadows?codegen_compile_shadows(program,env,NULL,NULL):codegen_compile(program,env,NULL,NULL));
        if(!seen)fprintf(stderr,"producer: %s\n%s\n",result.error_msg,source);
        CHECK(seen==1);CHECK(!result.ok && !result.module);
        CHECK(strstr(result.error_msg,"matched collection runtime cleanup"));
        size_t allocation_sites=allocation_attempts;CHECK(allocation_sites>0);
        for(size_t i=1;i<=allocation_sites;i++) {
            seen=0;allocation_attempts=0;fail_at=i;
            result=(shadows?codegen_compile_shadows(program,env,NULL,NULL):codegen_compile(program,env,NULL,NULL));
            CHECK(!seen && !result.ok && !result.module);
            CHECK(allocation_attempts==i && strstr(result.error_msg,"available contract storage"));
            fail_at=0;allocation_attempts=0;
            result=(shadows?codegen_compile_shadows(program,env,NULL,NULL):codegen_compile(program,env,NULL,NULL));
            CHECK(seen==1 && !result.ok && !result.module);
            CHECK(allocation_attempts==allocation_sites && strstr(result.error_msg,"matched collection runtime cleanup"));
        }
    }
    free_environment(env);free_ast(program);free_tokens(tokens,count);
}
static void run(const char *declarations,const char *body,bool accepted,unsigned sites) {
    run_source(declarations,body,accepted,sites,true);
}
int main(void) {
    run("effect E{pass:array<int>->array<int>}",
        "let a:array<int> =handle {perform E.pass([7])} with {pass a->{a}} return (at a 0)",true,0);
    run("effect E{pass:array<int>->int}",
        "return handle {perform E.pass([7])} with {pass a->{return (at a 0)}}",true,0);
    run("effect E{pass:array<int>->array<int>} fn relay(a:array<int>)->array<int>{return a} shadow relay{assert true}",
        "let a:array<int> =handle {perform E.pass((relay []))} with {pass a->{a}} return (array_length a)",true,1);
    run("effect E{pass:array<array<int>>->array<array<int>>}",
        "let a:array<array<int>> =handle {perform E.pass([[], [7]])} with {pass a->{a}} return (at (at a 1) 0)",true,1);
    run("struct Leaf{value:int} effect E{pass:array<Leaf>->array<Leaf>}",
        "let a:array<Leaf> =handle {perform E.pass([])} with {pass a->{a}} return (array_length a)",true,1);
    run_source("effect E{pass:array<Handle>->array<Handle>} "
        "fn relay(a:array<Handle>)->array<Handle>{return handle {perform E.pass(a)} with {pass a->{a}}} shadow relay{assert true} "
        "fn empty()->array<Handle>{return []} shadow empty{assert true}","return 7",true,1,false);
    run_source("effect E{pass:array<Handle>->array<Handle>} "
        "fn relay(a:array<Handle>)->array<Handle>{let n:int=(array_length a) return handle {perform E.pass((array_push a Handle{value:n}))} with {pass a->{a}}} shadow relay{assert true}",
        "return 7",true,0,false);
    run("effect E{pass:array<int>->array<int>}",
        "let a:array<int> =handle {perform E.pass([7, 8])} with {pass a->{a}} (array_set a 0 9) let b:array<int> =(array_push a 10) let c:array<int> =(array_remove_at b 1) return (array_pop c)",true,0);
    run("struct A{value:int} struct B{value:int} effect E{pass:array<A> array<B>->array<A>}",
        "let mut a:array<A> =handle {perform E.pass([] [])} with {pass a b->{a}} set a [A{value:7}] return (array_length a)",true,2);
    run_source("union Choice<T>{Some{value:T},None{}} effect E{pass:array<Choice<int>>->array<Choice<int>>}",
        "let a:array<Choice<int>> =handle {perform E.pass([Choice.Some{value:7}])} with {pass a->{a}} return (array_length a)",true,0,false);
    run("union Choice<T>{Some{value:T},None{}} effect E{pass:array<Choice<int>>->array<Choice<int>>}",
        "let value:Choice<int> =Choice.Some{value:7} let a:array<Choice<int>> =handle {perform E.pass([value])} with {pass a->{a}} return (array_length a)",true,0);
    run("effect E{pass:array<string>->array<string>}",
        "let a:array<string> =handle {perform E.pass([\"text\"])} with {pass a->{a}} return (array_length a)",true,0);
    run("effect E{pass:array<float>->array<float>}",
        "let a:array<float> =handle {perform E.pass([7.0])} with {pass a->{a}} assert (== (at a 0) 7.0) return 7",true,0);
    run("effect E{pass:array<bool>->array<bool>}",
        "let a:array<bool> =handle {perform E.pass([true])} with {pass a->{a}} assert (at a 0) return 7",true,0);
    run("union Pick{First{},Second{}} effect E{pass:array<int>->array<int>}",
        "let pick:Pick=Pick.First{} let a:array<int> =match pick{First(v)=>{[]} Second(v)=>{[7]}} let b:array<int> =handle {perform E.pass(a)} with {pass a->{a}} return (array_length b)",true,1);
    run("effect E{pass:array<int>->array<int>} fn empty()->array<int>{return []} shadow empty{assert true}",
        "let a:array<int> =handle {perform E.pass((empty))} with {pass a->{a}} return (array_length a)",true,1);
    printf("I passed %u C array producer checks.\n",checks);return 0;
}
