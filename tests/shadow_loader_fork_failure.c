/* I exercise both actual compiler callers with fork failing after preparation. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <unistd.h>
#include <errno.h>
#include <assert.h>
static unsigned fork_calls;
static pid_t shadow_fixture_fork(void) {
    ++fork_calls;
    errno = EAGAIN;
    return -1;
}
#define fork shadow_fixture_fork
#define main retained_compiler_main
#include "../src/main.c"
#undef main
#include "../src/nanovirt/shadow_runner.c"
#undef fork

int main(void) {
    CompilerOptions options = {0};
    ModuleList modules = {0};
    ASTNode program = {0};
    program.type = AST_PROGRAM;
    Environment *env = create_environment();
    assert(env);
    int previous;
    assert(!pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &previous));
    assert(!check_interpreted_shadows(&program, env, &modules, "fixture.nano", &options));
    assert(fork_calls == 1);
    FfiLoaderFork token = {0};
    assert(ffi_loader_shadow_prepare(&token));
    assert(ffi_loader_fork_parent(&token));
    assert(!check_shadows(&program, env, &modules, "fixture.nano", NULL, true));
    assert(fork_calls == 2);
    assert(ffi_loader_shadow_prepare(&token));
    assert(ffi_loader_fork_parent(&token));
    int restored;
    assert(!pthread_setcancelstate(previous, &restored));
    assert(restored == PTHREAD_CANCEL_ENABLE);
    free_environment(env);
    puts("I released both shadow preparations after actual fork refusal.");
    return 0;
}
