#include "nanolang.h"
#include "module_builder.h"
#include "nanovirt/codegen.h"
#include "nanovirt/wrapper_gen.h"
#include "nanoisa/nvm_format.h"
#include "nanoisa/verifier.h"
#include "../../modules/nanoisa/nanoisa.h"
#include "nanovm/vm.h"
#include "nanovm/vm_ffi.h"
#include "nanovm/value.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

#include "shadow_runner.h"

/* I already lower NanoLang imports into bytecode. Here I build only their
 * manifest-backed foreign support, including transitive imports. */


void free_ffi_bindings(FfiBinding *bindings, int count) {
    if (!bindings) return;
    for (int i = 0; i < count; i++) {
        free(bindings[i].artifact);
        module_metadata_free(bindings[i].metadata);
    }
    free(bindings);
}

bool bind_ffi_imports(NvmModule *module, ModuleList *modules,
                             FfiBinding *bindings, const char *input, Environment *env) {
    for (uint32_t i = 0; i < module->import_count; i++) {
        NvmImportEntry *imp = &module->imports[i];
        const char *name = nvm_get_string(module, imp->module_name_idx);
        if (!name || !name[0]) continue;
        const char *resolved = resolve_module_path(name, input);
        char *canonical = realpath(resolved ? resolved : name, NULL);
        free((void *)resolved);
        for (int j = 0; j < modules->count; j++) {
            if (!bindings[j].artifact) continue;
            char *candidate = realpath(modules->module_paths[j], NULL);
            bool match = strcmp(name, modules->module_paths[j]) == 0 ||
                         (canonical && candidate && strcmp(canonical, candidate) == 0);
            free(candidate);
            if (!match) continue;
            ModuleBuildMetadata *metadata = bindings[j].metadata;
            const char *symbol = nvm_get_string(module, imp->function_name_idx);
            for (size_t a = 0; a < metadata->callback_adapters_count; a++) {
                const ModuleCallbackAdapter *adapter = &metadata->callback_adapters[a];
                if (!symbol || strcmp(symbol, adapter->function_name)) continue;
                ASTNode *ast = get_cached_module_ast(modules->module_paths[j]);
                ASTNode *declaration = NULL;
                for (int d = 0; ast && d < ast->as.program.count; d++) {
                    ASTNode *item = ast->as.program.items[d];
                    if (item->type == AST_FUNCTION && item->as.function.is_extern &&
                        !strcmp(item->as.function.name, symbol)) {
                        if (declaration) { free(canonical); return false; }
                        declaration = item;
                    }
                }
                if (!codegen_bind_callback_contract(module, i, declaration, env,
                                                    adapter->adapter_symbol, adapter->worker_thread)) {
                    fprintf(stderr, "I cannot bind the retained callback declaration for %s\n", symbol);
                    free(canonical);
                    return false;
                }
            }
            uint32_t idx = nvm_add_string(module, bindings[j].artifact, (uint32_t)strlen(bindings[j].artifact));
            if (idx == UINT32_MAX) { free(canonical); return false; }
            imp->module_name_idx = idx;
            imp->kind = NVM_IMPORT_ARTIFACT;
            break;
        }
        free(canonical);
    }
    uint32_t contract = 0;
    for (uint32_t i = 0; i < module->import_count; i++) {
        while (contract < module->callback_contract_count &&
               module->callback_contracts[contract].import_idx < i) contract++;
        bool bound = contract < module->callback_contract_count &&
                     module->callback_contracts[contract].import_idx == i;
        for (uint16_t p = 0; !bound && p < module->imports[i].param_count; p++) {
            uint8_t tag = module->import_param_types[i][p];
            if (tag == TAG_FUNCTION || tag == TAG_CLOSURE) {
                fprintf(stderr, "I require an explicit retained callback adapter for %s\n",
                        nvm_get_string(module, module->imports[i].function_name_idx));
                return false;
            }
        }
    }
    return nvm_callback_contracts_valid(module);
}

bool build_ffi_modules(ModuleList *modules, FfiBinding *bindings) {
    for (int i = 0; i < modules->count; i++) {
        char *dir = strdup(modules->module_paths[i]);
        if (!dir) return false;
        char *slash = strrchr(dir, '/');
        if (slash == dir) slash[1] = '\0';
        else if (slash) *slash = '\0';
        else strcpy(dir, ".");

        char manifest[1024];
        int length = snprintf(manifest, sizeof(manifest), "%s/module.json", dir);
        if (length < 0 || (size_t)length >= sizeof(manifest)) {
            fprintf(stderr, "I could not represent the imported module manifest path\n");
            free(dir);
            return false;
        }
        ModuleBuildMetadata *meta = module_load_metadata(dir);
        free(dir);
        if (!meta) {
            if (access(manifest, F_OK) == 0 || errno != ENOENT) {
                fprintf(stderr, "I could not read imported module metadata: %s\n", manifest);
                return false;
            }
            continue;
        }
        if (!meta->name || !meta->name[0]) {
            fprintf(stderr, "I require a name in imported module metadata: %s\n", manifest);
            module_metadata_free(meta);
            return false;
        }
        bool ok = !meta->callback_adapters_count || meta->c_sources_count > 0;
        if (meta->c_sources_count > 0) {
            ModuleBuildInfo *info = module_build(NULL, meta);
            ok = info != NULL;
            if (ok) {
                /* I derive the library from the returned object generation,
                 * never from a second read of the mutable current pointer. */
                char *generation = info->object_file ? strdup(info->object_file) : NULL;
                char *end = generation ? strrchr(generation, '/') : NULL;
                ok = end != NULL;
                if (ok) {
                    *end = '\0';
                    char library[1024];
#ifdef __APPLE__
                    const char *extension = "dylib";
#else
                    const char *extension = "so";
#endif
                    int n = snprintf(library, sizeof(library), "%s/lib%s.%s", generation, meta->name, extension);
                    ok = n > 0 && (size_t)n < sizeof(library);
                    if (ok) {
                        bindings[i].artifact = realpath(library, NULL);
                        ok = bindings[i].artifact != NULL;
                    }
                }
                free(generation);
            }
            module_build_info_free(info);
        }
        bindings[i].metadata = meta;
        if (!ok) {
            fprintf(stderr, "I could not build foreign support for %s\n", modules->module_paths[i]);
            return false;
        }
    }
    return true;
}

bool check_shadows(ASTNode *program, Environment *env, ModuleList *modules,
                           const char *input, FfiBinding *bindings, bool include_imports) {
    bool present = false;
    for (int i = 0; i < program->as.program.count; i++) {
        if (program->as.program.items[i]->type == AST_SHADOW) present = true;
    }
    if (!present && !include_imports) return true;
    CodegenResult tests = codegen_compile_shadow_scope(program, env, modules, input, include_imports);
    if (!tests.ok) {
        fprintf(stderr, "I could not compile shadows at line %d: %s\n", tests.error_line, tests.error_msg);
        return false;
    }
    if (!bind_ffi_imports(tests.module, modules, bindings, input, env)) {
        fprintf(stderr, "I could not bind shadow foreign imports\n");
        nvm_module_free(tests.module);
        return false;
    }
    NvmVerifyResult verified = nvm_verify(tests.module);
    if (!verified.ok) {
        fprintf(stderr, "I could not verify shadow bytecode: %s\n", verified.error_msg);
        nvm_module_free(tests.module);
        return false;
    }
    int completion[2];
    if (pipe(completion) != 0) {
        fprintf(stderr, "I could not create the shadow completion channel\n");
        nvm_module_free(tests.module);
        return false;
    }
    if (fcntl(completion[0], F_SETFL, O_NONBLOCK) < 0 ||
        fcntl(completion[0], F_SETFD, FD_CLOEXEC) < 0 ||
        fcntl(completion[1], F_SETFD, FD_CLOEXEC) < 0) {
        close(completion[0]);
        close(completion[1]);
        nvm_module_free(tests.module);
        fprintf(stderr, "I could not configure the shadow completion channel\n");
        return false;
    }
    fflush(NULL);
    pid_t child = fork();
    if (child == 0) {
        close(completion[0]);
        /* I bound test execution, not its authority: this is not a sandbox. */
        signal(SIGALRM, SIG_DFL);
        alarm(10);
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) _exit(1);
        vm_ffi_set_env(env);
        VmState vm;
        vm_init(&vm, tests.module);
        VmResult status = vm_execute(&vm);
        if (status != VM_OK) {
            fprintf(stderr, "I failed a shadow: %s\n", vm.error_msg[0] ? vm.error_msg : vm_error_string(status));
            if (status == VM_ERR_ASSERT_FAILED) vm_stack_trace(&vm, stderr);
        }
        if (vm.cop_pid > 0) vm_ffi_cop_stop(&vm);
        vm_destroy(&vm);
        vm_ffi_shutdown();
        unsigned char done = 1;
        bool completed = status == VM_OK && write(completion[1], &done, 1) == 1;
        close(completion[1]);
        fflush(NULL);
        _exit(completed ? 0 : 1);
    }
    int fork_error = errno;
    close(completion[1]);
    int status = 0;
    pid_t waited = -1;
    bool timed_out = false;
    bool clock_failed = false;
    if (child > 0) {
        struct timespec start, now, pause = {0, 10000000};
        bool clock_ok = clock_gettime(CLOCK_MONOTONIC, &start) == 0;
        for (;;) {
            waited = waitpid(child, &status, WNOHANG);
            if (waited == child || (waited < 0 && errno != EINTR)) break;
            clock_failed = !clock_ok || clock_gettime(CLOCK_MONOTONIC, &now) != 0;
            timed_out = !clock_failed && (now.tv_sec - start.tv_sec > 10 ||
                (now.tv_sec - start.tv_sec == 10 && now.tv_nsec >= start.tv_nsec));
            if (clock_failed || timed_out) {
                kill(child, SIGKILL);
                do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
                break;
            }
            nanosleep(&pause, NULL);
        }
    }
    int supervision_error = child < 0 ? fork_error : errno;
    unsigned char done = 0;
    bool completed = read(completion[0], &done, 1) == 1 && done == 1;
    close(completion[0]);
    nvm_module_free(tests.module);
    if (child < 0 || waited < 0) {
        fprintf(stderr, "I could not supervise shadow execution: %s\n", strerror(supervision_error));
        return false;
    }
    if (clock_failed || timed_out || !completed || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        if (clock_failed)
            fprintf(stderr, "I could not measure the shadow execution deadline\n");
        else if (timed_out || (WIFSIGNALED(status) && WTERMSIG(status) == SIGALRM))
            fprintf(stderr, "I stopped shadow execution after 10 seconds\n");
        else if (WIFSIGNALED(status))
            fprintf(stderr, "I stopped shadow execution after signal %d\n", WTERMSIG(status));
        else
            fprintf(stderr, "I will not publish output after failed shadow execution\n");
        return false;
    }
    return true;
}

/* I select the callback-aware VM only for module graphs declaring retained
 * foreign policies. I never retry a failed shadow in a different backend. */
int check_callback_shadows(ASTNode *program, Environment *env, ModuleList *modules,
                           const char *input, bool include_imports) {
    bool needed = false;
    for (int i = 0; modules && i < modules->count; i++) {
        char *dir = strdup(modules->module_paths[i]);
        if (!dir) return -1;
        char *slash = strrchr(dir, '/');
        if (slash == dir) slash[1] = '\0';
        else if (slash) *slash = '\0';
        else strcpy(dir, ".");
        ModuleBuildMetadata *meta = module_load_metadata(dir);
        free(dir);
        if (meta && meta->callback_adapters_count) needed = true;
        module_metadata_free(meta);
    }
    if (!needed) return 0;
    FfiBinding *bindings = calloc((size_t)modules->count, sizeof(*bindings));
    if (!bindings) return -1;
    bool passed = build_ffi_modules(modules, bindings) &&
        check_shadows(program, env, modules, input, bindings, include_imports);
    free_ffi_bindings(bindings, modules->count);
    return passed ? 1 : -1;
}
