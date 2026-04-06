#include "nanolang.h"
#include "colors.h"
#include "version.h"
#include "module_builder.h"
#include "interpreter_ffi.h"
#include "reflection.h"
#include "emit_typed_ast.h"
#include "docgen_md.h"
#include "runtime/list_CompilerDiagnostic.h"
#include "toon_output.h"
#include "nanocore_subset.h"
#include "nanocore_export.h"
#include "wasm_backend.h"
#include "ptx_backend.h"
#include "opencl_backend.h"
#include "c_backend.h"
#include "riscv_backend.h"
#include "bench.h"
#include "tco_pass.h"
#include "cps_pass.h"
#include "pgo_pass.h"
#include "llvm_backend.h"
#include "sign.h"
#include <unistd.h>  /* For getpid(), execv() on all POSIX systems */
#include <limits.h>  /* For PATH_MAX */
#include <errno.h>   /* For errno/strerror in execv error reporting */

#ifdef __APPLE__
#include <mach-o/loader.h>
#include <fcntl.h>
#endif

/* Global argc/argv for runtime access by transpiled programs */
int g_argc = 0;
char **g_argv = NULL;

/* Compilation options */
typedef struct {
    bool verbose;
    bool keep_c;
    bool show_intermediate_code;
    bool save_asm;            /* -S flag: save generated C to .genC file */
    bool json_errors;         /* Output errors in JSON format for tooling */
    bool profile_gprof;       /* -pg flag: enable gprof profiling support */
    bool profile;             /* --profile: inject timing hooks into generated C */
    bool coverage;            /* --coverage: instrument compiled output for gcov/lcov line+branch coverage */
    bool profile_runtime;     /* --profile-runtime: also emit flamegraph collapsed-stack .nano.prof */

    const char *profile_output_path;     /* --profile-output <path>: write structured profile JSON to file */
    const char *profile_flamegraph_path; /* --profile-runtime [<path>]: flamegraph .nano.prof output path */
    const char *llm_diags_json_path; /* --llm-diags-json <path> (agent-only): write diagnostics as JSON */
    const char *llm_diags_toon_path; /* --llm-diags-toon <path> (agent-only): write diagnostics as TOON (~40% fewer tokens) */
    const char *llm_shadow_json_path; /* --llm-shadow-json <path> (agent-only): write shadow failure summary as JSON */
    const char *reflect_output_path;  /* --reflect <path>: emit module API as JSON */
    bool emit_typed_ast;              /* --emit-typed-ast-json: emit typed AST as JSON to stdout */
    char **include_paths;      /* -I flags */
    int include_count;
    char **library_paths;     /* -L flags */
    int library_path_count;
    char **libraries;         /* -l flags */
    int library_count;
    /* Phase 3: Module safety warnings */
    bool warn_unsafe_imports;  /* Warn when importing unsafe modules */
    bool warn_unsafe_calls;    /* Warn when calling functions from unsafe modules */
    bool warn_ffi;             /* Warn on any FFI call */
    bool forbid_unsafe;        /* Error (not warn) on unsafe modules */
    bool trust_report;         /* --trust-report: print formal verification trust levels */
    bool reference_eval;       /* --reference-eval: cross-check with Coq-extracted interpreter */
    const char *target;        /* --target <name>: compile target (default: native, wasm) */
    bool no_sourcemap;         /* --no-sourcemap: suppress .wasm.map generation for wasm target */
    bool tco;                  /* --tco: enable tail-call optimization pass */
    const char *pgo_profile;   /* --pgo <path>: apply profile-guided inlining from .nano.prof */
    bool llvm;                 /* --llvm: emit LLVM IR (.ll) instead of transpiled C */
    bool debug;                /* --debug / -g: emit DWARF v4 debug information */
    bool bench;                /* --bench: run @bench-annotated functions */
    uint64_t bench_n;          /* --bench-n <N>: fixed iteration count (0 = auto) */
    const char *bench_json;    /* --bench-json <path>: write JSON results to file */
    bool doc_md;               /* --doc-md / -dm: emit GFM Markdown API docs */
} CompilerOptions;

/* Return TMPDIR if set, otherwise "/tmp" (matches pattern in eval_io.c:177) */
static const char *get_tmp_dir(void) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    return tmp;
}

/* Resolve the nanolang project root from the compiler binary path.
 * The binary lives at <root>/bin/nanoc_c, so we go up two levels.
 * Falls back to CWD if resolution fails. */
char g_project_root[PATH_MAX] = "";

const char *get_project_root(void) {
    return g_project_root[0] ? g_project_root : ".";
}

static void resolve_project_root(const char *argv0) {
    char exe_path[PATH_MAX];
    if (realpath(argv0, exe_path) == NULL) {
        /* Fallback: use CWD */
        if (getcwd(g_project_root, sizeof(g_project_root)) == NULL) {
            strcpy(g_project_root, ".");
        }
        return;
    }
    /* Strip binary name: /path/to/bin/nanoc_c -> /path/to/bin */
    char *slash = strrchr(exe_path, '/');
    if (slash) {
        *slash = '\0';
        /* Strip bin/: /path/to/bin -> /path/to */
        slash = strrchr(exe_path, '/');
        if (slash) {
            *slash = '\0';
        }
    }
    strncpy(g_project_root, exe_path, sizeof(g_project_root) - 1);
    g_project_root[sizeof(g_project_root) - 1] = '\0';
}

static void json_escape(FILE *out, const char *s) {
    if (!s) return;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '\\': fputs("\\\\", out); break;
            case '"': fputs("\\\"", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (c < 0x20) fprintf(out, "\\u%04x", (unsigned int)c);
                else fputc((int)c, out);
        }
    }
}

static const char *phase_name(int phase) {
    switch (phase) {
        case CompilerPhase_PHASE_LEXER: return "lexer";
        case CompilerPhase_PHASE_PARSER: return "parser";
        case CompilerPhase_PHASE_TYPECHECK: return "typecheck";
        case CompilerPhase_PHASE_TRANSPILER: return "transpiler";
        case CompilerPhase_PHASE_RUNTIME: return "runtime";
        default: return "unknown";
    }
}

static const char *severity_name(int severity) {
    switch (severity) {
        case DiagnosticSeverity_DIAG_INFO: return "info";
        case DiagnosticSeverity_DIAG_WARNING: return "warning";
        case DiagnosticSeverity_DIAG_ERROR: return "error";
        default: return "unknown";
    }
}

static void llm_emit_diags_json(
    const char *path,
    const char *input_file,
    const char *output_file,
    int exit_code,
    List_CompilerDiagnostic *diags
) {
    if (!path || path[0] == '\0') return;

    FILE *f = fopen(path, "w");
    if (!f) return; /* best-effort */

    fprintf(f, "{");
    fprintf(f, "\"tool\":\"nanoc_c\",");
    fprintf(f, "\"success\":%s,", exit_code == 0 ? "true" : "false");
    fprintf(f, "\"exit_code\":%d,", exit_code);
    fprintf(f, "\"input_file\":\""); json_escape(f, input_file); fprintf(f, "\",");
    fprintf(f, "\"output_file\":\""); json_escape(f, output_file); fprintf(f, "\",");
    fprintf(f, "\"diagnostics\":[");

    int n = diags ? nl_list_CompilerDiagnostic_length(diags) : 0;
    for (int i = 0; i < n; i++) {
        CompilerDiagnostic d = nl_list_CompilerDiagnostic_get(diags, i);
        if (i > 0) fprintf(f, ",");
        fprintf(f, "{");
        fprintf(f, "\"code\":\""); json_escape(f, d.code); fprintf(f, "\",");
        fprintf(f, "\"message\":\""); json_escape(f, d.message); fprintf(f, "\",");
        fprintf(f, "\"phase\":%d,", d.phase);
        fprintf(f, "\"phase_name\":\"%s\",", phase_name(d.phase));
        fprintf(f, "\"severity\":%d,", d.severity);
        fprintf(f, "\"severity_name\":\"%s\",", severity_name(d.severity));
        fprintf(f, "\"location\":{");
        fprintf(f, "\"file\":\""); json_escape(f, d.location.file); fprintf(f, "\",");
        fprintf(f, "\"line\":%d,", d.location.line);
        fprintf(f, "\"column\":%d", d.location.column);
        fprintf(f, "}");
        fprintf(f, "}");
    }

    fprintf(f, "]}");
    fclose(f);
}

static void llm_emit_diags_toon(
    const char *path,
    const char *input_file,
    const char *output_file,
    int exit_code,
    List_CompilerDiagnostic *diags
) {
    if (!path || path[0] == '\0') return;
    if (!toon_diagnostics_enabled()) return;

    /* Populate TOON diagnostics from compiler diagnostics list */
    int n = diags ? nl_list_CompilerDiagnostic_length(diags) : 0;
    for (int i = 0; i < n; i++) {
        CompilerDiagnostic d = nl_list_CompilerDiagnostic_get(diags, i);
        toon_diagnostics_add(
            severity_name(d.severity),
            d.code,
            d.message,
            d.location.file,
            d.location.line,
            d.location.column
        );
    }

    /* Output to file */
    if (!toon_diagnostics_output_to_file(path, input_file, output_file, exit_code)) {
        /* best-effort, ignore failure */
    }

    toon_diagnostics_cleanup();
}

static void diags_push_simple(List_CompilerDiagnostic *diags, int phase, int severity, const char *code, const char *message) {
    if (!diags) return;
    CompilerDiagnostic d;
    d.phase = phase;
    d.severity = severity;
    d.code = (char*)(code ? code : "C0000");
    d.message = (char*)(message ? message : "");
    d.location.file = (char*)"";
    d.location.line = 0;
    d.location.column = 0;
    nl_list_CompilerDiagnostic_push(diags, d);
}

static bool deterministic_outputs_enabled(void) {
    const char *v = getenv("NANO_DETERMINISTIC");
    return v && (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "yes") == 0);
}

#ifdef __APPLE__
static int determinize_macho_uuid_and_signature(const char *path) {
    /* On modern macOS, Mach-O binaries are ad-hoc signed and include a randomized LC_UUID.
     * For deterministic bootstrap verification we:
     *  1) overwrite the LC_UUID bytes with a fixed value
     *  2) re-sign with a fixed identifier so the LC_CODE_SIGNATURE blob is deterministic
     */
    int fd = open(path, O_RDWR);
    if (fd < 0) return -1;

    struct mach_header_64 hdr;
    ssize_t n = pread(fd, &hdr, sizeof(hdr), 0);
    if (n != (ssize_t)sizeof(hdr) || hdr.magic != MH_MAGIC_64) {
        close(fd);
        return -1;
    }

    off_t off = (off_t)sizeof(hdr);
    for (uint32_t i = 0; i < hdr.ncmds; i++) {
        struct load_command lc;
        if (pread(fd, &lc, sizeof(lc), off) != (ssize_t)sizeof(lc) || lc.cmdsize < sizeof(lc)) {
            close(fd);
            return -1;
        }

        if (lc.cmd == LC_UUID) {
            struct uuid_command uc;
            if (lc.cmdsize < sizeof(uc) || pread(fd, &uc, sizeof(uc), off) != (ssize_t)sizeof(uc)) {
                close(fd);
                return -1;
            }

            static const uint8_t fixed_uuid[16] = {
                0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
            };
            memcpy(uc.uuid, fixed_uuid, sizeof(fixed_uuid));

            if (pwrite(fd, &uc, sizeof(uc), off) != (ssize_t)sizeof(uc)) {
                close(fd);
                return -1;
            }
            break;
        }

        off += (off_t)lc.cmdsize;
    }

    close(fd);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "codesign -s - --force -i nanolang.deterministic '%s' >/dev/null 2>&1", path);
    return system(cmd);
}
#endif

/* Compile nanolang source to executable */
static int compile_file(const char *input_file, const char *output_file, CompilerOptions *opts) {
    List_CompilerDiagnostic *diags = nl_list_CompilerDiagnostic_new();

    /* Read source file */
    FILE *file = fopen(input_file, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s'\n", input_file);
        diags_push_simple(diags, CompilerPhase_PHASE_LEXER, DiagnosticSeverity_DIAG_ERROR, "CIO01", "Could not open input file");
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *source = malloc(size + 1);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    fread(source, 1, size, file);
#pragma GCC diagnostic pop
    source[size] = '\0';
    fclose(file);

    if (opts->verbose) printf("Compiling %s...\n", input_file);

    /* Phase 1: Lexing */
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "Lexing failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_LEXER, DiagnosticSeverity_DIAG_ERROR, "CLEX01", "Lexing failed");
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }
    if (opts->verbose) printf("✓ Lexing complete (%d tokens)\n", token_count);

    /* Phase 2: Parsing */
    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "Parsing failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_PARSER, DiagnosticSeverity_DIAG_ERROR, "CPARSE01", "Parsing failed");
        free_tokens(tokens, token_count);
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }
    if (opts->verbose) printf("✓ Parsing complete\n");

    /* Phase 3: Create environment and process imports */
    clear_module_cache();  /* Clear cache from any previous compilation */
    Environment *env = create_environment();
    
    /* Set warning flags from compiler options */
    env->warn_unsafe_imports = opts->warn_unsafe_imports;
    env->warn_unsafe_calls = opts->warn_unsafe_calls;
    env->warn_ffi = opts->warn_ffi;
    env->forbid_unsafe = opts->forbid_unsafe;
    env->profile_gprof = opts->profile_gprof;
    env->profile = opts->profile;
    env->profile_runtime = opts->profile_runtime;
    env->profile_flamegraph_path = opts->profile_flamegraph_path;
    env->gpu_target = (opts->target &&
        (strcmp(opts->target, "ptx") == 0 || strcmp(opts->target, "opencl") == 0));


    env->profile_output_path = opts->profile_output_path;
    
    ModuleList *modules = create_module_list();
    if (!process_imports(program, env, modules, input_file)) {
        fprintf(stderr, "Module loading failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_PARSER, DiagnosticSeverity_DIAG_ERROR, "CIMPORT01", "Module loading failed");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }
    if (opts->verbose && modules->count > 0) {
        printf("✓ Loaded %d module(s)\n", modules->count);
    }

    /* Compile modules early so extern C functions are available for shadow tests (via FFI). */
    char module_objs[2048] = "";
    char module_compile_flags[2048] = "";

    /* Phase 4: Type Checking */
    typecheck_set_current_file(input_file);
    /* Use type_check_module if reflection or doc-md is requested (no main needed) */
    bool typecheck_success = (opts->reflect_output_path || opts->doc_md) ?
        type_check_module(program, env) :
        type_check(program, env);
    
    if (!typecheck_success) {
        fprintf(stderr, "Type checking failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_TYPECHECK, DiagnosticSeverity_DIAG_ERROR, "CTYPE01", "Type checking failed");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }
    if (opts->verbose) printf("✓ Type checking complete\n");

    /* ── Profile-Guided Optimization (inlining) ───────────────────────── */
    if (opts->pgo_profile) {
        PGOProfile *pgo = pgo_load_profile(opts->pgo_profile);
        if (pgo) {
            int sites = pgo_apply(program, pgo);
            if (opts->verbose || sites > 0)
                printf("✓ PGO pass: %d call site(s) inlined from %s\n",
                       sites, opts->pgo_profile);
            if (opts->verbose) pgo_print_report(pgo);
            pgo_profile_free(pgo);
        } else {
            fprintf(stderr, "warning: --pgo: could not load profile %s\n",
                    opts->pgo_profile);
        }
    }

    /* ── Tail-Call Optimization pass ─────────────────────────────────── */
    if (opts->tco) {
        tco_pass(program);
        if (opts->verbose) printf("✓ TCO pass complete\n");
    }

    /* ── CPS transform pass (async/await) — always runs ──────────────── */
    {
        int async_count = cps_pass(program);
        if (opts->verbose && async_count > 0)
            printf("✓ CPS pass: %d async function(s) transformed\n", async_count);
    }

    /* ── WASM target: emit .wasm binary and exit ─────────────────────── */
    if (opts->target && strcmp(opts->target, "wasm") == 0) {
        /* Determine output path: use -o flag or derive from input */
        const char *wasm_out = output_file;
        char wasm_out_buf[PATH_MAX];
        if (!wasm_out) {
            /* Replace .nano with .wasm */
            strncpy(wasm_out_buf, input_file, PATH_MAX - 6);
            wasm_out_buf[PATH_MAX - 6] = '\0';
            char *dot = strrchr(wasm_out_buf, '.');
            if (dot) *dot = '\0';
            strcat(wasm_out_buf, ".wasm");
            wasm_out = wasm_out_buf;
        }
        if (opts->verbose) printf("Emitting WASM → %s\n", wasm_out);
        /* Build source map path: <wasm_out>.map unless suppressed */
        char srcmap_buf[PATH_MAX + 8];
        const char *srcmap_path = NULL;
        if (!opts->no_sourcemap) {
            snprintf(srcmap_buf, sizeof(srcmap_buf), "%s.map", wasm_out);
            srcmap_path = srcmap_buf;
        }
        int wasm_rc = wasm_backend_emit(program, wasm_out, input_file, srcmap_path, opts->verbose);
        if (wasm_rc == 0 && opts->verbose) {
            printf("✓ WASM binary emitted to %s\n", wasm_out);
        }
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return wasm_rc;
    }

    /* ── PTX target: emit .ptx text and exit ────────────────────────── */
    if (opts->target && strcmp(opts->target, "ptx") == 0) {
        const char *ptx_out = output_file;
        char ptx_out_buf[PATH_MAX];
        if (!ptx_out) {
            strncpy(ptx_out_buf, input_file, PATH_MAX - 5);
            ptx_out_buf[PATH_MAX - 5] = '\0';
            char *dot = strrchr(ptx_out_buf, '.');
            if (dot) *dot = '\0';
            strcat(ptx_out_buf, ".ptx");
            ptx_out = ptx_out_buf;
        }
        if (opts->verbose) printf("Emitting PTX → %s\n", ptx_out);
        int ptx_rc = ptx_backend_emit(program, ptx_out, input_file, opts->verbose);
        if (ptx_rc == 0) {
            printf("✓ PTX assembly emitted to %s\n", ptx_out);
            printf("  Load with: cuModuleLoad(&mod, \"%s\");\n", ptx_out);
            printf("  Or compile: nvcc -ptx %s -o %s.cubin\n", ptx_out, ptx_out);
        }
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return ptx_rc;
    }

    /* ── OpenCL target: emit .cl kernel source and exit ─────────────── */
    if (opts->target && strcmp(opts->target, "opencl") == 0) {
        const char *cl_out = output_file;
        char cl_out_buf[PATH_MAX];
        if (!cl_out) {
            strncpy(cl_out_buf, input_file, PATH_MAX - 4);
            cl_out_buf[PATH_MAX - 4] = '\0';
            char *dot = strrchr(cl_out_buf, '.');
            if (dot) *dot = '\0';
            strcat(cl_out_buf, ".cl");
            cl_out = cl_out_buf;
        }
        if (opts->verbose) printf("Emitting OpenCL C → %s\n", cl_out);
        int cl_rc = ocl_backend_emit(program, cl_out, input_file, opts->verbose);
        if (cl_rc == 0) {
            printf("✓ OpenCL C kernel emitted to %s\n", cl_out);
            printf("  Load with: clCreateProgramWithSource + clBuildProgram\n");
            printf("  CPU fallback: set POCL_DEVICES=cpu (POCL required)\n");
        }
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return cl_rc;
    }

    /* ── C target: emit .c source file and exit ─────────────────────── */
    if (opts->target && strcmp(opts->target, "c") == 0) {
        const char *c_out = output_file;
        char c_out_buf[PATH_MAX];
        if (!c_out) {
            strncpy(c_out_buf, input_file, PATH_MAX - 3);
            c_out_buf[PATH_MAX - 3] = '\0';
            char *dot = strrchr(c_out_buf, '.');
            if (dot) *dot = '\0';
            strcat(c_out_buf, ".c");
            c_out = c_out_buf;
        }
        if (opts->verbose) printf("Emitting C → %s\n", c_out);
        CBOptions cb_opts = {0}; cb_opts.verbose = opts->verbose;
        int c_rc = c_backend_emit(program, c_out, input_file, &cb_opts);
        if (c_rc == 0 && opts->verbose) {
            printf("✓ C source emitted to %s\n", c_out);
            printf("  Compile with: gcc -std=c11 %s -o prog\n", c_out);
        }
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return c_rc;
    }

    /* ── LLVM IR: emit .ll and exit ──────────────────────────────────────── */
    if (opts->llvm) {
        const char *ll_out = output_file;
        char ll_out_buf[PATH_MAX];
        if (!ll_out) {
            strncpy(ll_out_buf, input_file, PATH_MAX - 4);
            ll_out_buf[PATH_MAX - 4] = '\0';
            char *dot = strrchr(ll_out_buf, '.');
            if (dot) *dot = '\0';
            strcat(ll_out_buf, ".ll");
            ll_out = ll_out_buf;
        }
        if (opts->verbose) printf("Emitting LLVM IR → %s\n", ll_out);
        int ll_rc = llvm_backend_emit(program, ll_out, input_file,
                                      opts->verbose, opts->debug);
        if (ll_rc == 0) {
            printf("✓ LLVM IR emitted to %s\n", ll_out);
            printf("  Compile: clang -O2 %s -o %s.out\n", ll_out, ll_out);
        } else {
            fprintf(stderr, "LLVM backend failed\n");
        }
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return ll_rc;
    }

    /* ── RISC-V target: emit .s assembly and exit ─────────────────────── */
    if (opts->target && strcmp(opts->target, "riscv") == 0) {
        const char *rv_out = output_file;
        char rv_out_buf[PATH_MAX];
        if (!rv_out) {
            strncpy(rv_out_buf, input_file, PATH_MAX - 3);
            rv_out_buf[PATH_MAX - 3] = '\0';
            char *dot = strrchr(rv_out_buf, '.');
            if (dot) *dot = '\0';
            strcat(rv_out_buf, ".s");
            rv_out = rv_out_buf;
        }
        if (opts->verbose) printf("Emitting RISC-V asm → %s\n", rv_out);
        int rv_rc = riscv_backend_emit(program, rv_out, input_file,
                                       opts->verbose, opts->debug);
        if (rv_rc == 0) {
            printf("✓ RISC-V assembly emitted to %s\n", rv_out);
            printf("  Assemble: riscv64-unknown-elf-gcc -nostdlib %s -o %s.elf\n",
                   rv_out, rv_out_buf);
        } else {
            fprintf(stderr, "RISC-V backend failed\n");
        }
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        clear_module_cache();
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return rv_rc;
    }

    /* ── Bench mode: run @bench-annotated functions ──────────────────── */
    if (opts->bench) {
        BenchOptions bopts = {
            .n_iters       = opts->bench_n,
            .output_format = opts->bench_json ? BENCH_FMT_JSON : BENCH_FMT_HUMAN,
            .json_out_path = opts->bench_json,
            .backend       = opts->target ? opts->target : "native",
            .verbose       = opts->verbose,
        };
        FILE *json_out = NULL;
        if (opts->bench_json) {
            json_out = fopen(opts->bench_json, "w");
            if (!json_out)
                fprintf(stderr, "[bench] Cannot open %s for writing\n",
                        opts->bench_json);
        }
        int bench_rc = bench_run_program(program, &bopts, input_file, json_out);
        if (json_out) fclose(json_out);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        clear_module_cache();
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return bench_rc;
    }

    /* Phase 4.1: Trust Report (if requested) */
    if (opts->trust_report) {
        TrustReport *report = nanocore_trust_report(program, env);
        if (report) {
            nanocore_print_trust_report(report, input_file);
            nanocore_free_trust_report(report);
        }
        /* Clean up and exit - trust report is an analysis-only mode */
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return 0;
    }

    /* Phase 4.2: Reference Eval — cross-check with Coq-extracted interpreter */
    if (opts->reference_eval) {
        printf("NanoCore Reference Evaluation: %s\n", input_file);
        printf("Cross-checking verified functions against Coq-extracted interpreter\n\n");

        /* Walk all functions and export verified ones */
        int checked = 0, matched = 0, failed = 0, skipped = 0;
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *item = program->as.program.items[i];
            if (!item || item->type != AST_FUNCTION) continue;
            if (item->as.function.is_extern) continue;

            const char *fname = item->as.function.name;
            TrustLevel level = nanocore_function_trust(item, env);

            if (level != TRUST_VERIFIED) {
                skipped++;
                continue;
            }

            /* Export entire function (wrapped in lambdas) to S-expression */
            char *sexpr = nanocore_export_sexpr(item, env);
            if (!sexpr) {
                printf("  %-30s [skip] could not export to S-expression\n", fname);
                skipped++;
                continue;
            }

            /* Call reference interpreter */
            char *ref_result = nanocore_reference_eval(sexpr, g_argv[0]);
            if (ref_result) {
                printf("  %-30s [ok]   ref=%s\n", fname, ref_result);
                matched++;
                free(ref_result);
            } else {
                printf("  %-30s [warn] reference interpreter unavailable\n", fname);
                failed++;
            }

            free(sexpr);
            checked++;
        }

        printf("\nSummary: %d checked, %d matched, %d warnings, %d skipped (not verified)\n",
               checked, matched, failed, skipped);

        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return 0;
    }

    /* Phase 4.4: Module Reflection (if requested) */
    if (opts->reflect_output_path) {
        /* Extract module name from input file */
        const char *module_name = strrchr(input_file, '/');
        module_name = module_name ? module_name + 1 : input_file;
        /* Remove .nano extension if present */
        char *name_copy = strdup(module_name);
        char *dot = strrchr(name_copy, '.');
        if (dot && strcmp(dot, ".nano") == 0) {
            *dot = '\0';
        }
        
        if (opts->verbose) printf("→ Emitting module reflection to %s\n", opts->reflect_output_path);
        
        if (!emit_module_reflection(opts->reflect_output_path, program, env, name_copy)) {
            fprintf(stderr, "Error: Failed to emit module reflection\n");
            free(name_copy);
            free_ast(program);
            free_tokens(tokens, token_count);
            free_environment(env);
            free_module_list(modules);
            free(source);
            llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
            nl_list_CompilerDiagnostic_free(diags);
            return 1;
        }
        
        if (opts->verbose) printf("✓ Module reflection complete\n");
        free(name_copy);
        
        /* Clean up and exit - no need to compile when reflecting */
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return 0;
    }

    /* Phase 4.42: Emit GFM Markdown docs (--doc-md) */
    if (opts->doc_md) {
        /* Extract module name from input file */
        const char *module_name = strrchr(input_file, '/');
        module_name = module_name ? module_name + 1 : input_file;
        char *name_copy = strdup(module_name);
        char *dot = strrchr(name_copy, '.');
        if (dot && strcmp(dot, ".nano") == 0) *dot = '\0';

        /* Determine output path: use -o if provided, else <module>.md */
        char md_out_buf[512];
        const char *md_out = output_file;
        /* output_file defaults to TMPDIR/nanoc_a.out; treat that as "not set" */
        {
            char default_prefix[64];
            snprintf(default_prefix, sizeof(default_prefix), "%s/nanoc_a.out",
                     get_tmp_dir());
            if (strcmp(md_out, default_prefix) == 0) {
                snprintf(md_out_buf, sizeof(md_out_buf), "%s.md", name_copy);
                md_out = md_out_buf;
            }
        }

        if (opts->verbose) printf("→ Emitting Markdown docs to %s\n", md_out);

        if (!emit_doc_md(md_out, program, source, name_copy)) {
            fprintf(stderr, "Error: Failed to emit Markdown docs\n");
            free(name_copy);
            free_ast(program);
            free_tokens(tokens, token_count);
            free_environment(env);
            free_module_list(modules);
            free(source);
            nl_list_CompilerDiagnostic_free(diags);
            return 1;
        }

        if (opts->verbose) printf("✓ Markdown docs written to %s\n", md_out);
        free(name_copy);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return 0;
    }

    /* Phase 4.45: Emit typed AST as JSON (if requested) */
    if (opts->emit_typed_ast) {
        emit_typed_ast_json(input_file, program, env);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return 0;
    }

    /* Phase 4.5: Build imported modules (object + shared libs) */
    if (modules->count > 0) {
        if (!compile_modules(modules, env, module_objs, sizeof(module_objs),
                             module_compile_flags, sizeof(module_compile_flags),
                             opts->verbose)) {
            fprintf(stderr, "Error: Failed to compile modules\n");
            diags_push_simple(diags, CompilerPhase_PHASE_PARSER, DiagnosticSeverity_DIAG_ERROR, "CMOD01", "Failed to compile imported modules");
            free_ast(program);
            free_tokens(tokens, token_count);
            free_environment(env);
            free_module_list(modules);
            clear_module_cache();
            free(source);
            llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
            llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
            nl_list_CompilerDiagnostic_free(diags);
            return 1;
        }
    }

    /* Phase 4.6: Initialize FFI and load module shared libraries for shadow tests */
    (void)ffi_init(opts->verbose);
    for (int i = 0; i < modules->count; i++) {
        const char *module_path = modules->module_paths[i];

        char *module_dir = strdup(module_path);
        char *last_slash = strrchr(module_dir, '/');
        if (last_slash) {
            *last_slash = '\0';
        } else {
            free(module_dir);
            module_dir = strdup(".");
        }

        ModuleBuildMetadata *meta = module_load_metadata(module_dir);

        char mod_name[256];
        if (meta && meta->name) {
            snprintf(mod_name, sizeof(mod_name), "%s", meta->name);
        } else {
            const char *base_name = last_slash ? last_slash + 1 : module_path;
            snprintf(mod_name, sizeof(mod_name), "%s", base_name);
            char *dot = strrchr(mod_name, '.');
            if (dot) *dot = '\0';
        }

        (void)ffi_load_module(mod_name, module_path, env, opts->verbose);

        if (meta) module_metadata_free(meta);
        free(module_dir);
    }

    /* Phase 5: Shadow-Test Execution (Compile-Time Function Execution) */
    if (opts->llm_shadow_json_path && opts->llm_shadow_json_path[0] != '\0') {
        setenv("NANO_LLM_SHADOW_JSON", opts->llm_shadow_json_path, 1);
    } else {
        unsetenv("NANO_LLM_SHADOW_JSON");
    }
    if (!run_shadow_tests(program, env, opts->verbose)) {
        fprintf(stderr, "Shadow tests failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_RUNTIME, DiagnosticSeverity_DIAG_ERROR, "CSHADOW01", "Shadow tests failed");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        ffi_cleanup();
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        unsetenv("NANO_LLM_SHADOW_JSON");
        return 1;
    }
    if (opts->verbose) printf("✓ Shadow tests passed\n");
    unsetenv("NANO_LLM_SHADOW_JSON");

    /* Phase 5.5: Ensure module ASTs are in cache for declaration generation */
    /* Module compilation uses isolated caches that get cleared, so we need to
     * re-load modules into the main cache before transpilation so that
     * generate_module_function_declarations() can find them. */
    if (modules->count > 0) {
        if (opts->verbose) printf("Ensuring module ASTs are cached for declaration generation...\n");
        for (int i = 0; i < modules->count; i++) {
            const char *module_path = modules->module_paths[i];
            if (module_path) {
                /* Load module into cache (won't re-parse if already loaded) */
                ASTNode *module_ast = load_module(module_path, env);
                if (!module_ast) {
                    fprintf(stderr, "Warning: Failed to load module '%s' for declaration generation\n", module_path);
                }
            }
        }
        if (opts->verbose) printf("✓ Module ASTs cached\n");
    }

    /* Phase 6: C Transpilation */
    if (opts->verbose) printf("Transpiling to C...\n");
    char *c_code = transpile_to_c(program, env, input_file);
    if (!c_code) {
        fprintf(stderr, "Transpilation failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_TRANSPILER, DiagnosticSeverity_DIAG_ERROR, "CTRANS01", "Transpilation failed");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        ffi_cleanup();
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }
    
    /* Calculate C code size */
    size_t c_code_size = strlen(c_code);
    int c_code_lines = 1;
    for (size_t i = 0; i < c_code_size; i++) {
        if (c_code[i] == '\n') c_code_lines++;
    }
    
    if (opts->verbose) {
        printf("✓ Transpilation complete (%d lines of C, %zu bytes)\n", c_code_lines, c_code_size);
    }

    if (opts->show_intermediate_code) {
        fwrite(c_code, 1, c_code_size, stdout);
        fflush(stdout);
    }
    
    /* Save generated C to .genC file if -S flag is set */
    if (opts->save_asm) {
        char gen_c_file[512];
        snprintf(gen_c_file, sizeof(gen_c_file), "%s.genC", input_file);
        FILE *gen_c = fopen(gen_c_file, "w");
        if (gen_c) {
            fprintf(gen_c, "%s", c_code);
            fclose(gen_c);
            if (opts->verbose) {
                printf("✓ Saved generated C to: %s\n", gen_c_file);
            }
        } else {
            fprintf(stderr, "Warning: Could not save generated C to %s\n", gen_c_file);
        }
    }

    /* Write C code to temporary file in /tmp (or keep in output dir if --keep-c) */
    char temp_c_file[512];
    if (opts->keep_c) {
        /* Keep in output directory if --keep-c is set */
        snprintf(temp_c_file, sizeof(temp_c_file), "%s.c", output_file);
    } else {
        /* Use TMPDIR (or /tmp) for temporary files */
        snprintf(temp_c_file, sizeof(temp_c_file), "%s/nanoc_%d_%s.c",
                 get_tmp_dir(), (int)getpid(), strrchr(output_file, '/') ? strrchr(output_file, '/') + 1 : output_file);
    }

    FILE *c_file = fopen(temp_c_file, "w");
    if (!c_file) {
        fprintf(stderr, "Error: Could not create C file '%s'\n", temp_c_file);
        diags_push_simple(diags, CompilerPhase_PHASE_TRANSPILER, DiagnosticSeverity_DIAG_ERROR, "CC01", "Could not create temporary C file");
        free(c_code);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }

    fprintf(c_file, "%s", c_code);
    fclose(c_file);
    if (opts->verbose) printf("✓ Generated C code: %s\n", temp_c_file);

    char compile_cmd[16384];  /* Increased to handle long command lines with many modules */
    
    /* Build include flags */
    char include_flags[8192];
    snprintf(include_flags, sizeof(include_flags), "-I%s/src", get_project_root());
    for (int i = 0; i < opts->include_count; i++) {
        char temp[512];
        snprintf(temp, sizeof(temp), " -I%s", opts->include_paths[i]);
        strncat(include_flags, temp, sizeof(include_flags) - strlen(include_flags) - 1);
    }
    
    /* Add module directories to include path (for FFI headers) */
    /* This enables standalone tools to import modules like "modules/std/fs.nano" */
    /* and have the C compiler find the corresponding "fs.h" header */
    if (modules && modules->count > 0) {
        /* Track unique directories to avoid duplicates */
        char **unique_dirs = malloc(sizeof(char*) * modules->count);
        int unique_count = 0;
        
        for (int i = 0; i < modules->count; i++) {
            const char *module_path = modules->module_paths[i];
            if (!module_path) continue;
            
            /* Extract directory from module path */
            char dir_path[512];
            strncpy(dir_path, module_path, sizeof(dir_path) - 1);
            dir_path[sizeof(dir_path) - 1] = '\0';
            
            /* Find last slash to get directory */
            char *last_slash = strrchr(dir_path, '/');
            if (last_slash) {
                *last_slash = '\0';  /* Trim filename to get directory */
                
                /* Check if this directory is already in the list */
                bool already_added = false;
                for (int j = 0; j < unique_count; j++) {
                    if (strcmp(unique_dirs[j], dir_path) == 0) {
                        already_added = true;
                        break;
                    }
                }
                
                if (!already_added) {
                    unique_dirs[unique_count] = strdup(dir_path);
                    unique_count++;
                    
                    /* Add -I flag for this directory */
                    char temp[1024];
                    snprintf(temp, sizeof(temp), " -I%s", dir_path);
                    strncat(include_flags, temp, sizeof(include_flags) - strlen(include_flags) - 1);
                    
                    if (opts->verbose) {
                        printf("Adding module include path: %s\n", dir_path);
                    }
                }
            }
        }
        
        /* Free unique_dirs */
        for (int i = 0; i < unique_count; i++) {
            free(unique_dirs[i]);
        }
        free(unique_dirs);
    }
    
    /* Add module compile flags (include paths from pkg-config) */
    if (module_compile_flags[0] != '\0') {
        strncat(include_flags, " ", sizeof(include_flags) - strlen(include_flags) - 1);
        strncat(include_flags, module_compile_flags, sizeof(include_flags) - strlen(include_flags) - 1);
    }
    
    /* Build library path flags */
    char lib_path_flags[2048] = "";
    for (int i = 0; i < opts->library_path_count; i++) {
        char temp[512];
        snprintf(temp, sizeof(temp), " -L%s", opts->library_paths[i]);
        strncat(lib_path_flags, temp, sizeof(lib_path_flags) - strlen(lib_path_flags) - 1);
    }
    
    /* Build library flags */
    char lib_flags[2048] = "-lm";
    for (int i = 0; i < opts->library_count; i++) {
        char temp[512];
        snprintf(temp, sizeof(temp), " -l%s", opts->libraries[i]);
        strncat(lib_flags, temp, sizeof(lib_flags) - strlen(lib_flags) - 1);
    }
    
    /* Detect and generate generic list types from the C code AND compiler_schema.h */
    char generated_lists[1024] = "";
    char detected_types[64][64]; /* Increased to handle more types */
    int detected_count = 0;
    
    /* First, scan compiler_schema.h if it exists */
    FILE *schema_h = fopen("src/generated/compiler_schema.h", "r");
    if (schema_h) {
        fseek(schema_h, 0, SEEK_END);
        long size = ftell(schema_h);
        fseek(schema_h, 0, SEEK_SET);
        char *schema_content = malloc(size + 1);
        if (schema_content) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
            fread(schema_content, 1, size, schema_h);
#pragma GCC diagnostic pop
            schema_content[size] = '\0';
            
            const char *ptr = schema_content;
            while ((ptr = strstr(ptr, "List_")) != NULL) {
                ptr += 5;
                const char *end = ptr;
                while ((*end >= 'A' && *end <= 'Z') || (*end >= 'a' && *end <= 'z') || (*end >= '0' && *end <= '9') || *end == '_') {
                    end++;
                }
                if (*end == '*' || *end == ' ' || *end == '\n' || *end == ';') {
                    int len = end - ptr;
                    char type_name[64];
                    strncpy(type_name, ptr, len);
                    type_name[len] = '\0';
                    
                    if (strcmp(type_name, "int") != 0 && strcmp(type_name, "string") != 0 && strcmp(type_name, "token") != 0 && strcmp(type_name, "Generic") != 0) {
                        bool found = false;
                        for (int i = 0; i < detected_count; i++) {
                            if (strcmp(detected_types[i], type_name) == 0) {
                                found = true;
                                break;
                            }
                        }
                        if (!found && detected_count < 64) {
                            strcpy(detected_types[detected_count++], type_name);
                        }
                    }
                }
            }
            free(schema_content);
        }
        fclose(schema_h);
    }

    const char *scan_ptr = c_code;
    /* Scan for List_TypeName* patterns in generated code too */
    while ((scan_ptr = strstr(scan_ptr, "List_")) != NULL) {
        scan_ptr += 5; /* Skip "List_" */
        const char *end_ptr = scan_ptr;
        
        /* Extract type name (alphanumeric + underscore) */
        while ((*end_ptr >= 'A' && *end_ptr <= 'Z') || 
               (*end_ptr >= 'a' && *end_ptr <= 'z') || 
               (*end_ptr >= '0' && *end_ptr <= '9') || 
               *end_ptr == '_') {
            end_ptr++;
        }
        
        /* Check if followed by * or space (valid list type) */
        if (*end_ptr == '*' || *end_ptr == ' ' || *end_ptr == '\n') {
            int len = end_ptr - scan_ptr;
            char type_name[64];
            strncpy(type_name, scan_ptr, len);
            type_name[len] = '\0';
            
            /* Skip built-in types */
            if (strcmp(type_name, "int") == 0 || 
                strcmp(type_name, "string") == 0 || 
                strcmp(type_name, "token") == 0) {
                continue;
            }
            
            /* Check if already detected */
            bool already_detected = false;
            for (int i = 0; i < detected_count; i++) {
                if (strcmp(detected_types[i], type_name) == 0) {
                    already_detected = true;
                    break;
                }
            }
            
            if (!already_detected && detected_count < 32) {
                snprintf(detected_types[detected_count], 64, "%s", type_name);
                detected_count++;
                
                /* Check if this type already has a runtime list implementation.
                 * If so, skip wrapper generation to avoid duplicate symbols —
                 * the runtime file is already linked via runtime_basenames[]. */
                char runtime_list_path[8192];
                snprintf(runtime_list_path, sizeof(runtime_list_path),
                         "%s/src/runtime/list_%s.c", get_project_root(), type_name);
                if (access(runtime_list_path, F_OK) == 0) {
                    if (opts->verbose) {
                        printf("Skipping List<%s> wrapper — runtime implementation exists at %s\n",
                               type_name, runtime_list_path);
                    }
                    continue;
                }
                
                /* Check if the transpiler already generated inline list code */
                char inline_check[128];
                snprintf(inline_check, sizeof(inline_check), "nl_list_%s_new", type_name);
                if (strstr(c_code, inline_check) != NULL) {
                    if (opts->verbose) {
                        printf("Skipping List<%s> wrapper — inline implementation in transpiled code\n",
                               type_name);
                    }
                    continue;
                }
                
                /* Generate list runtime files */
                const char *c_type = type_name;
                char nl_prefixed_type[128];
                if (strcmp(type_name, "LexerToken") == 0) c_type = "Token";
                else if (strcmp(type_name, "NSType") == 0) c_type = "NSType";
                else if (strncmp(type_name, "AST", 3) == 0 || strncmp(type_name, "Compiler", 8) == 0) {
                    c_type = type_name;
                } else {
                    snprintf(nl_prefixed_type, sizeof(nl_prefixed_type), "nl_%s", type_name);
                    c_type = nl_prefixed_type;
                }
                
                char gen_cmd[8192];
                snprintf(gen_cmd, sizeof(gen_cmd),
                        "%s/scripts/generate_list.sh %s %s %s > /dev/null 2>&1",
                        get_project_root(), type_name, get_tmp_dir(), c_type);
                if (opts->verbose) {
                    printf("Generating List<%s> runtime...\n", type_name);
                }
                int gen_result = system(gen_cmd);
                if (gen_result != 0 && opts->verbose) {
                    fprintf(stderr, "Warning: Failed to generate list_%s runtime\n", type_name);
                }
                
                /* Create wrapper that includes struct definition */
                char wrapper_file[512];
                snprintf(wrapper_file, sizeof(wrapper_file), "%s/list_%s_wrapper.c", get_tmp_dir(), type_name);
                FILE *wrapper = fopen(wrapper_file, "w");
                if (wrapper) {
                    /* Extract struct definition from generated C code */
                    const char *struct_search = c_code;
                    char struct_pattern[128];
                    snprintf(struct_pattern, sizeof(struct_pattern), "typedef struct nl_%s {", type_name);
                    const char *struct_start = strstr(struct_search, struct_pattern);
                    const char *struct_start_original = struct_start;
                    
                    /* Try to find guards if they exist */
                    char guard_pattern[128];
                    snprintf(guard_pattern, sizeof(guard_pattern), "#ifndef DEFINED_nl_%s", type_name);
                    const char *guard_start = strstr(struct_search, guard_pattern);
                    if (guard_start && guard_start < struct_start) {
                        struct_start = guard_start;
                    }
                    
                    if (struct_start) {
                        /* Find the end of the struct (closing brace + semicolon) */
                        const char *struct_end = struct_start;
                        int brace_count = 0;
                        bool found_open_brace = false;
                        while (*struct_end) {
                            if (*struct_end == '{') {
                                found_open_brace = true;
                                brace_count++;
                            } else if (*struct_end == '}' && found_open_brace) {
                                brace_count--;
                                if (brace_count == 0) {
                                    /* Found the closing brace, look for semicolon */
                                    struct_end++;
                                    while (*struct_end && *struct_end != ';') struct_end++;
                                    if (*struct_end == ';') struct_end++;
                                    
                                    /* Look for #endif if we started with a guard */
                                    if (guard_start && guard_start < struct_start_original) {
                                        const char *endif_search = strstr(struct_end, "#endif");
                                        if (endif_search) {
                                            struct_end = endif_search + 6;
                                        }
                                    }
                                    break;
                                }
                            }
                            struct_end++;
                        }
                        
                        /* Write the wrapper */
                        fprintf(wrapper, "#include <stdint.h>\n");
                        fprintf(wrapper, "#include <stdbool.h>\n");
                        fprintf(wrapper, "#include <stdlib.h>\n");
                        fprintf(wrapper, "#include <stdio.h>\n");
                        fprintf(wrapper, "#include <string.h>\n\n");
                        /* Needed for DynArray/nl_string_t/Token used by extracted structs */
                        fprintf(wrapper, "#include \"nanolang.h\"\n");
                        fprintf(wrapper, "#include \"generated/compiler_schema.h\"\n\n");
                        fprintf(wrapper, "/* Struct definition extracted from main file */\n");
                        fprintf(wrapper, "%.*s\n", (int)(struct_end - struct_start), struct_start);
                        /* Set guard macro to prevent typedef redefinition in list header */
                        char type_upper[128];
                        strncpy(type_upper, type_name, sizeof(type_upper) - 1);
                        type_upper[sizeof(type_upper) - 1] = '\0';
                        for (char *p = type_upper; *p; p++) {
                            *p = (char)toupper((unsigned char)*p);
                        }
                        fprintf(wrapper, "\n/* Guard macro set - typedef already defined above */\n");
                        fprintf(wrapper, "#define NL_%s_DEFINED\n\n", type_upper);
                        fprintf(wrapper, "/* Include list implementation */\n");
                        fprintf(wrapper, "#include \"%s/list_%s.c\"\n", get_tmp_dir(), type_name);
                    } else {
                        /* Fallback: just include the list file.
                         * We include schema headers in case it's a schema type. */
                        fprintf(wrapper, "#include <stdint.h>\n");
                        fprintf(wrapper, "#include <stdbool.h>\n");
                        fprintf(wrapper, "#include <stdlib.h>\n");
                        fprintf(wrapper, "#include <stdio.h>\n");
                        fprintf(wrapper, "#include <string.h>\n\n");
                        fprintf(wrapper, "#include \"nanolang.h\"\n");
                        fprintf(wrapper, "#include \"generated/compiler_schema.h\"\n\n");
                        fprintf(wrapper, "#include \"%s/list_%s.c\"\n", get_tmp_dir(), type_name);
                    }
                    fclose(wrapper);
                }
                
                /* Add wrapper to compile list */
                char list_file[256];
                snprintf(list_file, sizeof(list_file), " %s/list_%s_wrapper.c", get_tmp_dir(), type_name);
                strncat(generated_lists, list_file, sizeof(generated_lists) - strlen(generated_lists) - 1);
            }
        }
    }
    
    if (opts->verbose && detected_count > 0) {
        printf("Detected %d generic list type(s): ", detected_count);
        for (int i = 0; i < detected_count; i++) {
            printf("%s%s", detected_types[i], i < detected_count - 1 ? ", " : "");
        }
        printf("\n");
    }
    
    /* Build runtime files string (paths relative to project root) */
    /* Note: sdl_helpers.c is NOT included here - it's provided by the sdl_helpers module */
    static const char *runtime_basenames[] = {
        "runtime/list_int.c", "runtime/list_string.c", "runtime/list_LexerToken.c",
        "runtime/list_token.c", "runtime/list_CompilerDiagnostic.c",
        "runtime/list_CompilerSourceLocation.c", "runtime/list_ASTNumber.c",
        "runtime/list_ASTFloat.c", "runtime/list_ASTString.c", "runtime/list_ASTBool.c",
        "runtime/list_ASTIdentifier.c", "runtime/list_ASTBinaryOp.c",
        "runtime/list_ASTCall.c", "runtime/list_ASTModuleQualifiedCall.c",
        "runtime/list_ASTArrayLiteral.c", "runtime/list_ASTLet.c", "runtime/list_ASTSet.c",
        "runtime/list_ASTStmtRef.c", "runtime/list_ASTIf.c", "runtime/list_ASTWhile.c",
        "runtime/list_ASTFor.c", "runtime/list_ASTReturn.c", "runtime/list_ASTBlock.c",
        "runtime/list_ASTUnsafeBlock.c", "runtime/list_ASTPrint.c",
        "runtime/list_ASTAssert.c", "runtime/list_ASTFunction.c",
        "runtime/list_ASTShadow.c", "runtime/list_ASTStruct.c",
        "runtime/list_ASTStructLiteral.c", "runtime/list_ASTFieldAccess.c",
        "runtime/list_ASTEnum.c", "runtime/list_ASTUnion.c",
        "runtime/list_ASTUnionConstruct.c", "runtime/list_ASTMatch.c",
        "runtime/list_ASTImport.c", "runtime/list_ASTOpaqueType.c",
        "runtime/list_ASTTupleLiteral.c", "runtime/list_ASTTupleIndex.c",
        "runtime/token_helpers.c", "runtime/gc.c", "runtime/dyn_array.c",
        "runtime/gc_struct.c", "runtime/nl_string.c", "runtime/cli.c", "runtime/regex.c",
        "coroutine.c",
        NULL
    };
    char runtime_files[65536];
    runtime_files[0] = '\0';
    for (int i = 0; runtime_basenames[i]; i++) {
        char entry[8192];
        snprintf(entry, sizeof(entry), "%s%s/src/%s",
                 i > 0 ? " " : "", get_project_root(), runtime_basenames[i]);
        strncat(runtime_files, entry, sizeof(runtime_files) - strlen(runtime_files) - 1);
    }
    strncat(runtime_files, generated_lists, sizeof(runtime_files) - strlen(runtime_files) - 1);

    /* Add TMPDIR to include path for generated list headers */
    char include_flags_with_tmp[2560];
    snprintf(include_flags_with_tmp, sizeof(include_flags_with_tmp), "%s -I%s", include_flags, get_tmp_dir());
    
    const char *cc = getenv("NANO_CC");
    if (!cc) cc = getenv("CC");
    if (!cc) cc = "cc";

    const char *export_dynamic_flag = "";
#ifdef __linux__
    export_dynamic_flag = "-rdynamic";
#elif defined(__FreeBSD__)
    export_dynamic_flag = "-Wl,-E";
#endif

    /* Profiling flags for gprof support (-pg option) */
    const char *profile_flags = "";
    if (opts->profile_gprof) {
        profile_flags = "-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls";
        if (opts->verbose) {
            printf("Profiling enabled: adding %s\n", profile_flags);
        }
    }

    /* Coverage flags for gcov/lcov line+branch coverage (--coverage option) */
    const char *coverage_flags = "";
    if (opts->coverage) {
        coverage_flags = "-fprofile-arcs -ftest-coverage -fno-inline";
        if (opts->verbose) {
            printf("Coverage instrumentation enabled: adding %s\n", coverage_flags);
        }
    }

    int cmd_len = snprintf(compile_cmd, sizeof(compile_cmd),
            "%s -std=c99 -Wall -Wextra -Werror -Wno-error=unused-function -Wno-error=unused-parameter -Wno-error=unused-variable -Wno-error=unused-but-set-variable -Wno-error=logical-not-parentheses -Wno-error=duplicate-decl-specifier %s %s %s %s -o %s %s %s %s %s %s",
            cc, profile_flags, coverage_flags, include_flags_with_tmp, export_dynamic_flag, output_file, temp_c_file, module_objs, runtime_files, lib_path_flags, lib_flags);
    
    if (cmd_len >= (int)sizeof(compile_cmd)) {
        fprintf(stderr, "Error: Compile command too long (%d bytes, max %zu)\n", cmd_len, sizeof(compile_cmd));
        fprintf(stderr, "Try reducing the number of modules or shortening paths.\n");
        diags_push_simple(diags, CompilerPhase_PHASE_TRANSPILER, DiagnosticSeverity_DIAG_ERROR, "CCC02", "C compile command too long");
        free(c_code);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        if (!opts->keep_c) {
            remove(temp_c_file);
        }
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }

    if (opts->verbose) printf("Compiling C code: %s\n", compile_cmd);
    int result = system(compile_cmd);

    if (result == 0) {
        if (deterministic_outputs_enabled()) {
#ifdef __APPLE__
            (void)determinize_macho_uuid_and_signature(output_file);
#endif
        }
        if (opts->verbose) printf("✓ Compilation successful: %s\n", output_file);
    } else {
        fprintf(stderr, "C compilation failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_TRANSPILER, DiagnosticSeverity_DIAG_ERROR, "CCC01", "C compilation failed");
        /* Cleanup */
        free(c_code);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        /* Remove temporary C file unless --keep-c */
        if (!opts->keep_c) {
            remove(temp_c_file);
        }
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;  /* Return error if C compilation failed */
    }

    /* Remove temporary C file unless --keep-c (cleanup on both success and failure) */
    if (!opts->keep_c) {
        remove(temp_c_file);
    }

    llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 0, diags);
    llm_emit_diags_toon(opts->llm_diags_toon_path, input_file, output_file, 0, diags);
    nl_list_CompilerDiagnostic_free(diags);

    /* Cleanup */
    free(c_code);
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free_module_list(modules);
    clear_module_cache();  /* Free all cached module ASTs */
    free(source);
    ffi_cleanup();

    return 0;  /* Success */
}

/* Main entry point */
int main(int argc, char *argv[]) {
    /* Store argc/argv for runtime access */
    g_argc = argc;
    g_argv = argv;

    /* Resolve project root from binary location (enables compilation from any CWD) */
    resolve_project_root(argv[0]);
    
    /* Handle --version */
    if (argc >= 2 && (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0)) {
        printf("nanoc %s\n", NANOLANG_VERSION);
        printf("nanolang compiler\n");
        printf("Built: %s %s\n", NANOLANG_BUILD_DATE, NANOLANG_BUILD_TIME);
        return 0;
    }

    /* Handle --help */
    if (argc >= 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("nanoc - Compiler for the nanolang programming language\n\n");
        printf("Usage: %s <input.nano> [OPTIONS]\n\n", argv[0]);
        printf("Options:\n");
        printf("  -o <file>      Specify output file (default: $TMPDIR/nanoc_a.out)\n");
        printf("  --verbose      Show detailed compilation steps and commands\n");
        printf("                 (also enabled by NANO_VERBOSE_BUILD=1 env var)\n");
        printf("  --keep-c       Keep generated C file (saves to output dir instead of /tmp)\n");
        printf("  -fshow-intermediate-code  Print generated C to stdout\n");
        printf("  -S             Save generated C to <input>.genC (for inspection)\n");
        printf("  --json-errors  Output errors in JSON format for tool integration\n");
        printf("  --reflect <path>  Emit module API as JSON (for documentation generation)\n");
        printf("  -I <path>      Add include path for C compilation\n");
        printf("  -L <path>      Add library path for C linking\n");
        printf("  -l <lib>       Link against library (e.g., -lSDL2)\n");
        printf("  -pg            Enable gprof profiling (adds -g -fno-omit-frame-pointer)\n");
        printf("  --profile      Inject timing hooks; print hotspot report (sorted by total time)\n");
        printf("  --profile-output <p>  Write structured profiling JSON to file <p> (use with -pg)\n");
        printf("  --coverage     Instrument compiled output for gcov/lcov line+branch coverage\n");
        printf("                 Run: gcov <source.c>, or use 'make coverage-report' for HTML\n");
        printf("  --profile-runtime     Implies --profile; also write flamegraph collapsed-stack\n");
        printf("                 .nano.prof (default: <output_binary>.nano.prof). Compatible with\n");
        printf("                 flamegraph.pl: flamegraph.pl <bin>.nano.prof > flame.svg\n");
        printf("  --profile-runtime-output <p>  Set explicit path for flamegraph .nano.prof output\n");
        printf("  --target <t>   Compile target: native (default), wasm, ptx, opencl, c, riscv\n");
        printf("                 ptx:    emit NVIDIA PTX assembly for `gpu fn` functions\n");
        printf("                 opencl: emit OpenCL C kernel source (.cl) for `gpu fn` functions\n");
        printf("                         CPU fallback via POCL when no GPU present\n");
        printf("  --tco          Enable tail-call optimization (rewrite tail recursion to loops)\n");
        printf("  --llvm         Emit LLVM IR (.ll) instead of transpiled C\n");
        printf("  --debug / -g   Emit DWARF v4 debug info (with --llvm or --target riscv)\n");
        printf("                 Compile: llc -march=aarch64 prog.ll -o prog.s\n");
        printf("  --bench        Run @bench-annotated functions (micro-benchmark mode)\n");
        printf("  --bench-n <N>  Fixed iteration count (default: auto-calibrate to ~1s)\n");
        printf("  --bench-json <f> Write JSON benchmark results to file\n");
        printf("                          clang -O2 prog.ll -o prog\n");
        printf("  --pgo <file>   Profile-guided inlining: read .nano.prof from --profile-runtime,\n");
        printf("                 identify hot call sites and inline them. Combines with --tco.\n");
        printf("                 Example: nanoc --profile-runtime prog.nano -o prog\n");
        printf("                          ./prog                    # generates prog.nano.prof\n");
        printf("                          nanoc --pgo prog.nano.prof prog.nano -o prog_opt\n");
        printf("  publish <file> Compile to WASM and publish to AgentFS (nanoc-publish.sh)\n");
        printf("  install [pkg@range ...]\n");
        printf("                 Install nano packages from the registry (nanoc-install.sh)\n");
        printf("                 Reads nano.packages.json + writes nano.lock\n");
        printf("                 Example: nanoc install gpu-math@^1.0.0 nano-core@latest\n");
        printf("  --version, -v  Show version information\n");
        printf("  --help, -h     Show this help message\n");
        printf("\nVerification Options:\n");
        printf("  --trust-report         Show formal verification trust levels for all functions\n");
        printf("  --reference-eval       Cross-check verified functions with Coq-extracted interpreter\n");
        printf("\nSafety Options:\n");
        printf("  --warn-unsafe-imports  Warn when importing unsafe modules\n");
        printf("  --warn-unsafe-calls    Warn when calling functions from unsafe modules\n");
        printf("  --warn-ffi             Warn on any FFI (extern function) call\n");
        printf("  --forbid-unsafe        Error (not warn) on unsafe module imports\n");
        printf("\nAgent Options:\n");
        printf("  --llm-diags-json <p>   Write machine-readable diagnostics JSON (agent-only)\n");
        printf("  --llm-diags-toon <p>   Write diagnostics in TOON format (~40%% fewer tokens)\n");
        printf("  --llm-shadow-json <p>  Write machine-readable shadow failure summary JSON (agent-only)\n");
        printf("\nExamples:\n");
        printf("  %s hello.nano -o hello\n", argv[0]);
        printf("  %s program.nano --verbose -S          # Show steps and save C code\n", argv[0]);
        printf("  %s example.nano -o example --verbose\n", argv[0]);
        printf("  %s sdl_app.nano -o app -I/opt/homebrew/include/SDL2 -L/opt/homebrew/lib -lSDL2\n\n", argv[0]);
        return 0;
    }

    /* Handle 'publish' subcommand — delegates to scripts/nanoc-publish.sh */
    if (argc >= 2 && strcmp(argv[1], "publish") == 0) {
        /* Find the publish script relative to the binary.
         * Use dynamic allocation to avoid -Werror=format-truncation issues. */
        const char *suffix = "/scripts/nanoc-publish.sh";
        char *publish_script = NULL;
        const char *nano_root = getenv("NANOLANG_ROOT");
        if (nano_root) {
            size_t len = strlen(nano_root) + strlen(suffix) + 1;
            publish_script = malloc(len);
            if (!publish_script) { perror("malloc"); return 1; }
            strcpy(publish_script, nano_root);
            strcat(publish_script, suffix);
        } else {
            /* Derive from argv[0]: strip last path component to get bin dir */
            char *argv0_copy = strdup(argv[0]);
            if (!argv0_copy) { perror("strdup"); return 1; }
            char *last_slash = strrchr(argv0_copy, '/');
            if (last_slash) {
                *last_slash = '\0';
                size_t len = strlen(argv0_copy) + strlen(suffix) + 1;
                publish_script = malloc(len);
                if (!publish_script) { perror("malloc"); free(argv0_copy); return 1; }
                strcpy(publish_script, argv0_copy);
                strcat(publish_script, suffix);
            } else {
                publish_script = strdup("./scripts/nanoc-publish.sh");
                if (!publish_script) { perror("strdup"); return 1; }
            }
            free(argv0_copy);
        }
        /* Build argv for execv: [publish_script, argv[2], argv[3], ..., NULL] */
        const char **script_argv = malloc(((size_t)argc + 1) * sizeof(char *));
        if (!script_argv) { perror("malloc"); free(publish_script); return 1; }
        script_argv[0] = publish_script;
        for (int i = 2; i < argc; i++) {
            script_argv[i - 1] = argv[i];
        }
        script_argv[argc - 1] = NULL;
        execv(publish_script, (char *const *)script_argv);
        /* execv only returns on error */
        fprintf(stderr, "nanoc publish: could not exec %s: %s\n", publish_script, strerror(errno));
        fprintf(stderr, "Ensure nanoc-publish.sh is in scripts/ relative to NANOLANG_ROOT.\n");
        free(script_argv);
        free(publish_script);
        return 1;
    }

    /* Handle 'install' subcommand — delegates to scripts/nanoc-install.sh */
    if (argc >= 2 && strcmp(argv[1], "install") == 0) {
        const char *install_suffix = "/scripts/nanoc-install.sh";
        char *install_script = NULL;
        const char *nano_root = getenv("NANOLANG_ROOT");
        if (nano_root) {
            size_t len = strlen(nano_root) + strlen(install_suffix) + 1;
            install_script = malloc(len);
            if (install_script) { strcpy(install_script, nano_root); strcat(install_script, install_suffix); }
        }
        if (!install_script || 0) {
            /* Fall back: look relative to argv[0] */
            free(install_script);
            char *argv0_copy = strdup(argv[0]);
            char *last_slash = strrchr(argv0_copy, '/');
            if (last_slash) { *last_slash = '\0';
                size_t len = strlen(argv0_copy) + strlen(install_suffix) + 1;
                install_script = malloc(len);
                if (install_script) { strcpy(install_script, argv0_copy); strcat(install_script, install_suffix); }
            }
            free(argv0_copy);
            if (!install_script) install_script = strdup("./scripts/nanoc-install.sh");
        }
        /* Build argv for install script */
        const char **script_argv = malloc((size_t)(argc + 1) * sizeof(char *));
        if (!script_argv) { perror("malloc"); free(install_script); return 1; }
        script_argv[0] = install_script;
        for (int i = 2; i < argc; i++) script_argv[i - 1] = argv[i];
        script_argv[argc - 1] = NULL;
        execv(install_script, (char *const *)script_argv);
        fprintf(stderr, "nanoc install: could not exec %s: %s\n", install_script, strerror(errno));
        fprintf(stderr, "Ensure nanoc-install.sh is in scripts/ relative to NANOLANG_ROOT.\n");
        free(script_argv);
        free(install_script);
        return 1;
    }

    /* Handle 'sign' subcommand — Ed25519 WASM module signing */
    if (argc >= 2 && strcmp(argv[1], "sign") == 0) {
        return nanoc_sign_cmd(argc - 2, (char **)(argv + 2));
    }

    /* Handle 'verify' subcommand — Ed25519 WASM signature verification */
    if (argc >= 2 && strcmp(argv[1], "verify") == 0) {
        return nanoc_verify_cmd(argc - 2, (char **)(argv + 2));
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.nano> [OPTIONS]\n", argv[0]);
        fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    /* Default output to TMPDIR (or /tmp) to avoid polluting project dir */
    char default_output[512];
    snprintf(default_output, sizeof(default_output), "%s/nanoc_a.out", get_tmp_dir());
    const char *output_file = default_output;
    CompilerOptions opts = {
        .verbose = false,
        .keep_c = false,
        .show_intermediate_code = false,
        .save_asm = false,
        .json_errors = false,
        .profile_gprof = false,
        .profile = false,
        .coverage = false,
        .profile_runtime = false,

        .profile_output_path = NULL,
        .profile_flamegraph_path = NULL,
        .llm_diags_json_path = NULL,
        .llm_diags_toon_path = NULL,
        .llm_shadow_json_path = NULL,
        .reflect_output_path = NULL,
        .include_paths = NULL,
        .include_count = 0,
        .library_paths = NULL,
        .library_path_count = 0,
        .libraries = NULL,
        .library_count = 0,
        .warn_unsafe_imports = false,
        .warn_unsafe_calls = false,
        .warn_ffi = false,
        .forbid_unsafe = false,
        .trust_report = false,
        .reference_eval = false,
        .target = NULL,
        .no_sourcemap = false,
        .pgo_profile = NULL,
        .llvm = false,
        .debug = false,
        .doc_md = false
    };
    
    /* Allocate arrays for flags */
    char **include_paths = malloc(sizeof(char*) * 32);
    char **library_paths = malloc(sizeof(char*) * 32);
    char **libraries = malloc(sizeof(char*) * 32);
    int include_count = 0;
    int library_path_count = 0;
    int library_count = 0;

    /* Parse command-line options */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = true;
        } else if (strcmp(argv[i], "--keep-c") == 0) {
            opts.keep_c = true;
        } else if (strcmp(argv[i], "-fshow-intermediate-code") == 0) {
            opts.show_intermediate_code = true;
        } else if (strcmp(argv[i], "-S") == 0) {
            opts.save_asm = true;
        } else if (strcmp(argv[i], "--json-errors") == 0) {
            opts.json_errors = true;
        } else if (strcmp(argv[i], "-pg") == 0) {
            opts.profile_gprof = true;
        } else if (strcmp(argv[i], "--profile") == 0) {
            opts.profile = true;

        } else if (strcmp(argv[i], "--coverage") == 0) {
            opts.coverage = true;

        } else if (strcmp(argv[i], "--profile-runtime") == 0) {
            opts.profile = true;        /* --profile-runtime implies --profile */
            opts.profile_runtime = true;
            /* path via --profile-runtime-output; leave flamegraph_path NULL for auto-derive */

        } else if (strcmp(argv[i], "--profile-runtime-output") == 0 && i + 1 < argc) {
            opts.profile_flamegraph_path = argv[i + 1];
            i++;

        } else if (strcmp(argv[i], "--profile-output") == 0 && i + 1 < argc) {
            opts.profile_output_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-I") == 0 && i + 1 < argc) {
            if (include_count < 32) {
                include_paths[include_count++] = argv[i + 1];
            }
            i++;
        } else if (strncmp(argv[i], "-I", 2) == 0) {
            /* Handle -I/path form */
            if (include_count < 32) {
                include_paths[include_count++] = argv[i] + 2;
            }
        } else if (strcmp(argv[i], "-L") == 0 && i + 1 < argc) {
            if (library_path_count < 32) {
                library_paths[library_path_count++] = argv[i + 1];
            }
            i++;
        } else if (strncmp(argv[i], "-L", 2) == 0) {
            /* Handle -L/path form */
            if (library_path_count < 32) {
                library_paths[library_path_count++] = argv[i] + 2;
            }
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            if (library_count < 32) {
                libraries[library_count++] = argv[i + 1];
            }
            i++;
        } else if (strncmp(argv[i], "-l", 2) == 0) {
            /* Handle -llibname form */
            if (library_count < 32) {
                libraries[library_count++] = argv[i] + 2;
            }
        } else if (strcmp(argv[i], "--warn-unsafe-imports") == 0) {
            opts.warn_unsafe_imports = true;
        } else if (strcmp(argv[i], "--warn-unsafe-calls") == 0) {
            opts.warn_unsafe_calls = true;
        } else if (strcmp(argv[i], "--warn-ffi") == 0) {
            opts.warn_ffi = true;
        } else if (strcmp(argv[i], "--forbid-unsafe") == 0) {
            opts.forbid_unsafe = true;
        } else if (strcmp(argv[i], "--trust-report") == 0) {
            opts.trust_report = true;
        } else if (strcmp(argv[i], "--reference-eval") == 0) {
            opts.reference_eval = true;
        } else if (strcmp(argv[i], "--llm-diags-json") == 0 && i + 1 < argc) {
            opts.llm_diags_json_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--llm-diags-toon") == 0 && i + 1 < argc) {
            opts.llm_diags_toon_path = argv[i + 1];
            toon_diagnostics_enable();
            i++;
        } else if (strcmp(argv[i], "--llm-shadow-json") == 0 && i + 1 < argc) {
            opts.llm_shadow_json_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--reflect") == 0 && i + 1 < argc) {
            opts.reflect_output_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--emit-typed-ast-json") == 0) {
            opts.emit_typed_ast = true;
        } else if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            opts.target = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--no-sourcemap") == 0) {
            opts.no_sourcemap = true;
        } else if (strcmp(argv[i], "--tco") == 0) {
            opts.tco = true;
        } else if (strcmp(argv[i], "--pgo") == 0 && i + 1 < argc) {
            opts.pgo_profile = argv[++i];
        } else if (strcmp(argv[i], "--llvm") == 0) {
            opts.llvm = true;
        } else if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-g") == 0) {
            opts.debug = true;
        } else if (strcmp(argv[i], "--bench") == 0) {
            opts.bench = true;
        } else if (strcmp(argv[i], "--bench-n") == 0 && i + 1 < argc) {
            opts.bench_n = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--bench-json") == 0 && i + 1 < argc) {
            opts.bench_json = argv[++i];
        } else if (strcmp(argv[i], "--doc-md") == 0 ||
                   strcmp(argv[i], "-dm") == 0) {
            opts.doc_md = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
            free(include_paths);
            free(library_paths);
            free(libraries);
            return 1;
        }
    }
    
    /* Set parsed flags in options */
    opts.include_paths = include_paths;
    opts.include_count = include_count;
    opts.library_paths = library_paths;
    opts.library_path_count = library_path_count;
    opts.libraries = libraries;
    opts.library_count = library_count;
    
    /* Check for NANO_VERBOSE_BUILD environment variable */
    if (getenv("NANO_VERBOSE_BUILD")) {
        opts.verbose = true;
    }
    
    int result = compile_file(input_file, output_file, &opts);
    
    /* Cleanup */
    free(include_paths);
    free(library_paths);
    free(libraries);
    
    return result;
}