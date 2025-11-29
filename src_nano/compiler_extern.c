/* =============================================================================
 * Compiler Extern Functions - C Implementation
 * =============================================================================
 * Exposes C compiler internals to nanolang programs
 */

#include "../src/nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Opaque handles map to actual C types */
typedef Token* TokenHandle;
typedef ASTNode* ASTHandle;
typedef Environment* EnvHandle;

/* =============================================================================
 * PHASE 1: LEXING
 * ============================================================================= */

TokenHandle nl_compiler_tokenize(const char *source) {
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    
    /* Store count in first token's unused field for later retrieval */
    /* This is a hack but avoids needing a separate state structure */
    if (tokens && token_count > 0) {
        /* Store count at end of token array in a sentinel token */
        Token *extended = realloc(tokens, sizeof(Token) * (token_count + 1));
        if (extended) {
            extended[token_count].type = -1;  /* Sentinel */
            extended[token_count].line = token_count;  /* Store count */
            return extended;
        }
    }
    
    return tokens;
}

int nl_compiler_token_count(TokenHandle tokens) {
    if (!tokens) return 0;
    
    /* Find sentinel token with count */
    int i = 0;
    while (tokens[i].type != -1) {
        i++;
    }
    return tokens[i].line;  /* Count stored in sentinel */
}

void nl_compiler_free_tokens(TokenHandle tokens) {
    if (!tokens) return;
    
    int count = nl_compiler_token_count(tokens);
    free_tokens(tokens, count);
}

/* =============================================================================
 * PHASE 2: PARSING
 * ============================================================================= */

ASTHandle nl_compiler_parse(TokenHandle tokens, int token_count) {
    if (!tokens || token_count <= 0) return NULL;
    return parse_program(tokens, token_count);
}

void nl_compiler_free_ast(ASTHandle ast) {
    if (ast) free_ast(ast);
}

/* =============================================================================
 * PHASE 3: TYPE CHECKING
 * ============================================================================= */

EnvHandle nl_compiler_create_env(void) {
    return create_environment();
}

int nl_compiler_process_imports(ASTHandle ast, EnvHandle env, const char *input_file) {
    if (!ast || !env) return 0;
    
    clear_module_cache();
    ModuleList *modules = create_module_list();
    bool result = process_imports(ast, env, modules, input_file);
    free_module_list(modules);
    
    return result ? 1 : 0;
}

int nl_compiler_typecheck(ASTHandle ast, EnvHandle env) {
    if (!ast || !env) return 0;
    return type_check(ast, env) ? 1 : 0;
}

void nl_compiler_free_env(EnvHandle env) {
    if (env) free_environment(env);
}

/* =============================================================================
 * PHASE 4: SHADOW TESTS
 * ============================================================================= */

int nl_compiler_run_shadow_tests(ASTHandle ast, EnvHandle env) {
    if (!ast || !env) return 0;
    return run_shadow_tests(ast, env) ? 1 : 0;
}

/* =============================================================================
 * PHASE 5: CODE GENERATION
 * ============================================================================= */

char* nl_compiler_transpile(ASTHandle ast, EnvHandle env) {
    if (!ast || !env) return strdup("");
    
    char *code = transpile_to_c(ast, env);
    return code ? code : strdup("");
}

/* =============================================================================
 * PHASE 6: COMPILATION
 * ============================================================================= */

int nl_compiler_compile_c(const char *c_code, const char *output_file, int verbose) {
    if (!c_code || !output_file) return 1;
    
    /* Write C code to temporary file */
    char temp_c_file[512];
    snprintf(temp_c_file, sizeof(temp_c_file), "%s.temp.c", output_file);
    
    FILE *f = fopen(temp_c_file, "w");
    if (!f) {
        fprintf(stderr, "Error: Could not write temporary C file\n");
        return 1;
    }
    fprintf(f, "%s", c_code);
    fclose(f);
    
    /* Compile C to executable */
    char compile_cmd[2048];
    snprintf(compile_cmd, sizeof(compile_cmd),
        "gcc -std=c99 -Isrc %s -o %s -lm",
        temp_c_file, output_file);
    
    if (verbose) {
        printf("Compiling: %s\n", compile_cmd);
    }
    
    int result = system(compile_cmd);
    
    /* Clean up temp file */
    if (result == 0) {
        unlink(temp_c_file);
    }
    
    return result;
}

/* =============================================================================
 * ALL-IN-ONE COMPILATION
 * ============================================================================= */

int nl_compiler_compile_file(const char *input_file, const char *output_file, 
                             int verbose, int keep_c) {
    if (!input_file || !output_file) return 1;
    
    /* Read source */
    FILE *f = fopen(input_file, "r");
    if (!f) {
        fprintf(stderr, "Error: Could not open input file '%s'\n", input_file);
        return 1;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char *source = malloc(size + 1);
    fread(source, 1, size, f);
    source[size] = '\0';
    fclose(f);
    
    if (verbose) printf("Compiling %s...\n", input_file);
    
    /* Phase 1: Tokenize */
    TokenHandle tokens = nl_compiler_tokenize(source);
    free(source);
    
    if (!tokens) {
        fprintf(stderr, "Error: Lexing failed\n");
        return 1;
    }
    
    int token_count = nl_compiler_token_count(tokens);
    if (verbose) printf("✓ Lexing complete (%d tokens)\n", token_count);
    
    /* Phase 2: Parse */
    ASTHandle ast = nl_compiler_parse(tokens, token_count);
    if (!ast) {
        fprintf(stderr, "Error: Parsing failed\n");
        nl_compiler_free_tokens(tokens);
        return 1;
    }
    if (verbose) printf("✓ Parsing complete\n");
    
    /* Phase 3: Type check */
    EnvHandle env = nl_compiler_create_env();
    
    if (!nl_compiler_process_imports(ast, env, input_file)) {
        fprintf(stderr, "Error: Module loading failed\n");
        nl_compiler_free_ast(ast);
        nl_compiler_free_tokens(tokens);
        nl_compiler_free_env(env);
        return 1;
    }
    if (verbose) printf("✓ Imports processed\n");
    
    if (!nl_compiler_typecheck(ast, env)) {
        fprintf(stderr, "Error: Type checking failed\n");
        nl_compiler_free_ast(ast);
        nl_compiler_free_tokens(tokens);
        nl_compiler_free_env(env);
        return 1;
    }
    if (verbose) printf("✓ Type checking complete\n");
    
    /* Phase 4: Shadow tests */
    if (!nl_compiler_run_shadow_tests(ast, env)) {
        fprintf(stderr, "Error: Shadow tests failed\n");
        nl_compiler_free_ast(ast);
        nl_compiler_free_tokens(tokens);
        nl_compiler_free_env(env);
        return 1;
    }
    if (verbose) printf("✓ Shadow tests passed\n");
    
    /* Phase 5: Transpile */
    char *c_code = nl_compiler_transpile(ast, env);
    if (!c_code || strlen(c_code) == 0) {
        fprintf(stderr, "Error: Code generation failed\n");
        free(c_code);
        nl_compiler_free_ast(ast);
        nl_compiler_free_tokens(tokens);
        nl_compiler_free_env(env);
        return 1;
    }
    if (verbose) printf("✓ C code generated\n");
    
    /* Save C code if requested */
    if (keep_c) {
        char c_file[512];
        snprintf(c_file, sizeof(c_file), "%s.genC", input_file);
        FILE *cf = fopen(c_file, "w");
        if (cf) {
            fprintf(cf, "%s", c_code);
            fclose(cf);
            if (verbose) printf("✓ Saved C code to %s\n", c_file);
        }
    }
    
    /* Phase 6: Compile */
    int result = nl_compiler_compile_c(c_code, output_file, verbose);
    
    free(c_code);
    nl_compiler_free_ast(ast);
    nl_compiler_free_tokens(tokens);
    nl_compiler_free_env(env);
    
    if (result == 0 && verbose) {
        printf("✓ Compilation successful: %s\n", output_file);
    }
    
    return result;
}
