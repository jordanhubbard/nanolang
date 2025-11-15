#include "nanolang.h"
#include <sys/stat.h>
#include <unistd.h>

/* Create a new module list */
ModuleList *create_module_list(void) {
    ModuleList *list = malloc(sizeof(ModuleList));
    list->count = 0;
    list->capacity = 8;
    list->module_paths = malloc(sizeof(char*) * list->capacity);
    return list;
}

/* Free a module list */
void free_module_list(ModuleList *list) {
    if (!list) return;
    for (int i = 0; i < list->count; i++) {
        free(list->module_paths[i]);
    }
    free(list->module_paths);
    free(list);
}

/* Add a module path to the list */
void module_list_add(ModuleList *list, const char *module_path) {
    if (!list || !module_path) return;
    
    /* Check if already in list */
    for (int i = 0; i < list->count; i++) {
        if (strcmp(list->module_paths[i], module_path) == 0) {
            return;  /* Already added */
        }
    }
    
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->module_paths = realloc(list->module_paths, sizeof(char*) * list->capacity);
    }
    
    list->module_paths[list->count++] = strdup(module_path);
}

/* Resolve module path relative to current file */
char *resolve_module_path(const char *module_path, const char *current_file) {
    if (!module_path) return NULL;
    
    /* If module_path is absolute or starts with ./, use as-is */
    if (module_path[0] == '/' || (module_path[0] == '.' && module_path[1] == '/')) {
        return strdup(module_path);
    }
    
    /* Extract directory from current_file */
    char *resolved = NULL;
    if (current_file) {
        /* Find last '/' in current_file */
        const char *last_slash = strrchr(current_file, '/');
        if (last_slash) {
            int dir_len = last_slash - current_file + 1;
            resolved = malloc(dir_len + strlen(module_path) + 1);
            strncpy(resolved, current_file, dir_len);
            strcpy(resolved + dir_len, module_path);
        } else {
            /* No directory, use module_path as-is */
            resolved = strdup(module_path);
        }
    } else {
        /* No current file, use module_path as-is */
        resolved = strdup(module_path);
    }
    
    return resolved;
}

/* Load and parse a module file */
ASTNode *load_module(const char *module_path, Environment *env) {
    if (!module_path) return NULL;
    
    /* Read source file */
    FILE *file = fopen(module_path, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open module file '%s'\n", module_path);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char *source = malloc(size + 1);
    fread(source, 1, size, file);
    source[size] = '\0';
    fclose(file);
    
    /* Tokenize */
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "Error: Failed to tokenize module '%s'\n", module_path);
        free(source);
        return NULL;
    }
    
    /* Parse */
    ASTNode *module_ast = parse_program(tokens, token_count);
    if (!module_ast) {
        fprintf(stderr, "Error: Failed to parse module '%s'\n", module_path);
        free_tokens(tokens, token_count);
        free(source);
        return NULL;
    }
    
    /* Type check module (without requiring main) */
    if (!type_check_module(module_ast, env)) {
        fprintf(stderr, "Error: Type checking failed for module '%s'\n", module_path);
        free_ast(module_ast);
        free_tokens(tokens, token_count);
        free(source);
        return NULL;
    }
    
    free_tokens(tokens, token_count);
    free(source);
    return module_ast;
}

/* Process imports in a program */
bool process_imports(ASTNode *program, Environment *env, ModuleList *modules, const char *current_file) {
    if (!program || program->type != AST_PROGRAM) {
        return false;
    }
    
    /* First pass: collect all imports and resolve paths */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        if (item->type == AST_IMPORT) {
            char *module_path = resolve_module_path(item->as.import_stmt.module_path, current_file);
            if (!module_path) {
                fprintf(stderr, "Error at line %d, column %d: Failed to resolve module path '%s'\n",
                        item->line, item->column, item->as.import_stmt.module_path);
                return false;
            }
            
            /* Check if file exists */
            FILE *test = fopen(module_path, "r");
            if (!test) {
                fprintf(stderr, "Error at line %d, column %d: Module file '%s' not found\n",
                        item->line, item->column, module_path);
                free(module_path);
                return false;
            }
            fclose(test);
            
            /* Add to module list */
            if (modules) {
                module_list_add(modules, module_path);
            }
            
            /* Load the module */
            ASTNode *module_ast = load_module(module_path, env);
            if (!module_ast) {
                fprintf(stderr, "Error at line %d, column %d: Failed to load module '%s'\n",
                        item->line, item->column, module_path);
                free(module_path);
                return false;
            }
            
            /* Execute module definitions (functions, structs, etc.) */
            /* This makes module symbols available in the current environment */
            for (int j = 0; j < module_ast->as.program.count; j++) {
                ASTNode *module_item = module_ast->as.program.items[j];
                
                /* Skip imports, shadows, and statements in modules */
                if (module_item->type == AST_IMPORT || 
                    module_item->type == AST_SHADOW ||
                    module_item->type == AST_LET ||
                    module_item->type == AST_SET ||
                    module_item->type == AST_IF ||
                    module_item->type == AST_WHILE ||
                    module_item->type == AST_FOR ||
                    module_item->type == AST_RETURN ||
                    module_item->type == AST_PRINT ||
                    module_item->type == AST_ASSERT) {
                    continue;
                }
                
                /* Functions, structs, enums, unions are already registered during type_check */
            }
            
            free(module_path);
            /* Note: We don't free module_ast here - it's kept in memory for interpreter */
            /* For compiler, we'll track modules separately */
        }
    }
    
    return true;
}

/* Compile a single module to an object file */
bool compile_module_to_object(const char *module_path, const char *output_obj, Environment *env_unused, bool verbose) {
    (void)env_unused;  /* Not used - we create our own environment */
    if (!module_path || !output_obj) return false;
    
    /* Create a separate environment for module compilation to avoid symbol conflicts */
    Environment *module_env = create_environment();
    
    /* Load and type-check the module */
    ASTNode *module_ast = load_module(module_path, module_env);
    if (!module_ast) {
        fprintf(stderr, "Error: Failed to load module '%s' for compilation\n", module_path);
        free_environment(module_env);
        return false;
    }
    
    /* Transpile module to C */
    char *c_code = transpile_to_c(module_ast, module_env);
    if (!c_code) {
        fprintf(stderr, "Error: Failed to transpile module '%s'\n", module_path);
        free_ast(module_ast);
        free_environment(module_env);
        return false;
    }
    
    /* Write C code to temporary file */
    char temp_c_file[512];
    snprintf(temp_c_file, sizeof(temp_c_file), "%s.c", output_obj);
    
    FILE *c_file = fopen(temp_c_file, "w");
    if (!c_file) {
        fprintf(stderr, "Error: Could not create C file '%s'\n", temp_c_file);
        free(c_code);
        free_ast(module_ast);
        return false;
    }
    
    fprintf(c_file, "%s", c_code);
    fclose(c_file);
    
    if (verbose) {
        printf("✓ Generated module C code: %s\n", temp_c_file);
    }
    
    /* Compile C code to object file */
    /* Note: Runtime files are linked separately, not compiled into module objects */
    char compile_cmd[1024];
    if (verbose) {
        snprintf(compile_cmd, sizeof(compile_cmd),
                "gcc -std=c99 -Isrc -c -o %s %s",
                output_obj, temp_c_file);
    } else {
        snprintf(compile_cmd, sizeof(compile_cmd),
                "gcc -std=c99 -Isrc -c -o %s %s 2>/dev/null",
                output_obj, temp_c_file);
    }
    
    if (verbose) {
        printf("Compiling module: %s\n", compile_cmd);
    }
    
    int result = system(compile_cmd);
    
    /* Clean up temporary C file */
    remove(temp_c_file);
    
    if (result != 0) {
        fprintf(stderr, "Error: Failed to compile module '%s' to object file\n", module_path);
        free(c_code);
        free_ast(module_ast);
        free_environment(module_env);
        return false;
    }
    
    if (verbose) {
        printf("✓ Compiled module to object file: %s\n", output_obj);
    }
    
    free(c_code);
    free_ast(module_ast);
    free_environment(module_env);
    return true;
}

/* Compile all modules in the list to object files */
bool compile_modules(ModuleList *modules, Environment *env, char *module_objs_buffer, size_t buffer_size, bool verbose) {
    if (!modules || !module_objs_buffer || buffer_size == 0) {
        return false;
    }
    
    module_objs_buffer[0] = '\0';
    
    if (modules->count == 0) {
        return true;  /* No modules to compile */
    }
    
    if (verbose) {
        printf("Compiling %d module(s) to object files...\n", modules->count);
    }
    
    for (int i = 0; i < modules->count; i++) {
        const char *module_path = modules->module_paths[i];
        
        /* Generate object file name from module path */
        char obj_file[512];
        const char *last_slash = strrchr(module_path, '/');
        const char *base_name = last_slash ? last_slash + 1 : module_path;
        
        /* Remove .nano extension */
        char base_without_ext[256];
        snprintf(base_without_ext, sizeof(base_without_ext), "%s", base_name);
        char *dot = strrchr(base_without_ext, '.');
        if (dot && strcmp(dot, ".nano") == 0) {
            *dot = '\0';
        }
        
        /* Generate object file path */
        snprintf(obj_file, sizeof(obj_file), "obj/%s.o", base_without_ext);
        
        /* Ensure obj directory exists */
        system("mkdir -p obj 2>/dev/null");
        
        /* Compile module to object file */
        if (!compile_module_to_object(module_path, obj_file, env, verbose)) {
            fprintf(stderr, "Error: Failed to compile module '%s'\n", module_path);
            return false;
        }
        
        /* Add to module_objs buffer */
        if (strlen(module_objs_buffer) + strlen(obj_file) + 1 < buffer_size) {
            if (module_objs_buffer[0] != '\0') {
                strcat(module_objs_buffer, " ");
            }
            strcat(module_objs_buffer, obj_file);
        }
    }
    
    return true;
}

