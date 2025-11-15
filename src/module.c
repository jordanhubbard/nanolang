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

