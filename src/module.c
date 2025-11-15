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
    
    /* Extract module metadata before transpiling */
    const char *last_slash = strrchr(module_path, '/');
    const char *base_name = last_slash ? last_slash + 1 : module_path;
    char module_name[256];
    snprintf(module_name, sizeof(module_name), "%s", base_name);
    char *dot = strrchr(module_name, '.');
    if (dot) *dot = '\0';
    
    /* Extract module metadata - TODO: Fix bus error in extract_module_metadata */
    ModuleMetadata *meta = NULL;  // extract_module_metadata(module_env, module_name);
    
    /* Transpile module to C */
    char *c_code = transpile_to_c(module_ast, module_env);
    if (!c_code) {
        fprintf(stderr, "Error: Failed to transpile module '%s'\n", module_path);
        free_ast(module_ast);
        free_environment(module_env);
        if (meta) free_module_metadata(meta);
        return false;
    }
    
    /* Embed metadata in C code */
    /* TODO: Fix metadata embedding - currently disabled due to bus error */
    /* if (meta) {
        size_t c_code_len = strlen(c_code);
        size_t buffer_size = c_code_len * 2;
        char *c_code_with_meta = realloc(c_code, buffer_size);
        if (c_code_with_meta) {
            if (embed_metadata_in_module_c(c_code_with_meta, meta, buffer_size)) {
                c_code = c_code_with_meta;
            }
        }
    } */
    if (meta) {
        free_module_metadata(meta);
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
    snprintf(compile_cmd, sizeof(compile_cmd),
            "gcc -std=c99 -Isrc -c -o %s %s",
            output_obj, temp_c_file);
    
    if (verbose) {
        printf("Compiling module: %s\n", compile_cmd);
    }
    
    /* Compile and capture errors */
    char error_cmd[1152];
    snprintf(error_cmd, sizeof(error_cmd), "%s 2>&1", compile_cmd);
    FILE *pipe = popen(error_cmd, "r");
    char error_output[4096] = {0};
    if (pipe) {
        size_t bytes_read = fread(error_output, 1, sizeof(error_output) - 1, pipe);
        error_output[bytes_read] = '\0';
        pclose(pipe);
    }
    
    int result = system(compile_cmd);
    
    if (result != 0) {
        fprintf(stderr, "Error: Failed to compile module '%s' to object file\n", module_path);
        if (strlen(error_output) > 0) {
            fprintf(stderr, "Compilation errors:\n%s\n", error_output);
        }
        /* Keep C file for debugging */
        fprintf(stderr, "C file kept at: %s\n", temp_c_file);
        free(c_code);
        free_ast(module_ast);
        free_environment(module_env);
        return false;
    }
    
    /* Clean up temporary C file */
    remove(temp_c_file);
    
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


/* Extract module metadata from environment */
ModuleMetadata *extract_module_metadata(Environment *env, const char *module_name) {
    if (!env) return NULL;
    
    ModuleMetadata *meta = malloc(sizeof(ModuleMetadata));
    meta->module_name = module_name ? strdup(module_name) : strdup("unknown");
    
    /* Extract functions */
    meta->function_count = env->function_count;
    if (meta->function_count > 0) {
        meta->functions = malloc(sizeof(Function) * meta->function_count);
        for (int i = 0; i < meta->function_count; i++) {
            /* Copy function - note: we copy pointers, not deep copy */
            meta->functions[i] = env->functions[i];
            /* Deep copy name and params */
            if (env->functions[i].name) {
                meta->functions[i].name = strdup(env->functions[i].name);
            }
            if (env->functions[i].param_count > 0 && env->functions[i].params) {
                meta->functions[i].params = malloc(sizeof(Parameter) * env->functions[i].param_count);
                for (int j = 0; j < env->functions[i].param_count; j++) {
                    meta->functions[i].params[j] = env->functions[i].params[j];
                    if (env->functions[i].params[j].name) {
                        meta->functions[i].params[j].name = strdup(env->functions[i].params[j].name);
                    }
                    if (env->functions[i].params[j].struct_type_name) {
                        meta->functions[i].params[j].struct_type_name = strdup(env->functions[i].params[j].struct_type_name);
                    }
                    /* Note: fn_sig pointers are not deep copied - would need recursive copy */
                }
            }
            if (env->functions[i].return_struct_type_name) {
                meta->functions[i].return_struct_type_name = strdup(env->functions[i].return_struct_type_name);
            }
            /* Clear body and shadow_test - not needed in metadata */
            meta->functions[i].body = NULL;
            meta->functions[i].shadow_test = NULL;
        }
    } else {
        meta->functions = NULL;
    }
    
    /* Extract structs */
    meta->struct_count = env->struct_count;
    if (meta->struct_count > 0) {
        meta->structs = malloc(sizeof(StructDef) * meta->struct_count);
        for (int i = 0; i < meta->struct_count; i++) {
            meta->structs[i] = env->structs[i];
            if (env->structs[i].name) {
                meta->structs[i].name = strdup(env->structs[i].name);
            }
            if (env->structs[i].field_count > 0) {
                meta->structs[i].field_names = malloc(sizeof(char*) * env->structs[i].field_count);
                meta->structs[i].field_types = malloc(sizeof(Type) * env->structs[i].field_count);
                for (int j = 0; j < env->structs[i].field_count; j++) {
                    if (env->structs[i].field_names[j]) {
                        meta->structs[i].field_names[j] = strdup(env->structs[i].field_names[j]);
                    }
                    meta->structs[i].field_types[j] = env->structs[i].field_types[j];
                }
            }
        }
    } else {
        meta->structs = NULL;
    }
    
    /* Extract enums */
    meta->enum_count = env->enum_count;
    if (meta->enum_count > 0) {
        meta->enums = malloc(sizeof(EnumDef) * meta->enum_count);
        for (int i = 0; i < meta->enum_count; i++) {
            meta->enums[i] = env->enums[i];
            if (env->enums[i].name) {
                meta->enums[i].name = strdup(env->enums[i].name);
            }
            if (env->enums[i].variant_count > 0) {
                meta->enums[i].variant_names = malloc(sizeof(char*) * env->enums[i].variant_count);
                meta->enums[i].variant_values = malloc(sizeof(int) * env->enums[i].variant_count);
                for (int j = 0; j < env->enums[i].variant_count; j++) {
                    if (env->enums[i].variant_names[j]) {
                        meta->enums[i].variant_names[j] = strdup(env->enums[i].variant_names[j]);
                    }
                    meta->enums[i].variant_values[j] = env->enums[i].variant_values[j];
                }
            }
        }
    } else {
        meta->enums = NULL;
    }
    
    /* Extract unions */
    meta->union_count = env->union_count;
    if (meta->union_count > 0) {
        meta->unions = malloc(sizeof(UnionDef) * meta->union_count);
        for (int i = 0; i < meta->union_count; i++) {
            meta->unions[i] = env->unions[i];
            if (env->unions[i].name) {
                meta->unions[i].name = strdup(env->unions[i].name);
            }
            if (env->unions[i].variant_count > 0) {
                meta->unions[i].variant_names = malloc(sizeof(char*) * env->unions[i].variant_count);
                meta->unions[i].variant_field_counts = malloc(sizeof(int) * env->unions[i].variant_count);
                meta->unions[i].variant_field_names = malloc(sizeof(char**) * env->unions[i].variant_count);
                meta->unions[i].variant_field_types = malloc(sizeof(Type*) * env->unions[i].variant_count);
                for (int j = 0; j < env->unions[i].variant_count; j++) {
                    if (env->unions[i].variant_names[j]) {
                        meta->unions[i].variant_names[j] = strdup(env->unions[i].variant_names[j]);
                    }
                    meta->unions[i].variant_field_counts[j] = env->unions[i].variant_field_counts[j];
                    if (env->unions[i].variant_field_counts[j] > 0) {
                        meta->unions[i].variant_field_names[j] = malloc(sizeof(char*) * env->unions[i].variant_field_counts[j]);
                        meta->unions[i].variant_field_types[j] = malloc(sizeof(Type) * env->unions[i].variant_field_counts[j]);
                        for (int k = 0; k < env->unions[i].variant_field_counts[j]; k++) {
                            if (env->unions[i].variant_field_names[j][k]) {
                                meta->unions[i].variant_field_names[j][k] = strdup(env->unions[i].variant_field_names[j][k]);
                            }
                            meta->unions[i].variant_field_types[j][k] = env->unions[i].variant_field_types[j][k];
                        }
                    }
                }
            }
        }
    } else {
        meta->unions = NULL;
    }
    
    return meta;
}

/* Free module metadata */
void free_module_metadata(ModuleMetadata *meta) {
    if (!meta) return;
    
    if (meta->module_name) free(meta->module_name);
    
    /* Free functions */
    if (meta->functions) {
        for (int i = 0; i < meta->function_count; i++) {
            if (meta->functions[i].name) free(meta->functions[i].name);
            if (meta->functions[i].params) {
                for (int j = 0; j < meta->functions[i].param_count; j++) {
                    if (meta->functions[i].params[j].name) free(meta->functions[i].params[j].name);
                    if (meta->functions[i].params[j].struct_type_name) free(meta->functions[i].params[j].struct_type_name);
                }
                free(meta->functions[i].params);
            }
            if (meta->functions[i].return_struct_type_name) free(meta->functions[i].return_struct_type_name);
        }
        free(meta->functions);
    }
    
    /* Free structs */
    if (meta->structs) {
        for (int i = 0; i < meta->struct_count; i++) {
            if (meta->structs[i].name) free(meta->structs[i].name);
            if (meta->structs[i].field_names) {
                for (int j = 0; j < meta->structs[i].field_count; j++) {
                    if (meta->structs[i].field_names[j]) free(meta->structs[i].field_names[j]);
                }
                free(meta->structs[i].field_names);
            }
            if (meta->structs[i].field_types) free(meta->structs[i].field_types);
        }
        free(meta->structs);
    }
    
    /* Free enums */
    if (meta->enums) {
        for (int i = 0; i < meta->enum_count; i++) {
            if (meta->enums[i].name) free(meta->enums[i].name);
            if (meta->enums[i].variant_names) {
                for (int j = 0; j < meta->enums[i].variant_count; j++) {
                    if (meta->enums[i].variant_names[j]) free(meta->enums[i].variant_names[j]);
                }
                free(meta->enums[i].variant_names);
            }
            if (meta->enums[i].variant_values) free(meta->enums[i].variant_values);
        }
        free(meta->enums);
    }
    
    /* Free unions */
    if (meta->unions) {
        for (int i = 0; i < meta->union_count; i++) {
            if (meta->unions[i].name) free(meta->unions[i].name);
            if (meta->unions[i].variant_names) {
                for (int j = 0; j < meta->unions[i].variant_count; j++) {
                    if (meta->unions[i].variant_names[j]) free(meta->unions[i].variant_names[j]);
                    if (meta->unions[i].variant_field_names && meta->unions[i].variant_field_names[j]) {
                        for (int k = 0; k < meta->unions[i].variant_field_counts[j]; k++) {
                            if (meta->unions[i].variant_field_names[j][k]) free(meta->unions[i].variant_field_names[j][k]);
                        }
                        free(meta->unions[i].variant_field_names[j]);
                    }
                    if (meta->unions[i].variant_field_types && meta->unions[i].variant_field_types[j]) {
                        free(meta->unions[i].variant_field_types[j]);
                    }
                }
                free(meta->unions[i].variant_names);
                free(meta->unions[i].variant_field_counts);
                free(meta->unions[i].variant_field_names);
                free(meta->unions[i].variant_field_types);
            }
        }
        free(meta->unions);
    }
    
    free(meta);
}
