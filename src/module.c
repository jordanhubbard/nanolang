#define _POSIX_C_SOURCE 200809L  /* For mkdtemp */
#include "nanolang.h"
#include "module_builder.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <libgen.h>

/* mkdtemp declaration (if not available via headers) */
#ifndef _DARWIN_C_SOURCE
char *mkdtemp(char *template);
#endif

/* Module cache to prevent duplicate imports */
typedef struct {
    char **loaded_paths;
    int count;
    int capacity;
} ModuleCache;

static ModuleCache *module_cache = NULL;

static void init_module_cache(void) {
    if (!module_cache) {
        module_cache = malloc(sizeof(ModuleCache));
        module_cache->count = 0;
        module_cache->capacity = 16;
        module_cache->loaded_paths = malloc(sizeof(char*) * module_cache->capacity);
    }
}

static bool is_module_cached(const char *module_path) {
    if (!module_cache) return false;
    for (int i = 0; i < module_cache->count; i++) {
        if (strcmp(module_cache->loaded_paths[i], module_path) == 0) {
            return true;
        }
    }
    return false;
}

static void cache_module(const char *module_path) {
    init_module_cache();
    if (is_module_cached(module_path)) return;
    
    if (module_cache->count >= module_cache->capacity) {
        module_cache->capacity *= 2;
        module_cache->loaded_paths = realloc(module_cache->loaded_paths, 
                                             sizeof(char*) * module_cache->capacity);
    }
    module_cache->loaded_paths[module_cache->count++] = strdup(module_path);
}

void clear_module_cache(void) {
    if (module_cache) {
        for (int i = 0; i < module_cache->count; i++) {
            free(module_cache->loaded_paths[i]);
        }
        free(module_cache->loaded_paths);
        free(module_cache);
        module_cache = NULL;
    }
}

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

/* Get module search paths from NANO_MODULE_PATH environment variable */
static char **get_module_search_paths(int *count_out) {
    *count_out = 0;
    char **paths = NULL;
    int capacity = 4;
    paths = malloc(sizeof(char*) * capacity);
    
    /* Get NANO_MODULE_PATH environment variable */
    const char *module_path_env = getenv("NANO_MODULE_PATH");
    
    /* If NANO_MODULE_PATH is not set, use default module directory */
    if (!module_path_env || safe_strlen(module_path_env) == 0) {
        /* Default to "modules" directory in current working directory */
        *count_out = 1;
        free(paths);  /* Free the initial allocation */
        paths = malloc(sizeof(char*));
        paths[0] = strdup("modules");
        return paths;
    }
    
    if (module_path_env && safe_strlen(module_path_env) > 0) {
        /* Split by colon (Unix) or semicolon (Windows) */
        char *path_copy = strdup(module_path_env);
        char *token = strtok(path_copy, ":;");
        
        while (token) {
            /* Skip empty tokens */
            assert(token != NULL);
            if (safe_strlen(token) > 0) {
                if (*count_out >= capacity) {
                    capacity *= 2;
                    paths = realloc(paths, sizeof(char*) * capacity);
                }
                paths[*count_out] = strdup(token);
                (*count_out)++;
            }
            token = strtok(NULL, ":;");
        }
        
        free(path_copy);
    }
    
    /* Add default path: ~/.nanolang/modules */
    const char *home = getenv("HOME");
    if (home) {
        char default_path[512];
        snprintf(default_path, sizeof(default_path), "%s/.nanolang/modules", home);
        
        if (*count_out >= capacity) {
            capacity *= 2;
            paths = realloc(paths, sizeof(char*) * capacity);
        }
        paths[*count_out] = strdup(default_path);
        (*count_out)++;
    }
    
    return paths;
}

/* Free module search paths array */
static void free_module_search_paths(char **paths, int count) {
    if (!paths) return;
    for (int i = 0; i < count; i++) {
        free(paths[i]);
    }
    free(paths);
}

/* Find a module in NANO_MODULE_PATH directories
 * Returns path to module package (.nano.tar.zst) or NULL if not found
 */
char *find_module_in_paths(const char *module_name) {
    if (!module_name) return NULL;
    
    int path_count = 0;
    char **search_paths = get_module_search_paths(&path_count);
    
    char package_name[512];
    snprintf(package_name, sizeof(package_name), "%s.nano.tar.zst", module_name);
    
    for (int i = 0; i < path_count; i++) {
        char package_path[1024];
        snprintf(package_path, sizeof(package_path), "%s/%s", search_paths[i], package_name);
        
        FILE *test = fopen(package_path, "r");
        if (test) {
            fclose(test);
            char *result = strdup(package_path);
            free_module_search_paths(search_paths, path_count);
            return result;
        }
    }
    
    free_module_search_paths(search_paths, path_count);
    return NULL;
}

/* Unpack a module package to a temporary directory
 * Returns allocated string with temp directory path, or NULL on error
 * temp_dir_out must be at least 512 bytes (output parameter)
 */
char *unpack_module_package(const char *package_path, char *temp_dir_out, size_t temp_dir_size) {
    if (!package_path || !temp_dir_out || temp_dir_size < 512) {
        return NULL;
    }
    
    /* Create temporary directory */
    char temp_template[] = "/tmp/nanolang_module_XXXXXX";
    char *temp_dir = mkdtemp(temp_template);
    if (temp_dir == NULL || temp_dir != temp_template) {
        fprintf(stderr, "Error: Failed to create temporary directory\n");
        return NULL;
    }
    
    /* Extract package using tar with zstd */
    char extract_cmd[2048];
    snprintf(extract_cmd, sizeof(extract_cmd),
            "tar -I zstd -xf '%s' -C '%s' 2>/dev/null",
            package_path, temp_dir);
    
    int result = system(extract_cmd);
    if (result != 0) {
        fprintf(stderr, "Error: Failed to extract module package '%s'\n", package_path);
        rmdir(temp_dir);
        return NULL;
    }
    
    /* Copy temp directory path to output */
    strncpy(temp_dir_out, temp_dir, temp_dir_size - 1);
    temp_dir_out[temp_dir_size - 1] = '\0';
    
    return strdup(temp_dir);
}

/* Resolve module path relative to current file or in module search paths */
char *resolve_module_path(const char *module_path, const char *current_file) {
    if (!module_path) return NULL;
    
    /* If module_path is absolute or starts with ./, use as-is */
    if (module_path[0] == '/' || (module_path[0] == '.' && module_path[1] == '/')) {
        return strdup(module_path);
    }
    
    /* Check if this is a project-relative path (starts with "examples/" or "src/") */
    if (strncmp(module_path, "examples/", 9) == 0 || strncmp(module_path, "src/", 4) == 0) {
        /* Try to find project root by walking up from current_file */
        if (current_file) {
            char current_dir[1024];
            strncpy(current_dir, current_file, sizeof(current_dir) - 1);
            current_dir[sizeof(current_dir) - 1] = '\0';
            
            /* Remove filename to get directory */
            char *last_slash = strrchr(current_dir, '/');
            if (last_slash) {
                *last_slash = '\0';
                
                /* Walk up to find project root (directory containing "modules/" or "examples/") */
                char test_path[1024];
                for (int depth = 0; depth < 10; depth++) {
                    /* Check if this directory has modules/ or examples/ subdirectory */
                    snprintf(test_path, sizeof(test_path), "%s/modules", current_dir);
                    DIR *dir = opendir(test_path);
                    if (dir) {
                        closedir(dir);
                        /* Found project root - resolve path from here */
                        snprintf(test_path, sizeof(test_path), "%s/%s", current_dir, module_path);
                        FILE *test = fopen(test_path, "r");
                        if (test) {
                            fclose(test);
                            return strdup(test_path);
                        }
                        break;
                    }
                    
                    /* Try examples/ directory */
                    snprintf(test_path, sizeof(test_path), "%s/examples", current_dir);
                    dir = opendir(test_path);
                    if (dir) {
                        closedir(dir);
                        /* Found project root - resolve path from here */
                        snprintf(test_path, sizeof(test_path), "%s/%s", current_dir, module_path);
                        FILE *test = fopen(test_path, "r");
                        if (test) {
                            fclose(test);
                            return strdup(test_path);
                        }
                        break;
                    }
                    
                    /* Move up one directory */
                    last_slash = strrchr(current_dir, '/');
                    if (!last_slash) break;
                    *last_slash = '\0';
                }
            }
        }
    }
    
    /* First, try relative to current file */
    char *resolved = NULL;
    if (current_file) {
        /* Find last '/' in current_file */
        const char *last_slash = strrchr(current_file, '/');
        if (last_slash) {
            int dir_len = last_slash - current_file + 1;
            assert(module_path != NULL);
            resolved = malloc(dir_len + safe_strlen(module_path) + 1);
            strncpy(resolved, current_file, dir_len);
            safe_strncpy(resolved + dir_len, module_path, safe_strlen(module_path) + 1);
            
            /* Check if file exists */
            FILE *test = fopen(resolved, "r");
            if (test) {
                fclose(test);
                return resolved;
            }
            free(resolved);
            resolved = NULL;
        }
    }
    
    /* If not found relative to current file, try module search paths */
    /* Extract module name (remove .nano extension if present) */
    char module_name[256];
    strncpy(module_name, module_path, sizeof(module_name) - 1);
    module_name[sizeof(module_name) - 1] = '\0';
    
    char *dot = strrchr(module_name, '.');
    if (dot && strcmp(dot, ".nano") == 0) {
        *dot = '\0';
    }
    
    /* Look for package in module paths */
    char *package_path = find_module_in_paths(module_name);
    if (package_path) {
        /* Found package - will be unpacked by caller */
        return package_path;
    }
    
    /* Fallback: return module_path as-is (will fail if not found) */
    return strdup(module_path);
}

/* Load and parse a module file */
static ASTNode *load_module_internal(const char *module_path, Environment *env, bool use_cache) {
    if (!module_path) return NULL;
    
    /* Check if module is already loaded (only if using cache) */
    if (use_cache && is_module_cached(module_path)) {
        /* Module already loaded - skip to avoid duplicate definitions */
        return NULL;  /* NULL means "already loaded, skip processing" */
    }
    
    /* Mark module as loading to prevent circular imports (only if using cache) */
    if (use_cache) {
        cache_module(module_path);
    }
    
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
    
    /* Process imports first - modules may depend on symbols from imported modules */
    if (!process_imports(module_ast, env, NULL, module_path)) {
        fprintf(stderr, "Error: Failed to process imports for module '%s'\n", module_path);
        free_ast(module_ast);
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
    
    /* Load constants from C headers if module has module.json */
    char *module_dir_copy = strdup(module_path);
    char *dir_result = dirname(module_dir_copy);
    ModuleBuildMetadata *meta = module_load_metadata(dir_result);
    if (meta && meta->headers_count > 0) {
        /* Try to find and parse C headers for constants */
        for (size_t i = 0; i < meta->headers_count; i++) {
            /* Try to locate the header - check system include paths */
            char header_path[1024];
            
            /* Try common locations */
            const char *search_paths[] = {
                "/opt/homebrew/include",  /* macOS homebrew */
                "/usr/local/include",      /* Linux/macOS local */
                "/usr/include",            /* Linux system */
                NULL
            };
            
            bool found = false;
            for (int j = 0; search_paths[j] != NULL; j++) {
                snprintf(header_path, sizeof(header_path), "%s/%s", search_paths[j], meta->headers[i]);
                FILE *test = fopen(header_path, "r");
                if (test) {
                    fclose(test);
                    found = true;
                    break;
                }
            }
            
            if (found) {
                int const_count = 0;
                ConstantDef *constants = parse_c_header_constants(header_path, &const_count);
                
                if (constants && const_count > 0) {
                    /* Add constants to environment as immutable symbols */
                    for (int j = 0; j < const_count; j++) {
                        /* Check if symbol already exists (from manual declarations) */
                        bool exists = false;
                        for (int k = 0; k < env->symbol_count; k++) {
                            if (strcmp(env->symbols[k].name, constants[j].name) == 0) {
                                exists = true;
                                break;
                            }
                        }
                        
                        if (!exists) {
                            /* Add as new symbol */
                            if (env->symbol_count >= env->symbol_capacity) {
                                env->symbol_capacity = env->symbol_capacity == 0 ? 16 : env->symbol_capacity * 2;
                                env->symbols = realloc(env->symbols, sizeof(Symbol) * env->symbol_capacity);
                            }
                            
                            Symbol *sym = &env->symbols[env->symbol_count++];
                            sym->name = strdup(constants[j].name);
                            sym->type = constants[j].type;
                            sym->struct_type_name = NULL;
                            sym->element_type = TYPE_UNKNOWN;
                            sym->type_info = NULL;
                            sym->is_mut = false;  /* Constants are immutable */
                            sym->is_used = true;   /* Assume used (from header) */
                            sym->from_c_header = true;  /* Mark as from C header */
                            sym->def_line = 0;
                            sym->def_column = 0;
                            
                            /* Set value */
                            if (constants[j].type == TYPE_INT) {
                                sym->value.type = VAL_INT;
                                sym->value.as.int_val = constants[j].value;
                            } else if (constants[j].type == TYPE_FLOAT) {
                                sym->value.type = VAL_FLOAT;
                                union { double d; int64_t i; } u;
                                u.i = constants[j].value;
                                sym->value.as.float_val = u.d;
                            }
                        }
                    }
                    
                    /* Free constants array */
                    for (int j = 0; j < const_count; j++) {
                        free(constants[j].name);
                    }
                    free(constants);
                }
            }
        }
    }
    if (meta) {
        module_metadata_free(meta);
    }
    free(module_dir_copy);
    
    free_tokens(tokens, token_count);
    free(source);
    return module_ast;
}

/* Public wrapper for load_module that uses cache */
ASTNode *load_module(const char *module_path, Environment *env) {
    return load_module_internal(module_path, env, true);
}

/* Load module from a package file */
ASTNode *load_module_from_package(const char *package_path, Environment *env, char *temp_dir_out, size_t temp_dir_size) {
    if (!package_path || !env) return NULL;
    
    /* Unpack package to temporary directory */
    char temp_dir[512];
    char *unpacked_dir = unpack_module_package(package_path, temp_dir, sizeof(temp_dir));
    if (!unpacked_dir) {
        return NULL;
    }
    
    /* Copy temp directory to output if provided */
    if (temp_dir_out && temp_dir_size > 0) {
        strncpy(temp_dir_out, unpacked_dir, temp_dir_size - 1);
        temp_dir_out[temp_dir_size - 1] = '\0';
    }
    
    /* Find module.nano file in unpacked directory */
    char module_nano_path[1024];
    snprintf(module_nano_path, sizeof(module_nano_path), "%s/module.nano", unpacked_dir);
    
    /* Try alternative: look for any .nano file */
    FILE *test = fopen(module_nano_path, "r");
    if (!test) {
        /* Try to find any .nano file */
        DIR *dir = opendir(unpacked_dir);
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                if (strstr(entry->d_name, ".nano") != NULL) {
                    snprintf(module_nano_path, sizeof(module_nano_path), "%s/%s", unpacked_dir, entry->d_name);
                    break;
                }
            }
            closedir(dir);
        }
    } else {
        fclose(test);
    }
    
    /* Load the module */
    ASTNode *module_ast = load_module(module_nano_path, env);
    
    /* Note: We don't free unpacked_dir here - caller is responsible for cleanup */
    /* The temp directory will be cleaned up by the caller */
    
    return module_ast;
}

/* Process imports in a program */
bool process_imports(ASTNode *program, Environment *env, ModuleList *modules, const char *current_file) {
    if (!program || program->type != AST_PROGRAM) {
        return false;
    }
    
    /* Track unpacked package directories for cleanup */
    char **unpacked_dirs = NULL;
    int unpacked_count = 0;
    int unpacked_capacity = 4;
    unpacked_dirs = malloc(sizeof(char*) * unpacked_capacity);
    
    /* First pass: collect all imports and resolve paths */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        if (item->type == AST_IMPORT) {
            char *module_path = resolve_module_path(item->as.import_stmt.module_path, current_file);
            if (!module_path) {
                fprintf(stderr, "Error at line %d, column %d: Failed to resolve module path '%s'\n",
                        item->line, item->column, item->as.import_stmt.module_path);
                /* Cleanup unpacked directories */
                for (int j = 0; j < unpacked_count; j++) {
                    free(unpacked_dirs[j]);
                }
                free(unpacked_dirs);
                return false;
            }
            
            ASTNode *module_ast = NULL;
            char temp_dir[512] = {0};
            
            /* Check if this is a package file (.nano.tar.zst) */
            if (strstr(module_path, ".nano.tar.zst") != NULL) {
                /* Load from package */
                module_ast = load_module_from_package(module_path, env, temp_dir, sizeof(temp_dir));
                
                /* Track unpacked directory for cleanup */
                if (temp_dir[0] != '\0') {
                    if (unpacked_count >= unpacked_capacity) {
                        unpacked_capacity *= 2;
                        unpacked_dirs = realloc(unpacked_dirs, sizeof(char*) * unpacked_capacity);
                    }
                    unpacked_dirs[unpacked_count++] = strdup(temp_dir);
                }
                
                /* Use the unpacked .nano file path for module list */
                if (temp_dir[0] != '\0') {
                    char unpacked_nano[1024];
                    snprintf(unpacked_nano, sizeof(unpacked_nano), "%s/module.nano", temp_dir);
                    free(module_path);
                    module_path = strdup(unpacked_nano);
                }
            } else {
                /* Regular .nano file */
                FILE *test = fopen(module_path, "r");
                if (!test) {
                    fprintf(stderr, "Error at line %d, column %d: Module file '%s' not found\n",
                            item->line, item->column, module_path);
                    free(module_path);
                    /* Cleanup unpacked directories */
                    for (int j = 0; j < unpacked_count; j++) {
                        free(unpacked_dirs[j]);
                    }
                    free(unpacked_dirs);
                    return false;
                }
                fclose(test);
                
                module_ast = load_module(module_path, env);
            }
            
            /* NULL return means module was already loaded - this is OK */
            if (module_ast == NULL && !is_module_cached(module_path)) {
                /* Only error if module wasn't cached (i.e., actual failure) */
                fprintf(stderr, "Error at line %d, column %d: Failed to load module '%s'\n",
                        item->line, item->column, module_path);
                free(module_path);
                /* Cleanup unpacked directories */
                for (int j = 0; j < unpacked_count; j++) {
                    free(unpacked_dirs[j]);
                }
                free(unpacked_dirs);
                return false;
            }
            
            /* If module was already cached, skip processing */
            if (module_ast == NULL) {
                free(module_path);
                continue;
            }
            
            /* Add to module list */
            if (modules) {
                module_list_add(modules, module_path);
            }
            
            /* Execute module definitions (functions, structs, etc.) */
            /* This makes module symbols available in the current environment */
            for (int j = 0; j < module_ast->as.program.count; j++) {
                ASTNode *module_item = module_ast->as.program.items[j];
                
                /* Export top-level constants (immutable let statements) from modules */
                if (module_item->type == AST_LET && !module_item->as.let.is_mut) {
                    /* This is a constant - evaluate it and make it available in importing module */
                    Value val = create_void();
                    
                    /* Try to evaluate constant expressions (literals and simple expressions) */
                    if (module_item->as.let.value) {
                        ASTNode *value_node = module_item->as.let.value;
                        if (value_node->type == AST_NUMBER) {
                            val = create_int(value_node->as.number);
                        } else if (value_node->type == AST_FLOAT) {
                            val = create_float(value_node->as.float_val);
                        } else if (value_node->type == AST_BOOL) {
                            val = create_bool(value_node->as.bool_val);
                        } else if (value_node->type == AST_STRING) {
                            val = create_string(value_node->as.string_val);
                        }
                        /* For complex expressions, keep as void - transpiler will use variable name */
                    }
                    
                    env_define_var(env, module_item->as.let.name, 
                                   module_item->as.let.var_type, 
                                   false, val);
                    continue;
                }
                
                /* Skip imports, shadows, and executable statements in modules */
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
    
    /* Note: Unpacked directories are tracked but not cleaned up here */
    /* They will be cleaned up when the program exits or by the caller */
    /* For interpreter: cleaned up at end of session */
    /* For compiler: cleaned up after compilation */
    free(unpacked_dirs);
    
    return true;
}

/* Compile a single module to an object file */
bool compile_module_to_object(const char *module_path, const char *output_obj, Environment *env_unused, bool verbose) {
    (void)env_unused;  /* Not used - we create our own environment */
    if (!module_path || !output_obj) return false;
    
    /* Create a separate environment for module compilation to avoid symbol conflicts */
    Environment *module_env = create_environment();
    
    /* Clear cache temporarily - each module compilation gets fresh environment */
    /* But we want dependencies to load properly, so we'll save and restore */
    ModuleCache *saved_cache = module_cache;
    module_cache = NULL;  /* Start with clean cache for this compilation */
    
    /* Load and type-check the module */
    ASTNode *module_ast = load_module(module_path, module_env);
    if (!module_ast) {
        fprintf(stderr, "Error: Failed to load module '%s' for compilation\n", module_path);
        free_environment(module_env);
        /* Restore cache */
        clear_module_cache();
        module_cache = saved_cache;
        return false;
    }
    
    /* Clean up local cache and restore original */
    clear_module_cache();
    module_cache = saved_cache;
    
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
        assert(c_code != NULL);
        size_t c_code_len = safe_strlen(c_code);
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
    /* Add SDL flags if SDL modules are used */
    char sdl_flags[256] = "";
    FILE *sdl_check = popen("sdl2-config --cflags 2>/dev/null", "r");
    if (sdl_check) {
        fgets(sdl_flags, sizeof(sdl_flags), sdl_check);
        pclose(sdl_check);
        /* Remove trailing newline */
        char *newline = strchr(sdl_flags, '\n');
        if (newline) *newline = '\0';
    }
    
    char compile_cmd[1024];
    snprintf(compile_cmd, sizeof(compile_cmd),
            "gcc -std=c99 -Isrc -Imodules/sdl_helpers %s -c -o %s %s",
            sdl_flags, output_obj, temp_c_file);
    
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
        assert(error_output != NULL);
        if (safe_strlen(error_output) > 0) {
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

/* Check if a module is FFI-only (only extern declarations, no implementation) */
static bool is_ffi_only_module(const char *module_path) {
    FILE *file = fopen(module_path, "r");
    if (!file) return false;
    
    char line[1024];
    bool has_extern = false;
    bool has_implementation = false;
    
    while (fgets(line, sizeof(line), file)) {
        /* Skip comments and empty lines */
        char *trimmed = line;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
        if (*trimmed == '#' || *trimmed == '\n' || *trimmed == '\0') continue;
        
        /* Check for extern declarations */
        if (strstr(trimmed, "extern fn") != NULL) {
            has_extern = true;
        }
        
        /* Check for function implementations (non-extern functions) */
        if (strstr(trimmed, "fn ") != NULL && strstr(trimmed, "extern") == NULL) {
            has_implementation = true;
            break;
        }
        
        /* Check for struct/enum/union definitions */
        if (strstr(trimmed, "struct ") != NULL || 
            strstr(trimmed, "enum ") != NULL || 
            strstr(trimmed, "union ") != NULL) {
            has_implementation = true;
            break;
        }
    }
    
    fclose(file);
    
    /* FFI-only if it has extern declarations but no implementations */
    return has_extern && !has_implementation;
}

/* Helper: Extract module directory from module path */
static char* get_module_dir(const char *module_path) {
    char *dir = strdup(module_path);
    char *last_slash = strrchr(dir, '/');
    if (last_slash) {
        *last_slash = '\0';
    } else {
        free(dir);
        dir = strdup(".");
    }
    return dir;
}

/* Compile all modules in the list to object files using the module builder */
bool compile_modules(ModuleList *modules, Environment *env, char *module_objs_buffer, size_t buffer_size, char *compile_flags_buffer, size_t compile_flags_buffer_size, bool verbose) {
    if (!modules || !module_objs_buffer || buffer_size == 0) {
        return false;
    }
    
    /* Initialize buffers */
    module_objs_buffer[0] = '\0';
    if (compile_flags_buffer && compile_flags_buffer_size > 0) {
        compile_flags_buffer[0] = '\0';
    }
    
    if (modules->count == 0) {
        return true;  /* No modules to compile */
    }
    
    /* Check for verbose build flag */
    if (getenv("NANO_VERBOSE_BUILD")) {
        verbose = true;
        module_builder_verbose = true;
    }
    
    if (verbose) {
        printf("[Modules] Processing %d module(s)...\n", modules->count);
    }
    
    /* Create module builder */
    const char *module_path_env = getenv("NANO_MODULE_PATH");
    ModuleBuilder *builder = module_builder_new(module_path_env);
    if (!builder) {
        fprintf(stderr, "Error: Failed to create module builder\n");
        return false;
    }
    
    /* Track all build info for cleanup */
    ModuleBuildInfo **build_infos = calloc(modules->count, sizeof(ModuleBuildInfo*));
    int build_info_count = 0;
    
    int nanolang_compiled = 0;
    int c_modules_built = 0;
    int ffi_only = 0;
    
    for (int i = 0; i < modules->count; i++) {
        const char *module_path = modules->module_paths[i];
        
        /* Get module directory */
        char *module_dir = get_module_dir(module_path);
        
        /* Check if module has module.json (C sources) */
        ModuleBuildMetadata *meta = module_load_metadata(module_dir);
        
        if (meta) {
            /* Module has C sources - use module builder */
            if (verbose) {
                printf("[Modules] Building C module '%s' (%s)\n", meta->name, module_dir);
            }
            
            ModuleBuildInfo *info = module_build(builder, meta);
            if (!info) {
                fprintf(stderr, "Error: Failed to build module '%s'\n", meta->name);
                module_metadata_free(meta);
                free(module_dir);
                module_builder_free(builder);
                for (int j = 0; j < build_info_count; j++) {
                    module_build_info_free(build_infos[j]);
                }
                free(build_infos);
                return false;
            }
            
            /* Note: object file is included in link_flags, so we don't add it here */
            /* It will be added below when we collect all link flags */
            
            build_infos[build_info_count++] = info;
            c_modules_built++;
            module_metadata_free(meta);
        } else {
            /* No module.json - check if it's FFI-only (just extern declarations) */
            if (is_ffi_only_module(module_path)) {
                if (verbose) {
                    printf("[Modules] Skipping FFI-only module '%s'\n", module_path);
                }
                ffi_only++;
                free(module_dir);
                continue;
            }
            
            /* Pure nanolang module - compile using existing method */
            if (verbose) {
                printf("[Modules] Compiling nanolang module '%s'\n", module_path);
            }
            
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
                free(module_dir);
                module_builder_free(builder);
                for (int j = 0; j < build_info_count; j++) {
                    module_build_info_free(build_infos[j]);
                }
                free(build_infos);
                return false;
            }
            
            nanolang_compiled++;
            
            /* Add to module_objs buffer */
            if (strlen(module_objs_buffer) + strlen(obj_file) + 2 < buffer_size) {
                if (module_objs_buffer[0] != '\0') {
                    strcat(module_objs_buffer, " ");
                }
                strcat(module_objs_buffer, obj_file);
            }
        }
        
        free(module_dir);
    }
    
    /* Get all link flags from C modules */
    size_t link_flags_count = 0;
    char **link_flags = module_get_link_flags(build_infos, build_info_count, &link_flags_count);
    
    /* Add link flags to buffer */
    if (link_flags) {
        for (size_t i = 0; i < link_flags_count; i++) {
            if (strlen(module_objs_buffer) + strlen(link_flags[i]) + 2 < buffer_size) {
                if (module_objs_buffer[0] != '\0') {
                    strcat(module_objs_buffer, " ");
                }
                strcat(module_objs_buffer, link_flags[i]);
            }
            free(link_flags[i]);
        }
        free(link_flags);
    }
    
    /* Get all compile flags from C modules */
    if (compile_flags_buffer && compile_flags_buffer_size > 0) {
        size_t compile_flags_count = 0;
        char **compile_flags = module_get_compile_flags(build_infos, build_info_count, &compile_flags_count);
        
        /* Add compile flags to buffer */
        if (compile_flags) {
            for (size_t i = 0; i < compile_flags_count; i++) {
                if (strlen(compile_flags_buffer) + strlen(compile_flags[i]) + 2 < compile_flags_buffer_size) {
                    if (compile_flags_buffer[0] != '\0') {
                        strcat(compile_flags_buffer, " ");
                    }
                    strcat(compile_flags_buffer, compile_flags[i]);
                }
                free(compile_flags[i]);
            }
            free(compile_flags);
        }
    }
    
    /* Cleanup */
    for (int j = 0; j < build_info_count; j++) {
        module_build_info_free(build_infos[j]);
    }
    free(build_infos);
    module_builder_free(builder);
    
    if (verbose) {
        printf("[Modules] ✓ Complete: %d nanolang, %d C, %d FFI-only\n", 
               nanolang_compiled, c_modules_built, ffi_only);
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
    
    /* Extract constants from symbols (global let statements with constant values) */
    meta->constant_count = 0;
    meta->constants = NULL;
    
    /* Count global constants (immutable symbols with constant values at file scope) */
    int const_count = 0;
    for (int i = 0; i < env->symbol_count; i++) {
        Symbol *sym = &env->symbols[i];
        /* Only extract immutable int/float constants */
        if (!sym->is_mut && (sym->type == TYPE_INT || sym->type == TYPE_FLOAT)) {
            const_count++;
        }
    }
    
    if (const_count > 0) {
        meta->constants = malloc(sizeof(ConstantDef) * const_count);
        int const_idx = 0;
        for (int i = 0; i < env->symbol_count; i++) {
            Symbol *sym = &env->symbols[i];
            if (!sym->is_mut && (sym->type == TYPE_INT || sym->type == TYPE_FLOAT)) {
                meta->constants[const_idx].name = sym->name ? strdup(sym->name) : NULL;
                meta->constants[const_idx].type = sym->type;
                if (sym->type == TYPE_INT) {
                    meta->constants[const_idx].value = sym->value.as.int_val;
                } else if (sym->type == TYPE_FLOAT) {
                    /* Store float as int64 bit pattern - will need special handling */
                    union { double d; int64_t i; } u;
                    u.d = sym->value.as.float_val;
                    meta->constants[const_idx].value = u.i;
                }
                const_idx++;
            }
        }
        meta->constant_count = const_idx;
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
    
    /* Free constants */
    if (meta->constants) {
        for (int i = 0; i < meta->constant_count; i++) {
            if (meta->constants[i].name) free(meta->constants[i].name);
        }
        free(meta->constants);
    }
    
    free(meta);
}
