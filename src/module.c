#define _POSIX_C_SOURCE 200809L  /* For mkdtemp */
#include "nanolang.h"
#include "module_builder.h"
#include "stdlib_runtime.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <libgen.h>
#include <stdio.h>

/* mkdtemp declaration (if not available via headers) */
#ifndef _DARWIN_C_SOURCE
char *mkdtemp(char *template);
#endif

/* Module cache to prevent duplicate imports and preserve ASTs */
typedef struct {
    char **loaded_paths;
    ASTNode **loaded_asts;  /* Corresponding ASTs for each path */
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
        module_cache->loaded_asts = malloc(sizeof(ASTNode*) * module_cache->capacity);
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

ASTNode *get_cached_module_ast(const char *module_path) {
    if (!module_cache) return NULL;
    for (int i = 0; i < module_cache->count; i++) {
        if (strcmp(module_cache->loaded_paths[i], module_path) == 0) {
            return module_cache->loaded_asts[i];
        }
    }
    return NULL;
}

static void cache_module(const char *module_path) {
    init_module_cache();
    if (is_module_cached(module_path)) return;
    
    if (module_cache->count >= module_cache->capacity) {
        module_cache->capacity *= 2;
        module_cache->loaded_paths = realloc(module_cache->loaded_paths, 
                                             sizeof(char*) * module_cache->capacity);
        module_cache->loaded_asts = realloc(module_cache->loaded_asts,
                                            sizeof(ASTNode*) * module_cache->capacity);
    }
    module_cache->loaded_paths[module_cache->count] = strdup(module_path);
    module_cache->loaded_asts[module_cache->count] = NULL;  /* Set later */
    module_cache->count++;
}

static void cache_module_with_ast(const char *module_path, ASTNode *ast) {
    init_module_cache();
    
    /* Check if already cached - if so, update AST */
    for (int i = 0; i < module_cache->count; i++) {
        if (strcmp(module_cache->loaded_paths[i], module_path) == 0) {
            module_cache->loaded_asts[i] = ast;
            return;
        }
    }
    
    /* Not cached yet - add new entry */
    if (module_cache->count >= module_cache->capacity) {
        module_cache->capacity *= 2;
        module_cache->loaded_paths = realloc(module_cache->loaded_paths,
                                             sizeof(char*) * module_cache->capacity);
        module_cache->loaded_asts = realloc(module_cache->loaded_asts,
                                            sizeof(ASTNode*) * module_cache->capacity);
    }
    module_cache->loaded_paths[module_cache->count] = strdup(module_path);
    module_cache->loaded_asts[module_cache->count] = ast;
    module_cache->count++;
}

void clear_module_cache(void) {
    if (module_cache) {
        for (int i = 0; i < module_cache->count; i++) {
            free(module_cache->loaded_paths[i]);
            /* Free ASTs - they were allocated during module loading and are no longer needed */
            if (module_cache->loaded_asts[i]) {
                free_ast(module_cache->loaded_asts[i]);
            }
        }
        free(module_cache->loaded_paths);
        free(module_cache->loaded_asts);
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
    
    /* Create temporary directory (TMPDIR-aware) */
    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || tmpdir[0] == '\0') tmpdir = "/tmp";
    char temp_template[512];
    snprintf(temp_template, sizeof(temp_template), "%s/nanolang_module_XXXXXX", tmpdir);
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

/* Resolve module path relative to current file or in module search paths
 * Returns a malloc'd string that caller must free
 * Note: Returns const char* to match NanoLang's string type mapping
 */
const char *resolve_module_path(const char *module_path, const char *current_file) {
    if (!module_path) return NULL;
    
    /* If module_path is absolute or starts with ./, use as-is */
    if (module_path[0] == '/' || (module_path[0] == '.' && module_path[1] == '/')) {
        return strdup(module_path);
    }
    
    /* Check if this is a project-relative path (common prefixes like std/, stdlib/, examples/, src/) */
    if (strncmp(module_path, "examples/", 9) == 0 ||
        strncmp(module_path, "modules/", 8) == 0 ||
        strncmp(module_path, "src/", 4) == 0 ||
        strncmp(module_path, "src_nano/", 9) == 0 ||
        strncmp(module_path, "std/", 4) == 0 ||
        strncmp(module_path, "stdlib/", 7) == 0) {
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
                char test_path[2048];  /* Increased to handle long paths */
                for (int depth = 0; depth < 10; depth++) {
                    /* Check if this directory has modules/ or examples/ subdirectory */
                    snprintf(test_path, sizeof(test_path), "%s/modules", current_dir);
                    DIR *dir = opendir(test_path);
                    if (dir) {
                        closedir(dir);
                        /* Found project root - resolve path from here */
                        /* Prefer canonical stdlib/modules over legacy root-level paths */
                        const char *prefixes[] = {"stdlib/", "modules/", "", NULL};
                        for (int p = 0; prefixes[p] != NULL; p++) {
                            snprintf(test_path, sizeof(test_path), "%s/%s%s", current_dir, prefixes[p], module_path);
                            FILE *test = fopen(test_path, "r");
                            if (test) {
                                fclose(test);
                                return strdup(test_path);
                            }
                        }
                        break;
                    }
                    
                    /* Try examples/ directory */
                    snprintf(test_path, sizeof(test_path), "%s/examples", current_dir);
                    dir = opendir(test_path);
                    if (dir) {
                        closedir(dir);
                        /* Found project root - resolve path from here */
                        /* Prefer canonical stdlib/modules over legacy root-level paths */
                        const char *prefixes[] = {"stdlib/", "modules/", "", NULL};
                        for (int p = 0; prefixes[p] != NULL; p++) {
                            snprintf(test_path, sizeof(test_path), "%s/%s%s", current_dir, prefixes[p], module_path);
                            FILE *test = fopen(test_path, "r");
                            if (test) {
                                fclose(test);
                                return strdup(test_path);
                            }
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
            free((char*)resolved);
            resolved = NULL;
        }
    }

    /* Next, try common project-relative locations from current working directory.
     * This enables imports like "std/math/vector2d.nano" to resolve to
     * "stdlib/std/math/vector2d.nano" during tests (where current_file is often relative). */
    {
        /* Prefer canonical stdlib/modules over legacy root-level paths */
        const char *prefixes[] = {"stdlib/", "modules/", "", NULL};
        char test_path[2048];
        for (int p = 0; prefixes[p] != NULL; p++) {
            snprintf(test_path, sizeof(test_path), "%s%s", prefixes[p], module_path);
            FILE *test = fopen(test_path, "r");
            if (test) {
                fclose(test);
                return strdup(test_path);
            }
        }
    }
    
    /* If import path starts with "modules/", try NANO_MODULE_PATH with prefix stripped */
    if (strncmp(module_path, "modules/", 8) == 0) {
        const char *stripped = module_path + 8;  /* skip "modules/" */
        int path_count = 0;
        char **search_paths = get_module_search_paths(&path_count);
        for (int i = 0; i < path_count; i++) {
            char test_path[2048];
            snprintf(test_path, sizeof(test_path), "%s/%s", search_paths[i], stripped);
            FILE *test = fopen(test_path, "r");
            if (test) {
                fclose(test);
                free_module_search_paths(search_paths, path_count);
                return strdup(test_path);
            }
        }
        free_module_search_paths(search_paths, path_count);
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

static char *module_name_from_path(const char *module_path) {
    if (!module_path) {
        return NULL;
    }
    const char *last_slash = strrchr(module_path, '/');
    const char *last_dot = strrchr(module_path, '.');
    if (last_slash && last_dot && last_dot > last_slash) {
        size_t name_len = last_dot - (last_slash + 1);
        return strndup(last_slash + 1, name_len);
    }
    if (last_slash) {
        return strdup(last_slash + 1);
    }
    if (last_dot) {
        size_t name_len = last_dot - module_path;
        return strndup(module_path, name_len);
    }
    return strdup(module_path);
}

static Function *find_module_function(Environment *env, const char *module_name, const char *func_name) {
    if (!env || !func_name) {
        return NULL;
    }
    for (int i = 0; i < env->function_count; i++) {
        if (safe_strcmp(env->functions[i].name, func_name) == 0) {
            if (!module_name || !env->functions[i].module_name ||
                strcmp(env->functions[i].module_name, module_name) == 0) {
                return &env->functions[i];
            }
        }
    }
    return NULL;
}

/* Load and parse a module file */
static ASTNode *load_module_internal(const char *module_path, Environment *env, bool use_cache, ModuleList *modules_to_track) {
    if (!module_path) return NULL;
    
    /* Check if module is already loaded (only if using cache) */
    if (use_cache) {
        ASTNode *cached_ast = get_cached_module_ast(module_path);
        if (cached_ast) {
            /* Module already loaded - return cached AST */
            /* This allows compile_module_to_object to reuse the AST */
            /* without triggering a second parse (fixes nanolang-6h9) */
            return cached_ast;
        }
        
        /* Mark module as loading to prevent circular imports */
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
    if (!process_imports(module_ast, env, modules_to_track, module_path)) {
        fprintf(stderr, "Error: Failed to process imports for module '%s'\n", module_path);
        free_ast(module_ast);
        free_tokens(tokens, token_count);
        free(source);
        return NULL;
    }
    
    /* Type check module (without requiring main) */
    /* Save current module context before processing imported module */
    char *saved_current_module = env->current_module;
    
    /* Extract module name from path for function tagging */
    /* e.g., "modules/sdl/sdl.nano" -> "sdl" */
    char *module_name = NULL;
    const char *last_slash = strrchr(module_path, '/');
    const char *last_dot = strrchr(module_path, '.');
    if (last_slash && last_dot && last_dot > last_slash) {
        size_t name_len = last_dot - (last_slash + 1);
        module_name = strndup(last_slash + 1, name_len);
    } else if (last_slash) {
        module_name = strdup(last_slash + 1);
    } else if (last_dot) {
        size_t name_len = last_dot - module_path;
        module_name = strndup(module_path, name_len);
    } else {
        module_name = strdup(module_path);
    }
    
    env->current_module = module_name;  /* Set module context for function tagging */
    
    /* Register module for introspection BEFORE type checking so functions can be tracked */
    env_register_module(env, module_name, module_path, false);  /* is_unsafe will be updated later */
    
    if (!type_check_module(module_ast, env)) {
        fprintf(stderr, "Error: Type checking failed for module '%s'\n", module_path);
        /* NOTE: module_name may have been freed/overwritten by the module's own
         * `module <name>` declaration handler in the typechecker, so we must not
         * free it here.
         */
        env->current_module = saved_current_module;  /* Restore context */
        free_ast(module_ast);
        free_tokens(tokens, token_count);
        free(source);
        return NULL;
    }
    
    /* Restore original module context */
    /* NOTE: We intentionally DON'T free module_name here because:
     * 1. Functions registered during type_check have module_name pointers that reference it
     * 2. Those pointers are just shallow copies from the struct assignment
     * 3. Freeing would create dangling pointers
     * 4. This is a short-lived compiler process, so the memory leak is acceptable
     */
    env->current_module = saved_current_module;
    
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
    
    /* Cache the AST with the module path for reuse (fixes nanolang-6h9) */
    if (use_cache) {
        cache_module_with_ast(module_path, module_ast);
    }
    
    return module_ast;
}

/* Public wrapper for load_module that uses cache */
ASTNode *load_module(const char *module_path, Environment *env) {
    return load_module_internal(module_path, env, true, NULL);
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
            char *module_path = (char*)resolve_module_path(item->as.import_stmt.module_path, current_file);
            char *module_alias = item->as.import_stmt.module_alias;  /* NULL if no alias */
            
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

                /* Load module and track transitive imports for compilation */
                module_ast = load_module_internal(module_path, env, true, modules);
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
            
            /* Add to module list (even if already cached) */
            if (modules) {
                module_list_add(modules, module_path);
            }

            /* If module was already cached and returned NULL, try to grab cached AST for alias handling */
            if (module_ast == NULL) {
                module_ast = get_cached_module_ast(module_path);
                if (!module_ast) {
                    free(module_path);
                    continue;
                }
            }
            
            char *orig_module_name = NULL;
            for (int j = 0; j < module_ast->as.program.count; j++) {
                ASTNode *node = module_ast->as.program.items[j];
                if (node->type == AST_MODULE_DECL && !orig_module_name) {
                    orig_module_name = node->as.module_decl.name;
                }
            }
            
            /* Register namespace if module has an alias */
            if (module_alias) {
                /* Extract function names, struct names, enum names, union names from module */
                /* Count them first */
                int func_count = 0, struct_count = 0, enum_count = 0, union_count = 0;
                for (int j = 0; j < module_ast->as.program.count; j++) {
                    ASTNode *node = module_ast->as.program.items[j];
                    if (node->type == AST_FUNCTION) func_count++;
                    else if (node->type == AST_STRUCT_DEF) struct_count++;
                    else if (node->type == AST_ENUM_DEF) enum_count++;
                    else if (node->type == AST_UNION_DEF) union_count++;
                }
                
                /* Allocate arrays */
                char **func_names = malloc(sizeof(char*) * (func_count > 0 ? func_count : 1));
                char **struct_names = malloc(sizeof(char*) * (struct_count > 0 ? struct_count : 1));
                char **enum_names = malloc(sizeof(char*) * (enum_count > 0 ? enum_count : 1));
                char **union_names = malloc(sizeof(char*) * (union_count > 0 ? union_count : 1));
                
                /* Fill arrays */
                int f_idx = 0, s_idx = 0, e_idx = 0, u_idx = 0;
                for (int j = 0; j < module_ast->as.program.count; j++) {
                    ASTNode *node = module_ast->as.program.items[j];
                    if (node->type == AST_FUNCTION) {
                        func_names[f_idx++] = strdup(node->as.function.name);
                    } else if (node->type == AST_STRUCT_DEF) {
                        struct_names[s_idx++] = strdup(node->as.struct_def.name);
                    } else if (node->type == AST_ENUM_DEF) {
                        enum_names[e_idx++] = strdup(node->as.enum_def.name);
                    } else if (node->type == AST_UNION_DEF) {
                        union_names[u_idx++] = strdup(node->as.union_def.name);
                    }
                }
                
                /* Register the namespace */
                env_register_namespace(env, module_alias, orig_module_name,
                                      func_names, func_count,
                                      struct_names, struct_count,
                                      enum_names, enum_count,
                                      union_names, union_count);
            }

            /* Apply import aliases for selective imports: from "module" import foo as bar */
            if (item->as.import_stmt.is_selective &&
                item->as.import_stmt.import_symbols &&
                item->as.import_stmt.import_symbol_count > 0) {
                char *module_name_for_alias = NULL;
                if (orig_module_name) {
                    module_name_for_alias = strdup(orig_module_name);
                } else {
                    module_name_for_alias = module_name_from_path(module_path);
                }
                
                for (int j = 0; j < item->as.import_stmt.import_symbol_count; j++) {
                    const char *symbol = item->as.import_stmt.import_symbols[j];
                    const char *alias = item->as.import_stmt.import_aliases
                        ? item->as.import_stmt.import_aliases[j]
                        : NULL;
                    
                    if (!alias || alias[0] == '\0' || strcmp(alias, symbol) == 0) {
                        continue;
                    }
                    
                    if (env_get_function(env, alias)) {
                        fprintf(stderr, "Error at line %d, column %d: Import alias '%s' conflicts with existing function\n",
                                item->line, item->column, alias);
                        free(module_name_for_alias);
                        free(module_path);
                        for (int k = 0; k < unpacked_count; k++) {
                            free(unpacked_dirs[k]);
                        }
                        free(unpacked_dirs);
                        return false;
                    }
                    
                    Function *func = find_module_function(env, module_name_for_alias, symbol);
                    if (!func) {
                        fprintf(stderr, "Error at line %d, column %d: Symbol '%s' not found in module for alias '%s'\n",
                                item->line, item->column, symbol, alias);
                        free(module_name_for_alias);
                        free(module_path);
                        for (int k = 0; k < unpacked_count; k++) {
                            free(unpacked_dirs[k]);
                        }
                        free(unpacked_dirs);
                        return false;
                    }
                    
                    Function alias_func = *func;
                    alias_func.name = strdup(alias);
                    const char *orig_name = func->alias_of ? func->alias_of : func->name;
                    alias_func.alias_of = orig_name ? strdup(orig_name) : NULL;
                    env_define_function(env, alias_func);
                }
                
                if (module_name_for_alias) {
                    free(module_name_for_alias);
                }
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
bool compile_module_to_object(const char *module_path, const char *output_obj, Environment *env, bool verbose) {
    if (!module_path || !output_obj) return false;

    /* env is intentionally unused: module compilation is done in an isolated environment */
    (void)env;

    /* Module loading uses a global cache. Compile module objects with a fresh
     * cache, but restore the caller's cache afterward (needed for transpilation
     * and forward declarations in the main compilation).
     */
    ModuleCache *saved_cache = module_cache;
    module_cache = NULL;

    /*
     * IMPORTANT:
     * Module objects must be compiled in an isolated environment so that the
     * generated C does not accidentally include types/externs from the top-level
     * program (e.g. SDL externs leaking into stdlib objects).
     */
    Environment *module_env = create_environment();
    if (!module_env) {
        fprintf(stderr, "Error: Failed to create environment for module compilation\n");
        module_cache = saved_cache;
        return false;
    }
    module_env->emit_module_metadata = false;
    module_env->emit_c_main = false;

    /* Use a fresh module cache per module compilation to avoid cross-env AST reuse */

    ASTNode *module_ast = load_module_internal(module_path, module_env, true, NULL);
    if (!module_ast) {
        fprintf(stderr, "Error: Failed to load module '%s' for compilation\n", module_path);
        clear_module_cache();
        module_cache = saved_cache;
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
    ModuleMetadata *meta = NULL;  // extract_module_metadata(env, module_name);
    
    /*
     * Transpile module to C.
     *
     * IMPORTANT: We reuse the shared Environment to avoid module reload/AST corruption,
     * but that environment may contain a program-level `main` from the top-level compile.
     * When generating a module object, we must NOT emit a C main() wrapper.
     */
    Function *saved_main = env_get_function(module_env, "main");
    bool saved_main_is_extern = false;
    if (saved_main) {
        saved_main_is_extern = saved_main->is_extern;
        saved_main->is_extern = true;
    }

    char *c_code = transpile_to_c(module_ast, module_env, module_path);

    if (saved_main) {
        saved_main->is_extern = saved_main_is_extern;
    }
    if (!c_code) {
        fprintf(stderr, "Error: Failed to transpile module '%s'\n", module_path);
        if (meta) free_module_metadata(meta);
        clear_module_cache();
        module_cache = saved_cache;
        free_environment(module_env);
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
        /* Don't free AST - it's owned by the cache */
        clear_module_cache();
        module_cache = saved_cache;
        free_environment(module_env);
        return false;
    }
    
    fprintf(c_file, "%s", c_code);
    fclose(c_file);
    
    if (verbose) {
        printf("✓ Generated module C code: %s\n", temp_c_file);
    }

    /* Ensure any generic list headers needed by this module exist before compilation.
     * (The top-level compilation generates list runtimes later, but module objects
     * must be able to include the headers during their own cc -c step.)
     */
    {
        const char *scan_ptr = c_code;
        char detected_types[64][64];
        int detected_count = 0;

        while ((scan_ptr = strstr(scan_ptr, "List_")) != NULL) {
            scan_ptr += 5;
            const char *end_ptr = scan_ptr;

            while ((*end_ptr >= 'A' && *end_ptr <= 'Z') ||
                   (*end_ptr >= 'a' && *end_ptr <= 'z') ||
                   (*end_ptr >= '0' && *end_ptr <= '9') ||
                   *end_ptr == '_') {
                end_ptr++;
            }

            if (*end_ptr == '*' || *end_ptr == ' ' || *end_ptr == '\n') {
                int len = (int)(end_ptr - scan_ptr);
                if (len <= 0 || len >= 63) continue;

                char type_name[64];
                strncpy(type_name, scan_ptr, (size_t)len);
                type_name[len] = '\0';

                if (strcmp(type_name, "int") == 0 ||
                    strcmp(type_name, "string") == 0 ||
                    strcmp(type_name, "token") == 0) {
                    continue;
                }

                bool already_detected = false;
                for (int i = 0; i < detected_count; i++) {
                    if (strcmp(detected_types[i], type_name) == 0) {
                        already_detected = true;
                        break;
                    }
                }

                if (!already_detected && detected_count < 64) {
                    strncpy(detected_types[detected_count], type_name, 63);
                    detected_types[detected_count][63] = '\0';
                    detected_count++;

                    const char *c_type = type_name;
                    if (strcmp(type_name, "LexerToken") == 0) c_type = "Token";
                    else if (strcmp(type_name, "NSType") == 0) c_type = "NSType";

                    char gen_cmd[512];
                    snprintf(gen_cmd, sizeof(gen_cmd),
                             "./scripts/generate_list.sh %s /tmp %s > /dev/null 2>&1",
                             type_name, c_type);
                    if (verbose) {
                        printf("[Modules] Generating List<%s> runtime...\n", type_name);
                    }
                    (void)system(gen_cmd);
                }
            }
        }
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
    
    const char *cc = getenv("NANO_CC");
    if (!cc) cc = getenv("CC");
    if (!cc) cc = "cc";

    char compile_cmd[1024];
    snprintf(compile_cmd, sizeof(compile_cmd),
            "%s -std=c99 -Isrc -Imodules/std -Imodules/std/collections -Imodules/std/json -Imodules/std/io -Imodules/std/math -Imodules/std/peg -Imodules/std/string -Imodules/sdl_helpers %s -c -o %s %s",
            cc, sdl_flags, output_obj, temp_c_file);
    
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
        clear_module_cache();
        module_cache = saved_cache;
        /* Don't free AST or environment - they're owned by the cache/caller */
        free_environment(module_env);
        return false;
    }
    
    /* Clean up temporary C file */
    if (!verbose) {
        remove(temp_c_file);
    } else {
        printf("✓ Compiled module to object file: %s\n", output_obj);
        printf("  C source kept at: %s\n", temp_c_file);
    }
    
    free(c_code);
    clear_module_cache();
    module_cache = saved_cache;
    /* Don't free AST or environment - they're owned by the cache/caller */
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
            
            build_infos[build_info_count++] = info;
            c_modules_built++;

            /* 
             * IMPORTANT: If the module ALSO has a .nano file with implementations 
             * (not just externs), we must compile it too!
             */
            if (!is_ffi_only_module(module_path)) {
                if (verbose) {
                    printf("[Modules] Also compiling nanolang parts of C module '%s'\n", meta->name);
                }

                /*
                 * IMPORTANT:
                 * A single C-backed module directory (e.g. modules/std/) can contain multiple
                 * NanoLang source files that may be imported independently (env.nano, fs.nano,
                 * binary.nano, etc).
                 *
                 * If we always emit to obj/nano_modules/<module>_nano.o, later imports overwrite
                 * earlier ones and we end up linking only the last compiled Nano object, causing
                 * undefined references on strict linkers (Linux CI).
                 */
                system("mkdir -p obj/nano_modules 2>/dev/null");

                const char *last_slash = strrchr(module_path, '/');
                const char *base_name = last_slash ? last_slash + 1 : module_path;

                char base_without_ext[256];
                snprintf(base_without_ext, sizeof(base_without_ext), "%s", base_name);
                char *dot = strrchr(base_without_ext, '.');
                if (dot && strcmp(dot, ".nano") == 0) {
                    *dot = '\0';
                }

                char nano_obj[512];
                snprintf(nano_obj, sizeof(nano_obj), "obj/nano_modules/%s_nano_%s.o", meta->name, base_without_ext);
                
                if (!compile_module_to_object(module_path, nano_obj, env, verbose)) {
                    fprintf(stderr, "Error: Failed to compile nanolang parts of module '%s'\n", meta->name);
                    module_metadata_free(meta);
                    free(module_dir);
                    module_builder_free(builder);
                    for (int j = 0; j < build_info_count; j++) {
                        module_build_info_free(build_infos[j]);
                    }
                    free(build_infos);
                    return false;
                }
                
                /* Check if this object file is already in the buffer (avoid duplicates) */
                bool already_added = false;
                if (module_objs_buffer[0] != '\0') {
                    char *found = strstr(module_objs_buffer, nano_obj);
                    if (found) {
                        /* Verify it's a complete match, not a substring */
                        size_t nano_obj_len = strlen(nano_obj);
                        if ((found == module_objs_buffer || found[-1] == ' ') &&
                            (found[nano_obj_len] == '\0' || found[nano_obj_len] == ' ')) {
                            already_added = true;
                        }
                    }
                }
                
                if (!already_added && strlen(module_objs_buffer) + strlen(nano_obj) + 2 < buffer_size) {
                    if (module_objs_buffer[0] != '\0') {
                        strcat(module_objs_buffer, " ");
                    }
                    strcat(module_objs_buffer, nano_obj);
                }
            }
            
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
            
            /*
             * IMPORTANT:
             * Keep NanoLang-compiled module objects separate from C-compiled objects.
             * The C reference compiler builds src/lexer.c -> obj/lexer.o, etc.
             * If we also emit modules to obj/<name>.o (e.g. lexer.nano -> obj/lexer.o),
             * we can clobber C objects and break linking.
             */
            snprintf(obj_file, sizeof(obj_file), "obj/nano_modules/%s.o", base_without_ext);

            /* Ensure obj directory exists */
            system("mkdir -p obj/nano_modules 2>/dev/null");
            
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
            
            /* Add to module_objs buffer (check for duplicates) */
            bool already_added = false;
            if (module_objs_buffer[0] != '\0') {
                char *found = strstr(module_objs_buffer, obj_file);
                if (found) {
                    /* Verify it's a complete match, not a substring */
                    size_t obj_file_len = strlen(obj_file);
                    if ((found == module_objs_buffer || found[-1] == ' ') &&
                        (found[obj_file_len] == '\0' || found[obj_file_len] == ' ')) {
                        already_added = true;
                    }
                }
            }
            
            if (!already_added && strlen(module_objs_buffer) + strlen(obj_file) + 2 < buffer_size) {
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
            // Skip NULL or empty flags
            if (!link_flags[i] || link_flags[i][0] == '\0') {
                free(link_flags[i]);
                continue;
            }

            // Skip direct .tbd inputs (macOS text stubs) to avoid linker warnings
            size_t flag_len = strlen(link_flags[i]);
            if (flag_len > 4 && strcmp(link_flags[i] + flag_len - 4, ".tbd") == 0) {
                if (verbose) {
                    fprintf(stderr, "[Modules] Warning: Skipping direct .tbd link input %s\n", link_flags[i]);
                }
                free(link_flags[i]);
                continue;
            }
            
            // Validate flag is printable ASCII (no UTF-8 or binary garbage)
            bool valid = true;
            for (const char *p = link_flags[i]; *p; p++) {
                if ((unsigned char)*p < 32 || (unsigned char)*p > 126) {
                    // Non-printable or non-ASCII character found
                    if (verbose) {
                        fprintf(stderr, "[Modules] Warning: Skipping invalid link flag with non-ASCII character (byte 0x%02x)\n", (unsigned char)*p);
                    }
                    valid = false;
                    break;
                }
            }
            
            if (!valid) {
                free(link_flags[i]);
                continue;
            }
            
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
                // Skip NULL or empty flags
                if (!compile_flags[i] || compile_flags[i][0] == '\0') {
                    free(compile_flags[i]);
                    continue;
                }
                
                // Validate flag is printable ASCII (no UTF-8 or binary garbage)
                bool valid = true;
                for (const char *p = compile_flags[i]; *p; p++) {
                    if ((unsigned char)*p < 32 || (unsigned char)*p > 126) {
                        // Non-printable or non-ASCII character found
                        if (verbose) {
                            fprintf(stderr, "[Modules] Warning: Skipping invalid compile flag with non-ASCII character (byte 0x%02x)\n", (unsigned char)*p);
                        }
                        valid = false;
                        break;
                    }
                }
                
                if (!valid) {
                    free(compile_flags[i]);
                    continue;
                }
                
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

            /* Auto-generate memory semantics annotations based on return type */
            if (env->functions[i].return_type == TYPE_STRING) {
                /* String return types are GC-managed */
                meta->functions[i].returns_gc_managed = true;
                meta->functions[i].requires_manual_free = false;
                meta->functions[i].cleanup_function = NULL;
            } else if (env->functions[i].return_type == TYPE_OPAQUE) {
                /* Opaque types typically require manual free */
                meta->functions[i].returns_gc_managed = false;
                meta->functions[i].requires_manual_free = true;

                /* Infer cleanup function name from function name convention */
                /* Pattern: <type>_<action> -> <type>_free (e.g., regex_compile -> regex_free) */
                if (env->functions[i].name) {
                    const char *name = env->functions[i].name;
                    /* Find last underscore to extract type prefix */
                    const char *last_underscore = strrchr(name, '_');
                    if (last_underscore && last_underscore > name) {
                        /* Extract prefix (type name) */
                        size_t prefix_len = last_underscore - name;
                        char *cleanup_name = malloc(prefix_len + 6); /* prefix + "_free" + \0 */
                        if (cleanup_name) {
                            memcpy(cleanup_name, name, prefix_len);
                            cleanup_name[prefix_len] = '\0';
                            strcat(cleanup_name, "_free");
                            meta->functions[i].cleanup_function = cleanup_name;
                        } else {
                            meta->functions[i].cleanup_function = NULL;
                        }
                    } else {
                        meta->functions[i].cleanup_function = NULL;
                    }
                } else {
                    meta->functions[i].cleanup_function = NULL;
                }

                /* Detect borrowed reference returns (accessors vs constructors) */
                /* Pattern: functions with "get" in the name return borrowed refs */
                /* Constructors (parse, new_, compile) return owned refs */
                meta->functions[i].returns_borrowed = false;
                if (env->functions[i].name) {
                    const char *name = env->functions[i].name;
                    /* Check if it's an accessor function (gets existing data) */
                    if (strstr(name, "get") != NULL ||
                        strstr(name, "Get") != NULL ||
                        strstr(name, "as_") != NULL) {  /* as_string, as_bool, etc. also borrowed */
                        meta->functions[i].returns_borrowed = true;
                        /* Borrowed refs don't need cleanup */
                        meta->functions[i].requires_manual_free = false;
                        if (meta->functions[i].cleanup_function) {
                            free(meta->functions[i].cleanup_function);
                            meta->functions[i].cleanup_function = NULL;
                        }
                    }
                }
            } else {
                /* Other types don't require special memory management */
                meta->functions[i].returns_gc_managed = false;
                meta->functions[i].requires_manual_free = false;
                meta->functions[i].returns_borrowed = false;
                meta->functions[i].cleanup_function = NULL;
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
/* Get number of imports in a cached module
 * Returns -1 if module not found
 * Note: Returns int64_t to match NanoLang's int type mapping
 */
int64_t module_get_import_count(const char *module_path) {
    if (!module_path) return -1;
    
    ASTNode *module_ast = get_cached_module_ast(module_path);
    if (!module_ast || module_ast->type != AST_PROGRAM) {
        return -1;
    }
    
    int64_t count = 0;
    for (int i = 0; i < module_ast->as.program.count; i++) {
        ASTNode *item = module_ast->as.program.items[i];
        if (item && item->type == AST_IMPORT) {
            count++;
        }
    }
    
    return count;
}

/* Get the path of the nth import in a cached module
 * Returns a malloc'd string that caller must free, or NULL on error
 * Note: Returns const char* to match NanoLang's string type mapping
 * Note: index is int64_t to match NanoLang's int type mapping
 */
const char *module_get_import_path(const char *module_path, int64_t index) {
    if (!module_path || index < 0) return NULL;
    
    ASTNode *module_ast = get_cached_module_ast(module_path);
    if (!module_ast || module_ast->type != AST_PROGRAM) {
        return NULL;
    }
    
    int count = 0;
    for (int i = 0; i < module_ast->as.program.count; i++) {
        ASTNode *item = module_ast->as.program.items[i];
        if (item && item->type == AST_IMPORT) {
            if (count == index) {
                /* Resolve the import path */
                const char *resolved = resolve_module_path(
                    item->as.import_stmt.module_path, 
                    module_path
                );
                return resolved;
            }
            count++;
        }
    }
    
    return NULL;
}

/* Generate forward declarations for a module's public functions
 * Returns a malloc'd string with C forward declarations, or NULL on error
 * Caller must free the returned string
 * Note: Returns const char* to match NanoLang's string type mapping
 */
const char *module_generate_forward_declarations(const char *module_path) {
    if (!module_path) return NULL;
    
    ASTNode *module_ast = get_cached_module_ast(module_path);
    if (!module_ast || module_ast->type != AST_PROGRAM) {
        return NULL;
    }
    
    /* Find module name */
    const char *module_name = NULL;
    for (int i = 0; i < module_ast->as.program.count; i++) {
        ASTNode *item = module_ast->as.program.items[i];
        if (item && item->type == AST_MODULE_DECL && item->as.module_decl.name) {
            module_name = item->as.module_decl.name;
            break;
        }
    }
    
    /* Build forward declarations string */
    StringBuilder *sb = sb_create();
    if (!sb) return NULL;
    
    sb_append(sb, "/* Forward declarations from ");
    sb_append(sb, module_path);
    sb_append(sb, " */\n");
    
    /* Track emitted function names to avoid duplicates */
    int emitted_count = 0;
    int emitted_capacity = 64;
    char **emitted = malloc(sizeof(char*) * emitted_capacity);
    if (!emitted) {
        free(sb->buffer);
        free(sb);
        return NULL;
    }
    
    /* Iterate through functions in module */
    for (int i = 0; i < module_ast->as.program.count; i++) {
        ASTNode *item = module_ast->as.program.items[i];
        if (!item || item->type != AST_FUNCTION) continue;
        if (!item->as.function.is_pub) continue;  /* Only public functions */
        if (item->as.function.is_extern) continue;  /* Skip extern functions */
        if (strcmp(item->as.function.name, "main") == 0) continue;  /* Skip main */
        
        /* Build C function name with module prefix */
        char c_name_buf[512];
        const char *c_name = NULL;
        if (module_name) {
            snprintf(c_name_buf, sizeof(c_name_buf), "%s__%s", module_name, item->as.function.name);
            c_name = c_name_buf;
        } else {
            snprintf(c_name_buf, sizeof(c_name_buf), "nl_%s", item->as.function.name);
            c_name = c_name_buf;
        }
        
        /* Check if already emitted */
        bool seen = false;
        for (int j = 0; j < emitted_count; j++) {
            if (strcmp(emitted[j], c_name) == 0) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        
        /* Add to emitted list */
        if (emitted_count >= emitted_capacity) {
            emitted_capacity *= 2;
            char **new_emitted = realloc(emitted, sizeof(char*) * emitted_capacity);
            if (!new_emitted) {
                for (int j = 0; j < emitted_count; j++) free(emitted[j]);
                free(emitted);
                free(sb->buffer);
                free(sb);
                return NULL;
            }
            emitted = new_emitted;
        }
        emitted[emitted_count++] = strdup(c_name);
        
        /* Generate forward declaration */
        sb_append(sb, "extern ");
        
        /* Return type - simplified version */
        Type ret_type = item->as.function.return_type;
        if (ret_type == TYPE_INT) {
            sb_append(sb, "int64_t");
        } else if (ret_type == TYPE_FLOAT) {
            sb_append(sb, "double");
        } else if (ret_type == TYPE_BOOL) {
            sb_append(sb, "int");
        } else if (ret_type == TYPE_STRING) {
            sb_append(sb, "char*");
        } else if (ret_type == TYPE_VOID) {
            sb_append(sb, "void");
        } else if (ret_type == TYPE_STRUCT && item->as.function.return_struct_type_name) {
            char buf[256];
            snprintf(buf, sizeof(buf), "nl_%s", item->as.function.return_struct_type_name);
            sb_append(sb, buf);
        } else if (ret_type == TYPE_UNION && item->as.function.return_struct_type_name) {
            char buf[256];
            snprintf(buf, sizeof(buf), "nl_%s", item->as.function.return_struct_type_name);
            sb_append(sb, buf);
        } else if (ret_type == TYPE_LIST_GENERIC && item->as.function.return_struct_type_name) {
            char buf[256];
            snprintf(buf, sizeof(buf), "List_%s*", item->as.function.return_struct_type_name);
            sb_append(sb, buf);
        } else {
            sb_append(sb, "void");  /* Fallback */
        }
        
        char name_buf[1024];
        snprintf(name_buf, sizeof(name_buf), " %s(", c_name);
        sb_append(sb, name_buf);
        
        /* Parameters */
        for (int p = 0; p < item->as.function.param_count; p++) {
            if (p > 0) sb_append(sb, ", ");
            Parameter *param = &item->as.function.params[p];
            char param_buf[256];
            
            if (param->type == TYPE_INT) {
                sb_append(sb, "int64_t");
            } else if (param->type == TYPE_FLOAT) {
                sb_append(sb, "double");
            } else if (param->type == TYPE_BOOL) {
                sb_append(sb, "int");
            } else if (param->type == TYPE_STRING) {
                sb_append(sb, "char*");
            } else if (param->type == TYPE_STRUCT && param->struct_type_name) {
                snprintf(param_buf, sizeof(param_buf), "nl_%s", param->struct_type_name);
                sb_append(sb, param_buf);
            } else if (param->type == TYPE_UNION && param->struct_type_name) {
                snprintf(param_buf, sizeof(param_buf), "nl_%s", param->struct_type_name);
                sb_append(sb, param_buf);
            } else if (param->type == TYPE_LIST_GENERIC && param->struct_type_name) {
                snprintf(param_buf, sizeof(param_buf), "List_%s*", param->struct_type_name);
                sb_append(sb, param_buf);
            } else {
                sb_append(sb, "void");  /* Fallback */
            }
            
            if (param->name) {
                snprintf(param_buf, sizeof(param_buf), " %s", param->name);
                sb_append(sb, param_buf);
            } else {
                snprintf(param_buf, sizeof(param_buf), " param%d", p);
                sb_append(sb, param_buf);
            }
        }
        
        sb_append(sb, ");\n");
    }
    
    /* Cleanup */
    for (int i = 0; i < emitted_count; i++) {
        free(emitted[i]);
    }
    free(emitted);
    
    sb_append(sb, "\n");
    
    /* Return the string (caller must free) */
    char *result = strdup(sb->buffer);
    free(sb->buffer);
    free(sb);
    return result;
}

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
