#include "nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* FFI Binding Generator
 * Generates nanolang module files from C header files
 * 
 * Usage: nanoc-ffi <header_file.h> -o <module.nano> -l <library> -L <lib_path> -I <include_path>
 */

typedef struct {
    char **include_paths;
    int include_count;
    char **library_paths;
    int library_path_count;
    char **libraries;
    int library_count;
    const char *output_file;
    const char *module_name;
} FFIOptions;

/* Simple C tokenizer for parsing headers */
typedef struct {
    char *source;
    int pos;
    int line;
} CTokenizer;

static char *next_token(CTokenizer *t) {
    /* Skip whitespace and comments */
    while (t->source[t->pos] != '\0') {
        if (isspace(t->source[t->pos])) {
            if (t->source[t->pos] == '\n') t->line++;
            t->pos++;
            continue;
        }
        
        /* Skip single-line comments */
        if (t->source[t->pos] == '/' && t->source[t->pos + 1] == '/') {
            while (t->source[t->pos] != '\0' && t->source[t->pos] != '\n') {
                t->pos++;
            }
            continue;
        }
        
        /* Skip multi-line comments */
        if (t->source[t->pos] == '/' && t->source[t->pos + 1] == '*') {
            t->pos += 2;
            while (t->source[t->pos] != '\0') {
                if (t->source[t->pos] == '*' && t->source[t->pos + 1] == '/') {
                    t->pos += 2;
                    break;
                }
                if (t->source[t->pos] == '\n') t->line++;
                t->pos++;
            }
            continue;
        }
        
        break;
    }
    
    if (t->source[t->pos] == '\0') return NULL;
    
    int start = t->pos;
    
    /* Identifier or keyword */
    if (isalpha(t->source[t->pos]) || t->source[t->pos] == '_') {
        while (isalnum(t->source[t->pos]) || t->source[t->pos] == '_') {
            t->pos++;
        }
    }
    /* Number */
    else if (isdigit(t->source[t->pos])) {
        while (isdigit(t->source[t->pos]) || t->source[t->pos] == '.' || 
               t->source[t->pos] == 'e' || t->source[t->pos] == 'E' ||
               t->source[t->pos] == '+' || t->source[t->pos] == '-') {
            t->pos++;
        }
    }
    /* String literal */
    else if (t->source[t->pos] == '"') {
        t->pos++;
        while (t->source[t->pos] != '\0' && t->source[t->pos] != '"') {
            if (t->source[t->pos] == '\\') t->pos++;
            t->pos++;
        }
        if (t->source[t->pos] == '"') t->pos++;
    }
    /* Character literal */
    else if (t->source[t->pos] == '\'') {
        t->pos++;
        while (t->source[t->pos] != '\0' && t->source[t->pos] != '\'') {
            if (t->source[t->pos] == '\\') t->pos++;
            t->pos++;
        }
        if (t->source[t->pos] == '\'') t->pos++;
    }
    /* Punctuation */
    else {
        t->pos++;
    }
    
    int len = t->pos - start;
    char *token = malloc(len + 1);
    strncpy(token, t->source + start, len);
    token[len] = '\0';
    return token;
}

/* Check if token is a C type keyword */
static bool is_c_type_keyword(const char *tok) {
    static const char *type_keywords[] = {
        "unsigned", "signed", "const", "volatile",
        "char", "int", "long", "short", "float", "double", "void", "bool",
        "struct", "union", "enum", "static", "extern", "inline"
    };
    static int num_keywords = sizeof(type_keywords) / sizeof(type_keywords[0]);
    for (int i = 0; i < num_keywords; i++) {
        if (strcmp(tok, type_keywords[i]) == 0) return true;
    }
    return false;
}

/* Map C types to nanolang types */
static const char *map_c_type_to_nano(const char *c_type) {
    if (strcmp(c_type, "int") == 0 || strcmp(c_type, "int32_t") == 0) return "int";
    if (strcmp(c_type, "long") == 0 || strcmp(c_type, "int64_t") == 0 || 
        strcmp(c_type, "long long") == 0) return "int";
    if (strcmp(c_type, "float") == 0 || strcmp(c_type, "double") == 0) return "float";
    if (strcmp(c_type, "char") == 0 || strcmp(c_type, "char*") == 0 || 
        strcmp(c_type, "const char*") == 0) return "string";
    if (strcmp(c_type, "void") == 0) return "void";
    if (strcmp(c_type, "bool") == 0 || strcmp(c_type, "_Bool") == 0) return "bool";
    
    /* Pointer types - treat as int for now (we'll need better handling later) */
    if (strstr(c_type, "*") != NULL) return "int";
    
    /* Unknown type - return as-is (might be a struct/enum) */
    return c_type;
}

/* Parse a C function declaration and generate nanolang extern declaration */
static bool parse_function_declaration(CTokenizer *t, FILE *out) {
    /* Save position to restore if this isn't a function */
    int saved_pos = t->pos;
    int saved_line = t->line;
    
    /* Look for pattern: [static|extern|inline]* type function_name(params) [;|{] */
    /* Skip storage class specifiers */
    char *token = next_token(t);
    while (token && (strcmp(token, "static") == 0 || strcmp(token, "extern") == 0 || 
                     strcmp(token, "inline") == 0 || strcmp(token, "_inline") == 0)) {
        free(token);
        token = next_token(t);
    }
    
    if (!token) {
        t->pos = saved_pos;
        t->line = saved_line;
        return false;
    }
    
    /* Try to parse return type */
    char return_type[256] = "";
    strncpy(return_type, token, sizeof(return_type) - 1);
    free(token);
    
    /* Collect return type (may be multiple tokens: "unsigned", "long", "int") */
    token = next_token(t);
    while (token && (strcmp(token, "unsigned") == 0 || strcmp(token, "signed") == 0 ||
                     strcmp(token, "long") == 0 || strcmp(token, "short") == 0 ||
                     strcmp(token, "const") == 0)) {
        char temp[512];
        snprintf(temp, sizeof(temp), "%s %s", return_type, token);
        strncpy(return_type, temp, sizeof(return_type) - 1);
        free(token);
        token = next_token(t);
    }
    
    /* Now we should have the function name */
    if (!token) {
        t->pos = saved_pos;
        t->line = saved_line;
        return false;
    }
    
    char *func_name = token;
    
    /* Check if next token is '(' - if not, this isn't a function */
    token = next_token(t);
    if (!token || strcmp(token, "(") != 0) {
        free(func_name);
        if (token) free(token);
        t->pos = saved_pos;
        t->line = saved_line;
        return false;
    }
    free(token);
    
    /* Parse parameters */
    fprintf(out, "extern fn %s(", func_name);
    bool first_param = true;
    
    token = next_token(t);
    while (token && strcmp(token, ")") != 0) {
        if (strcmp(token, ",") == 0) {
            free(token);
            token = next_token(t);
            continue;
        }
        
        if (!first_param) {
            fprintf(out, ", ");
        }
        first_param = false;
        
        /* Parse parameter type */
        char param_type[256] = "";
        strncpy(param_type, token, sizeof(param_type) - 1);
        free(token);
        
        /* Collect type modifiers and base type */
        token = next_token(t);
        char param_name[128] = "";
        
        while (token && strcmp(token, ",") != 0 && strcmp(token, ")") != 0) {
            /* Check if this is a type keyword or modifier */
            if (is_c_type_keyword(token) || strcmp(token, "*") == 0 || strcmp(token, "**") == 0) {
                /* This is part of the type */
                char temp[512];
                if (strcmp(token, "*") == 0 || strcmp(token, "**") == 0) {
                    snprintf(temp, sizeof(temp), "%s%s", param_type, token);
                } else {
                    snprintf(temp, sizeof(temp), "%s %s", param_type, token);
                }
                strncpy(param_type, temp, sizeof(param_type) - 1);
                free(token);
                token = next_token(t);
            } else if ((isalpha(token[0]) || token[0] == '_') && strlen(param_type) > 0) {
                /* If we already have a type, this is likely the parameter name */
                /* But check if it might be a struct/enum name first */
                /* For now, assume it's the parameter name if we have a base type */
                snprintf(param_name, sizeof(param_name), "%s", token);
                free(token);
                token = next_token(t);
                break;  /* Parameter name found, stop collecting type */
            } else {
                /* Unknown - might be struct name or parameter name */
                /* If it looks like an identifier and we don't have a type yet, it might be a struct name */
                if (isalpha(token[0]) || token[0] == '_') {
                    char temp[512];
                    snprintf(temp, sizeof(temp), "%s %s", param_type, token);
                    strncpy(param_type, temp, sizeof(param_type) - 1);
                    free(token);
                    token = next_token(t);
                } else {
                    break;
                }
            }
        }
        
        /* Skip array brackets */
        while (token && strcmp(token, "[") == 0) {
            free(token);
            token = next_token(t);  /* Skip '[' */
            if (token) {
                free(token);
                token = next_token(t);  /* Skip size or ']' */
            }
            if (token && strcmp(token, "]") != 0) {
                free(token);
                token = next_token(t);  /* Skip ']' */
            } else if (token) {
                free(token);
            }
            if (token) {
                free(token);
                token = next_token(t);
            }
        }
        
        /* Map C type to nanolang type */
        const char *nano_type = map_c_type_to_nano(param_type);
        if (param_name[0] != '\0') {
            fprintf(out, "%s: %s", param_name, nano_type);
        } else {
            static int param_num = 0;
            fprintf(out, "param%d: %s", param_num++, nano_type);
        }
        
        if (token && strcmp(token, ",") == 0) {
            free(token);
            token = next_token(t);
        }
    }
    
    if (token) free(token);
    
    fprintf(out, ") -> %s\n", map_c_type_to_nano(return_type));
    
    /* Skip to semicolon or opening brace */
    token = next_token(t);
    while (token && strcmp(token, ";") != 0 && strcmp(token, "{") != 0) {
        free(token);
        token = next_token(t);
    }
    if (token) free(token);
    
    return true;
}

/* Generate nanolang module from C header */
static bool generate_module(const char *header_file, FFIOptions *opts) {
    FILE *header = fopen(header_file, "r");
    if (!header) {
        fprintf(stderr, "Error: Could not open header file '%s'\n", header_file);
        return false;
    }
    
    /* Read header file */
    fseek(header, 0, SEEK_END);
    long size = ftell(header);
    fseek(header, 0, SEEK_SET);
    
    char *source = malloc(size + 1);
    fread(source, 1, size, header);
    source[size] = '\0';
    fclose(header);
    
    /* Open output file */
    FILE *out = fopen(opts->output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: Could not create output file '%s'\n", opts->output_file);
        free(source);
        return false;
    }
    
    /* Write module header */
    fprintf(out, "# nanolang FFI module generated from %s\n", header_file);
    fprintf(out, "# Generated by nanoc-ffi\n\n");
    
    /* Write module metadata as comments */
    if (opts->include_count > 0) {
        fprintf(out, "# Include paths:\n");
        for (int i = 0; i < opts->include_count; i++) {
            fprintf(out, "#   -I%s\n", opts->include_paths[i]);
        }
        fprintf(out, "\n");
    }
    
    if (opts->library_path_count > 0) {
        fprintf(out, "# Library paths:\n");
        for (int i = 0; i < opts->library_path_count; i++) {
            fprintf(out, "#   -L%s\n", opts->library_paths[i]);
        }
        fprintf(out, "\n");
    }
    
    if (opts->library_count > 0) {
        fprintf(out, "# Libraries:\n");
        for (int i = 0; i < opts->library_count; i++) {
            fprintf(out, "#   -l%s\n", opts->libraries[i]);
        }
        fprintf(out, "\n");
    }
    
    /* Parse header and generate extern declarations */
    CTokenizer tokenizer;
    tokenizer.source = source;
    tokenizer.pos = 0;
    tokenizer.line = 1;
    
    fprintf(out, "# Extern function declarations (auto-generated)\n\n");
    
    int function_count = 0;
    size_t source_len = strlen(source);
    while ((size_t)tokenizer.pos < source_len) {
        if (parse_function_declaration(&tokenizer, out)) {
            function_count++;
        } else {
            /* Skip one token and try again */
            char *token = next_token(&tokenizer);
            if (!token) break;
            free(token);
        }
    }
    
    if (function_count == 0) {
        fprintf(out, "# No function declarations found in header.\n");
        fprintf(out, "# You may need to manually add extern declarations.\n");
        fprintf(out, "# Example:\n");
        fprintf(out, "# extern fn function_name(param1: int, param2: string) -> int\n");
    }
    
    fprintf(out, "\n");
    
    free(source);
    fclose(out);
    
    printf("Generated module template: %s\n", opts->output_file);
    printf("Note: Manual extern declarations are required.\n");
    printf("Future versions will auto-generate from C headers.\n");
    
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <header_file.h> -o <module.nano> [OPTIONS]\n", argv[0]);
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  -o <file>     Output nanolang module file\n");
        fprintf(stderr, "  -I <path>     Add include path\n");
        fprintf(stderr, "  -L <path>     Add library path\n");
        fprintf(stderr, "  -l <lib>      Link against library\n");
        fprintf(stderr, "  --name <name> Module name (default: derived from header)\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s SDL.h -o sdl.nano -I/opt/homebrew/include/SDL2 -L/opt/homebrew/lib -lSDL2\n", argv[0]);
        return 1;
    }
    
    FFIOptions opts = {0};
    const char *header_file = NULL;
    
    /* Allocate arrays */
    char **include_paths = malloc(sizeof(char*) * 32);
    char **library_paths = malloc(sizeof(char*) * 32);
    char **libraries = malloc(sizeof(char*) * 32);
    int include_count = 0;
    int library_path_count = 0;
    int library_count = 0;
    
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            opts.output_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-I") == 0 && i + 1 < argc) {
            if (include_count < 32) {
                include_paths[include_count++] = argv[i + 1];
            }
            i++;
        } else if (strncmp(argv[i], "-I", 2) == 0) {
            if (include_count < 32) {
                include_paths[include_count++] = argv[i] + 2;
            }
        } else if (strcmp(argv[i], "-L") == 0 && i + 1 < argc) {
            if (library_path_count < 32) {
                library_paths[library_path_count++] = argv[i + 1];
            }
            i++;
        } else if (strncmp(argv[i], "-L", 2) == 0) {
            if (library_path_count < 32) {
                library_paths[library_path_count++] = argv[i] + 2;
            }
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            if (library_count < 32) {
                libraries[library_count++] = argv[i + 1];
            }
            i++;
        } else if (strncmp(argv[i], "-l", 2) == 0) {
            if (library_count < 32) {
                libraries[library_count++] = argv[i] + 2;
            }
        } else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
            opts.module_name = argv[i + 1];
            i++;
        } else if (argv[i][0] != '-') {
            header_file = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            free(include_paths);
            free(library_paths);
            free(libraries);
            return 1;
        }
    }
    
    if (!header_file || !opts.output_file) {
        fprintf(stderr, "Error: Header file and output file are required\n");
        free(include_paths);
        free(library_paths);
        free(libraries);
        return 1;
    }
    
    opts.include_paths = include_paths;
    opts.include_count = include_count;
    opts.library_paths = library_paths;
    opts.library_path_count = library_path_count;
    opts.libraries = libraries;
    opts.library_count = library_count;
    
    bool success = generate_module(header_file, &opts);
    
    free(include_paths);
    free(library_paths);
    free(libraries);
    
    return success ? 0 : 1;
}

