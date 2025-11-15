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
    /* Look for function name pattern: type name(params) */
    /* This is simplified - real C parsing is more complex */
    
    /* Skip until we find something that looks like a function */
    char *token = next_token(t);
    if (!token) return false;
    
    /* Check if this looks like a function declaration */
    /* We'll look for patterns like: return_type function_name( */
    
    /* For now, we'll use a simpler approach: generate extern declarations manually */
    /* This is a placeholder - full C parsing would require a proper parser */
    
    free(token);
    return false;
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
    /* For now, we'll generate a template that users can fill in */
    fprintf(out, "# TODO: Add extern function declarations here\n");
    fprintf(out, "# Example:\n");
    fprintf(out, "# extern fn function_name(param1: int, param2: string) -> int\n\n");
    
    fprintf(out, "# Note: This is a template module.\n");
    fprintf(out, "# You need to manually add extern declarations based on the C header.\n");
    fprintf(out, "# Future versions will auto-generate these from the header file.\n");
    
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

