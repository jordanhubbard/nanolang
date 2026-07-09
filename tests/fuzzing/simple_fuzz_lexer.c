/*
 * Simple standalone fuzzer for nanolang lexer
 *
 * This is a basic fuzzer that reads input from files or stdin.
 * Works without special compiler features (libFuzzer/AFL++).
 *
 * Build:
 *   gcc -g -O1 -fsanitize=address -I../../src \
 *     simple_fuzz_lexer.c ../../src/lexer.c -o simple_fuzz_lexer
 *
 * Usage:
 *   # Test single file
 *   ./simple_fuzz_lexer test_input.nano
 *
 *   # Test all files in corpus (use shell glob)
 *   for f in corpus_lexer/seed*.nano; do ./simple_fuzz_lexer "$f"; done
 *
 *   # Test from stdin
 *   echo "fn main() -> int { return 0 }" | ./simple_fuzz_lexer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declarations */
typedef struct Token Token;

extern Token *tokenize(const char *source, int *token_count);
extern void free_tokens(Token *tokens, int count);

static char *read_file(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size < 0 || size > 1000000) {  /* Limit to 1MB */
        fclose(f);
        return NULL;
    }

    char *buffer = (char *)malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    size_t read_size = fread(buffer, 1, size, f);
    buffer[read_size] = '\0';
    fclose(f);

    return buffer;
}

static char *read_stdin(void) {
    size_t capacity = 4096;
    size_t size = 0;
    char *buffer = (char *)malloc(capacity);
    if (!buffer) {
        return NULL;
    }

    while (1) {
        if (size + 1024 > capacity) {
            capacity *= 2;
            char *new_buffer = (char *)realloc(buffer, capacity);
            if (!new_buffer) {
                free(buffer);
                return NULL;
            }
            buffer = new_buffer;
        }

        size_t read_size = fread(buffer + size, 1, 1024, stdin);
        if (read_size == 0) {
            break;
        }
        size += read_size;

        if (size > 100000) {  /* Limit to 100KB from stdin */
            break;
        }
    }

    buffer[size] = '\0';
    return buffer;
}

int main(int argc, char **argv) {
    char *input = NULL;
    const char *filename = "stdin";

    if (argc > 1) {
        /* Read from file */
        filename = argv[1];
        input = read_file(filename);
        if (!input) {
            fprintf(stderr, "Error: Could not read file '%s'\n", filename);
            return 1;
        }
    } else {
        /* Read from stdin */
        input = read_stdin();
        if (!input) {
            fprintf(stderr, "Error: Could not read from stdin\n");
            return 1;
        }
    }

    /* Fuzz the lexer */
    int token_count = 0;
    Token *tokens = tokenize(input, &token_count);
    if (tokens) {
        free_tokens(tokens, token_count);
    }

    free(input);
    return 0;
}
