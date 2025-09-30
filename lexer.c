#include "nanolang.h"

/* Helper function to create a token */
static Token create_token(TokenType type, const char *value, int line) {
    Token token;
    token.type = type;
    token.value = value ? strdup(value) : NULL;
    token.line = line;
    return token;
}

/* Check if character is part of an identifier */
static bool is_identifier_char(char c) {
    return isalnum(c) || c == '_';
}

/* Tokenize the source code */
Token *tokenize(const char *source, int *token_count) {
    int capacity = 16;
    int count = 0;
    Token *tokens = malloc(sizeof(Token) * capacity);
    int line = 1;
    int i = 0;
    
    while (source[i] != '\0') {
        /* Skip whitespace */
        if (isspace(source[i])) {
            if (source[i] == '\n') line++;
            i++;
            continue;
        }
        
        /* Skip comments */
        if (source[i] == '/' && source[i + 1] == '/') {
            while (source[i] != '\0' && source[i] != '\n') i++;
            continue;
        }
        
        /* Resize if needed */
        if (count >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, sizeof(Token) * capacity);
        }
        
        /* Numbers */
        if (isdigit(source[i])) {
            int start = i;
            while (isdigit(source[i])) i++;
            int len = i - start;
            char *num_str = malloc(len + 1);
            strncpy(num_str, source + start, len);
            num_str[len] = '\0';
            tokens[count++] = create_token(TOKEN_NUMBER, num_str, line);
            free(num_str);
            continue;
        }
        
        /* Identifiers and keywords */
        if (isalpha(source[i]) || source[i] == '_') {
            int start = i;
            while (is_identifier_char(source[i])) i++;
            int len = i - start;
            char *id_str = malloc(len + 1);
            strncpy(id_str, source + start, len);
            id_str[len] = '\0';
            
            /* Check for keywords */
            TokenType type = TOKEN_IDENTIFIER;
            if (strcmp(id_str, "print") == 0) type = TOKEN_PRINT;
            else if (strcmp(id_str, "let") == 0) type = TOKEN_LET;
            else if (strcmp(id_str, "if") == 0) type = TOKEN_IF;
            else if (strcmp(id_str, "else") == 0) type = TOKEN_ELSE;
            else if (strcmp(id_str, "while") == 0) type = TOKEN_WHILE;
            else if (strcmp(id_str, "true") == 0) type = TOKEN_TRUE;
            else if (strcmp(id_str, "false") == 0) type = TOKEN_FALSE;
            
            tokens[count++] = create_token(type, id_str, line);
            free(id_str);
            continue;
        }
        
        /* Single-character tokens */
        switch (source[i]) {
            case '+': tokens[count++] = create_token(TOKEN_PLUS, NULL, line); i++; break;
            case '-': tokens[count++] = create_token(TOKEN_MINUS, NULL, line); i++; break;
            case '*': tokens[count++] = create_token(TOKEN_STAR, NULL, line); i++; break;
            case '/': tokens[count++] = create_token(TOKEN_SLASH, NULL, line); i++; break;
            case '(': tokens[count++] = create_token(TOKEN_LPAREN, NULL, line); i++; break;
            case ')': tokens[count++] = create_token(TOKEN_RPAREN, NULL, line); i++; break;
            case ';': tokens[count++] = create_token(TOKEN_SEMICOLON, NULL, line); i++; break;
            case '{': tokens[count++] = create_token(TOKEN_LBRACE, NULL, line); i++; break;
            case '}': tokens[count++] = create_token(TOKEN_RBRACE, NULL, line); i++; break;
            case '=':
                if (source[i + 1] == '=') {
                    tokens[count++] = create_token(TOKEN_EQ, NULL, line);
                    i += 2;
                } else {
                    tokens[count++] = create_token(TOKEN_ASSIGN, NULL, line);
                    i++;
                }
                break;
            case '<': tokens[count++] = create_token(TOKEN_LT, NULL, line); i++; break;
            case '>': tokens[count++] = create_token(TOKEN_GT, NULL, line); i++; break;
            default:
                fprintf(stderr, "Error: Unknown character '%c' at line %d\n", source[i], line);
                i++;
                break;
        }
    }
    
    tokens[count++] = create_token(TOKEN_EOF, NULL, line);
    *token_count = count;
    return tokens;
}

/* Free token array */
void free_tokens(Token *tokens, int count) {
    for (int i = 0; i < count; i++) {
        free(tokens[i].value);
    }
    free(tokens);
}
