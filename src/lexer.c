#include "nanolang.h"

/* Helper function to create a token */
static Token create_token(TokenType type, const char *value, int line, int column) {
    Token token;
    token.type = type;
    token.value = value ? strdup(value) : NULL;
    token.line = line;
    token.column = column;
    return token;
}

/* Check if character is part of an identifier */
static bool is_identifier_start(char c) {
    return isalpha(c) || c == '_';
}

static bool is_identifier_char(char c) {
    return isalnum(c) || c == '_';
}

/* Check if string is a keyword and return appropriate token type */
static TokenType keyword_or_identifier(const char *str) {
    /* Keywords */
    if (strcmp(str, "extern") == 0) return TOKEN_EXTERN;
    if (strcmp(str, "fn") == 0) return TOKEN_FN;
    if (strcmp(str, "let") == 0) return TOKEN_LET;
    if (strcmp(str, "mut") == 0) return TOKEN_MUT;
    if (strcmp(str, "set") == 0) return TOKEN_SET;
    if (strcmp(str, "if") == 0) return TOKEN_IF;
    if (strcmp(str, "else") == 0) return TOKEN_ELSE;
    if (strcmp(str, "while") == 0) return TOKEN_WHILE;
    if (strcmp(str, "for") == 0) return TOKEN_FOR;
    if (strcmp(str, "in") == 0) return TOKEN_IN;
    if (strcmp(str, "return") == 0) return TOKEN_RETURN;
    if (strcmp(str, "assert") == 0) return TOKEN_ASSERT;
    if (strcmp(str, "shadow") == 0) return TOKEN_SHADOW;
    if (strcmp(str, "print") == 0) return TOKEN_PRINT;
    if (strcmp(str, "array") == 0) return TOKEN_ARRAY;
    if (strcmp(str, "struct") == 0) return TOKEN_STRUCT;
    if (strcmp(str, "enum") == 0) return TOKEN_ENUM;
    if (strcmp(str, "union") == 0) return TOKEN_UNION;
    if (strcmp(str, "match") == 0) return TOKEN_MATCH;

    /* Boolean literals */
    if (strcmp(str, "true") == 0) return TOKEN_TRUE;
    if (strcmp(str, "false") == 0) return TOKEN_FALSE;

    /* Types */
    if (strcmp(str, "int") == 0) return TOKEN_TYPE_INT;
    if (strcmp(str, "float") == 0) return TOKEN_TYPE_FLOAT;
    if (strcmp(str, "bool") == 0) return TOKEN_TYPE_BOOL;
    if (strcmp(str, "string") == 0) return TOKEN_TYPE_STRING;
    if (strcmp(str, "void") == 0) return TOKEN_TYPE_VOID;

    /* Operators (when used as identifiers in prefix position) */
    if (strcmp(str, "and") == 0) return TOKEN_AND;
    if (strcmp(str, "or") == 0) return TOKEN_OR;
    if (strcmp(str, "not") == 0) return TOKEN_NOT;
    if (strcmp(str, "range") == 0) return TOKEN_RANGE;

    return TOKEN_IDENTIFIER;
}

/* Tokenize the source code */
Token *tokenize(const char *source, int *token_count) {
    int capacity = 64;
    int count = 0;
    Token *tokens = malloc(sizeof(Token) * capacity);
    int line = 1;
    int column = 1;
    int line_start = 0;  /* Track start of current line for column calculation */
    int i = 0;

    while (source[i] != '\0') {
        /* Skip whitespace */
        if (isspace(source[i])) {
            if (source[i] == '\n') {
                line++;
                line_start = i + 1;
                column = 1;
            }
            i++;
            continue;
        }

        /* Skip comments (# to end of line) */
        if (source[i] == '#') {
            while (source[i] != '\0' && source[i] != '\n') i++;
            continue;
        }
        
        /* Skip C-style comments (slash-star ... star-slash) */
        if (source[i] == '/' && source[i+1] == '*') {
            i += 2;  /* Skip opening delimiter */
            while (source[i] != '\0') {
                if (source[i] == '*' && source[i+1] == '/') {
                    i += 2;  /* Skip closing delimiter */
                    break;
                }
                if (source[i] == '\n') {
                    line++;
                    line_start = i + 1;
                }
                i++;
            }
            continue;
        }

        /* Update column for this token */
        column = i - line_start + 1;

        /* Resize if needed */
        if (count >= capacity - 1) {
            capacity *= 2;
            tokens = realloc(tokens, sizeof(Token) * capacity);
        }

        /* String literals */
        if (source[i] == '"') {
            i++; /* Skip opening quote */
            int start = i;
            while (source[i] != '\0' && source[i] != '"') {
                if (source[i] == '\\' && source[i + 1] != '\0') {
                    i += 2; /* Skip escape sequences */
                } else {
                    i++;
                }
            }
            if (source[i] != '"') {
                fprintf(stderr, "Error: Unterminated string at line %d\n", line);
                free(tokens);
                return NULL;
            }
            int len = i - start;
            char *str = malloc(len + 1);
            strncpy(str, source + start, len);
            str[len] = '\0';
            tokens[count++] = create_token(TOKEN_STRING, str, line, column);
            free(str);
            i++; /* Skip closing quote */
            continue;
        }

        /* Numbers (integers and floats) */
        if (isdigit(source[i]) || (source[i] == '-' && isdigit(source[i + 1]))) {
            int start = i;
            if (source[i] == '-') i++;
            while (isdigit(source[i])) i++;

            /* Check for float */
            if (source[i] == '.' && isdigit(source[i + 1])) {
                i++;
                while (isdigit(source[i])) i++;
                int len = i - start;
                char *num_str = malloc(len + 1);
                strncpy(num_str, source + start, len);
                num_str[len] = '\0';
                tokens[count++] = create_token(TOKEN_FLOAT, num_str, line, column);
                free(num_str);
            } else {
                int len = i - start;
                char *num_str = malloc(len + 1);
                strncpy(num_str, source + start, len);
                num_str[len] = '\0';
                tokens[count++] = create_token(TOKEN_NUMBER, num_str, line, column);
                free(num_str);
            }
            continue;
        }

        /* Identifiers and keywords */
        if (is_identifier_start(source[i])) {
            int start = i;
            while (is_identifier_char(source[i])) i++;
            int len = i - start;
            char *id_str = malloc(len + 1);
            strncpy(id_str, source + start, len);
            id_str[len] = '\0';

            TokenType type = keyword_or_identifier(id_str);
            tokens[count++] = create_token(type, id_str, line, column);
            free(id_str);
            continue;
        }

        /* Two-character operators */
        if (source[i] == '-' && source[i + 1] == '>') {
            tokens[count++] = create_token(TOKEN_ARROW, NULL, line, column);
            i += 2;
            continue;
        }
        if (source[i] == '=' && source[i + 1] == '=') {
            tokens[count++] = create_token(TOKEN_EQ, NULL, line, column);
            i += 2;
            continue;
        }
        if (source[i] == '=' && source[i + 1] != '=') {
            tokens[count++] = create_token(TOKEN_ASSIGN, NULL, line, column);
            i++;
            continue;
        }
        if (source[i] == '!' && source[i + 1] == '=') {
            tokens[count++] = create_token(TOKEN_NE, NULL, line, column);
            i += 2;
            continue;
        }
        if (source[i] == '<' && source[i + 1] == '=') {
            tokens[count++] = create_token(TOKEN_LE, NULL, line, column);
            i += 2;
            continue;
        }
        if (source[i] == '>' && source[i + 1] == '=') {
            tokens[count++] = create_token(TOKEN_GE, NULL, line, column);
            i += 2;
            continue;
        }

        /* Single-character tokens */
        switch (source[i]) {
            case '(': tokens[count++] = create_token(TOKEN_LPAREN, NULL, line, column); i++; break;
            case ')': tokens[count++] = create_token(TOKEN_RPAREN, NULL, line, column); i++; break;
            case '{': tokens[count++] = create_token(TOKEN_LBRACE, NULL, line, column); i++; break;
            case '}': tokens[count++] = create_token(TOKEN_RBRACE, NULL, line, column); i++; break;
            case '[': tokens[count++] = create_token(TOKEN_LBRACKET, NULL, line, column); i++; break;
            case ']': tokens[count++] = create_token(TOKEN_RBRACKET, NULL, line, column); i++; break;
            case ',': tokens[count++] = create_token(TOKEN_COMMA, NULL, line, column); i++; break;
            case ':': tokens[count++] = create_token(TOKEN_COLON, NULL, line, column); i++; break;
            case '.': tokens[count++] = create_token(TOKEN_DOT, NULL, line, column); i++; break;
            case '+': tokens[count++] = create_token(TOKEN_PLUS, NULL, line, column); i++; break;
            case '-': tokens[count++] = create_token(TOKEN_MINUS, NULL, line, column); i++; break;
            case '*': tokens[count++] = create_token(TOKEN_STAR, NULL, line, column); i++; break;
            case '/': tokens[count++] = create_token(TOKEN_SLASH, NULL, line, column); i++; break;
            case '%': tokens[count++] = create_token(TOKEN_PERCENT, NULL, line, column); i++; break;
            case '<': tokens[count++] = create_token(TOKEN_LT, NULL, line, column); i++; break;
            case '>': tokens[count++] = create_token(TOKEN_GT, NULL, line, column); i++; break;
            default:
                fprintf(stderr, "Error: Unknown character '%c' at line %d\n", source[i], line);
                i++;
                break;
        }
    }

    tokens[count++] = create_token(TOKEN_EOF, NULL, line, column);
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

/* Get token type name for debugging */
const char *token_type_name(TokenType type) {
    switch (type) {
        case TOKEN_EOF: return "EOF";
        case TOKEN_NUMBER: return "NUMBER";
        case TOKEN_FLOAT: return "FLOAT";
        case TOKEN_STRING: return "STRING";
        case TOKEN_IDENTIFIER: return "IDENTIFIER";
        case TOKEN_TRUE: return "TRUE";
        case TOKEN_FALSE: return "FALSE";
        case TOKEN_LPAREN: return "LPAREN";
        case TOKEN_RPAREN: return "RPAREN";
        case TOKEN_LBRACE: return "LBRACE";
        case TOKEN_RBRACE: return "RBRACE";
        case TOKEN_LBRACKET: return "LBRACKET";
        case TOKEN_RBRACKET: return "RBRACKET";
        case TOKEN_COMMA: return "COMMA";
        case TOKEN_COLON: return "COLON";
        case TOKEN_ARROW: return "ARROW";
        case TOKEN_ASSIGN: return "ASSIGN";
        case TOKEN_DOT: return "DOT";
        case TOKEN_FN: return "FN";
        case TOKEN_LET: return "LET";
        case TOKEN_MUT: return "MUT";
        case TOKEN_SET: return "SET";
        case TOKEN_IF: return "IF";
        case TOKEN_ELSE: return "ELSE";
        case TOKEN_WHILE: return "WHILE";
        case TOKEN_FOR: return "FOR";
        case TOKEN_IN: return "IN";
        case TOKEN_RETURN: return "RETURN";
        case TOKEN_ASSERT: return "ASSERT";
        case TOKEN_SHADOW: return "SHADOW";
        case TOKEN_PRINT: return "PRINT";
        case TOKEN_EXTERN: return "EXTERN";
        case TOKEN_ARRAY: return "ARRAY";
        case TOKEN_STRUCT: return "STRUCT";
        case TOKEN_ENUM: return "ENUM";
        case TOKEN_UNION: return "UNION";
        case TOKEN_MATCH: return "MATCH";
        case TOKEN_TYPE_INT: return "TYPE_INT";
        case TOKEN_TYPE_FLOAT: return "TYPE_FLOAT";
        case TOKEN_TYPE_BOOL: return "TYPE_BOOL";
        case TOKEN_TYPE_STRING: return "TYPE_STRING";
        case TOKEN_TYPE_VOID: return "TYPE_VOID";
        case TOKEN_PLUS: return "PLUS";
        case TOKEN_MINUS: return "MINUS";
        case TOKEN_STAR: return "STAR";
        case TOKEN_SLASH: return "SLASH";
        case TOKEN_PERCENT: return "PERCENT";
        case TOKEN_EQ: return "EQ";
        case TOKEN_NE: return "NE";
        case TOKEN_LT: return "LT";
        case TOKEN_LE: return "LE";
        case TOKEN_GT: return "GT";
        case TOKEN_GE: return "GE";
        case TOKEN_AND: return "AND";
        case TOKEN_OR: return "OR";
        case TOKEN_NOT: return "NOT";
        case TOKEN_RANGE: return "RANGE";
        default: return "UNKNOWN";
    }
}