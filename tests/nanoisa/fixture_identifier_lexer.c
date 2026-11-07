/* I use my actual lexer to find reserved words in declaration-name positions. */
#include "nanolang.h"

static bool word(const char *value) {
    if (!value || !((*value >= 'A' && *value <= 'Z') ||
                    (*value >= 'a' && *value <= 'z') || *value == '_')) return false;
    for (const char *p = value; *p; ++p)
        if (!((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') ||
              (*p >= '0' && *p <= '9') || *p == '_')) return false;
    return true;
}

int main(void) {
    const size_t limit = 2u * 1024u * 1024u;
    char *source = malloc(limit + 1);
    if (!source) return 3;
    size_t length = fread(source, 1, limit, stdin);
    if (ferror(stdin) || (length == limit && fgetc(stdin) != EOF)) { free(source); return 3; }
    source[length] = '\0';
    int count = 0;
    Token *tokens = tokenize(source, &count);
    free(source);
    if (!tokens) return 2;
    for (int i = 0; i + 1 < count; ++i) {
        TokenType kind = tokens[i].token_type;
        if (kind != TOKEN_LET && kind != TOKEN_FN && kind != TOKEN_STRUCT &&
            kind != TOKEN_ENUM && kind != TOKEN_UNION && kind != TOKEN_SHADOW &&
            kind != TOKEN_EFFECT) continue;
        int next = i + 1;
        if (kind == TOKEN_LET && tokens[next].token_type == TOKEN_MUT) ++next;
        if (next >= count) continue;
        const Token *name = &tokens[next];
        if (name->token_type != TOKEN_IDENTIFIER && name->token_type != TOKEN_STRING && word(name->value))
            printf("%d\t%d\t%s\n", name->line, name->column, name->value);
    }
    free_tokens(tokens, count);
    return 0;
}
