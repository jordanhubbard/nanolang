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
    for (int i = 0; i < count; ++i) {
        const Token *name = &tokens[i];
        if (name->token_type == TOKEN_IDENTIFIER || name->token_type == TOKEN_STRING || !word(name->value))
            continue;
        TokenType prior = i ? tokens[i - 1].token_type : TOKEN_EOF;
        if (name->token_type == TOKEN_MUT && prior == TOKEN_LET &&
            (i + 1 == count || tokens[i + 1].token_type != TOKEN_COLON)) continue;
        bool declaration = prior == TOKEN_LET || prior == TOKEN_FN || prior == TOKEN_STRUCT ||
            prior == TOKEN_ENUM || prior == TOKEN_UNION || prior == TOKEN_SHADOW || prior == TOKEN_EFFECT ||
            (prior == TOKEN_MUT && i > 1 && tokens[i - 2].token_type == TOKEN_LET);
        bool typed_name = i + 1 < count && tokens[i + 1].token_type == TOKEN_COLON;
        if (declaration || typed_name)
            printf("%d\t%d\t%s\n", name->line, name->column, name->value);
    }
    free_tokens(tokens, count);
    return 0;
}
