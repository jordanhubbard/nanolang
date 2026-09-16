#include "nanolang.h"
#include <assert.h>

int main(void) {
    const int counts[] = {0, 1, 31, 32, 33, 63, 64, 65, 80, 128, 256};
    for (size_t test = 0; test < sizeof(counts) / sizeof(counts[0]); test++) {
        char source[4096] = "f\"";
        size_t length = 2;
        for (int i = 0; i < counts[test]; i++) {
            memcpy(source + length, "x{7}", 4);
            length += 4;
        }
        memcpy(source + length, "tail\"", 6);
        int count = 0;
        Token *tokens = tokenize(source, &count);
        assert(tokens && count > 0 && tokens[count - 1].token_type == TOKEN_EOF);
        int conversions = 0;
        for (int i = 0; i < count; i++)
            if (tokens[i].token_type == TOKEN_IDENTIFIER && tokens[i].value &&
                strcmp(tokens[i].value, "to_string") == 0) conversions++;
        assert(conversions == counts[test]);
        free_tokens(tokens, count);
    }
    puts("I preserved every f-string part across growth boundaries.");
    return 0;
}
