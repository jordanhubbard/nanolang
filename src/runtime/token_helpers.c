#include "list_LexerToken.h"
#include "../nanolang.h"
#include <stdint.h>

/* Helper functions to access LexerToken fields
 * These allow nanolang code to access LexerToken fields without redefining the LexerToken struct
 */

int64_t token_get_type(List_LexerToken *list, int64_t index) {
    LexerToken ptr = nl_list_LexerToken_get(list, (int)index);
    return (int64_t)ptr.token_type;
}

const char* token_get_value(List_LexerToken *list, int64_t index) {
    LexerToken ptr = nl_list_LexerToken_get(list, (int)index);
    return ptr.value;
}

int64_t token_get_line(List_LexerToken *list, int64_t index) {
    LexerToken ptr = nl_list_LexerToken_get(list, (int)index);
    return (int64_t)ptr.line;
}

int64_t token_get_column(List_LexerToken *list, int64_t index) {
    LexerToken ptr = nl_list_LexerToken_get(list, (int)index);
    return (int64_t)ptr.column;
}
