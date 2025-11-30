#include "list_token.h"
#include "../nanolang.h"
#include <stdint.h>

/* Helper functions to access Token fields
 * These allow nanolang code to access Token fields without redefining the Token struct
 */

int64_t token_get_type(List_token *list, int64_t index) {
    Token *ptr = list_token_get(list, (int)index);
    return (int64_t)ptr->type;
}

const char* token_get_value(List_token *list, int64_t index) {
    Token *ptr = list_token_get(list, (int)index);
    return ptr->value;
}

int64_t token_get_line(List_token *list, int64_t index) {
    Token *ptr = list_token_get(list, (int)index);
    return (int64_t)ptr->line;
}

int64_t token_get_column(List_token *list, int64_t index) {
    Token *ptr = list_token_get(list, (int)index);
    return (int64_t)ptr->column;
}
