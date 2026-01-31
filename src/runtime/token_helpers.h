#ifndef TOKEN_HELPERS_H
#define TOKEN_HELPERS_H

#include "list_token.h"
#include "../nanolang.h"
#include <stdint.h>

/* Helper functions to access Token fields
 * These allow nanolang code to access Token fields without redefining the Token struct
 * Note: Uses int64_t to match nanolang's int type
 */
int64_t token_get_type(List_token *list, int64_t index);
const char* token_get_value(List_token *list, int64_t index);
int64_t token_get_line(List_token *list, int64_t index);
int64_t token_get_column(List_token *list, int64_t index);

#endif /* TOKEN_HELPERS_H */
