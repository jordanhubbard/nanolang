#ifndef EVAL_STRING_H
#define EVAL_STRING_H

#include "../nanolang.h"

/* String operation built-in functions */
Value builtin_str_length(Value *args);
Value builtin_str_concat(Value *args);
Value builtin_str_substring(Value *args);
Value builtin_str_contains(Value *args);
Value builtin_str_equals(Value *args);
Value builtin_str_starts_with(Value *args);
Value builtin_str_ends_with(Value *args);
Value builtin_str_index_of(Value *args);
Value builtin_str_trim(Value *args);
Value builtin_str_trim_left(Value *args);
Value builtin_str_trim_right(Value *args);
Value builtin_str_to_lower(Value *args);
Value builtin_str_to_upper(Value *args);
Value builtin_str_replace(Value *args);

#endif /* EVAL_STRING_H */
