#ifndef EVAL_STRING_H
#define EVAL_STRING_H

#include "../nanolang.h"

/* String operation built-in functions */
Value builtin_str_length(Value *args);
Value builtin_str_concat(Value *args);
Value builtin_str_substring(Value *args);
Value builtin_str_contains(Value *args);
Value builtin_str_equals(Value *args);

#endif /* EVAL_STRING_H */
