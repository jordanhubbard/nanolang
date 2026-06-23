#ifndef EVAL_MATH_H
#define EVAL_MATH_H

#include "../nanolang.h"

/* Math built-in functions */
Value builtin_abs(Value *args);
Value builtin_min(Value *args);
Value builtin_max(Value *args);
Value builtin_sqrt(Value *args);
Value builtin_pow(Value *args);
Value builtin_floor(Value *args);
Value builtin_ceil(Value *args);
Value builtin_round(Value *args);
Value builtin_sin(Value *args);
Value builtin_cos(Value *args);
Value builtin_tan(Value *args);
Value builtin_atan2(Value *args);

#endif /* EVAL_MATH_H */
