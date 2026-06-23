#ifndef NANOLANG_NANO_TOOLS_H
#define NANOLANG_NANO_TOOLS_H

#include "nanolang.h"

int64_t eval(const char *source);

/* Eval with output capture - returns JSON result */
char* eval_with_output(const char *source);

/* Free the result from eval_with_output */
void eval_free_result(char *result);

#endif
