#ifndef NANO_EVAL_CALL_ARGUMENTS_H
#define NANO_EVAL_CALL_ARGUMENTS_H

#include "../nanolang.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define EVAL_CALL_INLINE_ARGUMENTS 16
#ifndef EVAL_CALL_ARGUMENT_MALLOC
#define EVAL_CALL_ARGUMENT_MALLOC malloc
#endif
#ifndef EVAL_CALL_ARGUMENT_FREE
#define EVAL_CALL_ARGUMENT_FREE free
#endif

/* I own only the carrier; its values keep their existing ownership. */
static Value *eval_call_arguments_acquire(Value *inline_args, int count) {
    if (count < 0 || (size_t)count > SIZE_MAX / sizeof(Value)) {
        fprintf(stderr, "I cannot represent this call's argument storage.\n");
        exit(EXIT_FAILURE);
    }
    if (count <= EVAL_CALL_INLINE_ARGUMENTS) return inline_args;
    Value *args = EVAL_CALL_ARGUMENT_MALLOC((size_t)count * sizeof(Value));
    if (!args) {
        fprintf(stderr, "I cannot allocate this call's argument storage.\n");
        exit(EXIT_FAILURE);
    }
    return args;
}

static void eval_call_arguments_release(Value *args, Value *inline_args) {
    if (args != inline_args) EVAL_CALL_ARGUMENT_FREE(args);
}

#endif
