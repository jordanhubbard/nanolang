#ifndef NANO_EVAL_FREEZE_H
#define NANO_EVAL_FREEZE_H

#include <stddef.h>
#include <stdint.h>

int nano_eval_freeze_defun(const char *source, int64_t point,
                           char *out, size_t outn);

#endif
