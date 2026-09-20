#ifndef NANOLANG_EVAL_U8_H
#define NANOLANG_EVAL_U8_H
#include "nanolang.h"
#include <stdint.h>
/* I retain control metadata and narrow an already checked scalar destination.
 * My interpreter represents both source INT and U8 payloads with VAL_INT. */
static inline Value eval_checked_scalar_destination(Type destination, Value value) {
    if (destination == TYPE_U8 && value.type == VAL_INT)
        value.as.int_val = (uint8_t)value.as.int_val;
    return value;
}
#endif
