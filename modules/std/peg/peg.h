#ifndef NANOLANG_STD_PEG_H
#define NANOLANG_STD_PEG_H

#include <stdint.h>

#include "nanolang.h"

/* Compile a PEG pattern. Returns NULL on error. */
void* nl_peg_compile(const char* pattern);

/* Full match: returns 1 on match, 0 on no match, -1 on error. */
int64_t nl_peg_match(void* peg, const char* input);

/* Captures from last match attempt (grouped subexpressions). */
DynArray* nl_peg_captures(void* peg, const char* input);

/* Free compiled PEG. */
void nl_peg_free(void* peg);

#endif
