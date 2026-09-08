/*
 * nvm2c — NanoISA to structured C11.
 *
 * This is an ISA translator, not a NanoLang compiler phase. It reads a
 * loaded NvmModule and writes C that computes with C operators. Embedding
 * the bytecode and calling nano_vm is wrapper_gen.c; that is not this.
 *
 * The generated process does not require nano_vm, nano_cop, or nano_vmd.
 *
 * Closed subset: i64 constants, locals, integer arithmetic and comparisons,
 * CALL/TAIL_CALL, JMP/JMP_FALSE, RET, HALT, PUSH_STR, STR_CONCAT, STR_LEN,
 * ARR_LITERAL, ARR_GET, ARR_LEN, ARR_PUSH of array<int> and array<string>,
 * AGG_PACK, AGG_GET, bool results as i64, PUSH_BOOL, BOOL_NOT, BOOL_AND,
 * BOOL_OR, PRINT, PRINTLN, ASSERT, STR_CONTAINS, CAST_STRING of i64,
 * EQ/NE of strings, STR_SUBSTR. Anything else is refused with an error.
 * CALL_EXTERN is refused because it is the VM FFI / co-process path, not a
 * host C ABI. Embedded NULs, nested arrays, nested records, variants, tuples,
 * printing arrays/records, array equality, STR_TRIM, and the rest of the
 * string and array libraries stay refused.
 */

#ifndef NANOISA_NVM2C_H
#define NANOISA_NVM2C_H

#include "nvm_format.h"
#include <stddef.h>

/* Returns a malloc'd C11 translation, or NULL on error.
 * On error, writes a message into err (if err_len > 0). */
char *nvm2c_emit(const NvmModule *mod, char *err, size_t err_len);

#endif /* NANOISA_NVM2C_H */
