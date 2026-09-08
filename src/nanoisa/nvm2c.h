/*
 * nvm2c — NanoISA to structured C11.
 *
 * This is an ISA translator, not a NanoLang compiler phase. It reads a
 * loaded NvmModule and writes C that computes with C operators. Embedding
 * the bytecode and calling nano_vm is wrapper_gen.c; that is not this.
 *
 * The generated process does not require nano_vm, nano_cop, or nano_vmd.
 *
 * Closed subset: i64 constants, locals, integer arithmetic and comparisons
 * (including generic LT from for-range, not only I64_LT_S),
 * CALL/TAIL_CALL, JMP/JMP_FALSE, RET, HALT, PUSH_STR, STR_CONCAT, STR_LEN,
 * ARR_LITERAL, ARR_GET, ARR_LEN, ARR_PUSH of array<int>, array<string>,
 * and lists of int/string records, ARR_NEW of array<int> or a string/record
 * list as a heap object so ARR_PUSH then POP mutates the local the way the
 * VM does (void list_int_push / list_string_push / list_Tok_push). ARR_SET
 * of array<int> mutates in place and pushes the same heap object (void
 * list_int_set). ARR_SET of a string list mutates in place (void
 * list_string_set). ARR_SET of a record list mutates nrec_t slots in place
 * (void list_Tok_set). The C
 * seed emits ARR_NEW tag 1 for every list_T_new; string and record lists
 * are classified from the pushed value.
 * AGG_PACK, AGG_GET of int and string fields, bool results as i64, PUSH_BOOL, BOOL_NOT, BOOL_AND,
 * BOOL_OR, PRINT, PRINTLN, ASSERT, STR_CONTAINS, CAST_STRING of i64,
 * EQ/NE of strings, STR_SUBSTR, STR_CHAR_AT, STR_STARTS_WITH, STR_ENDS_WITH,
 * and void functions (RET with no value; CALL of void does not POP),
 * and pin-record results as nrec_t (CALL/RET of TAG_STRUCT).
 * Anything else is refused with an error.
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
