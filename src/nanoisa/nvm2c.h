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
 * ARR_NEW of array<int> or a string list as a heap object so ARR_PUSH then
 * POP mutates the local the way the VM does (void list_int_push /
 * list_string_push). The C seed emits ARR_NEW tag 1 for every list_T_new;
 * string lists are classified from the pushed value.
 * AGG_PACK of records/variants, AGG_TAG, AGG_GET of int and string fields,
 * bool results as i64, PUSH_BOOL, BOOL_NOT, BOOL_AND,
 * BOOL_OR, PRINT, PRINTLN, ASSERT, STR_CONTAINS, CAST_STRING of i64,
 * EQ/NE of strings, STR_SUBSTR, STR_CHAR_AT. Anything else is refused with an error.
 * CALL_EXTERN is refused because it is the VM FFI / co-process path, not a
 * host C ABI. Embedded NULs, nested arrays, nested records, tuples,
 * printing arrays/records, array equality, STR_TRIM, and the rest of the
 * string and array libraries stay refused.
 * I track operand-stack joins and emit simultaneous transfers on taken
 * edges. I converge parameter kinds and flat aggregate field kinds across
 * direct calls before emitting prototypes and bodies. Flat record/variant
 * returns, forwarding and tail calls use C value returns. I reject conflicting
 * field layouts and unresolved packed fields instead of guessing their types.
 * Local classification remains function-wide, not generally flow-sensitive.
 * I preserve variant tags and runtime field kinds. AGG_TAG on a record,
 * out-of-range fields and mismatched field storage abort the emitted process.
 * Nested aggregates and field layouts that vary across calls remain outside
 * this representation. Runtime guards are not a complete type-system proof.
 */

#ifndef NANOISA_NVM2C_H
#define NANOISA_NVM2C_H

#include "nvm_format.h"
#include <stddef.h>

/* Returns a malloc'd C11 translation, or NULL on error.
 * On error, writes a message into err (if err_len > 0). */
char *nvm2c_emit(const NvmModule *mod, char *err, size_t err_len);

#endif /* NANOISA_NVM2C_H */
