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
 * EQ/NE of strings, STR_SUBSTR, STR_CHAR_AT, STR_STARTS_WITH and STR_ENDS_WITH.
 * Prefix/suffix predicates use byte comparisons, including empty patterns.
 * ARR_SET mutates int/string/flat-record arrays in place and returns the same
 * handle. I reject incompatible representations and abort on invalid indices.
 * I emit UTF-8 and control bytes with fixed-width C escapes; embedded NUL
 * remains refused because these helpers use NUL-terminated strings.
 * I also lower typed F64 arithmetic/negation/comparison, scalar float locals,
 * call/return/tail transport and checked tagged globals. Typed operations require
 * float operands; F64_DIV returns positive zero for either signed zero divisor.
 * CAST_STRING retains VM `%g` formatting, while printing keeps a decimal for
 * whole floats in the VM's bounded range. Explicit float-to-int conversion and
 * float arrays/record fields remain outside this scalar contract.
 * Anything else is refused with an error.
 * I lower exact builtin CALL_EXTERN signatures for get_argc, get_argv,
 * vm_getcwd, vm_tmp_dir, vm_getenv and nl_os_getenv to native host helpers.
 * I require the empty builtin namespace and FFI import kind; foreign modules,
 * co-process/artifact imports and other signatures remain refused. Arguments
 * include argv[0]; invalid argument indices and missing environment values
 * return empty strings. Returned copies currently live until process exit;
 * this is not a bounded-memory runtime. Cwd uses the VM's 1024-byte limit.
 * Embedded NULs, nested arrays, nested records, tuples,
 * printing arrays/records, array equality, STR_TRIM, and the rest of the
 * string and array libraries stay refused.
 * I track operand-stack joins and emit simultaneous transfers on taken
 * edges. I converge parameter kinds and flat aggregate field kinds across
 * direct calls before emitting prototypes and bodies. Flat record/variant
 * returns, forwarding and tail calls use C value returns. I reject conflicting
 * field layouts and unresolved packed fields instead of guessing their types.
 * Local classification remains function-wide, not generally flow-sensitive.
 * Self TAIL_CALL uses simultaneous parameter transfer and a local C restart,
 * resetting non-parameter locals. Cross-function tail calls still use C calls;
 * I do not yet guarantee bounded native stack for mutual tail recursion.
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
