/*
 * builtins_registry.h - Unified Builtin Function Registry
 *
 * Single authoritative table of all nanolang builtin functions.
 * Consumed by: env.c (typechecker), codegen.c (NanoISA), transpiler (C codegen).
 */
#ifndef BUILTINS_REGISTRY_H
#define BUILTINS_REGISTRY_H

#include <stdint.h>
#include <stdbool.h>

/* Forward-declare Type from nanolang.h to avoid circular include */
#ifndef NANOLANG_H
typedef enum {
    TYPE_INT = 0, TYPE_U8, TYPE_FLOAT, TYPE_BOOL, TYPE_STRING, TYPE_BSTRING,
    TYPE_VOID, TYPE_ARRAY, TYPE_STRUCT, TYPE_ENUM, TYPE_UNION,
    TYPE_GENERIC, TYPE_LIST_INT, TYPE_LIST_STRING, TYPE_LIST_TOKEN,
    TYPE_LIST_GENERIC, TYPE_HASHMAP, TYPE_FUNCTION, TYPE_TUPLE, TYPE_OPAQUE,
    TYPE_UNKNOWN
} Type;
#endif

/* Builtin flags */
#define BUILTIN_PURE       0x01  /* No side effects */
#define BUILTIN_INLINE_VM  0x02  /* Has a dedicated NanoISA opcode */
#define BUILTIN_FFI        0x04  /* Called via FFI (CALL_EXTERN) */
#define BUILTIN_IO         0x08  /* Performs I/O */
#define BUILTIN_LANG       0x10  /* Language-level builtin (known to typechecker) */

typedef struct {
    const char *name;          /* nanolang name: "str_length" */
    const char *c_name;        /* C runtime name for transpiler: "strlen" */
    uint8_t arity;
    Type param_types[4];
    Type return_type;
    uint8_t nano_opcode;       /* NanoISA opcode (0 = not inline) */
    uint32_t flags;
} BuiltinEntry;

extern const BuiltinEntry builtin_registry[];
extern const int builtin_registry_count;

/* Lookup functions */
const BuiltinEntry *builtin_find(const char *name);
const char *builtin_c_name(const char *name);
bool builtin_is_known(const char *name);

#endif /* BUILTINS_REGISTRY_H */
