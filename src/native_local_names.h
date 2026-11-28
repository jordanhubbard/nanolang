#ifndef NANOLANG_NATIVE_LOCAL_NAMES_H
#define NANOLANG_NATIVE_LOCAL_NAMES_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NL_NATIVE_LOCAL_PREFIX "__nl_keyword_"

/* I reserve my emitted prefix as well as C keywords, so a source spelling
 * cannot impersonate another binding's generated storage name. */
static inline bool native_c_name_needs_alias(const char *name) {
    static const char *const keywords[] = {
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "inline", "int", "long", "register", "restrict", "return", "short",
        "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
        "unsigned", "void", "volatile", "while", "_Alignas", "_Alignof",
        "_Atomic", "_BitInt", "_Bool", "_Complex", "_Decimal32", "_Decimal64",
        "_Decimal128", "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
        "_Thread_local", "alignas", "alignof", "bool", "constexpr", "false",
        "nullptr", "static_assert", "thread_local", "true", "typeof",
        "typeof_unqual", "__asm", "__asm__", "__attribute", "__attribute__",
        "__auto_type", "__complex", "__complex__", "__const", "__const__",
        "__extension__", "__imag", "__imag__", "__inline", "__inline__",
        "__label__", "__real", "__real__", "__restrict", "__restrict__",
        "__signed", "__signed__", "__thread", "__typeof", "__typeof__",
        "__volatile", "__volatile__"
    };
    if (!name) return false;
    if (strncmp(name, NL_NATIVE_LOCAL_PREFIX,
                sizeof(NL_NATIVE_LOCAL_PREFIX) - 1) == 0) return true;
    for (size_t i = 0; i < sizeof(keywords) / sizeof(keywords[0]); ++i)
        if (strcmp(name, keywords[i]) == 0) return true;
    return false;
}

/* I return owned C storage spelling. Source names and foreign symbols retain
 * their own identities; parameter names do not participate in a C ABI. */
static inline char *native_c_parameter_name(const char *name) {
    static const char prefix[] = NL_NATIVE_LOCAL_PREFIX "p_";
    static const char hex[] = "0123456789abcdef";
    if (!name) return NULL;
    size_t length = strlen(name);
    if (!native_c_name_needs_alias(name)) {
        if (length == SIZE_MAX) return NULL;
        char *out = malloc(length + 1);
        if (out) memcpy(out, name, length + 1);
        return out;
    }
    if (length > (SIZE_MAX - sizeof(prefix)) / 2) return NULL;
    char *out = malloc(sizeof(prefix) + length * 2);
    if (!out) return NULL;
    memcpy(out, prefix, sizeof(prefix) - 1);
    for (size_t i = 0; i < length; ++i) {
        unsigned char byte = (unsigned char)name[i];
        out[sizeof(prefix) - 1 + i * 2] = hex[byte >> 4];
        out[sizeof(prefix) + i * 2] = hex[byte & 15];
    }
    out[sizeof(prefix) - 1 + length * 2] = '\0';
    return out;
}

#endif
