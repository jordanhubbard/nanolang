#ifndef NL_MODULE_SYMBOL_H
#define NL_MODULE_SYMBOL_H
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* I preserve bytes in NUL-terminated metadata, without C escape continuation
 * or trigraph interpretation. Three-digit octal escapes have fixed width. */
static inline const char *module_c_literal(const char *text) {
    static _Thread_local char buffer[32768];
    if (!text) text = "";
    size_t length = strlen(text);
    if (length > (sizeof(buffer) - 3) / 4) {
        fprintf(stderr, "I cannot represent this metadata string.\n");
        exit(1);
    }
    char *out = buffer;
    *out++ = '"';
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p >= 32 && *p < 127 && *p != '"' && *p != '\\' && *p != '?') {
            *out++ = (char)*p;
        } else {
            *out++ = '\\';
            *out++ = (char)('0' + (*p >> 6));
            *out++ = (char)('0' + ((*p >> 3) & 7));
            *out++ = (char)('0' + (*p & 7));
        }
    }
    *out++ = '"';
    *out = '\0';
    return buffer;
}

/* I preserve ordinary suffixes; the reserved prefix is itself encoded so
 * distinct byte strings cannot collide with a pre-encoded-looking name. */
static inline const char *module_symbol_suffix(const char *name) {
    static _Thread_local char buffer[4096];
    const char prefix[] = "__nano_hex_";
    const char hex[] = "0123456789abcdef";
    bool plain = name[0] && strncmp(name, prefix, sizeof(prefix) - 1) != 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) {
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
              (*p >= '0' && *p <= '9') || *p == '_')) plain = false;
    }
    if (plain) return name;
    size_t length = strlen(name);
    if (length > (sizeof(buffer) - sizeof(prefix)) / 2) {
        fprintf(stderr, "I cannot represent this module symbol.\n");
        exit(1);
    }
    memcpy(buffer, prefix, sizeof(prefix) - 1);
    char *out = buffer + sizeof(prefix) - 1;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) {
        *out++ = hex[*p >> 4];
        *out++ = hex[*p & 15];
    }
    *out = '\0';
    return buffer;
}

static inline const char *module_helper_c_name(const char *name) {
    static _Thread_local char buffer[8192];
    const char *prefixes[] = {"___module_is_unsafe_", "___module_has_ffi_",
        "___module_name_", "___module_path_", "___module_function_count_",
        "___module_function_name_", "___module_struct_count_", "___module_struct_name_"};
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
        size_t length = strlen(prefixes[i]);
        if (strncmp(name, prefixes[i], length) != 0) continue;
        const char *suffix = module_symbol_suffix(name + length);
        if (suffix == name + length) return name;
        snprintf(buffer, sizeof(buffer), "%s%s", prefixes[i], suffix);
        return buffer;
    }
    return name;
}
#endif
