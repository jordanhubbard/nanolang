#ifndef NANO_BSTRING_CONTRACTS_H
#define NANO_BSTRING_CONTRACTS_H

#include "nanoisa/isa.h"
#include <string.h>

typedef enum {
    NANO_BSTR_NEW, NANO_BSTR_BINARY, NANO_BSTR_LENGTH, NANO_BSTR_CONCAT,
    NANO_BSTR_SUBSTRING, NANO_BSTR_EQUALS, NANO_BSTR_BYTE,
    NANO_BSTR_VALIDATE, NANO_BSTR_UTF8_LENGTH, NANO_BSTR_UTF8_CHAR,
    NANO_BSTR_CSTR, NANO_BSTR_FREE
} NanoBstringOperation;

typedef struct {
    const char *source_name, *host_name;
    NanoBstringOperation operation;
    uint8_t argc, result, parameters[3];
} NanoBstringContract;

/* I keep my new length-aware ABI distinct from legacy vm_bstr_* STRING calls. */
static const NanoBstringContract nano_bstring_contracts[] = {
    {"bstr_new", "vm_bstring_new", NANO_BSTR_NEW, 1, TAG_BSTRING, {TAG_STRING}},
    {"bstr_new_binary", "vm_bstring_new_binary", NANO_BSTR_BINARY, 1, TAG_BSTRING, {TAG_ARRAY}},
    {"bstr_length", "vm_bstring_length", NANO_BSTR_LENGTH, 1, TAG_INT, {TAG_BSTRING}},
    {"bstr_concat", "vm_bstring_concat", NANO_BSTR_CONCAT, 2, TAG_BSTRING, {TAG_BSTRING, TAG_BSTRING}},
    {"bstr_substring", "vm_bstring_substring", NANO_BSTR_SUBSTRING, 3, TAG_BSTRING, {TAG_BSTRING, TAG_INT, TAG_INT}},
    {"bstr_equals", "vm_bstring_equals", NANO_BSTR_EQUALS, 2, TAG_BOOL, {TAG_BSTRING, TAG_BSTRING}},
    {"bstr_byte_at", "vm_bstring_byte_at", NANO_BSTR_BYTE, 2, TAG_INT, {TAG_BSTRING, TAG_INT}},
    {"bstr_validate_utf8", "vm_bstring_validate_utf8", NANO_BSTR_VALIDATE, 1, TAG_BOOL, {TAG_BSTRING}},
    {"bstr_utf8_length", "vm_bstring_utf8_length", NANO_BSTR_UTF8_LENGTH, 1, TAG_INT, {TAG_BSTRING}},
    {"bstr_utf8_char_at", "vm_bstring_utf8_char_at", NANO_BSTR_UTF8_CHAR, 2, TAG_INT, {TAG_BSTRING, TAG_INT}},
    {"bstr_to_cstr", "vm_bstring_to_cstr", NANO_BSTR_CSTR, 1, TAG_STRING, {TAG_BSTRING}},
    {"bstr_free", "vm_bstring_free", NANO_BSTR_FREE, 1, TAG_VOID, {TAG_BSTRING}},
};

static inline const NanoBstringContract *nano_bstring_source_contract(const char *name) {
    if (!name || strncmp(name, "bstr_", 5)) return NULL;
    for (size_t i = 0; i < sizeof nano_bstring_contracts / sizeof nano_bstring_contracts[0]; ++i)
        if (!strcmp(name, nano_bstring_contracts[i].source_name)) return &nano_bstring_contracts[i];
    return NULL;
}

static inline const NanoBstringContract *nano_bstring_host_contract(const char *name) {
    if (!name || strncmp(name, "vm_bstring_", 11)) return NULL;
    for (size_t i = 0; i < sizeof nano_bstring_contracts / sizeof nano_bstring_contracts[0]; ++i)
        if (!strcmp(name, nano_bstring_contracts[i].host_name)) return &nano_bstring_contracts[i];
    return NULL;
}

#endif
