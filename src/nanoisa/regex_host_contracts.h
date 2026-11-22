#ifndef NANO_REGEX_HOST_CONTRACTS_H
#define NANO_REGEX_HOST_CONTRACTS_H
#include "isa.h"
#include <string.h>

/* I admit only these builtin signatures, including each array element family. */
#define NANO_REGEX_HOST_CONTRACTS(X) \
    X(nl_regex_compile, nhost_regex_compile, 1, TAG_OPAQUE, TAG_STRING, TAG_VOID, TAG_VOID, TAG_VOID) \
    X(nl_regex_match, nhost_regex_match, 2, TAG_INT, TAG_OPAQUE, TAG_STRING, TAG_VOID, TAG_VOID) \
    X(nl_regex_find, nhost_regex_find, 2, TAG_INT, TAG_OPAQUE, TAG_STRING, TAG_VOID, TAG_VOID) \
    X(nl_regex_find_all, nhost_regex_find_all, 2, TAG_ARRAY, TAG_OPAQUE, TAG_STRING, TAG_VOID, TAG_INT) \
    X(nl_regex_groups, nhost_regex_groups, 2, TAG_ARRAY, TAG_OPAQUE, TAG_STRING, TAG_VOID, TAG_STRING) \
    X(nl_regex_replace, nhost_regex_replace, 3, TAG_STRING, TAG_OPAQUE, TAG_STRING, TAG_STRING, TAG_VOID) \
    X(nl_regex_replace_all, nhost_regex_replace_all, 3, TAG_STRING, TAG_OPAQUE, TAG_STRING, TAG_STRING, TAG_VOID) \
    X(nl_regex_split, nhost_regex_split, 2, TAG_ARRAY, TAG_OPAQUE, TAG_STRING, TAG_VOID, TAG_STRING) \
    X(nl_regex_free, nhost_regex_free, 1, TAG_VOID, TAG_OPAQUE, TAG_VOID, TAG_VOID, TAG_VOID)

typedef struct {
    const char *name;
    uint8_t argc, result, parameters[3], array_element;
} NanoRegexHostContract;
#define NANO_REGEX_CONTRACT_ROW(name, native, argc, result, p0, p1, p2, element) \
    {#name, argc, result, {p0, p1, p2}, element},
static const NanoRegexHostContract nano_regex_host_contracts[] = {
    NANO_REGEX_HOST_CONTRACTS(NANO_REGEX_CONTRACT_ROW)
};
#undef NANO_REGEX_CONTRACT_ROW
static inline const NanoRegexHostContract *nano_regex_host_contract(const char *name) {
    if (!name || strncmp(name, "nl_regex_", 9)) return NULL;
    for (size_t i = 0; i < sizeof nano_regex_host_contracts / sizeof nano_regex_host_contracts[0]; ++i)
        if (!strcmp(name, nano_regex_host_contracts[i].name)) return &nano_regex_host_contracts[i];
    return NULL;
}
#endif
