#ifndef NL_NSI_H
#define NL_NSI_H

#include <stddef.h>
#include <stdbool.h>

/* Nano Service Interface v0: identifiers only. Direction, ownership,
 * generation, and adapters are later NSI versions. */

#define NL_NSI_VERSION 0

typedef struct {
    char *id;
    char *name;
} NlNsiNamed;

typedef struct {
    int version;
    NlNsiNamed iface;
    NlNsiNamed *methods;
    size_t method_count;
    NlNsiNamed *types;
    size_t type_count;
    NlNsiNamed *errors;
    size_t error_count;
    NlNsiNamed *capabilities;
    size_t capability_count;
} NlNsi;

void nl_nsi_free(NlNsi *nsi);
NlNsi *nl_nsi_load_path(const char *path);

const char *nl_nsi_interface_id(const NlNsi *nsi);
size_t nl_nsi_method_count(const NlNsi *nsi);
const char *nl_nsi_method_id(const NlNsi *nsi, size_t i);

#endif
