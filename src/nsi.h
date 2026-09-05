#ifndef NL_NSI_H
#define NL_NSI_H

#include <stddef.h>
#include <stdbool.h>

/* Nano Service Interface v0: identifiers plus an optional parameter
 * contract (direction, ownership, lifetime, mutability, optionality,
 * streaming). Records, compatibility, generation, and adapters are later. */

#define NL_NSI_VERSION 0

typedef enum {
    NL_NSI_DIR_IN = 0,
    NL_NSI_DIR_OUT,
    NL_NSI_DIR_INOUT,
    NL_NSI_DIR_RETURN
} NlNsiDirection;

typedef enum {
    NL_NSI_OWN_BORROW = 0,
    NL_NSI_OWN_TRANSFER,
    NL_NSI_OWN_COPY
} NlNsiOwnership;

typedef enum {
    NL_NSI_LIFE_CALL = 0,
    NL_NSI_LIFE_CALLER,
    NL_NSI_LIFE_CALLEE,
    NL_NSI_LIFE_RESOURCE
} NlNsiLifetime;

typedef enum {
    NL_NSI_MUT_IMMUTABLE = 0,
    NL_NSI_MUT_MUTABLE
} NlNsiMutability;

typedef enum {
    NL_NSI_STREAM_NONE = 0,
    NL_NSI_STREAM_IN,
    NL_NSI_STREAM_OUT,
    NL_NSI_STREAM_BIDI
} NlNsiStreaming;

typedef struct {
    char *id;
    char *name;
} NlNsiNamed;

typedef struct {
    char *id;
    char *name;
    char *type_id;
    NlNsiDirection direction;
    NlNsiOwnership ownership;
    NlNsiLifetime lifetime;
    NlNsiMutability mutability;
    int optional;
    NlNsiStreaming streaming;
} NlNsiParam;

typedef struct {
    char *id;
    char *name;
    NlNsiParam *params;
    size_t param_count;
} NlNsiMethod;

typedef struct {
    int version;
    NlNsiNamed iface;
    NlNsiMethod *methods;
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
size_t nl_nsi_param_count(const NlNsi *nsi, size_t method_i);
const NlNsiParam *nl_nsi_param(const NlNsi *nsi, size_t method_i, size_t param_i);

#endif
