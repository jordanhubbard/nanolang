#ifndef NL_NSI_SERVICES_VALUES_H
#define NL_NSI_SERVICES_VALUES_H
#include "nsi_file_values.h"
#include "nsi_socket_values.h"

/* I retain one independent lifetime core per nominal instance. This private
 * adapter grants no source/import authority. My trusted caller supplies the
 * catalog table and serializes every call, including creation and destruction.
 * Caller objects are valid and disjoint from context/input/output storage.
 * Values and borrows are checked identities; copying them adds no ownership. */
#define NL_SERVICES_VALUE_INSTANCES 64u
typedef enum { NL_SERVICES_FILE=1, NL_SERVICES_TCP=2 } NlServicesCatalog;
typedef enum {
    NL_SERVICES_VALUE_OK, NL_SERVICES_VALUE_ARGUMENT, NL_SERVICES_VALUE_STALE,
    NL_SERVICES_VALUE_TYPE, NL_SERVICES_VALUE_BORROWED, NL_SERVICES_VALUE_LIMIT,
    NL_SERVICES_VALUE_MEMORY, NL_SERVICES_VALUE_DISPOSED, NL_SERVICES_VALUE_STATE
} NlServicesValueStatus;
typedef struct NlServicesValues NlServicesValues;
typedef struct {
    uint32_t instance; /* One-based; zero denotes an empty value. */
    NlServicesCatalog catalog;
    union { NlFileValue file; NlSocketValue tcp; } value;
} NlServicesValue;
typedef struct {
    uint32_t instance;
    NlServicesCatalog catalog;
    union { NlFileValueBorrow file; NlSocketValueBorrow tcp; } borrow;
} NlServicesBorrow;
typedef struct {
    NlServicesCatalog catalog;
    bool ok, pending;
    union { NlFileResult file; NlSocketResult tcp; } error;
} NlServicesOpenView;
typedef struct {
    NlServicesCatalog catalog;
    union { NlFileScalarResult file; NlSocketScalarResult tcp; } result;
} NlServicesScalarResult;
typedef struct {
    NlServicesValueStatus execution;
    uint32_t count;
    uint64_t cleanup_failures;
    struct {
        NlServicesCatalog catalog;
        union { NlFileValuesFinish file; NlSocketValuesFinish tcp; } finish;
    } instances[NL_SERVICES_VALUE_INSTANCES];
} NlServicesFinish;
/* I copy the table, allocate all cores, and preserve *out on failure. Creation
 * requires *out==NULL. No resource is acquired until acquire. */
bool nl_services_values_storage_bound(const NlServicesCatalog *,uint32_t,size_t *);
NlServicesValueStatus nl_services_values_create(const NlServicesCatalog *,uint32_t,NlServicesValues **);
/* Instance arguments are zero-based table positions. File requires no Endpoint;
 * TCP requires one. An accepted host error publishes an owned Error Result. */
NlServicesValueStatus nl_services_values_acquire(NlServicesValues *,uint32_t,const NlSocketEndpoint *,NlServicesValue *);
NlServicesValueStatus nl_services_value_move(NlServicesValues *,NlServicesValue *,NlServicesValue *);
NlServicesValueStatus nl_services_value_view(NlServicesValues *,const NlServicesValue *,NlServicesOpenView *);
NlServicesValueStatus nl_services_value_take_ok(NlServicesValues *,NlServicesValue *,NlServicesValue *);
NlServicesValueStatus nl_services_value_take_error(NlServicesValues *,NlServicesValue *,NlServicesOpenView *);
NlServicesValueStatus nl_services_value_borrow(NlServicesValues *,const NlServicesValue *,NlServicesBorrow *);
NlServicesValueStatus nl_services_value_end_borrow(NlServicesValues *,NlServicesBorrow *);
/* Methods 1/2/3 are write-or-send/rewind-or-finish/read-or-receive. Only method1
 * accepts a byte argument; other methods require zero. Instance mismatch refuses
 * before host I/O and preserves both the borrow and result. */
NlServicesValueStatus nl_services_value_call(NlServicesValues *,uint32_t,uint32_t,const NlServicesBorrow *,int64_t,NlServicesScalarResult *);
NlServicesValueStatus nl_services_value_close(NlServicesValues *,uint32_t,NlServicesValue *,NlServicesScalarResult *);
/* I expose private validation and exact per-instance owner/borrow masks for
 * checked frame transfers. These queries acquire no host authority. */
NlServicesValueStatus nl_services_value_validate(NlServicesValues *,const NlServicesValue *,bool);
NlServicesValueStatus nl_services_borrow_validate(NlServicesValues *,const NlServicesBorrow *);
NlServicesValueStatus nl_services_values_live_slots(NlServicesValues *,uint32_t,uint64_t *,uint64_t *);
NlServicesValueStatus nl_services_value_drop(NlServicesValues *,NlServicesValue *);
/* Finish drains every core, including outstanding borrows, and caches the first
 * execution status and every instance's cleanup evidence. Later finish calls
 * return that same report. Destroy clears the pointer after draining. */
bool nl_services_values_report(const NlServicesValues *,NlServicesFinish *);
NlServicesValueStatus nl_services_values_finish(NlServicesValues *,NlServicesValueStatus,NlServicesFinish *);
NlServicesValueStatus nl_services_values_destroy(NlServicesValues **,NlServicesValueStatus,NlServicesFinish *);
#endif
