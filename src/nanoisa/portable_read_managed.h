#ifndef NANOISA_PORTABLE_READ_MANAGED_H
#define NANOISA_PORTABLE_READ_MANAGED_H
#include "managed_strings.h"
#include "portable_read_host.h"

typedef struct {
    NprStatus host_status;
    NmsStatus managed_status;
    NmsHandle value;
} NprManagedResult;
/* I borrow the argument in its active runtime on every path. Only two OK
 * statuses publish an owned, copied STRING root. Every failure returns zero.
 * NPR_OK with a managed failure does not claim that the callback ran.
 * The trusted callback may not retain scratch or reenter/mutate this runtime.
 * I do not begin, finish, collect, dispose, or consume caller roots. */
NprManagedResult npr_read_managed(NmsRuntime *, NmsHandle,
                                 const NprHostBinding *);
#endif
