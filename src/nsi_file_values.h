#ifndef NL_NSI_FILE_VALUES_H
#define NL_NSI_FILE_VALUES_H
#include "nsi_file.h"

/* Private, non-admitting lifetime core. All calls, including creation and calls
 * to the underlying private File adapter, require external serialization. I do
 * not claim thread safety or detect concurrent entry with an ordinary flag.
 * Context/output memory must not overlap. No pointer use follows destruction. */
typedef struct NlFileValues NlFileValues;
#define NL_FILE_VALUE_SLOTS 64u

typedef enum {
    NL_FILE_VALUE_OK, NL_FILE_VALUE_ARGUMENT, NL_FILE_VALUE_STALE,
    NL_FILE_VALUE_TYPE, NL_FILE_VALUE_BORROWED, NL_FILE_VALUE_LIMIT,
    NL_FILE_VALUE_MEMORY, NL_FILE_VALUE_DISPOSED, NL_FILE_VALUE_STATE,
    NL_FILE_VALUE_STATUS_COUNT
} NlFileValueStatus;
/* Zero initializes an empty handle. These checked identities expose no token,
 * host pointer or descriptor. C copies confer no additional ownership. */
typedef struct { uint64_t invocation, generation; uint32_t slot; } NlFileValue;
typedef struct { NlFileValue value; uint64_t epoch; } NlFileValueBorrow;
typedef struct { bool ok; NlFileResult error; } NlFileOpenView;
typedef enum {
    NL_FILE_VALUE_WRITE, NL_FILE_VALUE_POSITION, NL_FILE_VALUE_READ,
    NL_FILE_VALUE_CLOSE
} NlFileScalarKind;
typedef struct {
    NlFileScalarKind kind;
    bool ok;
    NlFileResult detail;
    int64_t value; /* Write progress, read byte, or zero for unit/Error. */
    bool eof;     /* Read Ok only; every other payload is canonical false. */
} NlFileScalarResult;
typedef struct {
    NlFileValueStatus execution;
    uint64_t cleanup_failures;
    NlFileResult first_cleanup, next_cleanup;
} NlFileValuesFinish;

/* *out must initially be NULL. Every failure preserves it. */
NlFileValueStatus nl_file_values_create(NlFileValues **out);
/* Output must be empty. I publish OpenResult.Ok or .Error only on VALUE_OK;
 * a capacity/execution failure publishes nothing and acquires no stream. */
NlFileValueStatus nl_file_values_temp(NlFileValues *, NlFileValue *out);
/* Source/destination must be disjoint; destination must be empty. Every failed
 * transfer preserves both; success clears source and invalidates old copies. */
NlFileValueStatus nl_file_value_move(NlFileValues *, NlFileValue *source, NlFileValue *out);
NlFileValueStatus nl_file_open_view(NlFileValues *, const NlFileValue *, NlFileOpenView *out);
NlFileValueStatus nl_file_open_take_ok(NlFileValues *, NlFileValue *, NlFileValue *out);
/* Differently typed input/output objects must be disjoint. */
NlFileValueStatus nl_file_open_take_error(NlFileValues *, NlFileValue *, NlFileResult *out);
NlFileValueStatus nl_file_value_borrow(NlFileValues *, const NlFileValue *, NlFileValueBorrow *out);
NlFileValueStatus nl_file_value_end_borrow(NlFileValues *, NlFileValueBorrow *);
/* Only an exact live exclusive borrow may call these operations. Error Results
 * retain File. Checked API refusal preserves every output. */
NlFileValueStatus nl_file_value_write_byte(NlFileValues *, const NlFileValueBorrow *, int64_t, NlFileScalarResult *out);
NlFileValueStatus nl_file_value_rewind(NlFileValues *, const NlFileValueBorrow *, NlFileScalarResult *out);
NlFileValueStatus nl_file_value_read_byte(NlFileValues *, const NlFileValueBorrow *, NlFileScalarResult *out);
NlFileValueStatus nl_file_value_close(NlFileValues *, NlFileValue *, NlFileScalarResult *out);
/* I drop File or either OpenResult arm; an empty handle is an idempotent no-op.
 * A stale nonempty handle is refused, without closing a different stream. */
NlFileValueStatus nl_file_value_drop(NlFileValues *, NlFileValue *);
/* Finish is terminal, cleans individual roots before disposal, then caches the
 * first supplied execution status and cleanup report. Later finish calls do not
 * replace it. Nonzero cleanup_failures forbids clean overall publication even
 * when execution is OK. Destroy returns the report before freeing C storage. */
NlFileValuesFinish nl_file_values_finish(NlFileValues *, NlFileValueStatus first_execution_error);
NlFileValuesFinish nl_file_values_destroy(NlFileValues *, NlFileValueStatus first_execution_error);
#endif
