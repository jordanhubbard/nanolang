#ifndef NANOISA_MANAGED_STRINGS_H
#define NANOISA_MANAGED_STRINGS_H
#include <stddef.h>
#include <stdint.h>

/* My private runtime API does not itself grant bytecode/profile admission. */
typedef enum {
    NMS_OK = 0, NMS_TYPE = 1, NMS_ASSERT = 2, NMS_MEMORY = 3,
    NMS_BUSY = 4, NMS_DISPOSED = 5, NMS_STATE = 6
} NmsStatus;
typedef uint64_t NmsHandle;
#define NMS_DYNAMIC (UINT64_C(1) << 63)
typedef struct { const unsigned char *data; uint32_t length; } NmsView;
typedef struct {
    unsigned char *data;
    uint64_t references;
    uint32_t length, next_free;
} NmsSlot;
typedef struct {
    const NmsView *literals; /* Borrowed immutable storage, alive until disposal. */
    NmsSlot *slots;
    uint64_t live_bytes, live_objects;
    uint32_t literal_count, capacity, free_head;
    unsigned active, disposed;
#ifdef NMS_TESTING
    uint64_t fail_after;
#endif
} NmsRuntime;

/* init requires fresh storage or a previously disposed instance. Handles and
 * views never cross runtime instances; a view borrows its handle's lifetime. */
void nms_init(NmsRuntime *, const NmsView *, uint32_t);
NmsStatus nms_create(NmsRuntime *, const unsigned char *, uint64_t, NmsHandle *);
/* I consume one source owner on every path; output changes only on success.
 * Trim also allocates a fresh result when no bytes change. */
NmsStatus nms_trim_owned(NmsRuntime *, NmsHandle, NmsHandle *);
NmsStatus nms_substr_owned(NmsRuntime *, NmsHandle, uint32_t, uint32_t, NmsHandle *);
/* I consume one owned reference per input on success or failure. Equal inputs
 * require two references. Other aliases survive; out is unchanged on failure. */
NmsStatus nms_concat_owned(NmsRuntime *, NmsHandle, NmsHandle, NmsHandle *);
/* I create one owner from scalar bits; strings transfer in lowering. */
NmsStatus nms_format_scalar(NmsRuntime *, uint64_t, uint32_t, NmsHandle *);
/* I borrow the handle, parse C-locale decimal bytes, and allocate nothing. */
NmsStatus nms_parse_f64(const NmsRuntime *, NmsHandle, uint64_t *);
NmsStatus nms_parse_i64(const NmsRuntime *, NmsHandle, int64_t *);
/* I borrow both handles and allocate nothing; output changes only on success. */
typedef enum { NMS_CONTAINS = 0, NMS_STARTS_WITH = 1, NMS_ENDS_WITH = 2 } NmsPredicate;
NmsStatus nms_predicate(const NmsRuntime *, NmsHandle, NmsHandle, uint32_t, uint32_t *);
NmsStatus nms_view(const NmsRuntime *, NmsHandle, NmsView *);
NmsStatus nms_retain(NmsRuntime *, NmsHandle);
NmsStatus nms_release(NmsRuntime *, NmsHandle);
/* These guard exported entry; generated frames own their separate cleanup. */
NmsStatus nms_begin(NmsRuntime *);
uint64_t nms_finish(NmsRuntime *, NmsStatus, int32_t);
NmsStatus nms_dispose(NmsRuntime *);
int nms_reserved_entry(const char *);
#ifdef NMS_TESTING
void nms_test_fail_after(NmsRuntime *, uint64_t);
uint64_t nms_test_live_allocations(void);
uint64_t nms_test_memory_pages(void);
#endif
#endif
