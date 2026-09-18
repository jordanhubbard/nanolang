#ifndef NANOISA_MANAGED_STRINGS_H
#define NANOISA_MANAGED_STRINGS_H
#include <stddef.h>
#include <stdint.h>

/* My private runtime API does not itself grant bytecode/profile admission. */
typedef enum {
    NMS_OK = 0, NMS_TYPE = 1, NMS_ASSERT = 2, NMS_MEMORY = 3,
    NMS_BUSY = 4, NMS_DISPOSED = 5, NMS_STATE = 6, NMS_BOUNDS = 7
} NmsStatus;
typedef uint64_t NmsHandle;
/* I share this value tag with the ISA; the emitter asserts its ABI. */
#define NMS_ARRAY_TAG 7
#define NMS_DYNAMIC (UINT64_C(1) << 63)
typedef struct { const unsigned char *data; uint32_t length; } NmsView;
typedef enum { NMS_SLOT_FREE = 0, NMS_SLOT_STRING = 1, NMS_SLOT_STRING_ARRAY = 2, NMS_SLOT_BOXED_ARRAY = 3, NMS_SLOT_BOXED_LEAF_ARRAY = NMS_SLOT_BOXED_ARRAY, NMS_SLOT_PACKED_SCALAR_ARRAY = 4 } NmsSlotKind;
typedef struct {
    unsigned char *data;
    uint64_t references;
    uint32_t length, next_free, capacity, kind, element_tag, vm_array_policy;
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
/* My arrays own string children only. Append borrows both arguments and
 * retains one child on success; get returns an owner, or zero for a missing
 * unsigned index. Outputs and array contents change only on success. */
/* I consume both string owners and publish a complete split array only on success. */
NmsStatus nms_split_owned(NmsRuntime *, NmsHandle, NmsHandle, NmsHandle *);
NmsStatus nms_string_array_create(NmsRuntime *, NmsHandle *);
NmsStatus nms_string_array_append(NmsRuntime *, NmsHandle, NmsHandle);
NmsStatus nms_string_array_get(NmsRuntime *, NmsHandle, uint64_t, NmsHandle *);
NmsStatus nms_string_array_length(const NmsRuntime *, NmsHandle, uint32_t *);
/* My boxed-value API preserves scalar bits and owns string/array handles.
 * Other heap/callable tags remain outside this private graph foundation. */
typedef struct { uint64_t payload; uint32_t tag; } NmsValue;
NmsStatus nms_value_retain(NmsRuntime *, NmsValue);
NmsStatus nms_value_release(NmsRuntime *, NmsValue);
/* I retain a declared int/U8/float/bool kind; this API grants no opcode admission. */
NmsStatus nms_packed_array_create(NmsRuntime *, uint32_t, NmsHandle *);
NmsStatus nms_value_array_create(NmsRuntime *, NmsHandle *);
/* I prepare capacity8 and VM checked-doubling semantics before publication. */
NmsStatus nms_vm_array_create(NmsRuntime *, uint32_t, NmsHandle *);
/* I borrow input roots and publish only a complete fresh VM-policy array.
 * Literal count is uint16-bounded; slice endpoints are already uint32 values.
 * Input vectors remain valid for the call (outside relocating slot storage).
 * Failure leaves inputs and output unchanged. These APIs grant no admission. */
NmsStatus nms_vm_array_literal(NmsRuntime *, uint32_t, const uint64_t *, const uint32_t *, uint32_t, NmsHandle *);
NmsStatus nms_vm_array_slice(NmsRuntime *, NmsHandle, uint32_t, uint32_t, NmsHandle *);
/* I consume both strings and build boxed tagged children before publication. */
NmsStatus nms_split_values_owned(NmsRuntime *, NmsHandle, NmsHandle, NmsHandle *);
NmsStatus nms_value_array_append(NmsRuntime *, NmsHandle, NmsValue);
NmsStatus nms_value_array_set(NmsRuntime *, NmsHandle, uint64_t, NmsValue);
NmsStatus nms_value_array_get(NmsRuntime *, NmsHandle, uint64_t, NmsValue *);
NmsStatus nms_value_array_pop(NmsRuntime *, NmsHandle, NmsValue *);
NmsStatus nms_value_array_length(const NmsRuntime *, NmsHandle, uint32_t *);
/* I consume one owner and publish a fresh ASCII-mapped result; upper is 0/1.
 * Output changes only on success. */
NmsStatus nms_case_owned(NmsRuntime *, NmsHandle, uint32_t, NmsHandle *);
/* I consume one source owner on every path; output changes only on success.
 * Trim also allocates a fresh result when no bytes change. */
NmsStatus nms_trim_owned(NmsRuntime *, NmsHandle, NmsHandle *);
NmsStatus nms_substr_owned(NmsRuntime *, NmsHandle, uint32_t, uint32_t, NmsHandle *);
/* I consume three owners, including one per equal handle; failed output is unchanged. */
NmsStatus nms_replace_owned(NmsRuntime *, NmsHandle, NmsHandle, NmsHandle, NmsHandle *);
/* I consume one owned reference per input on success or failure. Equal inputs
 * require two references. Other aliases survive; out is unchanged on failure. */
NmsStatus nms_concat_owned(NmsRuntime *, NmsHandle, NmsHandle, NmsHandle *);
/* I create one owner from scalar bits; strings transfer in lowering. */
NmsStatus nms_format_scalar(NmsRuntime *, uint64_t, uint32_t, NmsHandle *);
/* I borrow the handle, parse C-locale decimal bytes, and allocate nothing. */
NmsStatus nms_parse_f64(const NmsRuntime *, NmsHandle, uint64_t *);
NmsStatus nms_parse_i64(const NmsRuntime *, NmsHandle, int64_t *);
typedef enum { NMS_CONTAINS = 0, NMS_STARTS_WITH = 1, NMS_ENDS_WITH = 2 } NmsPredicate;
/* I borrow the source, allocate nothing, and publish only on success.
 * Non-integer indices use zero; signed negative/out-of-range returns -1. */
NmsStatus nms_char_at(const NmsRuntime *, NmsHandle, uint64_t, uint32_t, int64_t *);
/* I borrow both handles and allocate nothing; output changes only on success. */
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
