#ifndef NANOISA_RECORD_ARRAY_GENERATED_PRIVATE_H
#define NANOISA_RECORD_ARRAY_GENERATED_PRIVATE_H
#include "managed_strings.h"
#include <stdbool.h>
/* Private generated products only. This interface grants no public admission. */
#ifdef NANO_RECORD_ARRAY_GENERATED_PRIVATE
#define NRG_ABI 1u
#define NRG_FRAMES 1024u
#define NRG_ROOTS (NRG_FRAMES*512u+1u)
#define NRG_EXTRA_BYTES (UINT64_C(128)*1024*1024)
#define NRG_EXTRA_STEPS UINT64_C(33554432)
typedef enum {
    NRG_OK, NRG_TYPE, NRG_BOUNDS, NRG_ASSERT, NRG_MEMORY, NRG_ARITHMETIC,
    NRG_FRAMES_EXHAUSTED, NRG_BUSY, NRG_STATE
} NrgStatus;
typedef struct NrgInstance NrgInstance;
/* A generated function runs real labels until a call, return or error. */
typedef void (*NrgBody)(NrgInstance *);
typedef struct {
    uint32_t locals, arity, result_count, result_tag, maximum_stack;
    const uint8_t *parameters; /* NULL retains the original unknown signature. */
    NrgBody body;
} NrgFunction;
typedef struct {
    uint32_t tag, nested_layout, element;
} NrgField;
typedef struct {
    uint32_t abi, value_size, value_tag_offset, frame_limit;
    uint32_t function_count, entry, initializer, global_count;
    uint32_t literal_count, record_count, field_count;
    const NrgFunction *functions;
    const NmsView *literals;
    const NmsRecordDescriptor *records;
    const uint32_t *field_starts;
    const NrgField *fields;
} NrgProgram;
/* Program tables are immutable generated storage alive through disposal.
 * Host preparation separately proves exact copied-plan correspondence. */
NrgStatus nrg_create(const NrgProgram *, NrgInstance **);
NrgStatus nrg_run(NrgInstance *);
void nrg_destroy(NrgInstance *);
NrgStatus nrg_status(const NrgInstance *);
void nrg_fail(NrgInstance *, NrgStatus);
uint32_t nrg_resume(const NrgInstance *);
void nrg_call(NrgInstance *, uint32_t function, uint32_t continuation);
void nrg_return(NrgInstance *);
/* Borrowed operand views never survive an allocating helper or transfer.
 * push_move clears the input only on success; replace consumes operand roots
 * after the new result has been acquired. All failures retain live roots. */
bool nrg_peek(NrgInstance *, uint32_t distance, NmsValue *);
bool nrg_push_move(NrgInstance *, NmsValue *);
bool nrg_replace(NrgInstance *, uint32_t consumed, NmsValue *);
bool nrg_load(NrgInstance *, bool global, uint32_t index);
bool nrg_store(NrgInstance *, bool global, uint32_t index);
bool nrg_drop(NrgInstance *);
bool nrg_dup(NrgInstance *);
bool nrg_swap(NrgInstance *);
bool nrg_safe_point(NrgInstance *);
/* These operations borrow all operands until a complete result is published. */
bool nrg_record_new(NrgInstance *, uint32_t ordinal);
bool nrg_record_get(NrgInstance *, uint32_t field, bool aggregate);
bool nrg_record_set(NrgInstance *, uint32_t field, bool aggregate);
bool nrg_array_new(NrgInstance *, uint32_t tag, uint32_t count, bool literal);
bool nrg_array_get(NrgInstance *);
bool nrg_array_set(NrgInstance *);
bool nrg_array_push(NrgInstance *);
bool nrg_array_pop(NrgInstance *);
bool nrg_array_slice(NrgInstance *);
/* Shared primitive helpers contain no bytecode or instruction dispatch. */
bool nrg_truth(NrgInstance *,NmsValue);
bool nrg_equal(NrgInstance *,NmsValue,NmsValue);
int nrg_order(NrgInstance *,NmsValue,NmsValue);
bool nrg_cast_int(NrgInstance *);
bool nrg_cast_float(NrgInstance *);
bool nrg_cast_string(NrgInstance *);
bool nrg_format(NrgInstance *,uint32_t tag);
bool nrg_concat(NrgInstance *);
bool nrg_substring(NrgInstance *);
bool nrg_trim(NrgInstance *);
bool nrg_case(NrgInstance *,bool upper);
bool nrg_split(NrgInstance *);
bool nrg_string_replace(NrgInstance *);
bool nrg_length(NrgInstance *,bool array);
bool nrg_character(NrgInstance *);
bool nrg_predicate(NrgInstance *,uint32_t predicate);
/* Snapshot observations borrow roots; no retainable handles cross instances. */
typedef struct {
    uint64_t epoch, preparation_bytes, live_bytes, live_objects;
    uint32_t frames, maximum_frames;
    NrgStatus status;
    bool has_result;
} NrgStats;
bool nrg_stats(const NrgInstance *, NrgStats *);
#endif
#endif
