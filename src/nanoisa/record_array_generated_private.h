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
    NRG_FRAMES_EXHAUSTED, NRG_BUSY, NRG_STATE, NRG_UNDEFINED_FUNCTION
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
    uint32_t function_count, entry, initializer, global_count, has_main;
    uint32_t literal_count, record_count, field_count;
    const NrgFunction *functions;
    const NmsView *literals;
    const NmsRecordDescriptor *records;
    const uint32_t *field_starts;
    const NrgField *fields;
} NrgProgram;
/* Pure target ABI facts: no instance, allocation, table dereference or effects.
 * LLVM consumers compare all fields before constructing an instance. */
typedef enum {
    NRG_LAYOUT_REVISION,
    NRG_LAYOUT_FRAMES,
    NRG_LAYOUT_ROOTS,
    NRG_LAYOUT_STATUS_SIZE,
    NRG_LAYOUT_BOOL_SIZE,
    NRG_LAYOUT_VALUE_SIZE,
    NRG_LAYOUT_VALUE_ALIGN,
    NRG_LAYOUT_VALUE_PAYLOAD,
    NRG_LAYOUT_VALUE_TAG,
    NRG_LAYOUT_FUNCTION_SIZE,
    NRG_LAYOUT_FUNCTION_ALIGN,
    NRG_LAYOUT_FUNCTION_LOCALS,
    NRG_LAYOUT_FUNCTION_ARITY,
    NRG_LAYOUT_FUNCTION_RESULT_COUNT,
    NRG_LAYOUT_FUNCTION_RESULT_TAG,
    NRG_LAYOUT_FUNCTION_MAXIMUM_STACK,
    NRG_LAYOUT_FUNCTION_PARAMETERS,
    NRG_LAYOUT_FUNCTION_BODY,
    NRG_LAYOUT_PROGRAM_SIZE,
    NRG_LAYOUT_PROGRAM_ALIGN,
    NRG_LAYOUT_PROGRAM_ABI,
    NRG_LAYOUT_PROGRAM_VALUE_SIZE,
    NRG_LAYOUT_PROGRAM_VALUE_TAG_OFFSET,
    NRG_LAYOUT_PROGRAM_FRAME_LIMIT,
    NRG_LAYOUT_PROGRAM_FUNCTION_COUNT,
    NRG_LAYOUT_PROGRAM_ENTRY,
    NRG_LAYOUT_PROGRAM_INITIALIZER,
    NRG_LAYOUT_PROGRAM_GLOBAL_COUNT,
    NRG_LAYOUT_PROGRAM_HAS_MAIN,
    NRG_LAYOUT_PROGRAM_LITERAL_COUNT,
    NRG_LAYOUT_PROGRAM_RECORD_COUNT,
    NRG_LAYOUT_PROGRAM_FIELD_COUNT,
    NRG_LAYOUT_PROGRAM_FUNCTIONS,
    NRG_LAYOUT_PROGRAM_LITERALS,
    NRG_LAYOUT_PROGRAM_RECORDS,
    NRG_LAYOUT_PROGRAM_FIELD_STARTS,
    NRG_LAYOUT_PROGRAM_FIELDS,
    NRG_LAYOUT_VIEW_SIZE,
    NRG_LAYOUT_VIEW_ALIGN,
    NRG_LAYOUT_VIEW_DATA,
    NRG_LAYOUT_VIEW_LENGTH,
    NRG_LAYOUT_RECORD_SIZE,
    NRG_LAYOUT_RECORD_ALIGN,
    NRG_LAYOUT_RECORD_GLOBAL_LAYOUT_INDEX,
    NRG_LAYOUT_RECORD_FIELD_COUNT,
    NRG_LAYOUT_FIELD_SIZE,
    NRG_LAYOUT_FIELD_ALIGN,
    NRG_LAYOUT_FIELD_TAG,
    NRG_LAYOUT_FIELD_NESTED_LAYOUT,
    NRG_LAYOUT_FIELD_ELEMENT,
    NRG_LAYOUT_COUNT
} NrgLayoutField;
uint32_t nrg_layout_field(uint32_t field); /* UINT32_MAX rejects an unknown field. */
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
 * after the new result has been acquired. replace always consumes its supplied
 * result owner; on failure it releases that owner and leaves operand roots
 * available to unwind. push_move leaves its supplied owner intact on failure. */
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
bool nrg_truth(NrgInstance *,const NmsValue *);
bool nrg_equal(NrgInstance *,const NmsValue *,const NmsValue *);
int nrg_order(NrgInstance *,const NmsValue *,const NmsValue *);
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
typedef struct {
    uint64_t epoch, identity, scalar_bits;
    uint32_t tag, length, layout, element;
} NrgObservation;
bool nrg_observe(const NrgInstance *,bool global,uint32_t,const uint32_t *,uint16_t,NrgObservation *);
bool nrg_string(const NrgInstance *,bool global,uint32_t,const uint32_t *,uint16_t,uint32_t,uint32_t,void *);
bool nrg_stats(const NrgInstance *, NrgStats *);
#endif
#endif
