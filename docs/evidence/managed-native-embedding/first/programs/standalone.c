#ifndef NANOISA_BINARY64_PARSE_H
#define NANOISA_BINARY64_PARSE_H
#include <stdint.h>
/* I parse closed C-locale binary64 values with fixed-capacity integer math.
 * My input is a valid byte view; NUL terminates its numeric prefix. */
typedef struct {
    uint32_t limb[128];
    unsigned used;
} NbpBig;
static inline void nbp_zero(NbpBig *a) { *a = (NbpBig){{0}, 0}; }
static inline void nbp_normalize(NbpBig *a) {
    while (a->used && !a->limb[a->used - 1])
        a->used--;
}
static inline int nbp_multiply(NbpBig *a, unsigned base, unsigned add) {
    uint64_t carry = add;
    for (unsigned i = 0; i < a->used; i++) {
        uint64_t value = (uint64_t)a->limb[i] * base + carry;
        a->limb[i] = (uint32_t)value;
        carry = value >> 32;
    }
    if (carry) {
        if (a->used == 128)
            return 0;
        a->limb[a->used++] = (uint32_t)carry;
    }
    return 1;
}
static inline unsigned nbp_bits(const NbpBig *a) {
    if (!a->used)
        return 0;
    uint32_t top = a->limb[a->used - 1];
    unsigned n = 0;
    while (top) {
        top >>= 1;
        n++;
    }
    return (a->used - 1) * 32 + n;
}
static inline int nbp_shift(NbpBig *a, unsigned shift) {
    if (!a->used || !shift)
        return 1;
    unsigned words = shift / 32, bits = shift % 32;
    unsigned carry = bits && (a->limb[a->used - 1] >> (32 - bits));
    if (words >= 128 || a->used + words + carry > 128)
        return 0;
    NbpBig result;
    nbp_zero(&result);
    result.used = a->used + words + carry;
    for (unsigned i = 0; i < a->used; i++) {
        uint64_t value = (uint64_t)a->limb[i] << bits;
        result.limb[i + words] |= (uint32_t)value;
        if (value >> 32)
            result.limb[i + words + 1] |= (uint32_t)(value >> 32);
    }
    *a = result;
    return 1;
}
static inline int nbp_compare(const NbpBig *a, const NbpBig *b) {
    if (a->used != b->used)
        return a->used > b->used ? 1 : -1;
    for (unsigned i = a->used; i > 0; i--)
        if (a->limb[i - 1] != b->limb[i - 1])
            return a->limb[i - 1] > b->limb[i - 1] ? 1 : -1;
    return 0;
}
static inline void nbp_subtract(NbpBig *a, const NbpBig *b) {
    uint64_t borrow = 0;
    for (unsigned i = 0; i < a->used; i++) {
        uint64_t sub = (i < b->used ? b->limb[i] : 0) + borrow, value = a->limb[i];
        a->limb[i] = (uint32_t)(value - sub);
        borrow = value < sub;
    }
    nbp_normalize(a);
}
static inline int nbp_rational(NbpBig numerator, NbpBig denominator, int sticky, uint64_t *out) {
    int exponent = (int)nbp_bits(&numerator) - (int)nbp_bits(&denominator);
    NbpBig probe = exponent < 0 ? numerator : denominator;
    if (!nbp_shift(&probe, (unsigned)(exponent < 0 ? -exponent : exponent)))
        return 0;
    if ((exponent < 0 ? nbp_compare(&probe, &denominator) : nbp_compare(&numerator, &probe)) < 0)
        exponent--;
    if (exponent > 1023) {
        *out = UINT64_C(0x7ff0000000000000);
        return 1;
    }
    int normal = exponent >= -1022, shift = normal ? 52 - exponent : 1074;
    if (!nbp_shift(shift >= 0 ? &numerator : &denominator, (unsigned)(shift >= 0 ? shift : -shift)))
        return 0;
    int bit = (int)nbp_bits(&numerator) - (int)nbp_bits(&denominator);
    if (bit >= 64)
        return 0;
    uint64_t quotient = 0;
    for (; bit >= 0; bit--) {
        probe = denominator;
        if (!nbp_shift(&probe, (unsigned)bit))
            return 0;
        if (nbp_compare(&numerator, &probe) >= 0) {
            nbp_subtract(&numerator, &probe);
            quotient |= UINT64_C(1) << bit;
        }
    }
    if (!nbp_shift(&numerator, 1))
        return 0;
    int half = nbp_compare(&numerator, &denominator);
    if (half > 0 || (half == 0 && (sticky || (quotient & 1))))
        quotient++;
    if (normal) {
        if (quotient == (UINT64_C(1) << 53)) {
            quotient >>= 1;
            exponent++;
        }
        if (exponent > 1023) {
            *out = UINT64_C(0x7ff0000000000000);
            return 1;
        }
        if (quotient < (UINT64_C(1) << 52) || quotient >= (UINT64_C(1) << 53))
            return 0;
        *out = ((uint64_t)(exponent + 1023) << 52) | (quotient - (UINT64_C(1) << 52));
    } else {
        if (quotient > (UINT64_C(1) << 52))
            return 0;
        *out = quotient;
    }
    return 1;
}
static inline unsigned char nbp_lower(unsigned char c) {
    return c >= 'A' && c <= 'Z' ? (unsigned char)(c + 32) : c;
}
static inline int nbp_digit(unsigned char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    c = nbp_lower(c);
    return c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
}
static inline int nbp_word(const unsigned char *text, uint32_t length, uint32_t start,
                           const char *word, unsigned count) {
    if (start > length || length - start < count)
        return 0;
    for (unsigned i = 0; i < count; i++)
        if (nbp_lower(text[start + i]) != (unsigned char)word[i])
            return 0;
    return 1;
}
static inline uint64_t nbp_nan_payload(const unsigned char *text, uint32_t length, uint32_t start,
                                       uint32_t *consumed) {
    if (start >= length || text[start] != '(')
        return 0;
    uint32_t end = start + 1;
    while (end < length) {
        unsigned char c = nbp_lower(text[end]);
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'))
            break;
        end++;
    }
    if (end == length || text[end] != ')')
        return 0;
    *consumed = end + 1;
    uint32_t i = start + 1;
    unsigned base = 10;
    if (end - i >= 2 && text[i] == '0' && nbp_lower(text[i + 1]) == 'x') {
        base = 16;
        i += 2;
    } else if (end - i > 1 && text[i] == '0')
        base = 8;
    if (i == end)
        return 0;
    uint64_t value = 0;
    for (; i < end; i++) {
        int digit = nbp_digit(text[i]);
        if (digit < 0 || (unsigned)digit >= base)
            return 0;
        if (value > (UINT64_MAX - (unsigned)digit) / base)
            value = UINT64_MAX;
        else
            value = value * base + (unsigned)digit;
    }
    return value;
}
static inline int nbp_parse_impl(const unsigned char *text, uint32_t length, uint64_t *out,
                                 uint32_t *consumed) {
    if (!out || (!text && length))
        return 0;
    *consumed = 0;
    uint32_t i = 0;
    while (i < length && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r' ||
                          text[i] == '\v' || text[i] == '\f'))
        i++;
    uint64_t sign = 0;
    if (i < length && (text[i] == '-' || text[i] == '+')) {
        if (text[i] == '-')
            sign = UINT64_C(1) << 63;
        i++;
    }
    if (nbp_word(text, length, i, "inf", 3)) {
        *consumed = i + (nbp_word(text, length, i, "infinity", 8) ? 8 : 3);
        *out = sign | UINT64_C(0x7ff0000000000000);
        return 1;
    }
    if (nbp_word(text, length, i, "nan", 3)) {
        *consumed = i + 3;
        *out = sign | UINT64_C(0x7ff8000000000000) |
               (nbp_nan_payload(text, length, i + 3, consumed) & UINT64_C(0xfffffffffffff));
        return 1;
    }
    unsigned radix = 10, cap = 800;
    if (length - i >= 3 && text[i] == '0' && nbp_lower(text[i + 1]) == 'x' &&
        (nbp_digit(text[i + 2]) >= 0 ||
         (length - i >= 4 && text[i + 2] == '.' && nbp_digit(text[i + 3]) >= 0))) {
        radix = 16;
        cap = 32;
        i += 2;
    }
    NbpBig numerator;
    nbp_zero(&numerator);
    unsigned kept = 0;
    int point = 0, saw_digit = 0, started = 0, sticky = 0;
    int64_t fractional = 0, omitted = 0;
    while (i < length) {
        int digit = nbp_digit(text[i]);
        if (digit >= 0 && (unsigned)digit < radix) {
            saw_digit = 1;
            if (point)
                fractional++;
            if (digit)
                started = 1;
            if (started) {
                if (kept < cap) {
                    if (!nbp_multiply(&numerator, radix, (unsigned)digit))
                        return 0;
                    kept++;
                } else {
                    omitted++;
                    sticky |= digit != 0;
                }
            }
            i++;
            continue;
        }
        if (text[i] == '.' && !point) {
            point = 1;
            i++;
            continue;
        }
        break;
    }
    if (!saw_digit) {
        *out = 0;
        return 1;
    }
    int64_t exponent = 0;
    if (i < length && nbp_lower(text[i]) == (radix == 16 ? 'p' : 'e')) {
        uint32_t next = i + 1;
        int minus = 0;
        if (next < length && (text[next] == '+' || text[next] == '-')) {
            minus = text[next] == '-';
            next++;
        }
        if (next < length && text[next] >= '0' && text[next] <= '9') {
            const int64_t cap_exponent = INT64_C(1) << 40;
            for (; next < length && text[next] >= '0' && text[next] <= '9'; next++) {
                if (exponent < cap_exponent) {
                    exponent = exponent * 10 + (text[next] - '0');
                    if (exponent > cap_exponent)
                        exponent = cap_exponent;
                }
            }
            i = next;
            if (minus)
                exponent = -exponent;
        }
    }
    *consumed = i;
    if (!started) {
        *out = sign;
        return 1;
    }
    int64_t scale = exponent + (omitted - fractional) * (radix == 16 ? 4 : 1);
    int64_t order = (radix == 16 ? (int64_t)nbp_bits(&numerator) - 1 : (int64_t)kept - 1) + scale;
    if (order > (radix == 16 ? 1023 : 308)) {
        *out = sign | UINT64_C(0x7ff0000000000000);
        return 1;
    }
    if (order < (radix == 16 ? -1075 : -324)) {
        *out = sign;
        return 1;
    }
    NbpBig denominator;
    nbp_zero(&denominator);
    denominator.used = 1;
    denominator.limb[0] = 1;
    NbpBig *scaled = scale < 0 ? &denominator : &numerator;
    unsigned power = (unsigned)(scale < 0 ? -scale : scale);
    if (radix == 16) {
        if (!nbp_shift(scaled, power))
            return 0;
    } else
        for (unsigned step = 0; step < power; step++)
            if (!nbp_multiply(scaled, 10, 0))
                return 0;
    uint64_t result;
    if (!nbp_rational(numerator, denominator, sticky, &result))
        return 0;
    *out = sign | result;
    return 1;
}
/* I publish both outputs only after checked parsing succeeds. An input with
 * no numeric prefix has endpoint zero; optional endpoints preserve values. */
static inline int nbp_parse_end(const unsigned char *text, uint32_t length, uint64_t *out,
                                uint32_t *consumed) {
    if (!out || (!text && length))
        return 0;
    uint64_t bits;
    uint32_t end;
    if (!nbp_parse_impl(text, length, &bits, &end))
        return 0;
    *out = bits;
    if (consumed)
        *consumed = end;
    return 1;
}
static inline int nbp_parse(const unsigned char *text, uint32_t length, uint64_t *out) {
    return nbp_parse_end(text, length, out, 0);
}
#endif
/* I canonicalize only scalar binary arithmetic results, never transported bits. */
#ifndef NANOLANG_BINARY64_ARITHMETIC_H
#define NANOLANG_BINARY64_ARITHMETIC_H
#include <float.h>
#include <stdint.h>
#include <string.h>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "I require ordinary IEEE arithmetic without fast-math."
#endif
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || DBL_MIN_EXP != -1021
#error "I require binary64 double arithmetic."
#endif
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "I require operations evaluated in their binary64 type."
#endif
/* I retain a compile-time storage check in both C99 and C11 output. */
typedef char nano_rt_binary64_storage_guard[
    sizeof(double) == 8 && sizeof(uint64_t) == 8 ? 1 : -1];

/* I inspect a rounded result with integer operations, not another FP operation. */
static inline double nano_rt_f64_arithmetic_result(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) &&
        (bits & UINT64_C(0x000fffffffffffff)) != 0) {
        bits = UINT64_C(0x7ff8000000000000);
        memcpy(&value, &bits, sizeof(value));
    }
    return value;
}

/* Each volatile store/load is a binary64 rounding and noncontraction boundary. */
static inline double nano_rt_f64_add(double a, double b) {
    volatile double rounded = a + b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_sub(double a, double b) {
    volatile double rounded = a - b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_mul(double a, double b) {
    volatile double rounded = a * b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_div(double a, double b) {
    uint64_t divisor;
    memcpy(&divisor, &b, sizeof(divisor));
    /* Either signed zero takes precedence even over a signaling NaN numerator. */
    if ((divisor & UINT64_C(0x7fffffffffffffff)) == 0) return 0.0;
    volatile double rounded = a / b;
    return nano_rt_f64_arithmetic_result(rounded);
}
#endif
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
#define NMS_RECORD_TAG 8
#define NMS_DYNAMIC (UINT64_C(1) << 63)
typedef struct { const unsigned char *data; uint32_t length; } NmsView;
typedef enum { NMS_SLOT_FREE = 0, NMS_SLOT_STRING = 1, NMS_SLOT_STRING_ARRAY = 2, NMS_SLOT_BOXED_ARRAY = 3, NMS_SLOT_BOXED_LEAF_ARRAY = NMS_SLOT_BOXED_ARRAY, NMS_SLOT_PACKED_SCALAR_ARRAY = 4, NMS_SLOT_RECORD = 5 } NmsSlotKind;
typedef struct {
    unsigned char *data;
    uint64_t references;
    uint32_t length, next_free, capacity, kind, element_tag, vm_array_policy, record_ordinal;
} NmsSlot;
typedef struct { uint32_t global_layout_index, field_count; } NmsRecordDescriptor;
typedef struct {
    const NmsRecordDescriptor *record_descriptors; /* Immutable through disposal. */
    uint32_t record_count;
    unsigned records_bound;
    const NmsView *literals; /* Borrowed immutable storage, alive until disposal. */
    NmsSlot *slots;
    void *collection_workspace;
    uint32_t collection_capacity;
    unsigned collection_prepared;
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
/* My boxed-value API preserves scalar bits and owns string/array/record handles.
 * Other heap/callable tags remain outside this private graph foundation. */
typedef struct { uint64_t payload; uint32_t tag; } NmsValue;
NmsStatus nms_value_retain(NmsRuntime *, NmsValue);
NmsStatus nms_value_release(NmsRuntime *, NmsValue);
/* Private description only: binding supplies no ordinary/resource authority.
 * I bind once before dynamic allocations. The descriptor table stays alive and
 * immutable through disposal; construction borrows ordered values. Failure
 * leaves output and owners unchanged. GET retains; SET borrows both inputs. */
NmsStatus nms_bind_records(NmsRuntime *, const NmsRecordDescriptor *, uint32_t);
NmsStatus nms_record_create(NmsRuntime *, uint32_t, const NmsValue *, uint32_t, NmsHandle *);
NmsStatus nms_record_identity(const NmsRuntime *, NmsHandle, uint32_t *, uint32_t *);
NmsStatus nms_record_get(NmsRuntime *, NmsHandle, uint64_t, NmsValue *);
NmsStatus nms_record_set(NmsRuntime *, NmsHandle, uint64_t, NmsValue);

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
/* I reclaim unreachable array graphs explicitly; failure leaves owners intact.
 * This synchronous private operation grants no emitted collection safe point. */
NmsStatus nms_collect(NmsRuntime *);
/* I reserve reusable workspace transactionally; prepared collection allocates
 * nothing. These private operations do not admit generated graph execution. */
NmsStatus nms_prepare_collection(NmsRuntime *);
NmsStatus nms_collect_prepared(NmsRuntime *);
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
#ifndef NANOISA_BINARY64_PARSE_H
#define NANOISA_BINARY64_PARSE_H
#include <stdint.h>
/* I parse closed C-locale binary64 values with fixed-capacity integer math.
 * My input is a valid byte view; NUL terminates its numeric prefix. */
typedef struct {
    uint32_t limb[128];
    unsigned used;
} NbpBig;
static inline void nbp_zero(NbpBig *a) { *a = (NbpBig){{0}, 0}; }
static inline void nbp_normalize(NbpBig *a) {
    while (a->used && !a->limb[a->used - 1])
        a->used--;
}
static inline int nbp_multiply(NbpBig *a, unsigned base, unsigned add) {
    uint64_t carry = add;
    for (unsigned i = 0; i < a->used; i++) {
        uint64_t value = (uint64_t)a->limb[i] * base + carry;
        a->limb[i] = (uint32_t)value;
        carry = value >> 32;
    }
    if (carry) {
        if (a->used == 128)
            return 0;
        a->limb[a->used++] = (uint32_t)carry;
    }
    return 1;
}
static inline unsigned nbp_bits(const NbpBig *a) {
    if (!a->used)
        return 0;
    uint32_t top = a->limb[a->used - 1];
    unsigned n = 0;
    while (top) {
        top >>= 1;
        n++;
    }
    return (a->used - 1) * 32 + n;
}
static inline int nbp_shift(NbpBig *a, unsigned shift) {
    if (!a->used || !shift)
        return 1;
    unsigned words = shift / 32, bits = shift % 32;
    unsigned carry = bits && (a->limb[a->used - 1] >> (32 - bits));
    if (words >= 128 || a->used + words + carry > 128)
        return 0;
    NbpBig result;
    nbp_zero(&result);
    result.used = a->used + words + carry;
    for (unsigned i = 0; i < a->used; i++) {
        uint64_t value = (uint64_t)a->limb[i] << bits;
        result.limb[i + words] |= (uint32_t)value;
        if (value >> 32)
            result.limb[i + words + 1] |= (uint32_t)(value >> 32);
    }
    *a = result;
    return 1;
}
static inline int nbp_compare(const NbpBig *a, const NbpBig *b) {
    if (a->used != b->used)
        return a->used > b->used ? 1 : -1;
    for (unsigned i = a->used; i > 0; i--)
        if (a->limb[i - 1] != b->limb[i - 1])
            return a->limb[i - 1] > b->limb[i - 1] ? 1 : -1;
    return 0;
}
static inline void nbp_subtract(NbpBig *a, const NbpBig *b) {
    uint64_t borrow = 0;
    for (unsigned i = 0; i < a->used; i++) {
        uint64_t sub = (i < b->used ? b->limb[i] : 0) + borrow, value = a->limb[i];
        a->limb[i] = (uint32_t)(value - sub);
        borrow = value < sub;
    }
    nbp_normalize(a);
}
static inline int nbp_rational(NbpBig numerator, NbpBig denominator, int sticky, uint64_t *out) {
    int exponent = (int)nbp_bits(&numerator) - (int)nbp_bits(&denominator);
    NbpBig probe = exponent < 0 ? numerator : denominator;
    if (!nbp_shift(&probe, (unsigned)(exponent < 0 ? -exponent : exponent)))
        return 0;
    if ((exponent < 0 ? nbp_compare(&probe, &denominator) : nbp_compare(&numerator, &probe)) < 0)
        exponent--;
    if (exponent > 1023) {
        *out = UINT64_C(0x7ff0000000000000);
        return 1;
    }
    int normal = exponent >= -1022, shift = normal ? 52 - exponent : 1074;
    if (!nbp_shift(shift >= 0 ? &numerator : &denominator, (unsigned)(shift >= 0 ? shift : -shift)))
        return 0;
    int bit = (int)nbp_bits(&numerator) - (int)nbp_bits(&denominator);
    if (bit >= 64)
        return 0;
    uint64_t quotient = 0;
    for (; bit >= 0; bit--) {
        probe = denominator;
        if (!nbp_shift(&probe, (unsigned)bit))
            return 0;
        if (nbp_compare(&numerator, &probe) >= 0) {
            nbp_subtract(&numerator, &probe);
            quotient |= UINT64_C(1) << bit;
        }
    }
    if (!nbp_shift(&numerator, 1))
        return 0;
    int half = nbp_compare(&numerator, &denominator);
    if (half > 0 || (half == 0 && (sticky || (quotient & 1))))
        quotient++;
    if (normal) {
        if (quotient == (UINT64_C(1) << 53)) {
            quotient >>= 1;
            exponent++;
        }
        if (exponent > 1023) {
            *out = UINT64_C(0x7ff0000000000000);
            return 1;
        }
        if (quotient < (UINT64_C(1) << 52) || quotient >= (UINT64_C(1) << 53))
            return 0;
        *out = ((uint64_t)(exponent + 1023) << 52) | (quotient - (UINT64_C(1) << 52));
    } else {
        if (quotient > (UINT64_C(1) << 52))
            return 0;
        *out = quotient;
    }
    return 1;
}
static inline unsigned char nbp_lower(unsigned char c) {
    return c >= 'A' && c <= 'Z' ? (unsigned char)(c + 32) : c;
}
static inline int nbp_digit(unsigned char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    c = nbp_lower(c);
    return c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
}
static inline int nbp_word(const unsigned char *text, uint32_t length, uint32_t start,
                           const char *word, unsigned count) {
    if (start > length || length - start < count)
        return 0;
    for (unsigned i = 0; i < count; i++)
        if (nbp_lower(text[start + i]) != (unsigned char)word[i])
            return 0;
    return 1;
}
static inline uint64_t nbp_nan_payload(const unsigned char *text, uint32_t length, uint32_t start,
                                       uint32_t *consumed) {
    if (start >= length || text[start] != '(')
        return 0;
    uint32_t end = start + 1;
    while (end < length) {
        unsigned char c = nbp_lower(text[end]);
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'))
            break;
        end++;
    }
    if (end == length || text[end] != ')')
        return 0;
    *consumed = end + 1;
    uint32_t i = start + 1;
    unsigned base = 10;
    if (end - i >= 2 && text[i] == '0' && nbp_lower(text[i + 1]) == 'x') {
        base = 16;
        i += 2;
    } else if (end - i > 1 && text[i] == '0')
        base = 8;
    if (i == end)
        return 0;
    uint64_t value = 0;
    for (; i < end; i++) {
        int digit = nbp_digit(text[i]);
        if (digit < 0 || (unsigned)digit >= base)
            return 0;
        if (value > (UINT64_MAX - (unsigned)digit) / base)
            value = UINT64_MAX;
        else
            value = value * base + (unsigned)digit;
    }
    return value;
}
static inline int nbp_parse_impl(const unsigned char *text, uint32_t length, uint64_t *out,
                                 uint32_t *consumed) {
    if (!out || (!text && length))
        return 0;
    *consumed = 0;
    uint32_t i = 0;
    while (i < length && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r' ||
                          text[i] == '\v' || text[i] == '\f'))
        i++;
    uint64_t sign = 0;
    if (i < length && (text[i] == '-' || text[i] == '+')) {
        if (text[i] == '-')
            sign = UINT64_C(1) << 63;
        i++;
    }
    if (nbp_word(text, length, i, "inf", 3)) {
        *consumed = i + (nbp_word(text, length, i, "infinity", 8) ? 8 : 3);
        *out = sign | UINT64_C(0x7ff0000000000000);
        return 1;
    }
    if (nbp_word(text, length, i, "nan", 3)) {
        *consumed = i + 3;
        *out = sign | UINT64_C(0x7ff8000000000000) |
               (nbp_nan_payload(text, length, i + 3, consumed) & UINT64_C(0xfffffffffffff));
        return 1;
    }
    unsigned radix = 10, cap = 800;
    if (length - i >= 3 && text[i] == '0' && nbp_lower(text[i + 1]) == 'x' &&
        (nbp_digit(text[i + 2]) >= 0 ||
         (length - i >= 4 && text[i + 2] == '.' && nbp_digit(text[i + 3]) >= 0))) {
        radix = 16;
        cap = 32;
        i += 2;
    }
    NbpBig numerator;
    nbp_zero(&numerator);
    unsigned kept = 0;
    int point = 0, saw_digit = 0, started = 0, sticky = 0;
    int64_t fractional = 0, omitted = 0;
    while (i < length) {
        int digit = nbp_digit(text[i]);
        if (digit >= 0 && (unsigned)digit < radix) {
            saw_digit = 1;
            if (point)
                fractional++;
            if (digit)
                started = 1;
            if (started) {
                if (kept < cap) {
                    if (!nbp_multiply(&numerator, radix, (unsigned)digit))
                        return 0;
                    kept++;
                } else {
                    omitted++;
                    sticky |= digit != 0;
                }
            }
            i++;
            continue;
        }
        if (text[i] == '.' && !point) {
            point = 1;
            i++;
            continue;
        }
        break;
    }
    if (!saw_digit) {
        *out = 0;
        return 1;
    }
    int64_t exponent = 0;
    if (i < length && nbp_lower(text[i]) == (radix == 16 ? 'p' : 'e')) {
        uint32_t next = i + 1;
        int minus = 0;
        if (next < length && (text[next] == '+' || text[next] == '-')) {
            minus = text[next] == '-';
            next++;
        }
        if (next < length && text[next] >= '0' && text[next] <= '9') {
            const int64_t cap_exponent = INT64_C(1) << 40;
            for (; next < length && text[next] >= '0' && text[next] <= '9'; next++) {
                if (exponent < cap_exponent) {
                    exponent = exponent * 10 + (text[next] - '0');
                    if (exponent > cap_exponent)
                        exponent = cap_exponent;
                }
            }
            i = next;
            if (minus)
                exponent = -exponent;
        }
    }
    *consumed = i;
    if (!started) {
        *out = sign;
        return 1;
    }
    int64_t scale = exponent + (omitted - fractional) * (radix == 16 ? 4 : 1);
    int64_t order = (radix == 16 ? (int64_t)nbp_bits(&numerator) - 1 : (int64_t)kept - 1) + scale;
    if (order > (radix == 16 ? 1023 : 308)) {
        *out = sign | UINT64_C(0x7ff0000000000000);
        return 1;
    }
    if (order < (radix == 16 ? -1075 : -324)) {
        *out = sign;
        return 1;
    }
    NbpBig denominator;
    nbp_zero(&denominator);
    denominator.used = 1;
    denominator.limb[0] = 1;
    NbpBig *scaled = scale < 0 ? &denominator : &numerator;
    unsigned power = (unsigned)(scale < 0 ? -scale : scale);
    if (radix == 16) {
        if (!nbp_shift(scaled, power))
            return 0;
    } else
        for (unsigned step = 0; step < power; step++)
            if (!nbp_multiply(scaled, 10, 0))
                return 0;
    uint64_t result;
    if (!nbp_rational(numerator, denominator, sticky, &result))
        return 0;
    *out = sign | result;
    return 1;
}
/* I publish both outputs only after checked parsing succeeds. An input with
 * no numeric prefix has endpoint zero; optional endpoints preserve values. */
static inline int nbp_parse_end(const unsigned char *text, uint32_t length, uint64_t *out,
                                uint32_t *consumed) {
    if (!out || (!text && length))
        return 0;
    uint64_t bits;
    uint32_t end;
    if (!nbp_parse_impl(text, length, &bits, &end))
        return 0;
    *out = bits;
    if (consumed)
        *consumed = end;
    return 1;
}
static inline int nbp_parse(const unsigned char *text, uint32_t length, uint64_t *out) {
    return nbp_parse_end(text, length, out, 0);
}
#endif
/* I retain byte strings with context-local handles. My managed lowering
 * supplies frame/global ownership and cleanup around this allocator core. */
#include <limits.h>
#ifndef __wasm32__
#include <stdlib.h>
_Static_assert(sizeof(void *) == 8 && sizeof(size_t) == 8,
               "I require the declared native 64-bit allocator ABI");
#endif

#ifdef NMS_TESTING
static uint64_t live_allocations;
#endif
static void copy_bytes(unsigned char *to, const unsigned char *from, uint64_t n) {
    /* Volatile byte accesses keep the freestanding Wasm core independent of
     * compiler-created memcpy/memmove imports, including optimized builds. */
    volatile unsigned char *dst = to;
    const volatile unsigned char *src = from;
    for (uint64_t i = 0; i < n; i++) dst[i] = src[i];
}

#ifdef __wasm32__
/* Free-block headers live inside memory owned by this allocator. The sole
 * external boundary is wasm-ld's heap base, after data and reserved stack. */
extern unsigned char __heap_base;
typedef struct FreeBlock { uint64_t size; struct FreeBlock *next; } FreeBlock;
_Static_assert(sizeof(FreeBlock) <= 16, "I reserve a 16-byte block header");
static FreeBlock *free_blocks;
static unsigned pool_initialized;

static void pool_insert(FreeBlock *block) {
    FreeBlock *previous = NULL, *next = free_blocks;
    while (next && (uintptr_t)next < (uintptr_t)block) {
        previous = next; next = next->next;
    }
    block->next = next;
    if (next && (uint64_t)(uintptr_t)block + block->size == (uintptr_t)next) {
        block->size += next->size;
        block->next = next->next;
    }
    if (previous) {
        if ((uint64_t)(uintptr_t)previous + previous->size == (uintptr_t)block) {
            previous->size += block->size;
            previous->next = block->next;
        } else previous->next = block;
    } else free_blocks = block;
}
static void pool_init(void) {
    if (pool_initialized) return;
    pool_initialized = 1;
    uint64_t start = ((uint64_t)(uintptr_t)&__heap_base + 15) & ~UINT64_C(15);
    uint64_t end = (uint64_t)__builtin_wasm_memory_size(0) * 65536;
    if (start < end && end - start >= 32) {
        FreeBlock *block = (FreeBlock *)(uintptr_t)start;
        block->size = end - start;
        block->next = NULL;
        free_blocks = block;
    }
}
static void *backend_allocate(uint64_t bytes) {
    /* Widen before header/alignment arithmetic; no wrapped request reaches
     * the free list or memory.grow. All published block sizes are aligned. */
    if (bytes > UINT32_MAX - UINT64_C(31)) return NULL;
    uint64_t need = (bytes + 16 + 15) & ~UINT64_C(15);
    pool_init();
    for (unsigned attempt = 0; attempt < 2; attempt++) {
        FreeBlock **link = &free_blocks;
        for (FreeBlock *block = *link; block; link = &block->next, block = *link) {
            if (block->size < need) continue;
            uint64_t remaining = block->size - need;
            if (remaining >= 32) {
                FreeBlock *tail = (FreeBlock *)((unsigned char *)block + (size_t)need);
                tail->size = remaining; tail->next = block->next;
                *link = tail; block->size = need;
            } else *link = block->next;
            block->next = NULL;
            return (unsigned char *)block + 16;
        }
        if (attempt) break;
        uint64_t current = __builtin_wasm_memory_size(0);
        uint64_t missing = need;
        FreeBlock *tail = free_blocks;
        while (tail && tail->next) tail = tail->next;
        if (tail && (uint64_t)(uintptr_t)tail + tail->size == current * 65536)
            missing -= tail->size; /* No free block fitted, so this is positive. */
        uint64_t pages = (missing + 65535) / 65536;
        if (current > 65536 || pages > 65536 - current) return NULL;
        size_t old = __builtin_wasm_memory_grow(0, (size_t)pages);
        if (old == (size_t)-1) return NULL;
        FreeBlock *added = (FreeBlock *)(uintptr_t)((uint64_t)old * 65536);
        added->size = pages * 65536; added->next = NULL;
        pool_insert(added);
    }
    return NULL;
}
static void backend_free(void *memory) {
    if (memory) pool_insert((FreeBlock *)((unsigned char *)memory - 16));
}
#else
static void *backend_allocate(uint64_t bytes) {
    if (bytes > SIZE_MAX) return NULL;
    return malloc((size_t)bytes);
}
static void backend_free(void *memory) { free(memory); }
#endif

static void *allocate(NmsRuntime *runtime, uint64_t bytes) {
#ifdef NMS_TESTING
    if (!runtime->fail_after) return NULL;
    if (runtime->fail_after != UINT64_MAX) runtime->fail_after--;
#else
    (void)runtime;
#endif
    void *memory = backend_allocate(bytes);
#ifdef NMS_TESTING
    if (memory) live_allocations++;
#endif
    return memory;
}
static void deallocate(void *memory) {
    if (!memory) return;
#ifdef NMS_TESTING
    live_allocations--;
#endif
    backend_free(memory);
}
void nms_init(NmsRuntime *runtime, const NmsView *literals, uint32_t count) {
    runtime->record_descriptors = NULL;
    runtime->record_count = runtime->records_bound = 0;
    runtime->literals = literals;
    runtime->literal_count = count;
    runtime->slots = NULL;
    runtime->collection_workspace = NULL;
    runtime->collection_capacity = runtime->collection_prepared = 0;
    runtime->capacity = runtime->free_head = 0;
    runtime->live_bytes = runtime->live_objects = 0;
    runtime->active = runtime->disposed = 0;
#ifdef NMS_TESTING
    runtime->fail_after = UINT64_MAX;
#endif
}
static NmsStatus slot_for(const NmsRuntime *runtime, NmsHandle handle, uint32_t *index) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    uint64_t raw = handle & ~NMS_DYNAMIC;
    if (!(handle & NMS_DYNAMIC) || !raw || raw > runtime->capacity ||
        !runtime->slots || !runtime->slots[raw].references) return NMS_STATE;
    *index = (uint32_t)raw;
    return NMS_OK;
}
NmsStatus nms_view(const NmsRuntime *runtime, NmsHandle handle, NmsView *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (handle & NMS_DYNAMIC) {
        uint32_t index;
        NmsStatus status = slot_for(runtime, handle, &index);
        if (status != NMS_OK) return status;
        if (runtime->slots[index].kind != NMS_SLOT_STRING) return NMS_TYPE;
        out->data = runtime->slots[index].data;
        out->length = runtime->slots[index].length;
    } else {
        if (!handle || handle > runtime->literal_count || !runtime->literals)
            return NMS_STATE;
        const NmsView *literal = &runtime->literals[handle - 1];
        if (!literal->data) return NMS_STATE;
        out->data = literal->data; out->length = literal->length;
    }
    return NMS_OK;
}
/* uint64 trial counts, byte marks, then aligned uint32 queue. All arithmetic
 * is widened before checking the target size_t limit. */
static int collection_layout(uint32_t capacity, uint64_t *mark_offset,
                             uint64_t *queue_offset, uint64_t *bytes) {
    uint64_t count = (uint64_t)capacity + 1;
    uint64_t marks = count * sizeof(uint64_t);
    uint64_t queue = (marks + count + 3) & ~UINT64_C(3);
    uint64_t total = queue + count * sizeof(uint32_t);
    if (total > SIZE_MAX) return 0;
    *mark_offset = marks; *queue_offset = queue; *bytes = total;
    return 1;
}
static void *collection_allocate(NmsRuntime *runtime, uint32_t capacity) {
    uint64_t marks, queue, bytes;
    if (!collection_layout(capacity, &marks, &queue, &bytes)) return NULL;
    return allocate(runtime, bytes);
}
NmsStatus nms_prepare_collection(NmsRuntime *runtime) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (runtime->collection_prepared) return NMS_OK;
    void *workspace = collection_allocate(runtime, runtime->capacity);
    if (!workspace) return NMS_MEMORY;
    runtime->collection_workspace = workspace;
    runtime->collection_capacity = runtime->capacity;
    runtime->collection_prepared = 1;
    return NMS_OK;
}
/* Publication borrows prepared storage until success; I never publish a
 * partial table or consume that storage on allocation failure. */
static NmsStatus publish_slot(NmsRuntime *runtime, unsigned char *bytes,
                              uint32_t length, uint32_t capacity, uint32_t kind,
                              uint64_t storage_bytes, NmsHandle *out) {
    if (storage_bytes > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    uint32_t new_capacity = runtime->capacity;
    if (!runtime->free_head) {
        if (runtime->capacity >= UINT32_MAX - 1) return NMS_MEMORY;
        new_capacity = runtime->capacity ?
            (runtime->capacity > (UINT32_MAX - 1) / 2 ? UINT32_MAX - 1 : runtime->capacity * 2) : 8;
        if ((uint64_t)new_capacity + 1 > SIZE_MAX / sizeof(NmsSlot)) return NMS_MEMORY;
    }
    NmsSlot *slots = runtime->slots;
    if (new_capacity != runtime->capacity) {
        slots = allocate(runtime, ((uint64_t)new_capacity + 1) * sizeof(NmsSlot));
        if (!slots) return NMS_MEMORY;
        void *workspace = NULL;
        if (runtime->collection_prepared) {
            workspace = collection_allocate(runtime, new_capacity);
            if (!workspace) { deallocate(slots); return NMS_MEMORY; }
        }
        for (uint64_t i = 0; i <= new_capacity; i++) {
            if (runtime->slots && i <= runtime->capacity) {
                slots[i].data = runtime->slots[i].data;
                slots[i].references = runtime->slots[i].references;
                slots[i].length = runtime->slots[i].length;
                slots[i].next_free = runtime->slots[i].next_free;
                slots[i].element_tag = runtime->slots[i].element_tag;
                slots[i].vm_array_policy = runtime->slots[i].vm_array_policy;
                slots[i].record_ordinal = runtime->slots[i].record_ordinal;
                slots[i].kind = runtime->slots[i].kind;
                slots[i].capacity = runtime->slots[i].capacity;
            } else {
                slots[i].data = NULL; slots[i].references = 0; slots[i].length = 0;
                slots[i].kind = NMS_SLOT_FREE; slots[i].capacity = 0; slots[i].element_tag = 0; slots[i].vm_array_policy = 0; slots[i].record_ordinal = 0;
                slots[i].next_free = i < new_capacity ? (uint32_t)i + 1 : 0;
            }
        }
        NmsSlot *old = runtime->slots;
        if (runtime->collection_prepared) {
            void *old_workspace = runtime->collection_workspace;
            runtime->collection_workspace = workspace;
            runtime->collection_capacity = new_capacity;
            deallocate(old_workspace);
        }
        runtime->slots = slots;
        runtime->free_head = runtime->capacity + 1;
        runtime->capacity = new_capacity;
        deallocate(old);
    }
    uint32_t index = runtime->free_head;
    runtime->free_head = slots[index].next_free;
    slots[index].data = bytes; slots[index].length = (uint32_t)length;
    slots[index].element_tag = 0; slots[index].vm_array_policy = 0; slots[index].record_ordinal = 0;
    slots[index].kind = kind; slots[index].capacity = capacity;
    slots[index].references = 1; slots[index].next_free = 0;
    runtime->live_bytes += storage_bytes; runtime->live_objects++;
    *out = NMS_DYNAMIC | index;
    return NMS_OK;
}
static NmsStatus create_parts(NmsRuntime *runtime,
                              const unsigned char *data, uint64_t first_length,
                              const unsigned char *second, uint64_t second_length,
                              NmsHandle *out) {
    if (!runtime || !out || (!data && first_length) || (!second && second_length))
        return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (first_length > UINT32_MAX || second_length > UINT32_MAX) return NMS_MEMORY;
    uint64_t length = first_length + second_length;
    if (length > UINT32_MAX || length == SIZE_MAX ||
        length > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    unsigned char *bytes = allocate(runtime, length + 1);
    if (!bytes) return NMS_MEMORY;
    copy_bytes(bytes, data, first_length);
    copy_bytes(bytes + first_length, second, second_length);
    bytes[length] = 0;
    NmsStatus status = publish_slot(runtime, bytes, (uint32_t)length, 0,
                                   NMS_SLOT_STRING, length, out);
    if (status != NMS_OK) deallocate(bytes);
    return status;
}
static NmsStatus string_array_slot(const NmsRuntime *runtime, NmsHandle handle,
                                   uint32_t *index) {
    if (!(handle & NMS_DYNAMIC)) {
        NmsView view;
        NmsStatus status = nms_view(runtime, handle, &view);
        return status == NMS_OK ? NMS_TYPE : status;
    }
    NmsStatus status = slot_for(runtime, handle, index);
    if (status != NMS_OK) return status;
    return runtime->slots[*index].kind == NMS_SLOT_STRING_ARRAY ? NMS_OK : NMS_TYPE;
}
NmsStatus nms_string_array_create(NmsRuntime *runtime, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    return publish_slot(runtime, NULL, 0, 0, NMS_SLOT_STRING_ARRAY, 0, out);
}
NmsStatus nms_string_array_append(NmsRuntime *runtime, NmsHandle array, NmsHandle child) {
    uint32_t index;
    NmsStatus status = string_array_slot(runtime, array, &index);
    if (status != NMS_OK) return status;
    NmsView view;
    status = nms_view(runtime, child, &view);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[index];
    if (slot->length == UINT32_MAX) return NMS_MEMORY;
    uint32_t capacity = slot->capacity;
    NmsHandle *buffer = (NmsHandle *)slot->data;
    if (slot->length == capacity) {
        capacity = capacity ? (capacity > UINT32_MAX / 2 ? UINT32_MAX : capacity * 2) : 4;
        uint64_t bytes = (uint64_t)capacity * sizeof(NmsHandle);
        uint64_t added = (uint64_t)(capacity - slot->capacity) * sizeof(NmsHandle);
        if (bytes > SIZE_MAX || added > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
        buffer = allocate(runtime, bytes);
        if (!buffer) return NMS_MEMORY;
        copy_bytes((unsigned char *)buffer, slot->data, (uint64_t)slot->length * sizeof(NmsHandle));
    }
    status = nms_retain(runtime, child);
    if (status != NMS_OK) {
        if ((unsigned char *)buffer != slot->data) deallocate(buffer);
        return status;
    }
    if ((unsigned char *)buffer != slot->data) {
        runtime->live_bytes += (uint64_t)(capacity - slot->capacity) * sizeof(NmsHandle);
        deallocate(slot->data);
        slot->data = (unsigned char *)buffer;
        slot->capacity = capacity;
    }
    buffer[slot->length++] = child;
    return NMS_OK;
}
NmsStatus nms_string_array_get(NmsRuntime *runtime, NmsHandle array, uint64_t index,
                              NmsHandle *out) {
    if (!out) return NMS_STATE;
    uint32_t slot_index;
    NmsStatus status = string_array_slot(runtime, array, &slot_index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[slot_index];
    if (index >= slot->length) { *out = 0; return NMS_OK; }
    NmsHandle child = ((NmsHandle *)slot->data)[index];
    status = nms_retain(runtime, child);
    if (status == NMS_OK) *out = child;
    return status;
}
NmsStatus nms_string_array_length(const NmsRuntime *runtime, NmsHandle array, uint32_t *out) {
    if (!out) return NMS_STATE;
    uint32_t index;
    NmsStatus status = string_array_slot(runtime, array, &index);
    if (status == NMS_OK) *out = runtime->slots[index].length;
    return status;
}
NmsStatus nms_create(NmsRuntime *runtime, const unsigned char *data, uint64_t length,
                     NmsHandle *out) {
    return create_parts(runtime, data, length, NULL, 0, out);
}
_Static_assert(offsetof(NmsValue, payload) == 0 && offsetof(NmsValue, tag) == 8 &&
               sizeof(NmsValue) == 16, "I require the private boxed leaf-value ABI");
static NmsStatus value_array_slot(const NmsRuntime *, NmsHandle, uint32_t *);
static NmsStatus record_slot(const NmsRuntime *runtime, NmsHandle handle, uint32_t *index) {
    NmsStatus status = slot_for(runtime, handle, index);
    if (status != NMS_OK) return status;
    const NmsSlot *slot = &runtime->slots[*index];
    if (slot->kind != NMS_SLOT_RECORD) return NMS_TYPE;
    if (!runtime->records_bound || !runtime->record_descriptors ||
        slot->record_ordinal >= runtime->record_count) return NMS_STATE;
    uint32_t fields = runtime->record_descriptors[slot->record_ordinal].field_count;
    return slot->length == fields && slot->capacity == fields &&
        (!fields || slot->data) ? NMS_OK : NMS_STATE;
}
static int value_is_reference(NmsValue value) {
    return value.tag == 5 || value.tag == NMS_ARRAY_TAG || value.tag == NMS_RECORD_TAG;
}
static int slot_has_children(const NmsSlot *slot) {
    return slot->kind == NMS_SLOT_STRING_ARRAY || slot->kind == NMS_SLOT_BOXED_ARRAY ||
        slot->kind == NMS_SLOT_RECORD;
}

static NmsStatus value_valid(const NmsRuntime *runtime, NmsValue value) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (value.tag == 5) {
        NmsView view;
        return nms_view(runtime, value.payload, &view);
    }
    if (value.tag == NMS_ARRAY_TAG) {
        uint32_t index;
        return value_array_slot(runtime, value.payload, &index);
    }
    if (value.tag == NMS_RECORD_TAG) {
        uint32_t index;
        return record_slot(runtime, value.payload, &index);
    }
    return value.tag < 5 || value.tag == 9 ? NMS_OK : NMS_TYPE;
}
NmsStatus nms_value_retain(NmsRuntime *runtime, NmsValue value) {
    NmsStatus status = value_valid(runtime, value);
    if (status != NMS_OK) return status;
    return value_is_reference(value) ? nms_retain(runtime, value.payload) : NMS_OK;
}
NmsStatus nms_value_release(NmsRuntime *runtime, NmsValue value) {
    NmsStatus status = value_valid(runtime, value);
    if (status != NMS_OK) return status;
    return value_is_reference(value) ? nms_release(runtime, value.payload) : NMS_OK;
}
static NmsStatus value_array_slot(const NmsRuntime *runtime, NmsHandle array,
                                  uint32_t *index) {
    if (!(array & NMS_DYNAMIC)) {
        NmsView view;
        NmsStatus status = nms_view(runtime, array, &view);
        return status == NMS_OK ? NMS_TYPE : status;
    }
    NmsStatus status = slot_for(runtime, array, index);
    if (status != NMS_OK) return status;
    uint32_t kind = runtime->slots[*index].kind;
    return kind == NMS_SLOT_STRING_ARRAY || kind == NMS_SLOT_BOXED_ARRAY || kind == NMS_SLOT_PACKED_SCALAR_ARRAY ? NMS_OK : NMS_TYPE;
}
static uint32_t packed_width(uint32_t tag) {
    return tag == 1 || tag == 3 ? 8 : tag == 2 || tag == 4 ? 1 : 0;
}
/* I distinguish VM logical growth from older private empty-buffer factories. */
static NmsStatus array_next_capacity(uint32_t old, uint32_t width, uint32_t vm_policy,
                                     uint32_t *out) {
    uint64_t next = old ? (uint64_t)old * 2 : 4;
    if (vm_policy) {
        if (!width || next <= old || next > UINT32_MAX / width) return NMS_MEMORY;
    } else if (next > UINT32_MAX) next = UINT32_MAX;
    *out = (uint32_t)next;
    return NMS_OK;
}
static NmsStatus vm_array_create_capacity(NmsRuntime *runtime, uint32_t tag,
                                           uint32_t capacity, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!(tag <= 5 || tag == NMS_ARRAY_TAG || tag == 9)) return NMS_TYPE;
    uint32_t width = packed_width(tag);
    uint32_t kind = width ? NMS_SLOT_PACKED_SCALAR_ARRAY : NMS_SLOT_BOXED_ARRAY;
    if (!width) width = sizeof(NmsValue);
    if (capacity < 8) capacity = 8;
    uint64_t bytes = (uint64_t)capacity * width;
    if (bytes > SIZE_MAX || bytes > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    unsigned char *buffer = allocate(runtime, bytes);
    if (!buffer) return NMS_MEMORY;
    NmsHandle result = 0;
    NmsStatus status = publish_slot(runtime, buffer, 0, capacity, kind, bytes, &result);
    if (status != NMS_OK) { deallocate(buffer); return status; }
    NmsSlot *slot = &runtime->slots[(uint32_t)result];
    slot->element_tag = tag; slot->vm_array_policy = 1;
    *out = result;
    return NMS_OK;
}
NmsStatus nms_vm_array_create(NmsRuntime *runtime, uint32_t tag, NmsHandle *out) {
    return vm_array_create_capacity(runtime, tag, 8, out);
}
static NmsValue packed_value(const NmsSlot *slot, uint32_t index) {
    uint32_t width = packed_width(slot->element_tag);
    uint64_t offset = (uint64_t)index * width;
    NmsValue value = {0, slot->element_tag};
    for (uint32_t i = 0; i < width; i++) value.payload |= (uint64_t)slot->data[offset + i] << (8 * i);
    return value;
}
static void packed_publish(NmsSlot *slot, uint32_t index, uint64_t bits) {
    uint32_t width = packed_width(slot->element_tag);
    uint64_t offset = (uint64_t)index * width;
    for (uint32_t i = 0; i < width; i++) slot->data[offset + i] = (unsigned char)(bits >> (8 * i));
}
static NmsStatus packed_prepare(uint32_t tag, NmsValue value, uint64_t *bits) {
    if ((value.tag == 2 && value.payload > 255) || (value.tag == 4 && value.payload > 1)) return NMS_TYPE;
    if (tag == value.tag && packed_width(tag)) { *bits = value.payload; return NMS_OK; }
    if (tag == 1 && value.tag == 2) { *bits = value.payload; return NMS_OK; }
    if (tag == 2 && value.tag == 1) { *bits = value.payload & 255; return NMS_OK; }
    if (tag == 3 && value.tag == 1) {
        /* I avoid an implementation-defined unsigned-to-signed conversion. */
        int64_t integer = value.payload <= INT64_MAX ? (int64_t)value.payload :
                          -1 - (int64_t)(UINT64_MAX - value.payload);
        double converted = (double)integer;
        _Static_assert(sizeof(double) == sizeof(uint64_t), "I require binary64 scalar storage");
        copy_bytes((unsigned char *)bits, (const unsigned char *)&converted, sizeof converted);
        return NMS_OK;
    }
    return NMS_TYPE;
}
NmsStatus nms_packed_array_create(NmsRuntime *runtime, uint32_t tag, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!packed_width(tag)) return NMS_TYPE;
    NmsHandle result = 0;
    NmsStatus status = publish_slot(runtime, NULL, 0, 0, NMS_SLOT_PACKED_SCALAR_ARRAY, 0, &result);
    if (status != NMS_OK) return status;
    runtime->slots[(uint32_t)result].element_tag = tag;
    *out = result;
    return NMS_OK;
}
static NmsStatus packed_write(NmsRuntime *runtime, NmsSlot *slot, uint64_t index,
                               NmsValue value, int append) {
    uint64_t bits = 0;
    NmsStatus status = packed_prepare(slot->element_tag, value, &bits);
    if (status != NMS_OK) return status;
    if (!append && index >= slot->length) return NMS_STATE;
    if (append && slot->length == UINT32_MAX) return NMS_MEMORY;
    uint32_t width = packed_width(slot->element_tag);
    if (!width) return NMS_STATE;
    uint32_t target = append ? slot->length : (uint32_t)index;
    if (append && slot->length == slot->capacity) {
        uint32_t capacity = 0;
        status = array_next_capacity(slot->capacity, width, slot->vm_array_policy, &capacity);
        if (status != NMS_OK) return status;
        uint64_t bytes = (uint64_t)capacity * width;
        uint64_t delta = (uint64_t)(capacity - slot->capacity) * width;
        if (bytes > SIZE_MAX || delta > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
        unsigned char *buffer = allocate(runtime, bytes);
        if (!buffer) return NMS_MEMORY;
        copy_bytes(buffer, slot->data, (uint64_t)slot->length * width);
        unsigned char *old = slot->data;
        slot->data = buffer; slot->capacity = capacity;
        runtime->live_bytes += delta;
        deallocate(old);
    }
    packed_publish(slot, target, bits);
    if (append) slot->length++;
    return NMS_OK;
}
static NmsValue slot_value(const NmsSlot *slot, uint32_t index) {
    if (slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY) return packed_value(slot, index);
    if (slot->kind == NMS_SLOT_STRING_ARRAY) {
        NmsValue value = {((const NmsHandle *)slot->data)[index], 5};
        return value;
    }
    return ((const NmsValue *)slot->data)[index];
}
static uint64_t slot_storage_bytes(const NmsSlot *slot) {
    return (uint64_t)slot->capacity *
        (slot->kind == NMS_SLOT_STRING_ARRAY ? sizeof(NmsHandle) :
         slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY ? packed_width(slot->element_tag) : sizeof(NmsValue));
}
NmsStatus nms_bind_records(NmsRuntime *runtime, const NmsRecordDescriptor *descriptors,
                            uint32_t count) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (runtime->active) return NMS_BUSY;
    if (runtime->records_bound || runtime->capacity || runtime->live_objects ||
        runtime->live_bytes || (count && !descriptors)) return NMS_STATE;
    if (count > 256) return NMS_TYPE;
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (descriptors[i].global_layout_index >= 256 ||
            (i && descriptors[i].global_layout_index <= descriptors[i-1].global_layout_index) ||
            descriptors[i].field_count > UINT16_MAX ||
            descriptors[i].field_count > 65536u - total) return NMS_TYPE;
        total += descriptors[i].field_count;
    }
    runtime->record_descriptors = descriptors;
    runtime->record_count = count;
    runtime->records_bound = 1;
    return NMS_OK;
}
NmsStatus nms_record_create(NmsRuntime *runtime, uint32_t ordinal, const NmsValue *values,
                             uint32_t count, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!runtime->records_bound || !runtime->record_descriptors ||
        ordinal >= runtime->record_count) return NMS_TYPE;
    if (count != runtime->record_descriptors[ordinal].field_count) return NMS_TYPE;
    if (count && !values) return NMS_STATE;
    uint64_t bytes = (uint64_t)count * sizeof(NmsValue);
    if (bytes > SIZE_MAX || bytes > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    for (uint32_t i = 0; i < count; i++) {
        NmsStatus status = value_valid(runtime, values[i]);
        if (status != NMS_OK) return status;
    }
    NmsValue *fields = bytes ? allocate(runtime, bytes) : NULL;
    if (bytes && !fields) return NMS_MEMORY;
    uint32_t retained = 0;
    NmsStatus status = NMS_OK;
    for (; retained < count; retained++) {
        fields[retained] = values[retained];
        status = nms_value_retain(runtime, fields[retained]);
        if (status != NMS_OK) break;
    }
    NmsHandle result;
    if (status == NMS_OK)
        status = publish_slot(runtime, (unsigned char *)fields, count, count,
                              NMS_SLOT_RECORD, bytes, &result);
    if (status != NMS_OK) {
        while (retained) nms_value_release(runtime, fields[--retained]);
        deallocate(fields);
        return status;
    }
    runtime->slots[(uint32_t)result].record_ordinal = ordinal;
    *out = result;
    return NMS_OK;
}
NmsStatus nms_record_identity(const NmsRuntime *runtime, NmsHandle record,
                               uint32_t *ordinal, uint32_t *global_layout) {
    if (!ordinal || !global_layout) return NMS_STATE;
    uint32_t index;
    NmsStatus status = record_slot(runtime, record, &index);
    if (status != NMS_OK) return status;
    uint32_t definition = runtime->slots[index].record_ordinal;
    *ordinal = definition;
    *global_layout = runtime->record_descriptors[definition].global_layout_index;
    return NMS_OK;
}
NmsStatus nms_record_get(NmsRuntime *runtime, NmsHandle record, uint64_t field, NmsValue *out) {
    if (!out) return NMS_STATE;
    uint32_t index;
    NmsStatus status = record_slot(runtime, record, &index);
    if (status != NMS_OK) return status;
    if (field >= runtime->slots[index].length) return NMS_BOUNDS;
    NmsValue value = slot_value(&runtime->slots[index], (uint32_t)field);
    status = nms_value_retain(runtime, value);
    if (status == NMS_OK) *out = value;
    return status;
}
NmsStatus nms_record_set(NmsRuntime *runtime, NmsHandle record, uint64_t field, NmsValue value) {
    uint32_t index;
    NmsStatus status = record_slot(runtime, record, &index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[index];
    if (field >= slot->length) return NMS_BOUNDS;
    status = nms_value_retain(runtime, value);
    if (status != NMS_OK) return status;
    NmsValue *fields = (NmsValue *)slot->data;
    NmsValue previous = fields[field];
    fields[field] = value;
    return nms_value_release(runtime, previous);
}
NmsStatus nms_value_array_create(NmsRuntime *runtime, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    return publish_slot(runtime, NULL, 0, 0, NMS_SLOT_BOXED_ARRAY, 0, out);
}
static NmsStatus value_array_write(NmsRuntime *runtime, NmsHandle array,
                                    uint64_t requested_index, NmsValue value, int append) {
    uint32_t index;
    NmsStatus status = value_array_slot(runtime, array, &index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[index];
    if (slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY)
        return packed_write(runtime, slot, requested_index, value, append);
    status = value_valid(runtime, value);
    if (status != NMS_OK) return status;
    if (!append && requested_index >= slot->length) return NMS_STATE;
    if (append && slot->length == UINT32_MAX) return NMS_MEMORY;
    uint32_t target = append ? slot->length : (uint32_t)requested_index;
    uint32_t capacity = slot->capacity;
    if (append && slot->length == capacity) {
        status = array_next_capacity(slot->capacity, sizeof(NmsValue), slot->vm_array_policy, &capacity);
        if (status != NMS_OK) return status;
    }
    uint64_t bytes = (uint64_t)capacity * sizeof(NmsValue);
    uint64_t old_bytes = slot_storage_bytes(slot);
    if (bytes > SIZE_MAX || bytes - old_bytes > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    NmsValue *buffer = (NmsValue *)slot->data;
    int replacement = slot->kind == NMS_SLOT_STRING_ARRAY || capacity != slot->capacity;
    if (replacement) {
        buffer = allocate(runtime, bytes);
        if (!buffer) return NMS_MEMORY;
        /* I move existing edge ownership without changing reference counts. */
        for (uint32_t i = 0; i < slot->length; i++) buffer[i] = slot_value(slot, i);
    }
    status = nms_value_retain(runtime, value);
    if (status != NMS_OK) {
        if (replacement) deallocate(buffer);
        return status;
    }
    NmsValue previous = {0, 0};
    if (!append) previous = slot_value(slot, target);
    if (replacement) {
        unsigned char *old = slot->data;
        slot->data = (unsigned char *)buffer;
        slot->capacity = capacity;
        slot->kind = NMS_SLOT_BOXED_ARRAY;
        runtime->live_bytes += bytes - old_bytes;
        deallocate(old);
    }
    buffer[target] = value;
    if (append) slot->length++;
    /* My new edge is visible before I drop the old child's owner. */
    return append ? NMS_OK : nms_value_release(runtime, previous);
}
NmsStatus nms_value_array_append(NmsRuntime *runtime, NmsHandle array, NmsValue value) {
    return value_array_write(runtime, array, 0, value, 1);
}
NmsStatus nms_value_array_set(NmsRuntime *runtime, NmsHandle array, uint64_t index, NmsValue value) {
    return value_array_write(runtime, array, index, value, 0);
}
NmsStatus nms_value_array_get(NmsRuntime *runtime, NmsHandle array, uint64_t index, NmsValue *out) {
    if (!out) return NMS_STATE;
    uint32_t slot_index;
    NmsStatus status = value_array_slot(runtime, array, &slot_index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[slot_index];
    if (index >= slot->length) { *out = (NmsValue){0, 0}; return NMS_OK; }
    NmsValue value = slot_value(slot, (uint32_t)index);
    status = nms_value_retain(runtime, value);
    if (status == NMS_OK) *out = value;
    return status;
}
NmsStatus nms_value_array_pop(NmsRuntime *runtime, NmsHandle array, NmsValue *out) {
    if (!out) return NMS_STATE;
    uint32_t index;
    NmsStatus status = value_array_slot(runtime, array, &index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[index];
    if (!slot->length) { *out = (NmsValue){0, 0}; return NMS_OK; }
    NmsValue value = slot_value(slot, slot->length - 1);
    slot->length--;
    if (slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY) packed_publish(slot, slot->length, 0);
    else if (slot->kind == NMS_SLOT_STRING_ARRAY) ((NmsHandle *)slot->data)[slot->length] = 0;
    else ((NmsValue *)slot->data)[slot->length] = (NmsValue){0, 0};
    *out = value;
    return NMS_OK;
}
NmsStatus nms_value_array_length(const NmsRuntime *runtime, NmsHandle array, uint32_t *out) {
    if (!out) return NMS_STATE;
    uint32_t index;
    NmsStatus status = value_array_slot(runtime, array, &index);
    if (status == NMS_OK) *out = runtime->slots[index].length;
    return status;
}
/* I borrow complete input arrays; my result remains private until all edges exist. */
NmsStatus nms_vm_array_literal(NmsRuntime *runtime, uint32_t tag,
                                const uint64_t *payloads, const uint32_t *tags,
                                uint32_t count, NmsHandle *out) {
    if (!runtime || !out || count > UINT16_MAX || (count && (!payloads || !tags))) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!(tag <= 5 || tag == NMS_ARRAY_TAG || tag == 9)) return NMS_TYPE;
    for (uint32_t i = 0; i < count; i++) {
        NmsValue value = {payloads[i], tags[i]};
        uint64_t ignored;
        NmsStatus status = packed_width(tag) ? packed_prepare(tag, value, &ignored) : value_valid(runtime, value);
        if (status != NMS_OK) return status;
    }
    NmsHandle result = 0;
    NmsStatus status = vm_array_create_capacity(runtime, tag, count, &result);
    if (status != NMS_OK) return status;
    for (uint32_t i = 0; i < count; i++) {
        status = nms_value_array_append(runtime, result, (NmsValue){payloads[i], tags[i]});
        if (status != NMS_OK) { nms_release(runtime, result); return status; }
    }
    *out = result;
    return NMS_OK;
}
NmsStatus nms_vm_array_slice(NmsRuntime *runtime, NmsHandle source,
                              uint32_t start, uint32_t end, NmsHandle *out) {
    if (!out) return NMS_STATE;
    uint32_t source_index;
    NmsStatus status = value_array_slot(runtime, source, &source_index);
    if (status != NMS_OK) return status;
    const NmsSlot *input = &runtime->slots[source_index];
    uint32_t length = input->length;
    uint32_t tag = input->kind == NMS_SLOT_STRING_ARRAY ? 5 : input->element_tag;
    if (start > length) start = length;
    if (end > length) end = length;
    uint32_t count = end > start ? end - start : 0;
    NmsHandle result = 0;
    status = vm_array_create_capacity(runtime, tag, count, &result);
    if (status != NMS_OK) return status;
    /* Descriptor growth may relocate the source slot; its immutable buffer
     * remains owned by the borrowed source handle. Reacquire both slots. */
    input = &runtime->slots[source_index];
    NmsSlot *output = &runtime->slots[(uint32_t)result];
    if (input->kind == NMS_SLOT_PACKED_SCALAR_ARRAY) {
        uint32_t width = packed_width(tag);
        if (count) copy_bytes(output->data, input->data + (uint64_t)start * width,
                              (uint64_t)count * width);
        output->length = count;
    } else {
        for (uint32_t i = 0; i < count; i++) {
            NmsValue value = slot_value(input, start + i);
            status = nms_value_array_append(runtime, result, value);
            if (status != NMS_OK) { nms_release(runtime, result); return status; }
        }
    }
    *out = result;
    return NMS_OK;
}

/* I format the exact binary64 rational with decimal integer arithmetic. The
 * largest coefficient needs fewer than 800 digits; 1100 is an explicit cap. */
static NmsStatus nms_format_binary64(NmsRuntime *runtime, uint64_t bits, NmsHandle *out) {
    unsigned char output[32];
    unsigned used = 0;
    int negative = (int)(bits >> 63);
    if (negative) output[used++] = '-';
    uint32_t exponent_bits = (uint32_t)((bits >> 52) & 2047);
    uint64_t significand = bits & ((UINT64_C(1) << 52) - 1);
    if (exponent_bits == 2047) {
        const char *word = significand ? "nan" : "inf";
        for (unsigned i = 0; i < 3; i++) output[used++] = (unsigned char)word[i];
        return nms_create(runtime, output, used, out);
    }
    if (!exponent_bits && !significand) {
        output[used++] = '0';
        return nms_create(runtime, output, used, out);
    }
    int binary_exponent = exponent_bits ? (int)exponent_bits - 1023 - 52 : -1074;
    if (exponent_bits) significand |= UINT64_C(1) << 52;
    unsigned char digits[1100]; /* Little endian, one decimal digit per byte. */
    unsigned count = 0;
    do { digits[count++] = (unsigned char)(significand % 10); significand /= 10; }
    while (significand);
    unsigned steps = (unsigned)(binary_exponent < 0 ? -binary_exponent : binary_exponent);
    unsigned multiplier = binary_exponent < 0 ? 5 : 2;
    for (unsigned step = 0; step < steps; step++) {
        unsigned carry = 0;
        for (unsigned i = 0; i < count; i++) {
            unsigned value = digits[i] * multiplier + carry;
            digits[i] = (unsigned char)(value % 10);
            carry = value / 10;
        }
        if (carry) {
            if (count == sizeof digits) return NMS_STATE;
            digits[count++] = (unsigned char)carry;
        }
    }
    int exponent = (int)count - 1 + (binary_exponent < 0 ? binary_exponent : 0);
    uint32_t rounded = 0;
    for (unsigned i = 0; i < 6; i++)
        rounded = rounded * 10 + (i < count ? digits[count - 1 - i] : 0);
    if (count > 6) {
        unsigned next = digits[count - 7];
        int lower_nonzero = 0;
        for (unsigned i = 0; i + 7 < count; i++) lower_nonzero |= digits[i] != 0;
        if (next > 5 || (next == 5 && (lower_nonzero || (rounded & 1)))) rounded++;
        if (rounded == 1000000) { rounded = 100000; exponent++; }
    }
    unsigned char leading[6];
    for (unsigned i = 6; i > 0; i--) { leading[i-1] = (unsigned char)('0' + rounded % 10); rounded /= 10; }
    unsigned length = 6;
    while (length > 1 && leading[length - 1] == '0') length--;
    /* At most 14 bytes: sign + six digits + point + e + sign + three exponent
     * digits. Fixed notation is smaller because its exponent lies in [-4,5]. */
    if (exponent < -4 || exponent >= 6) {
        output[used++] = leading[0];
        if (length > 1) {
            output[used++] = '.';
            for (unsigned i = 1; i < length; i++) output[used++] = leading[i];
        }
        output[used++] = 'e';
        output[used++] = exponent < 0 ? '-' : '+';
        unsigned magnitude = (unsigned)(exponent < 0 ? -exponent : exponent);
        if (magnitude >= 100) output[used++] = (unsigned char)('0' + magnitude / 100);
        output[used++] = (unsigned char)('0' + (magnitude / 10) % 10);
        output[used++] = (unsigned char)('0' + magnitude % 10);
    } else if (exponent < 0) {
        output[used++] = '0'; output[used++] = '.';
        for (int i = -1; i > exponent; i--) output[used++] = '0';
        for (unsigned i = 0; i < length; i++) output[used++] = leading[i];
    } else {
        unsigned integer_length = (unsigned)exponent + 1;
        for (unsigned i = 0; i < integer_length; i++) output[used++] = leading[i];
        if (length > integer_length) {
            output[used++] = '.';
            for (unsigned i = integer_length; i < length; i++) output[used++] = leading[i];
        }
    }
    return nms_create(runtime, output, used, out);
}
NmsStatus nms_format_scalar(NmsRuntime *runtime, uint64_t bits, uint32_t tag, NmsHandle *out) {
    if (tag == 3) return nms_format_binary64(runtime, bits, out);
    if (tag == 0 || tag == NMS_ARRAY_TAG || tag == 9) return nms_create(runtime, NULL, 0, out);
    if (tag == 4) return nms_create(runtime,
        (const unsigned char *)(bits ? "true" : "false"), bits ? 4 : 5, out);
    if (tag != 1 && tag != 2) return NMS_TYPE;
    int negative = tag == 1 && (bits >> 63);
    uint64_t magnitude = tag == 2 ? bits & 255 : negative ? UINT64_C(0) - bits : bits;
    unsigned char digits[20];
    unsigned position = sizeof digits;
    do { digits[--position] = (unsigned char)('0' + magnitude % 10); magnitude /= 10; }
    while (magnitude);
    if (negative) digits[--position] = '-';
    return nms_create(runtime, digits + position, sizeof digits - position, out);
}
NmsStatus nms_parse_f64(const NmsRuntime *runtime, NmsHandle source, uint64_t *out) {
    NmsView view;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status != NMS_OK) return status;
    return nbp_parse(view.data, view.length, out) ? NMS_OK : NMS_STATE;
}
NmsStatus nms_parse_i64(const NmsRuntime *runtime, NmsHandle source, int64_t *out) {
    NmsView view;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status != NMS_OK) return status;
    if (!out) return NMS_STATE;
    uint32_t i = 0;
    while (i < view.length) {
        unsigned char c = view.data[i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\v' && c != '\f') break;
        i++;
    }
    int negative = 0;
    if (i < view.length && (view.data[i] == '-' || view.data[i] == '+')) {
        negative = view.data[i] == '-';
        i++;
    }
    uint64_t limit = negative ? (UINT64_C(1) << 63) : INT64_MAX;
    uint64_t value = 0;
    while (i < view.length) {
        unsigned char c = view.data[i++];
        if (c < '0' || c > '9') break;
        uint64_t digit = c - '0';
        if (value > (limit - digit) / 10) { value = limit; break; }
        value = value * 10 + digit;
    }
    /* I never cast 2^63 to signed or negate INT64_MIN. */
    *out = negative ? (value == (UINT64_C(1) << 63) ? INT64_MIN : -(int64_t)value)
                    : (int64_t)value;
    return NMS_OK;
}
NmsStatus nms_case_owned(NmsRuntime *runtime, NmsHandle source, uint32_t upper,
                         NmsHandle *out) {
    NmsView view;
    NmsHandle result = 0;
    NmsStatus status = (!out || upper > 1) ? NMS_STATE : nms_view(runtime, source, &view);
    if (status == NMS_OK) status = nms_create(runtime, view.data, view.length, &result);
    if (status == NMS_OK) {
        /* Creation can move the descriptor table. This new owner is private;
         * I reacquire its slot after allocation and transform before publish. */
        NmsSlot *slot = &runtime->slots[(uint32_t)(result & ~NMS_DYNAMIC)];
        for (uint32_t i = 0; i < slot->length; i++) {
            unsigned char byte = slot->data[i];
            slot->data[i] = upper ? (byte >= 'a' && byte <= 'z' ? byte - 32 : byte)
                                  : (byte >= 'A' && byte <= 'Z' ? byte + 32 : byte);
        }
    }
    NmsStatus released = nms_release(runtime, source);
    if (status != NMS_OK) return status;
    if (released != NMS_OK) { nms_release(runtime, result); return released; }
    *out = result;
    return NMS_OK;
}
static int trim_space(unsigned char byte) {
    return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r';
}
NmsStatus nms_trim_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle *out) {
    NmsView view;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status != NMS_OK) { nms_release(runtime, source); return status; }
    uint32_t start = 0, end = view.length;
    while (start < end && trim_space(view.data[start])) start++;
    while (end > start && trim_space(view.data[end - 1])) end--;
    return nms_substr_owned(runtime, source, start, end - start, out);
}
NmsStatus nms_substr_owned(NmsRuntime *runtime, NmsHandle source,
                           uint32_t start, uint32_t length, NmsHandle *out) {
    NmsView view;
    NmsHandle result = 0;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status == NMS_OK) {
        if (start >= view.length) { start = 0; length = 0; }
        else if (length > view.length - start) length = view.length - start;
        status = out ? nms_create(runtime, length ? view.data + start : NULL, length, &result) : NMS_STATE;
    }
    NmsStatus released = nms_release(runtime, source);
    if (status != NMS_OK) return status;
    if (released != NMS_OK) { nms_release(runtime, result); return released; }
    *out = result;
    return NMS_OK;
}
NmsStatus nms_concat_owned(NmsRuntime *runtime, NmsHandle left, NmsHandle right,
                           NmsHandle *out) {
    NmsView a, b;
    NmsHandle result = 0;
    NmsStatus status = nms_view(runtime, left, &a);
    if (status == NMS_OK) status = nms_view(runtime, right, &b);
    if (status == NMS_OK)
        status = out ? create_parts(runtime, a.data, a.length, b.data, b.length, &result) : NMS_STATE;
    /* Each input represents a transferred owner, including equal handles.
     * Copying finishes before release or descriptor-table replacement. */
    NmsStatus left_status = nms_release(runtime, left);
    NmsStatus right_status = nms_release(runtime, right);
    if (status != NMS_OK) return status;
    /* Ownership-correct callers cannot fail these releases. */
    if (left_status != NMS_OK || right_status != NMS_OK) {
        nms_release(runtime, result);
        return left_status != NMS_OK ? left_status : right_status;
    }
    *out = result;
    return NMS_OK;
}
NmsStatus nms_retain(NmsRuntime *runtime, NmsHandle handle) {
    if (!(handle & NMS_DYNAMIC)) { NmsView view; return nms_view(runtime, handle, &view); }
    uint32_t index;
    NmsStatus status = slot_for(runtime, handle, &index);
    if (status != NMS_OK) return status;
    if (runtime->slots[index].references == UINT64_MAX) return NMS_MEMORY;
    runtime->slots[index].references++;
    return NMS_OK;
}
NmsStatus nms_release(NmsRuntime *runtime, NmsHandle handle) {
    if (!(handle & NMS_DYNAMIC)) { NmsView view; return nms_view(runtime, handle, &view); }
    uint32_t index;
    NmsStatus status = slot_for(runtime, handle, &index);
    if (status != NMS_OK) return status;
    if (--runtime->slots[index].references) return NMS_OK;
    /* Zero-count slots are unavailable to live-handle lookup. I use their
     * next_free field as a private worklist until all outgoing edges are gone.
     * No allocation or table relocation occurs in this loop. */
    uint32_t pending = index;
    runtime->slots[index].next_free = 0;
    NmsStatus first_error = NMS_OK;
    while (pending) {
        index = pending;
        NmsSlot *slot = &runtime->slots[index];
        pending = slot->next_free;
        if (slot_has_children(slot)) {
            for (uint32_t i = 0; i < slot->length; i++) {
                NmsValue child = slot_value(slot, i);
                if (!value_is_reference(child) || !(child.payload & NMS_DYNAMIC)) continue;
                uint32_t child_index;
                NmsStatus child_status = slot_for(runtime, child.payload, &child_index);
                if (child_status != NMS_OK) {
                    if (first_error == NMS_OK) first_error = child_status;
                    continue;
                }
                NmsSlot *child_slot = &runtime->slots[child_index];
                if (!--child_slot->references) {
                    child_slot->next_free = pending;
                    pending = child_index;
                }
            }
        }
        runtime->live_bytes -= slot->kind == NMS_SLOT_STRING ? slot->length : slot_storage_bytes(slot);
        runtime->live_objects--;
        deallocate(slot->data);
        slot->data = NULL; slot->length = 0; slot->capacity = 0;
        slot->kind = NMS_SLOT_FREE; slot->element_tag = 0; slot->vm_array_policy = 0; slot->record_ordinal = 0;
        slot->next_free = runtime->free_head; runtime->free_head = index;
    }
    return first_error;
}
/* I complete validation and scratch allocation before changing any owner.
 * Trial counts identify external roots; graph edges do not become roots. */
static NmsStatus collect_with_workspace(NmsRuntime *runtime, uint64_t *trial,
                                        unsigned char *marked, uint32_t *queue) {
    uint64_t count = (uint64_t)runtime->capacity + 1;
    uint64_t bytes = 0, objects = 0;
    NmsStatus status = NMS_STATE;
    for (uint64_t i = 0; i < count; i++) {
        const NmsSlot *slot = &runtime->slots[i];
        trial[i] = slot->references; marked[i] = 0;
        if (!slot->references) {
            if (slot->kind != NMS_SLOT_FREE || slot->data || slot->length || slot->capacity) goto done;
            continue;
        }
        if (!i || slot->kind < NMS_SLOT_STRING || slot->kind > NMS_SLOT_RECORD) goto done;
        uint64_t storage;
        if (slot->kind == NMS_SLOT_STRING) {
            if (!slot->data) goto done;
            storage = slot->length;
        } else {
            if (slot->length > slot->capacity || (slot->capacity && !slot->data)) goto done;
            if (slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY && !packed_width(slot->element_tag)) goto done;
            if (slot->kind == NMS_SLOT_RECORD) {
                uint32_t index;
                if (record_slot(runtime, NMS_DYNAMIC | i, &index) != NMS_OK) goto done;
            }
            storage = slot_storage_bytes(slot);
            if (storage > SIZE_MAX) goto done;
        }
        if (storage > UINT64_MAX - bytes) goto done;
        bytes += storage; objects++;
    }
    if (objects != runtime->live_objects || bytes != runtime->live_bytes) goto done;
    for (uint64_t i = 1; i < count; i++) {
        const NmsSlot *slot = &runtime->slots[i];
        if (!slot_has_children(slot)) continue;
        for (uint32_t j = 0; j < slot->length; j++) {
            NmsValue child = slot_value(slot, j);
            if (value_valid(runtime, child) != NMS_OK) goto done;
            if (value_is_reference(child) && (child.payload & NMS_DYNAMIC)) {
                uint32_t index = (uint32_t)child.payload;
                if (!trial[index]) goto done;
                trial[index]--;
            }
        }
    }
    uint64_t head = 0, tail = 0;
    for (uint64_t i = 1; i < count; i++) if (trial[i]) {
        marked[i] = 1; queue[tail++] = (uint32_t)i;
    }
    while (head < tail) {
        const NmsSlot *slot = &runtime->slots[queue[head++]];
        if (!slot_has_children(slot)) continue;
        for (uint32_t j = 0; j < slot->length; j++) {
            NmsValue child = slot_value(slot, j);
            if (!value_is_reference(child) || !(child.payload & NMS_DYNAMIC)) continue;
            uint32_t index = (uint32_t)child.payload;
            if (!marked[index]) { marked[index] = 1; queue[tail++] = index; }
        }
    }
    /* Every marked slot is reached from an external owner. Removing only
     * dead-to-live edges therefore cannot remove its final live owner. */
    for (uint64_t i = 1; i < count; i++) {
        const NmsSlot *slot = &runtime->slots[i];
        if (marked[i] || !slot->references ||
            !slot_has_children(slot)) continue;
        for (uint32_t j = 0; j < slot->length; j++) {
            NmsValue child = slot_value(slot, j);
            if (value_is_reference(child) && (child.payload & NMS_DYNAMIC)) {
                uint32_t index = (uint32_t)child.payload;
                if (marked[index]) runtime->slots[index].references--;
            }
        }
    }
    for (uint64_t i = 1; i < count; i++) {
        NmsSlot *slot = &runtime->slots[i];
        if (marked[i] || !slot->references) continue;
        runtime->live_bytes -= slot->kind == NMS_SLOT_STRING ? slot->length : slot_storage_bytes(slot);
        runtime->live_objects--; deallocate(slot->data);
        slot->data = NULL; slot->references = 0; slot->length = 0; slot->capacity = 0;
        slot->kind = NMS_SLOT_FREE; slot->element_tag = 0; slot->vm_array_policy = 0; slot->record_ordinal = 0;
        slot->next_free = runtime->free_head; runtime->free_head = (uint32_t)i;
    }
    status = NMS_OK;
 done:
    return status;
}
static NmsStatus collection_ready(const NmsRuntime *runtime) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!runtime->capacity)
        return !runtime->slots && !runtime->live_objects && !runtime->live_bytes ? NMS_OK : NMS_STATE;
    return runtime->slots ? NMS_OK : NMS_STATE;
}
NmsStatus nms_collect(NmsRuntime *runtime) {
    NmsStatus status = collection_ready(runtime);
    if (status != NMS_OK || !runtime->capacity) return status;
    uint64_t count = (uint64_t)runtime->capacity + 1;
    if (count > SIZE_MAX / sizeof(uint64_t) || count > SIZE_MAX / sizeof(uint32_t)) return NMS_MEMORY;
    uint64_t *trial = allocate(runtime, count * sizeof(uint64_t));
    unsigned char *marked = allocate(runtime, count);
    uint32_t *queue = allocate(runtime, count * sizeof(uint32_t));
    status = trial && marked && queue ? collect_with_workspace(runtime, trial, marked, queue) : NMS_MEMORY;
    deallocate(queue); deallocate(marked); deallocate(trial);
    return status;
}
NmsStatus nms_collect_prepared(NmsRuntime *runtime) {
    NmsStatus status = collection_ready(runtime);
    if (status != NMS_OK) return status;
    if (!runtime->collection_prepared || !runtime->collection_workspace ||
        runtime->collection_capacity < runtime->capacity) return NMS_STATE;
    if (!runtime->capacity) return NMS_OK;
    uint64_t marks, queue, bytes;
    if (!collection_layout(runtime->collection_capacity, &marks, &queue, &bytes)) return NMS_STATE;
    unsigned char *workspace = runtime->collection_workspace;
    return collect_with_workspace(runtime, (uint64_t *)workspace, workspace + marks,
                                  (uint32_t *)(workspace + queue));
}
NmsStatus nms_begin(NmsRuntime *runtime) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (runtime->active) return NMS_BUSY;
    runtime->active = 1;
    return NMS_OK;
}
uint64_t nms_finish(NmsRuntime *runtime, NmsStatus status, int32_t result) {
    if (!runtime || !runtime->active) return (uint64_t)NMS_STATE << 32;
    runtime->active = 0;
    if ((unsigned)status > NMS_BOUNDS) status = NMS_STATE;
    return ((uint64_t)status << 32) | (status == NMS_OK ? (uint32_t)result : 0);
}
NmsStatus nms_dispose(NmsRuntime *runtime) {
    if (!runtime) return NMS_STATE;
    if (runtime->active) return NMS_BUSY;
    if (runtime->disposed) return NMS_OK;
    /* Terminal instance disposal invalidates all context-local handles.
     * Later lowering must first release frame/global roots normally. */
    for (uint64_t i = 1; i <= runtime->capacity; i++)
        if (runtime->slots[i].references) deallocate(runtime->slots[i].data);
    deallocate(runtime->slots);
    deallocate(runtime->collection_workspace);
    runtime->collection_workspace = NULL;
    runtime->collection_capacity = runtime->collection_prepared = 0;
    runtime->slots = NULL; runtime->capacity = runtime->free_head = 0;
    runtime->live_bytes = runtime->live_objects = 0;
    runtime->record_descriptors = NULL;
    runtime->record_count = runtime->records_bound = 0;
    runtime->disposed = 1;
    return NMS_OK;
}
int nms_reserved_entry(const char *name) {
    if (!name) return 0;
    const char *reserved[] = {"nano_try_entry", "nano_dispose", "nano_runtime_", "nms_"};
    for (unsigned i = 0; i < sizeof reserved / sizeof reserved[0]; i++) {
        unsigned n = 0;
        while (reserved[i][n] && name[n] == reserved[i][n]) n++;
        if (!reserved[i][n] && (i >= 2 || !name[n])) return 1;
    }
    return 0;
}
#ifdef NMS_TESTING
void nms_test_fail_after(NmsRuntime *runtime, uint64_t count) { runtime->fail_after = count; }
uint64_t nms_test_live_allocations(void) { return live_allocations; }
uint64_t nms_test_memory_pages(void) {
#ifdef __wasm32__
    return __builtin_wasm_memory_size(0);
#else
    return 0;
#endif
}
#endif

/* I compare complete byte views, including embedded zero bytes. */
static int equal_bytes(const unsigned char *left, const unsigned char *right, uint32_t length) {
    for (uint32_t i = 0; i < length; i++)
        if (left[i] != right[i]) return 0;
    return 1;
}
static int find_bytes(NmsView source, NmsView needle, uint32_t start, uint32_t *found) {
    if (!needle.length || start > source.length || needle.length > source.length - start)
        return 0;
    uint32_t last = source.length - needle.length;
    for (uint32_t position = start; position <= last; position++) {
        if (equal_bytes(source.data + position, needle.data, needle.length)) {
            *found = position;
            return 1;
        }
    }
    return 0;
}
static NmsStatus append_segment(NmsRuntime *runtime, NmsHandle array,
                                 const unsigned char *bytes, uint32_t length, int values) {
    NmsHandle child = 0;
    NmsStatus status = nms_create(runtime, bytes, length, &child);
    if (status != NMS_OK) return status;
    status = values ? nms_value_array_append(runtime, array, (NmsValue){child, 5}) :
                      nms_string_array_append(runtime, array, child);
    NmsStatus released = nms_release(runtime, child);
    return status != NMS_OK ? status : released;
}
static NmsStatus split_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle delimiter,
                               NmsHandle *out, int values) {
    NmsView a, b;
    NmsHandle array = 0;
    NmsStatus status = out ? nms_view(runtime, source, &a) : NMS_STATE;
    if (status == NMS_OK) status = nms_view(runtime, delimiter, &b);
    if (status == NMS_OK) status = values ? nms_vm_array_create(runtime, 5, &array) :
                                          nms_string_array_create(runtime, &array);
    if (status == NMS_OK && !b.length) {
        for (uint32_t i = 0; i < a.length && status == NMS_OK; i++)
            status = append_segment(runtime, array, a.data + i, 1, values);
    } else if (status == NMS_OK) {
        uint32_t position = 0, found = 0;
        while (status == NMS_OK && find_bytes(a, b, position, &found)) {
            status = append_segment(runtime, array, a.data + position, found - position, values);
            position = found + b.length;
        }
        if (status == NMS_OK)
            status = append_segment(runtime, array, a.data + position, a.length - position, values);
    }
    /* Views point at retained byte buffers, never relocated slot-table cells.
     * I finish every read before consuming either input owner. */
    NmsStatus left = nms_release(runtime, source);
    NmsStatus right = nms_release(runtime, delimiter);
    if (status == NMS_OK) status = left != NMS_OK ? left : right;
    if (status != NMS_OK) {
        if (array) nms_release(runtime, array);
        return status;
    }
    *out = array;
    return NMS_OK;
}
NmsStatus nms_split_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle delimiter,
                          NmsHandle *out) {
    return split_owned(runtime, source, delimiter, out, 0);
}
NmsStatus nms_split_values_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle delimiter,
                                 NmsHandle *out) {
    return split_owned(runtime, source, delimiter, out, 1);
}
NmsStatus nms_replace_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle needle,
                            NmsHandle replacement, NmsHandle *out) {
    NmsView a, b, c;
    NmsHandle result = 0;
    unsigned char *scratch = NULL;
    NmsStatus status = out ? nms_view(runtime, source, &a) : NMS_STATE;
    if (status == NMS_OK) status = nms_view(runtime, needle, &b);
    if (status == NMS_OK) status = nms_view(runtime, replacement, &c);
    if (status == NMS_OK && !b.length) {
        status = nms_create(runtime, a.data, a.length, &result);
    } else if (status == NMS_OK) {
        uint64_t count = 0;
        uint32_t position = 0, found;
        while (find_bytes(a, b, position, &found)) {
            count++;
            position = found + b.length;
        }
        uint32_t length = 0;
        if (count > a.length / b.length) status = NMS_STATE;
        else {
            uint32_t remaining = a.length - (uint32_t)count * b.length;
            if (c.length && count > (UINT32_MAX - remaining) / c.length) status = NMS_MEMORY;
            else length = remaining + (uint32_t)count * c.length;
        }
        if (status == NMS_OK && (uint64_t)length + 1 > SIZE_MAX) status = NMS_MEMORY;
        if (status == NMS_OK) {
            scratch = allocate(runtime, (uint64_t)length + 1);
            if (!scratch) status = NMS_MEMORY;
        }
        if (status == NMS_OK) {
            uint32_t written = 0;
            position = 0;
            while (find_bytes(a, b, position, &found)) {
                uint32_t segment = found - position;
                copy_bytes(scratch + written, a.data + position, segment);
                written += segment;
                copy_bytes(scratch + written, c.data, c.length);
                written += c.length;
                position = found + b.length;
            }
            copy_bytes(scratch + written, a.data + position, a.length - position);
            scratch[length] = 0;
            /* All three immutable byte views remain owned through this copy;
             * descriptor-table relocation cannot invalidate their storage. */
            status = nms_create(runtime, scratch, length, &result);
        }
    }
    deallocate(scratch);
    NmsStatus sa = nms_release(runtime, source);
    NmsStatus sb = nms_release(runtime, needle);
    NmsStatus sc = nms_release(runtime, replacement);
    if (status != NMS_OK) return status;
    if (sa != NMS_OK || sb != NMS_OK || sc != NMS_OK) {
        nms_release(runtime, result);
        return sa != NMS_OK ? sa : sb != NMS_OK ? sb : sc;
    }
    *out = result;
    return NMS_OK;
}
NmsStatus nms_char_at(const NmsRuntime *runtime, NmsHandle source,
                      uint64_t index_bits, uint32_t is_integer, int64_t *out) {
    if (!out || is_integer > 1) return NMS_STATE;
    NmsView view;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status != NMS_OK) return status;
    uint64_t index = is_integer ? index_bits : 0;
    *out = (index & (UINT64_C(1) << 63)) || index >= view.length
         ? -1 : (int64_t)view.data[(uint32_t)index];
    return NMS_OK;
}
NmsStatus nms_predicate(const NmsRuntime *runtime, NmsHandle source, NmsHandle affix,
                        uint32_t operation, uint32_t *out) {
    if (!out || operation > NMS_ENDS_WITH) return NMS_STATE;
    NmsView haystack, needle;
    NmsStatus status = nms_view(runtime, source, &haystack);
    if (status != NMS_OK) return status;
    status = nms_view(runtime, affix, &needle);
    if (status != NMS_OK) return status;
    uint32_t answer = 0;
    if (needle.length == 0) answer = 1;
    else if (needle.length <= haystack.length) {
        uint32_t last = haystack.length - needle.length;
        if (operation == NMS_STARTS_WITH)
            answer = equal_bytes(haystack.data, needle.data, needle.length);
        else if (operation == NMS_ENDS_WITH)
            answer = equal_bytes(haystack.data + last, needle.data, needle.length);
        else {
            /* A nonempty needle makes last < UINT32_MAX; increment cannot wrap. */
            for (uint32_t position = 0; position <= last; position++) {
                if (equal_bytes(haystack.data + position, needle.data, needle.length)) {
                    answer = 1;
                    break;
                }
            }
        }
    }
    *out = answer;
    return NMS_OK;
}

#include <assert.h>
int main(void) {
    static const NmsRecordDescriptor descriptors[]={{7,1}};
    NmsRuntime runtime;
    nms_init(&runtime,NULL,0);
    assert(nms_bind_records(&runtime,descriptors,1)==NMS_OK);
    assert(nms_begin(&runtime)==NMS_OK);
    NmsHandle array=0,record=0;
    assert(nms_vm_array_create(&runtime,3,&array)==NMS_OK);
    NmsValue one={UINT64_C(0x3ff8000000000000),3};
    assert(nms_value_array_append(&runtime,array,one)==NMS_OK);
    NmsValue child={array,NMS_ARRAY_TAG};
    assert(nms_record_create(&runtime,0,&child,1,&record)==NMS_OK);
    assert(nms_release(&runtime,array)==NMS_OK);
    uint32_t ordinal=99,global=99;
    assert(nms_record_identity(&runtime,record,&ordinal,&global)==NMS_OK);
    assert(ordinal==0 && global==7);
    NmsValue alias={0,0};
    assert(nms_record_get(&runtime,record,0,&alias)==NMS_OK);
    assert(nms_release(&runtime,record)==NMS_OK);
    NmsValue got={0,0};
    assert(nms_value_array_get(&runtime,alias.payload,0,&got)==NMS_OK);
    assert(got.tag==3 && got.payload==one.payload);
    assert(nms_value_release(&runtime,got)==NMS_OK);
    assert(nms_value_array_get(&runtime,alias.payload,UINT64_MAX,&got)==NMS_OK);
    assert(got.tag==0);
    assert(nms_value_release(&runtime,alias)==NMS_OK);
    /* I check real roots before terminal disposal can sweep them. */
    assert(runtime.live_objects==0 && runtime.live_bytes==0);
    assert(nms_finish(&runtime,NMS_OK,0)==0);
    assert(!runtime.active);
    assert(nms_dispose(&runtime)==NMS_OK);
    assert(nano_rt_f64_add(1.5,2.5)==4.0);
    return 0;
}
