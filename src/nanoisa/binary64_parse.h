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
