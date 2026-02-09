/*
 * NanoVM Value operations
 */

#define _POSIX_C_SOURCE 200809L  /* For strdup() */

#include "value.h"
#include "heap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Printing
 * ======================================================================== */

void val_print(NanoValue v, FILE *out) {
    switch (v.tag) {
        case TAG_VOID:
            fprintf(out, "void");
            break;
        case TAG_INT:
            fprintf(out, "%lld", (long long)v.as.i64);
            break;
        case TAG_U8:
            fprintf(out, "%u", v.as.u8);
            break;
        case TAG_FLOAT: {
            /* Print without trailing zeros, but always with at least one decimal */
            double d = v.as.f64;
            if (d == (long long)d && d >= -1e15 && d <= 1e15) {
                fprintf(out, "%.1f", d);
            } else {
                fprintf(out, "%g", d);
            }
            break;
        }
        case TAG_BOOL:
            fprintf(out, "%s", v.as.boolean ? "true" : "false");
            break;
        case TAG_STRING:
            if (v.as.string) {
                fprintf(out, "%s", vmstring_cstr(v.as.string));
            } else {
                fprintf(out, "null");
            }
            break;
        case TAG_ENUM:
            fprintf(out, "enum(%d)", v.as.enum_val);
            break;
        case TAG_ARRAY:
            if (v.as.array) {
                fprintf(out, "[");
                for (uint32_t i = 0; i < v.as.array->length; i++) {
                    if (i > 0) fprintf(out, ", ");
                    val_print(v.as.array->elements[i], out);
                }
                fprintf(out, "]");
            } else {
                fprintf(out, "[]");
            }
            break;
        case TAG_STRUCT:
            if (v.as.sval) {
                fprintf(out, "{");
                for (uint32_t i = 0; i < v.as.sval->field_count; i++) {
                    if (i > 0) fprintf(out, ", ");
                    if (v.as.sval->field_names && v.as.sval->field_names[i]) {
                        fprintf(out, "%s: ", vmstring_cstr(v.as.sval->field_names[i]));
                    }
                    val_print(v.as.sval->fields[i], out);
                }
                fprintf(out, "}");
            } else {
                fprintf(out, "{}");
            }
            break;
        case TAG_UNION:
            if (v.as.uval) {
                fprintf(out, "variant(%u", v.as.uval->variant);
                for (uint32_t i = 0; i < v.as.uval->field_count; i++) {
                    fprintf(out, ", ");
                    val_print(v.as.uval->fields[i], out);
                }
                fprintf(out, ")");
            } else {
                fprintf(out, "union(null)");
            }
            break;
        case TAG_TUPLE:
            if (v.as.tuple) {
                fprintf(out, "(");
                for (uint32_t i = 0; i < v.as.tuple->count; i++) {
                    if (i > 0) fprintf(out, ", ");
                    val_print(v.as.tuple->elements[i], out);
                }
                fprintf(out, ")");
            } else {
                fprintf(out, "()");
            }
            break;
        case TAG_HASHMAP:
            fprintf(out, "hashmap(...)");
            break;
        case TAG_FUNCTION:
            fprintf(out, "fn(%u)", v.as.fn_idx);
            break;
        case TAG_OPAQUE:
            fprintf(out, "opaque(%u)", v.as.proxy_id);
            break;
        default:
            fprintf(out, "unknown(%u)", v.tag);
            break;
    }
}

void val_println(NanoValue v) {
    val_print(v, stdout);
    printf("\n");
}

/* ========================================================================
 * Comparison
 * ======================================================================== */

bool val_equal(NanoValue a, NanoValue b) {
    /* Allow enum ↔ int comparison (enum values are integers) */
    if (a.tag == TAG_ENUM && b.tag == TAG_INT)
        return (int64_t)a.as.enum_val == b.as.i64;
    if (a.tag == TAG_INT && b.tag == TAG_ENUM)
        return a.as.i64 == (int64_t)b.as.enum_val;
    /* Allow int ↔ float cross-type comparison */
    if (a.tag == TAG_INT && b.tag == TAG_FLOAT)
        return (double)a.as.i64 == b.as.f64;
    if (a.tag == TAG_FLOAT && b.tag == TAG_INT)
        return a.as.f64 == (double)b.as.i64;
    if (a.tag != b.tag) return false;
    switch (a.tag) {
        case TAG_VOID:   return true;
        case TAG_INT:    return a.as.i64 == b.as.i64;
        case TAG_U8:     return a.as.u8 == b.as.u8;
        case TAG_FLOAT:  return a.as.f64 == b.as.f64;
        case TAG_BOOL:   return a.as.boolean == b.as.boolean;
        case TAG_ENUM:   return a.as.enum_val == b.as.enum_val;
        case TAG_STRING:
            if (a.as.string == b.as.string) return true;
            if (!a.as.string || !b.as.string) return false;
            return vmstring_equal(a.as.string, b.as.string);
        default:
            /* Heap objects: pointer equality */
            return a.as.obj == b.as.obj;
    }
}

int val_compare(NanoValue a, NanoValue b) {
    /* Cross-type comparisons */
    if (a.tag == TAG_ENUM && b.tag == TAG_INT) {
        int64_t av = (int64_t)a.as.enum_val;
        return av < b.as.i64 ? -1 : av > b.as.i64 ? 1 : 0;
    }
    if (a.tag == TAG_INT && b.tag == TAG_ENUM) {
        int64_t bv = (int64_t)b.as.enum_val;
        return a.as.i64 < bv ? -1 : a.as.i64 > bv ? 1 : 0;
    }
    if (a.tag == TAG_INT && b.tag == TAG_FLOAT) {
        double da = (double)a.as.i64;
        return da < b.as.f64 ? -1 : da > b.as.f64 ? 1 : 0;
    }
    if (a.tag == TAG_FLOAT && b.tag == TAG_INT) {
        double db = (double)b.as.i64;
        return a.as.f64 < db ? -1 : a.as.f64 > db ? 1 : 0;
    }
    if (a.tag != b.tag) return (int)a.tag - (int)b.tag;
    switch (a.tag) {
        case TAG_INT:
            if (a.as.i64 < b.as.i64) return -1;
            if (a.as.i64 > b.as.i64) return 1;
            return 0;
        case TAG_FLOAT:
            if (a.as.f64 < b.as.f64) return -1;
            if (a.as.f64 > b.as.f64) return 1;
            return 0;
        case TAG_BOOL:
            return (int)a.as.boolean - (int)b.as.boolean;
        case TAG_STRING:
            if (a.as.string == b.as.string) return 0;
            if (!a.as.string) return -1;
            if (!b.as.string) return 1;
            return vmstring_compare(a.as.string, b.as.string);
        default:
            return 0;
    }
}

bool val_truthy(NanoValue v) {
    switch (v.tag) {
        case TAG_VOID:   return false;
        case TAG_INT:    return v.as.i64 != 0;
        case TAG_U8:     return v.as.u8 != 0;
        case TAG_FLOAT:  return v.as.f64 != 0.0;
        case TAG_BOOL:   return v.as.boolean;
        case TAG_STRING: return v.as.string != NULL;
        default:         return v.as.obj != NULL;
    }
}

char *val_to_cstring(NanoValue v) {
    char buf[256];
    switch (v.tag) {
        case TAG_VOID:
            return strdup("void");
        case TAG_INT:
            snprintf(buf, sizeof(buf), "%lld", (long long)v.as.i64);
            return strdup(buf);
        case TAG_FLOAT:
            snprintf(buf, sizeof(buf), "%g", v.as.f64);
            return strdup(buf);
        case TAG_BOOL:
            return strdup(v.as.boolean ? "true" : "false");
        case TAG_STRING:
            if (v.as.string) return strdup(vmstring_cstr(v.as.string));
            return strdup("null");
        default:
            snprintf(buf, sizeof(buf), "<%s>", isa_tag_name(v.tag));
            return strdup(buf);
    }
}
