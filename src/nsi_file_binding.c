#include "nsi_file_binding.h"
#include "nsi_internal.h"
#include "nsi_file_plan.h"
#include "utf8.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct NlFileBindingPlan {
    size_t interface_size, source_size, storage_size, peak_bound;
    unsigned char bytes[];
};

typedef struct {
    const unsigned char *bytes;
    size_t size, pos, tokens, objects;
    NlFileBindingStatus status;
} BindingScan;

static bool fb_add(size_t *total, size_t n) {
    if (n > SIZE_MAX - *total) return false;
    *total += n;
    return true;
}
static bool fb_product(size_t *total, size_t n, size_t width) {
    return (!n || width <= SIZE_MAX / n) && fb_add(total, n * width);
}
bool nl_file_binding_allocation_bound(size_t *out) {
    size_t total = sizeof(NlFileBindingPlan);
    size_t elements = 0;
    if (!out || !fb_add(&elements, sizeof(NlNsiNamed)) ||
        !fb_add(&elements, sizeof(NlNsiParam)) || !fb_add(&elements, sizeof(NlNsiMethod)) ||
        !fb_add(&elements, sizeof(NlNsiMember)) || !fb_add(&elements, sizeof(NlNsiType)) ||
        !fb_add(&elements, sizeof(NlNsiError)) ||
        !fb_product(&total, 2, NL_FILE_BINDING_MAX_BYTES + 1u) ||
        !fb_add(&total, NL_FILE_BINDING_MAX_BYTES + 1u) ||
        !fb_product(&total, NL_FILE_BINDING_MAX_TOKENS + 1u, sizeof(cJSON)) ||
        !fb_product(&total, 2, NL_FILE_BINDING_MAX_BYTES + 2u * NL_FILE_BINDING_MAX_TOKENS) ||
        !fb_add(&total, NL_FILE_BINDING_MAX_LEXEME + 1u) ||
        !fb_add(&total, sizeof(NlNsi)) ||
        !fb_product(&total, NL_FILE_BINDING_MAX_TOKENS, elements) ||
        !fb_product(&total, NL_FILE_BINDING_MAX_OBJECTS + 1u, sizeof(const char *)) ||
        !fb_add(&total, nl_file_plan_storage_size()) ||
        total > NL_FILE_BINDING_MAX_ALLOCATION) return false;
    *out = total;
    return true;
}
static bool fb_fail(BindingScan *s, NlFileBindingStatus status) {
    s->status = status;
    return false;
}
static bool fb_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}
static void fb_skip(BindingScan *s) {
    while (s->pos < s->size && fb_space(s->bytes[s->pos])) s->pos++;
}
static bool fb_token(BindingScan *s) {
    if (s->tokens == NL_FILE_BINDING_MAX_TOKENS) return fb_fail(s, NL_FILE_BINDING_LIMIT);
    s->tokens++;
    return true;
}
static int fb_hex(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static bool fb_hex4(BindingScan *s, unsigned *out) {
    unsigned n = 0;
    if (s->size - s->pos < 4) return fb_fail(s, NL_FILE_BINDING_INVALID);
    for (size_t i = 0; i < 4; i++) {
        int v = fb_hex(s->bytes[s->pos++]);
        if (v < 0) return fb_fail(s, NL_FILE_BINDING_INVALID);
        n = (n << 4) | (unsigned)v;
    }
    *out = n;
    return true;
}
static bool fb_string(BindingScan *s) {
    if (!fb_token(s)) return false;
    if (s->pos == s->size || s->bytes[s->pos++] != '"') return fb_fail(s, NL_FILE_BINDING_INVALID);
    size_t start = s->pos;
    while (s->pos < s->size) {
        unsigned char c = s->bytes[s->pos++];
        if (c == '"') {
            if (s->pos - start - 1 > NL_FILE_BINDING_MAX_LEXEME) return fb_fail(s, NL_FILE_BINDING_LIMIT);
            return true;
        }
        if (s->pos - start > NL_FILE_BINDING_MAX_LEXEME) return fb_fail(s, NL_FILE_BINDING_LIMIT);
        if (c < 0x20) return fb_fail(s, NL_FILE_BINDING_INVALID);
        if (c != '\\') continue;
        if (s->pos == s->size) return fb_fail(s, NL_FILE_BINDING_INVALID);
        c = s->bytes[s->pos++];
        if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') continue;
        if (c != 'u') return fb_fail(s, NL_FILE_BINDING_INVALID);
        unsigned value;
        if (!fb_hex4(s, &value)) return false;
        if (!value || (value >= 0xdc00 && value <= 0xdfff)) return fb_fail(s, NL_FILE_BINDING_INVALID);
        if (value >= 0xd800 && value <= 0xdbff) {
            if (s->size - s->pos < 2 || s->bytes[s->pos] != '\\' || s->bytes[s->pos + 1] != 'u')
                return fb_fail(s, NL_FILE_BINDING_INVALID);
            s->pos += 2;
            if (!fb_hex4(s, &value)) return false;
            if (value < 0xdc00 || value > 0xdfff) return fb_fail(s, NL_FILE_BINDING_INVALID);
        }
    }
    return fb_fail(s, NL_FILE_BINDING_INVALID);
}
static bool fb_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static bool fb_number(BindingScan *s) {
    size_t start = s->pos;
    if (!fb_token(s)) return false;
    if (s->bytes[s->pos] == '-') s->pos++;
    if (s->pos == s->size || !fb_digit(s->bytes[s->pos])) return fb_fail(s, NL_FILE_BINDING_INVALID);
    if (s->bytes[s->pos] == '0') s->pos++;
    else while (s->pos < s->size && fb_digit(s->bytes[s->pos])) s->pos++;
    if (s->pos < s->size && s->bytes[s->pos] == '.') {
        s->pos++;
        if (s->pos == s->size || !fb_digit(s->bytes[s->pos])) return fb_fail(s, NL_FILE_BINDING_INVALID);
        while (s->pos < s->size && fb_digit(s->bytes[s->pos])) s->pos++;
    }
    if (s->pos < s->size && (s->bytes[s->pos] == 'e' || s->bytes[s->pos] == 'E')) {
        s->pos++;
        if (s->pos < s->size && (s->bytes[s->pos] == '+' || s->bytes[s->pos] == '-')) s->pos++;
        if (s->pos == s->size || !fb_digit(s->bytes[s->pos])) return fb_fail(s, NL_FILE_BINDING_INVALID);
        while (s->pos < s->size && fb_digit(s->bytes[s->pos])) s->pos++;
    }
    return s->pos - start <= NL_FILE_BINDING_MAX_LEXEME || fb_fail(s, NL_FILE_BINDING_LIMIT);
}
static bool fb_value(BindingScan *s, size_t depth) {
    fb_skip(s);
    if (s->pos == s->size) return fb_fail(s, NL_FILE_BINDING_INVALID);
    unsigned char c = s->bytes[s->pos];
    if (c == '"') return fb_string(s);
    if (c == '-' || fb_digit(c)) return fb_number(s);
    if (c == '{' || c == '[') {
        bool object = c == '{';
        unsigned char end = object ? '}' : ']';
        size_t count = 0;
        if (depth == NL_FILE_BINDING_MAX_DEPTH) return fb_fail(s, NL_FILE_BINDING_LIMIT);
        if (object && ++s->objects > NL_FILE_BINDING_MAX_OBJECTS) return fb_fail(s, NL_FILE_BINDING_LIMIT);
        if (!fb_token(s)) return false;
        s->pos++;
        fb_skip(s);
        if (s->pos < s->size && s->bytes[s->pos] == end) { s->pos++; return fb_token(s); }
        for (;;) {
            if (++count > (object ? NL_FILE_BINDING_MAX_MEMBERS : NL_FILE_BINDING_MAX_ELEMENTS))
                return fb_fail(s, NL_FILE_BINDING_LIMIT);
            if (object) {
                if (!fb_string(s)) return false;
                fb_skip(s);
                if (s->pos == s->size || s->bytes[s->pos++] != ':') return fb_fail(s, NL_FILE_BINDING_INVALID);
                if (!fb_token(s)) return false;
            }
            if (!fb_value(s, depth + 1)) return false;
            fb_skip(s);
            if (s->pos == s->size) return fb_fail(s, NL_FILE_BINDING_INVALID);
            c = s->bytes[s->pos++];
            if (!fb_token(s)) return false;
            if (c == end) return true;
            if (c != ',') return fb_fail(s, NL_FILE_BINDING_INVALID);
            fb_skip(s);
        }
    }
    const char *literal = c == 't' ? "true" : c == 'f' ? "false" : c == 'n' ? "null" : NULL;
    if (!literal || s->size - s->pos < strlen(literal) ||
        memcmp(s->bytes + s->pos, literal, strlen(literal))) return fb_fail(s, NL_FILE_BINDING_INVALID);
    s->pos += strlen(literal);
    return fb_token(s);
}
static NlFileBindingStatus fb_preflight(const unsigned char *bytes, size_t size) {
    if (!bytes || !size) return NL_FILE_BINDING_INVALID;
    if (size > NL_FILE_BINDING_MAX_BYTES) return NL_FILE_BINDING_LIMIT;
    if (memchr(bytes, 0, size) || !nl_utf8_validate((const char *)bytes, size, NULL)) return NL_FILE_BINDING_INVALID;
    BindingScan s = { bytes, size, 0, 0, 0, NL_FILE_BINDING_OK };
    fb_skip(&s);
    if (s.pos == size || bytes[s.pos] != '{') return NL_FILE_BINDING_INVALID;
    if (!fb_value(&s, 0)) return s.status;
    fb_skip(&s);
    return s.pos == size ? NL_FILE_BINDING_OK : NL_FILE_BINDING_INVALID;
}
static bool fb_unique_keys(const cJSON *value, size_t depth) {
    if (depth > NL_FILE_BINDING_MAX_DEPTH) return false;
    if (cJSON_IsObject(value)) {
        for (const cJSON *a = value->child; a; a = a->next) {
            if (!a->string) return false;
            for (const cJSON *b = a->next; b; b = b->next)
                if (!b->string || strcmp(a->string, b->string) == 0) return false;
        }
    }
    for (const cJSON *child = value->child; child; child = child->next)
        if (!fb_unique_keys(child, depth + 1)) return false;
    return true;
}
/* I decode only; canonical roundtrip calls this once without invoking renderer. */
static NlFileBindingStatus fb_decode(const unsigned char *bytes, size_t size, NlNsi **out) {
    NlFileBindingStatus status = fb_preflight(bytes, size);
    if (status != NL_FILE_BINDING_OK) return status;
    char *copy = malloc(size + 1);
    if (!copy) return NL_FILE_BINDING_MEMORY;
    memcpy(copy, bytes, size); copy[size] = 0;
    const char *end = NULL;
    cJSON *json = cJSON_ParseWithLengthOpts(copy, size + 1, &end, 1);
    if (!json) { free(copy); return NL_FILE_BINDING_UNRESOLVED; }
    if (end != copy + size || !fb_unique_keys(json, 0)) {
        cJSON_Delete(json); free(copy); return NL_FILE_BINDING_INVALID;
    }
    NlNsi *nsi = nl_nsi_decode_object(json);
    cJSON_Delete(json); free(copy);
    if (!nsi) return NL_FILE_BINDING_UNRESOLVED;
    NlFilePlan *catalog = NULL;
    NlFilePlanStatus checked = nl_file_plan_build(nsi, &catalog);
    nl_file_plan_free(catalog);
    if (checked != NL_FILE_PLAN_OK) {
        nl_nsi_free(nsi);
        return checked == NL_FILE_PLAN_MEMORY ? NL_FILE_BINDING_MEMORY : NL_FILE_BINDING_INVALID;
    }
    *out = nsi;
    return NL_FILE_BINDING_OK;
}

typedef struct { unsigned char *data; size_t size, capacity; bool ok; } BindingWriter;
static void fb_write(BindingWriter *w, const char *text) {
    size_t n = strlen(text);
    if (!w->ok || n > NL_FILE_BINDING_MAX_BYTES - w->size) { w->ok = false; return; }
    if (w->data) {
        if (n > w->capacity - w->size) { w->ok = false; return; }
        memcpy(w->data + w->size, text, n);
    }
    w->size += n;
}
static void fb_quote(BindingWriter *w, const char *text) {
    static const char hex[] = "0123456789abcdef";
    fb_write(w, "\"");
    for (const unsigned char *p = (const unsigned char *)text; *p && w->ok; p++) {
        char escaped[7] = {0};
        if (*p == '"' || *p == '\\') { escaped[0] = '\\'; escaped[1] = (char)*p; }
        else if (*p < 0x20) {
            memcpy(escaped, "\\u00", 4); escaped[4] = hex[*p >> 4]; escaped[5] = hex[*p & 15];
        } else escaped[0] = (char)*p;
        fb_write(w, escaped);
    }
    fb_write(w, "\"");
}
static void fb_named(BindingWriter *w, const char *id, const char *name) {
    fb_write(w, "{\"id\":"); fb_quote(w, id);
    fb_write(w, ",\"name\":"); fb_quote(w, name);
}
static void fb_render_interface(BindingWriter *w, const NlNsi *n) {
    static const char *const directions[] = {"in","out","inout","return"};
    static const char *const ownerships[] = {"borrow","transfer","copy"};
    static const char *const lifetimes[] = {"call","caller","callee","resource"};
    static const char *const mutability[] = {"immutable","mutable"};
    fb_write(w, "{\"nsi_version\":0,\"interface\":");
    fb_named(w, n->iface.id, n->iface.name); fb_write(w, "},\"methods\":[");
    for (size_t i = 0; i < n->method_count; i++) {
        const NlNsiMethod *m = &n->methods[i];
        if (i) fb_write(w, ",");
        fb_named(w, m->id, m->name); fb_write(w, ",\"idempotent\":false,\"params\":[");
        for (size_t j = 0; j < m->param_count; j++) {
            const NlNsiParam *p = &m->params[j];
            if (j) fb_write(w, ",");
            fb_named(w, p->id, p->name); fb_write(w, ",\"type\":"); fb_quote(w, p->type_id);
            fb_write(w, ",\"direction\":"); fb_quote(w, directions[p->direction]);
            fb_write(w, ",\"ownership\":"); fb_quote(w, ownerships[p->ownership]);
            fb_write(w, ",\"lifetime\":"); fb_quote(w, lifetimes[p->lifetime]);
            fb_write(w, ",\"mutability\":"); fb_quote(w, mutability[p->mutability]);
            fb_write(w, ",\"optional\":false,\"streaming\":\"none\"}");
        }
        fb_write(w, "]}");
    }
    fb_write(w, "],\"types\":[");
    for (size_t i = 0; i < n->type_count; i++) {
        const NlNsiType *t = &n->types[i];
        if (i) fb_write(w, ",");
        fb_named(w, t->id, t->name); fb_write(w, ",\"kind\":");
        fb_quote(w, t->kind == NL_NSI_TYPE_RECORD ? "record" : t->kind == NL_NSI_TYPE_VARIANT ? "variant" : "resource");
        if (t->kind != NL_NSI_TYPE_RESOURCE) {
            fb_write(w, t->kind == NL_NSI_TYPE_RECORD ? ",\"fields\":[" : ",\"cases\":[");
            for (size_t j = 0; j < t->member_count; j++) {
                const NlNsiMember *m = &t->members[j];
                if (j) fb_write(w, ",");
                fb_named(w, m->id, m->name);
                if (m->type_id) { fb_write(w, ",\"type\":"); fb_quote(w, m->type_id); }
                fb_write(w, "}");
            }
            fb_write(w, "]");
        }
        fb_write(w, "}");
    }
    fb_write(w, "],\"errors\":[");
    fb_named(w, n->errors[0].id, n->errors[0].name);
    fb_write(w, ",\"version\":"); fb_quote(w, n->errors[0].version);
    fb_write(w, "}],\"capabilities\":[");
    fb_named(w, n->capabilities[0].id, n->capabilities[0].name);
    fb_write(w, "}]}\n");
}
/* I render the reviewed forward source surface, not an ordinary int wrapper.
 * Later paired parsing/lowering and actual shadows must qualify these bytes. */
static void fb_render_source(BindingWriter *w, const NlNsi *n) {
    fb_write(w, "# I declare catalog1; this declaration alone grants no host authority.\nservice ");
    fb_quote(w, n->iface.id);
    fb_write(w, " catalog 1 from \"interface.nsi.json\"\n\n");
    for (size_t i = 0; i < n->method_count; i++) {
        fb_write(w, "shadow "); fb_write(w, n->methods[i].name);
        fb_write(w, " {\n    let opened: OpenResult = (temp)\n    match opened {\n        Ok(file) => {\n            let mut owned: File = file\n");
        if (i == 1) {
            fb_write(w, "            let zero: WriteResult = (write_byte &mut owned 0)\n            match zero {\n                Ok(count) => { assert (== count 1) }\n                Error(error) => { assert false }\n            }\n");
        }
        if (i >= 1 && i <= 3) {
            fb_write(w, "            let written: WriteResult = (write_byte &mut owned ");
            fb_write(w, i == 3 ? "0" : "255");
            fb_write(w, ")\n            match written {\n                Ok(count) => { assert (== count 1) }\n                Error(error) => { assert false }\n            }\n");
        }
        if (i == 2 || i == 3) {
            fb_write(w, "            let positioned: PositionResult = (rewind &mut owned)\n            match positioned {\n                Ok() => {}\n                Error(error) => { assert false }\n            }\n            let read: ReadResult = (read_byte &mut owned)\n            match read {\n                Ok(byte) => {\n                    assert (== byte.value ");
            fb_write(w, i == 3 ? "0" : "255");
            fb_write(w, ")\n                    assert (not byte.eof)\n                }\n                Error(error) => { assert false }\n            }\n");
        }
        if (i == 3) {
            fb_write(w, "            let ended: ReadResult = (read_byte &mut owned)\n            match ended {\n                Ok(byte) => { assert byte.eof assert (== byte.value 0) }\n                Error(error) => { assert false }\n            }\n");
        }
        fb_write(w, "            let closed: CloseResult = (close owned)\n            match closed {\n                Ok() => {}\n                Error(error) => { assert false }\n            }\n        }\n        Error(error) => { assert false }\n    }\n}\n\n");
    }
}
NlFileBindingStatus nl_file_binding_prepare(const unsigned char *bytes, size_t size,
                                          NlFileBindingPlan **out) {
    size_t bound;
    if (!out) return NL_FILE_BINDING_INVALID;
    if (!nl_file_binding_allocation_bound(&bound)) return NL_FILE_BINDING_LIMIT;
    NlNsi *nsi = NULL;
    NlFileBindingStatus status = fb_decode(bytes, size, &nsi);
    if (status != NL_FILE_BINDING_OK) return status;
    BindingWriter json = {NULL, 0, 0, true}, source = {NULL, 0, 0, true};
    fb_render_interface(&json, nsi); fb_render_source(&source, nsi);
    size_t allocation = sizeof(NlFileBindingPlan);
    if (!json.ok || !source.ok || !fb_add(&allocation, json.size + 1) ||
        !fb_add(&allocation, source.size + 1)) { nl_nsi_free(nsi); return NL_FILE_BINDING_LIMIT; }
    NlFileBindingPlan *plan = malloc(allocation);
    if (!plan) { nl_nsi_free(nsi); return NL_FILE_BINDING_MEMORY; }
    plan->interface_size = json.size; plan->source_size = source.size;
    plan->storage_size = allocation; plan->peak_bound = bound;
    json.data = plan->bytes; json.capacity = json.size; json.size = 0;
    source.data = plan->bytes + plan->interface_size + 1;
    source.capacity = source.size; source.size = 0;
    fb_render_interface(&json, nsi); fb_render_source(&source, nsi);
    nl_nsi_free(nsi); nsi = NULL;
    if (!json.ok || !source.ok || json.size != plan->interface_size || source.size != plan->source_size) {
        free(plan); return NL_FILE_BINDING_UNRESOLVED;
    }
    json.data[json.size] = 0; source.data[source.size] = 0;
    /* I revalidate once via the lower decoder; it never calls prepare/render. */
    status = fb_decode(plan->bytes, plan->interface_size, &nsi);
    nl_nsi_free(nsi);
    if (status != NL_FILE_BINDING_OK) { free(plan); return status; }
    *out = plan;
    return NL_FILE_BINDING_OK;
}
void nl_file_binding_free(NlFileBindingPlan *plan) { free(plan); }
const unsigned char *nl_file_binding_interface_bytes(const NlFileBindingPlan *plan, size_t *size) {
    if (!plan || !size) return NULL;
    *size = plan->interface_size;
    return plan->bytes;
}
const unsigned char *nl_file_binding_source_bytes(const NlFileBindingPlan *plan, size_t *size) {
    if (!plan || !size) return NULL;
    *size = plan->source_size;
    return plan->bytes + plan->interface_size + 1;
}
size_t nl_file_binding_storage_size(const NlFileBindingPlan *plan) { return plan ? plan->storage_size : 0; }
size_t nl_file_binding_peak_bound(const NlFileBindingPlan *plan) { return plan ? plan->peak_bound : 0; }
