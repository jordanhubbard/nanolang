#include "nsi.h"
#include "utf8.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void free_named(NlNsiNamed *items, size_t n) {
    size_t i;
    if (!items) return;
    for (i = 0; i < n; i++) {
        free(items[i].id);
        free(items[i].name);
    }
    free(items);
}

static void free_params(NlNsiParam *items, size_t n) {
    size_t i;
    if (!items) return;
    for (i = 0; i < n; i++) {
        free(items[i].id);
        free(items[i].name);
        free(items[i].type_id);
    }
    free(items);
}

static void free_methods(NlNsiMethod *items, size_t n) {
    size_t i;
    if (!items) return;
    for (i = 0; i < n; i++) {
        free(items[i].id);
        free(items[i].name);
        free_params(items[i].params, items[i].param_count);
    }
    free(items);
}

static void free_members(NlNsiMember *items, size_t n) {
    size_t i;
    if (!items) return;
    for (i = 0; i < n; i++) {
        free(items[i].id);
        free(items[i].name);
        free(items[i].type_id);
    }
    free(items);
}

static void free_types(NlNsiType *items, size_t n) {
    size_t i;
    if (!items) return;
    for (i = 0; i < n; i++) {
        free(items[i].id);
        free(items[i].name);
        free_members(items[i].members, items[i].member_count);
        free(items[i].element_id);
        free(items[i].method_id);
        free(items[i].result_id);
    }
    free(items);
}

static void free_errors(NlNsiError *items, size_t n) {
    size_t i;
    if (!items) return;
    for (i = 0; i < n; i++) {
        free(items[i].id);
        free(items[i].name);
        free(items[i].version);
    }
    free(items);
}

void nl_nsi_free(NlNsi *nsi) {
    if (!nsi) return;
    free(nsi->iface.id);
    free(nsi->iface.name);
    free_methods(nsi->methods, nsi->method_count);
    free_types(nsi->types, nsi->type_count);
    free_errors(nsi->errors, nsi->error_count);
    free_named(nsi->capabilities, nsi->capability_count);
    free(nsi);
}

static int id_char_ok(unsigned char c) {
    return nl_ascii_isalnum((int)c) || c == ':' || c == '/' || c == '.' ||
           c == '_' || c == '-' || c == '#';
}

static bool id_ascii_ok(const char *id) {
    size_t i;
    if (!id || !id[0]) return false;
    for (i = 0; id[i]; i++) {
        if (!id_char_ok((unsigned char)id[i])) return false;
    }
    return true;
}

static bool starts_with(const char *s, const char *pfx) {
    return s && pfx && strncmp(s, pfx, strlen(pfx)) == 0;
}

static bool parse_named(cJSON *obj, NlNsiNamed *out) {
    cJSON *id;
    cJSON *name;
    if (!obj || !cJSON_IsObject(obj) || !out) return false;
    id = cJSON_GetObjectItemCaseSensitive(obj, "id");
    name = cJSON_GetObjectItemCaseSensitive(obj, "name");
    if (!cJSON_IsString(id) || !cJSON_IsString(name)) return false;
    if (!id_ascii_ok(id->valuestring)) return false;
    if (!name->valuestring || !name->valuestring[0]) return false;
    out->id = strdup(id->valuestring);
    out->name = strdup(name->valuestring);
    return out->id && out->name;
}

static bool parse_named_array(cJSON *arr, NlNsiNamed **out, size_t *count) {
    int n;
    int i;
    if (!arr) {
        *out = NULL;
        *count = 0;
        return true;
    }
    if (!cJSON_IsArray(arr)) return false;
    n = cJSON_GetArraySize(arr);
    if (n < 0) return false;
    if (n == 0) {
        *out = NULL;
        *count = 0;
        return true;
    }
    *out = calloc((size_t)n, sizeof(**out));
    if (!*out) return false;
    *count = 0;
    for (i = 0; i < n; i++) {
        if (!parse_named(cJSON_GetArrayItem(arr, i), &(*out)[i])) {
            free_named(*out, (size_t)i);
            *out = NULL;
            *count = 0;
            return false;
        }
        (*count)++;
    }
    return true;
}

static bool parse_enum(cJSON *obj, const char *key, const char *const *names, int n, int *out) {
    cJSON *item;
    int i;
    if (!obj || !key || !names || !out || n <= 0) return false;
    item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!cJSON_IsString(item) || !item->valuestring) return false;
    for (i = 0; i < n; i++) {
        if (strcmp(item->valuestring, names[i]) == 0) {
            *out = i;
            return true;
        }
    }
    return false;
}

static bool streaming_matches_direction(NlNsiDirection dir, NlNsiStreaming stream) {
    if (stream == NL_NSI_STREAM_NONE) return true;
    if (stream == NL_NSI_STREAM_IN)
        return dir == NL_NSI_DIR_IN || dir == NL_NSI_DIR_INOUT;
    if (stream == NL_NSI_STREAM_OUT)
        return dir == NL_NSI_DIR_OUT || dir == NL_NSI_DIR_INOUT ||
               dir == NL_NSI_DIR_RETURN;
    if (stream == NL_NSI_STREAM_BIDI) return dir == NL_NSI_DIR_INOUT;
    return false;
}

static const char *const k_direction[] = { "in", "out", "inout", "return" };
static const char *const k_ownership[] = { "borrow", "transfer", "copy" };
static const char *const k_lifetime[] = { "call", "caller", "callee", "resource" };
static const char *const k_mutability[] = { "immutable", "mutable" };
static const char *const k_streaming[] = { "none", "in", "out", "bidi" };

static bool parse_param(cJSON *obj, NlNsiParam *out) {
    NlNsiNamed named = {0};
    cJSON *optional;
    cJSON *type;
    int direction = 0;
    int ownership = 0;
    int lifetime = 0;
    int mutability = 0;
    int streaming = 0;

    if (!parse_named(obj, &named)) return false;
    out->id = named.id;
    out->name = named.name;
    type = cJSON_GetObjectItemCaseSensitive(obj, "type");
    if (!cJSON_IsString(type) || !id_ascii_ok(type->valuestring)) {
        free(out->id);
        free(out->name);
        out->id = NULL;
        out->name = NULL;
        return false;
    }
    if (!starts_with(type->valuestring, "nsi:")) {
        free(out->id);
        free(out->name);
        out->id = NULL;
        out->name = NULL;
        return false;
    }
    if (!parse_enum(obj, "direction", k_direction, 4, &direction) ||
        !parse_enum(obj, "ownership", k_ownership, 3, &ownership) ||
        !parse_enum(obj, "lifetime", k_lifetime, 4, &lifetime) ||
        !parse_enum(obj, "mutability", k_mutability, 2, &mutability) ||
        !parse_enum(obj, "streaming", k_streaming, 4, &streaming)) {
        free(out->id);
        free(out->name);
        out->id = NULL;
        out->name = NULL;
        return false;
    }
    optional = cJSON_GetObjectItemCaseSensitive(obj, "optional");
    if (!cJSON_IsBool(optional)) {
        free(out->id);
        free(out->name);
        out->id = NULL;
        out->name = NULL;
        return false;
    }
    out->type_id = strdup(type->valuestring);
    if (!out->type_id) {
        free(out->id);
        free(out->name);
        out->id = NULL;
        out->name = NULL;
        return false;
    }
    out->direction = (NlNsiDirection)direction;
    out->ownership = (NlNsiOwnership)ownership;
    out->lifetime = (NlNsiLifetime)lifetime;
    out->mutability = (NlNsiMutability)mutability;
    out->optional = cJSON_IsTrue(optional) ? 1 : 0;
    out->streaming = (NlNsiStreaming)streaming;
    if (!streaming_matches_direction(out->direction, out->streaming)) {
        free(out->id);
        free(out->name);
        free(out->type_id);
        out->id = NULL;
        out->name = NULL;
        out->type_id = NULL;
        return false;
    }
    return true;
}

static bool parse_param_array(cJSON *arr, NlNsiParam **out, size_t *count) {
    int n;
    int i;
    if (!arr) {
        *out = NULL;
        *count = 0;
        return true;
    }
    if (!cJSON_IsArray(arr)) return false;
    n = cJSON_GetArraySize(arr);
    if (n < 0) return false;
    if (n == 0) {
        *out = NULL;
        *count = 0;
        return true;
    }
    *out = calloc((size_t)n, sizeof(**out));
    if (!*out) return false;
    *count = 0;
    for (i = 0; i < n; i++) {
        if (!parse_param(cJSON_GetArrayItem(arr, i), &(*out)[i])) {
            free_params(*out, (size_t)i);
            *out = NULL;
            *count = 0;
            return false;
        }
        (*count)++;
    }
    return true;
}

static bool parse_method(cJSON *obj, NlNsiMethod *out) {
    NlNsiNamed named = {0};
    if (!parse_named(obj, &named)) return false;
    out->id = named.id;
    out->name = named.name;
    out->params = NULL;
    out->param_count = 0;
    if (!parse_param_array(cJSON_GetObjectItemCaseSensitive(obj, "params"),
                           &out->params, &out->param_count)) {
        free(out->id);
        free(out->name);
        out->id = NULL;
        out->name = NULL;
        return false;
    }
    return true;
}

static bool parse_method_array(cJSON *arr, NlNsiMethod **out, size_t *count) {
    int n;
    int i;
    if (!arr) {
        *out = NULL;
        *count = 0;
        return true;
    }
    if (!cJSON_IsArray(arr)) return false;
    n = cJSON_GetArraySize(arr);
    if (n < 0) return false;
    if (n == 0) {
        *out = NULL;
        *count = 0;
        return true;
    }
    *out = calloc((size_t)n, sizeof(**out));
    if (!*out) return false;
    *count = 0;
    for (i = 0; i < n; i++) {
        if (!parse_method(cJSON_GetArrayItem(arr, i), &(*out)[i])) {
            free_methods(*out, (size_t)i);
            *out = NULL;
            *count = 0;
            return false;
        }
        (*count)++;
    }
    return true;
}

static bool type_ref_ok(const char *id) {
    return id_ascii_ok(id) && starts_with(id, "nsi:");
}

static bool version_ok(const char *v) {
    size_t i;
    if (!v || !v[0]) return false;
    for (i = 0; v[i]; i++) {
        unsigned char c = (unsigned char)v[i];
        if (!nl_ascii_isalnum((int)c) && c != '.' && c != '_' && c != '-')
            return false;
    }
    return true;
}

static bool parse_member(cJSON *obj, NlNsiMember *out, int require_type) {
    NlNsiNamed named = {0};
    cJSON *type;
    if (!parse_named(obj, &named)) return false;
    out->id = named.id;
    out->name = named.name;
    out->type_id = NULL;
    type = cJSON_GetObjectItemCaseSensitive(obj, "type");
    if (type) {
        if (!cJSON_IsString(type) || !type_ref_ok(type->valuestring)) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
        out->type_id = strdup(type->valuestring);
        if (!out->type_id) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
    } else if (require_type) {
        free(out->id);
        free(out->name);
        out->id = NULL;
        out->name = NULL;
        return false;
    }
    return true;
}

static bool parse_member_array(cJSON *arr, NlNsiMember **out, size_t *count, int require_type) {
    int n;
    int i;
    if (!arr || !cJSON_IsArray(arr)) return false;
    n = cJSON_GetArraySize(arr);
    if (n < 0) return false;
    if (n == 0) {
        *out = NULL;
        *count = 0;
        return true;
    }
    *out = calloc((size_t)n, sizeof(**out));
    if (!*out) return false;
    *count = 0;
    for (i = 0; i < n; i++) {
        if (!parse_member(cJSON_GetArrayItem(arr, i), &(*out)[i], require_type)) {
            free_members(*out, (size_t)i);
            *out = NULL;
            *count = 0;
            return false;
        }
        (*count)++;
    }
    return true;
}

static const char *const k_type_kind[] = {
    "opaque", "record", "variant", "array", "string", "binary",
    "resource", "callback", "async"
};

static bool parse_type(cJSON *obj, NlNsiType *out) {
    NlNsiNamed named = {0};
    cJSON *kind_item;
    int kind = NL_NSI_TYPE_OPAQUE;

    if (!parse_named(obj, &named)) return false;
    out->id = named.id;
    out->name = named.name;
    out->members = NULL;
    out->member_count = 0;
    out->element_id = NULL;
    out->method_id = NULL;
    out->result_id = NULL;
    kind_item = cJSON_GetObjectItemCaseSensitive(obj, "kind");
    if (kind_item) {
        if (!parse_enum(obj, "kind", k_type_kind, 9, &kind)) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
    }
    out->kind = (NlNsiTypeKind)kind;
    if (out->kind == NL_NSI_TYPE_RECORD) {
        if (!parse_member_array(cJSON_GetObjectItemCaseSensitive(obj, "fields"),
                                &out->members, &out->member_count, 1)) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
    } else if (out->kind == NL_NSI_TYPE_VARIANT) {
        if (!parse_member_array(cJSON_GetObjectItemCaseSensitive(obj, "cases"),
                                &out->members, &out->member_count, 0) ||
            out->member_count == 0) {
            free(out->id);
            free(out->name);
            free_members(out->members, out->member_count);
            out->id = NULL;
            out->name = NULL;
            out->members = NULL;
            out->member_count = 0;
            return false;
        }
    } else if (out->kind == NL_NSI_TYPE_ARRAY) {
        cJSON *el = cJSON_GetObjectItemCaseSensitive(obj, "element");
        if (!cJSON_IsString(el) || !type_ref_ok(el->valuestring)) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
        out->element_id = strdup(el->valuestring);
        if (!out->element_id) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
    } else if (out->kind == NL_NSI_TYPE_CALLBACK) {
        cJSON *m = cJSON_GetObjectItemCaseSensitive(obj, "method");
        if (!cJSON_IsString(m) || !id_ascii_ok(m->valuestring)) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
        out->method_id = strdup(m->valuestring);
        if (!out->method_id) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
    } else if (out->kind == NL_NSI_TYPE_ASYNC) {
        cJSON *r = cJSON_GetObjectItemCaseSensitive(obj, "result");
        if (!cJSON_IsString(r) || !type_ref_ok(r->valuestring)) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
        out->result_id = strdup(r->valuestring);
        if (!out->result_id) {
            free(out->id);
            free(out->name);
            out->id = NULL;
            out->name = NULL;
            return false;
        }
    }
    return true;
}

static bool parse_type_array(cJSON *arr, NlNsiType **out, size_t *count) {
    int n;
    int i;
    if (!arr) {
        *out = NULL;
        *count = 0;
        return true;
    }
    if (!cJSON_IsArray(arr)) return false;
    n = cJSON_GetArraySize(arr);
    if (n < 0) return false;
    if (n == 0) {
        *out = NULL;
        *count = 0;
        return true;
    }
    *out = calloc((size_t)n, sizeof(**out));
    if (!*out) return false;
    *count = 0;
    for (i = 0; i < n; i++) {
        if (!parse_type(cJSON_GetArrayItem(arr, i), &(*out)[i])) {
            free_types(*out, (size_t)i);
            *out = NULL;
            *count = 0;
            return false;
        }
        (*count)++;
    }
    return true;
}

static bool parse_error(cJSON *obj, NlNsiError *out) {
    NlNsiNamed named = {0};
    cJSON *ver;
    if (!parse_named(obj, &named)) return false;
    out->id = named.id;
    out->name = named.name;
    out->version = NULL;
    ver = cJSON_GetObjectItemCaseSensitive(obj, "version");
    if (!ver) return true;
    if (!cJSON_IsString(ver) || !version_ok(ver->valuestring)) {
        free(out->id);
        free(out->name);
        out->id = NULL;
        out->name = NULL;
        return false;
    }
    out->version = strdup(ver->valuestring);
    if (!out->version) {
        free(out->id);
        free(out->name);
        out->id = NULL;
        out->name = NULL;
        return false;
    }
    return true;
}

static bool parse_error_array(cJSON *arr, NlNsiError **out, size_t *count) {
    int n;
    int i;
    if (!arr) {
        *out = NULL;
        *count = 0;
        return true;
    }
    if (!cJSON_IsArray(arr)) return false;
    n = cJSON_GetArraySize(arr);
    if (n < 0) return false;
    if (n == 0) {
        *out = NULL;
        *count = 0;
        return true;
    }
    *out = calloc((size_t)n, sizeof(**out));
    if (!*out) return false;
    *count = 0;
    for (i = 0; i < n; i++) {
        if (!parse_error(cJSON_GetArrayItem(arr, i), &(*out)[i])) {
            free_errors(*out, (size_t)i);
            *out = NULL;
            *count = 0;
            return false;
        }
        (*count)++;
    }
    return true;
}

static bool ids_unique(const NlNsi *nsi) {
    size_t n = 1u;
    size_t i;
    size_t j;
    size_t k = 0;
    const char **ids;
    bool ok = true;

    n += nsi->method_count + nsi->type_count + nsi->error_count +
         nsi->capability_count;
    for (i = 0; i < nsi->method_count; i++) n += nsi->methods[i].param_count;
    for (i = 0; i < nsi->type_count; i++) n += nsi->types[i].member_count;
    ids = malloc(n * sizeof(*ids));
    if (!ids) return false;
    ids[k++] = nsi->iface.id;
    for (i = 0; i < nsi->method_count; i++) {
        ids[k++] = nsi->methods[i].id;
        for (j = 0; j < nsi->methods[i].param_count; j++)
            ids[k++] = nsi->methods[i].params[j].id;
    }
    for (i = 0; i < nsi->type_count; i++) {
        ids[k++] = nsi->types[i].id;
        for (j = 0; j < nsi->types[i].member_count; j++)
            ids[k++] = nsi->types[i].members[j].id;
    }
    for (i = 0; i < nsi->error_count; i++) ids[k++] = nsi->errors[i].id;
    for (i = 0; i < nsi->capability_count; i++) ids[k++] = nsi->capabilities[i].id;
    for (i = 0; i < k && ok; i++) {
        for (j = i + 1; j < k; j++) {
            if (strcmp(ids[i], ids[j]) == 0) {
                ok = false;
                break;
            }
        }
    }
    free(ids);
    return ok;
}

static bool fragment_of_interface(const char *iface, const char *id) {
    size_t n;
    if (!iface || !id) return false;
    n = strlen(iface);
    if (strncmp(id, iface, n) != 0) return false;
    return id[n] == '#' && id[n + 1] != '\0';
}

static bool validate_prefixes(const NlNsi *nsi) {
    size_t i;
    size_t j;
    if (!starts_with(nsi->iface.id, "nsi:")) return false;
    if (strchr(nsi->iface.id, '#') != NULL) return false;
    for (i = 0; i < nsi->method_count; i++) {
        if (!fragment_of_interface(nsi->iface.id, nsi->methods[i].id)) return false;
        for (j = 0; j < nsi->methods[i].param_count; j++) {
            if (!fragment_of_interface(nsi->iface.id, nsi->methods[i].params[j].id))
                return false;
        }
    }
    for (i = 0; i < nsi->type_count; i++) {
        if (!fragment_of_interface(nsi->iface.id, nsi->types[i].id)) return false;
        for (j = 0; j < nsi->types[i].member_count; j++) {
            if (!fragment_of_interface(nsi->iface.id, nsi->types[i].members[j].id))
                return false;
        }
        if (nsi->types[i].kind == NL_NSI_TYPE_CALLBACK) {
            int found = 0;
            size_t m;
            for (m = 0; m < nsi->method_count; m++) {
                if (nsi->types[i].method_id &&
                    strcmp(nsi->types[i].method_id, nsi->methods[m].id) == 0) {
                    found = 1;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    for (i = 0; i < nsi->error_count; i++) {
        if (!fragment_of_interface(nsi->iface.id, nsi->errors[i].id)) return false;
    }
    for (i = 0; i < nsi->capability_count; i++) {
        if (!starts_with(nsi->capabilities[i].id, "cap:")) return false;
        if (strchr(nsi->capabilities[i].id, '#') != NULL) return false;
    }
    return true;
}

NlNsi *nl_nsi_load_path(const char *path) {
    FILE *fp;
    long size;
    char *buf;
    cJSON *json;
    cJSON *ver;
    NlNsi *nsi;

    if (!path) return NULL;
    fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    size = ftell(fp);
    if (size < 0) { fclose(fp); return NULL; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return NULL; }
    buf = malloc((size_t)size + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)size, fp) != (size_t)size) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    buf[size] = '\0';
    fclose(fp);

    if (!nl_utf8_validate(buf, (size_t)size, NULL)) {
        free(buf);
        return NULL;
    }

    json = cJSON_Parse(buf);
    free(buf);
    if (!json || !cJSON_IsObject(json)) {
        if (json) cJSON_Delete(json);
        return NULL;
    }

    nsi = calloc(1, sizeof(*nsi));
    if (!nsi) {
        cJSON_Delete(json);
        return NULL;
    }

    ver = cJSON_GetObjectItemCaseSensitive(json, "nsi_version");
    if (!cJSON_IsNumber(ver) || ver->valuedouble != (double)NL_NSI_VERSION) {
        nl_nsi_free(nsi);
        cJSON_Delete(json);
        return NULL;
    }
    nsi->version = NL_NSI_VERSION;

    if (!parse_named(cJSON_GetObjectItemCaseSensitive(json, "interface"), &nsi->iface) ||
        !parse_method_array(cJSON_GetObjectItemCaseSensitive(json, "methods"),
                            &nsi->methods, &nsi->method_count) ||
        !parse_type_array(cJSON_GetObjectItemCaseSensitive(json, "types"),
                           &nsi->types, &nsi->type_count) ||
        !parse_error_array(cJSON_GetObjectItemCaseSensitive(json, "errors"),
                           &nsi->errors, &nsi->error_count) ||
        !parse_named_array(cJSON_GetObjectItemCaseSensitive(json, "capabilities"),
                           &nsi->capabilities, &nsi->capability_count) ||
        !ids_unique(nsi) ||
        !validate_prefixes(nsi)) {
        nl_nsi_free(nsi);
        cJSON_Delete(json);
        return NULL;
    }

    cJSON_Delete(json);
    return nsi;
}

const char *nl_nsi_interface_id(const NlNsi *nsi) {
    return nsi && nsi->iface.id ? nsi->iface.id : "";
}

size_t nl_nsi_method_count(const NlNsi *nsi) {
    return nsi ? nsi->method_count : 0;
}

const char *nl_nsi_method_id(const NlNsi *nsi, size_t i) {
    if (!nsi || i >= nsi->method_count) return "";
    return nsi->methods[i].id ? nsi->methods[i].id : "";
}

size_t nl_nsi_param_count(const NlNsi *nsi, size_t method_i) {
    if (!nsi || method_i >= nsi->method_count) return 0;
    return nsi->methods[method_i].param_count;
}

const NlNsiParam *nl_nsi_param(const NlNsi *nsi, size_t method_i, size_t param_i) {
    if (!nsi || method_i >= nsi->method_count) return NULL;
    if (param_i >= nsi->methods[method_i].param_count) return NULL;
    return &nsi->methods[method_i].params[param_i];
}
