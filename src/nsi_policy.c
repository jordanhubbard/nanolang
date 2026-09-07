#include "nsi_policy.h"
#include "nsi_cap.h"
#include "cJSON.h"
#include "utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void bounded_copy(char *dest, size_t dest_size, const char *src) {
    size_t n;
    if (!dest || dest_size == 0) return;
    if (!src) { dest[0] = 0; return; }
    n = strlen(src);
    if (n >= dest_size) n = dest_size - 1;
    memcpy(dest, src, n);
    dest[n] = 0;
}

static char *read_file(const char *path, size_t *out_n) {
    FILE *fp;
    long sz;
    char *buf;
    size_t n;
    if (out_n) *out_n = 0;
    fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    sz = ftell(fp);
    if (sz < 0) { fclose(fp); return NULL; }
    rewind(fp);
    buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    n = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[n] = '\0';
    if (out_n) *out_n = n;
    return buf;
}

static uint32_t parse_right(const char *s) {
    if (!s) return 0;
    if (strcmp(s, "READ") == 0) return NL_CAP_READ;
    if (strcmp(s, "WRITE") == 0) return NL_CAP_WRITE;
    if (strcmp(s, "MAP") == 0) return NL_CAP_MAP;
    if (strcmp(s, "SEAL") == 0) return NL_CAP_SEAL;
    if (strcmp(s, "TRANSFER") == 0) return NL_CAP_TRANSFER;
    if (strcmp(s, "EVAL") == 0) return NL_CAP_EVAL;
    return 0;
}

static int add_unique(char names[][NL_POLICY_MAX_NAME], int *n, int maxn,
                      size_t elemsz, const char *s) {
    int i;
    if (!s || !s[0]) return 1;
    for (i = 0; i < *n; i++) {
        if (strcmp(names[i], s) == 0) return 1;
    }
    if (*n >= maxn) return 0;
    bounded_copy(names[*n], elemsz, s);
    (*n)++;
    return 1;
}

static int add_unique16(char names[][16], int *n, int maxn, const char *s) {
    int i;
    if (!s || !s[0]) return 1;
    for (i = 0; i < *n; i++) {
        if (strcmp(names[i], s) == 0) return 1;
    }
    if (*n >= maxn) return 0;
    bounded_copy(names[*n], 16, s);
    (*n)++;
    return 1;
}

static int add_unique32(char names[][32], int *n, int maxn, const char *s) {
    int i;
    if (!s || !s[0]) return 1;
    for (i = 0; i < *n; i++) {
        if (strcmp(names[i], s) == 0) return 1;
    }
    if (*n >= maxn) return 0;
    bounded_copy(names[*n], 32, s);
    (*n)++;
    return 1;
}

static int add_unique64(char names[][64], int *n, int maxn, const char *s) {
    int i;
    if (!s || !s[0]) return 1;
    for (i = 0; i < *n; i++) {
        if (strcmp(names[i], s) == 0) return 1;
    }
    if (*n >= maxn) return 0;
    bounded_copy(names[*n], 64, s);
    (*n)++;
    return 1;
}

NlEffectMap *nl_effect_map_load(const char *path) {
    size_t n = 0;
    char *raw;
    cJSON *root;
    cJSON *layers;
    cJSON *map;
    NlEffectMap *m;
    int i;
    if (!path) return NULL;
    raw = read_file(path, &n);
    if (!raw) return NULL;
    if (!nl_utf8_validate(raw, n, NULL)) { free(raw); return NULL; }
    root = cJSON_Parse(raw);
    free(raw);
    if (!root) return NULL;
    m = calloc(1, sizeof(*m));
    if (!m) { cJSON_Delete(root); return NULL; }
    layers = cJSON_GetObjectItemCaseSensitive(root, "layers");
    map = cJSON_GetObjectItemCaseSensitive(root, "map");
    if (!cJSON_IsArray(layers) || !cJSON_IsArray(map)) {
        free(m);
        cJSON_Delete(root);
        return NULL;
    }
    for (i = 0; i < cJSON_GetArraySize(layers) && i < 8; i++) {
        cJSON *it = cJSON_GetArrayItem(layers, i);
        if (!cJSON_IsString(it) || !it->valuestring) {
            free(m);
            cJSON_Delete(root);
            return NULL;
        }
        bounded_copy(m->layers[m->layer_n], sizeof(m->layers[0]), it->valuestring);
        m->layer_n++;
    }
    for (i = 0; i < cJSON_GetArraySize(map) && i < NL_POLICY_MAX_ROWS; i++) {
        cJSON *row = cJSON_GetArrayItem(map, i);
        cJSON *effect, *op, *trap, *method, *cap, *rights, *r;
        NlEffectMapRow *dst;
        if (!cJSON_IsObject(row)) {
            free(m);
            cJSON_Delete(root);
            return NULL;
        }
        effect = cJSON_GetObjectItemCaseSensitive(row, "effect");
        op = cJSON_GetObjectItemCaseSensitive(row, "op");
        trap = cJSON_GetObjectItemCaseSensitive(row, "trap");
        method = cJSON_GetObjectItemCaseSensitive(row, "method");
        cap = cJSON_GetObjectItemCaseSensitive(row, "capability");
        rights = cJSON_GetObjectItemCaseSensitive(row, "rights");
        if (!cJSON_IsString(effect) || !cJSON_IsString(op) ||
            !cJSON_IsString(trap) || !cJSON_IsString(method) ||
            !cJSON_IsString(cap) || !cJSON_IsArray(rights)) {
            free(m);
            cJSON_Delete(root);
            return NULL;
        }
        dst = &m->rows[m->row_n];
        bounded_copy(dst->effect, sizeof(dst->effect), effect->valuestring);
        bounded_copy(dst->op, sizeof(dst->op), op->valuestring);
        bounded_copy(dst->trap, sizeof(dst->trap), trap->valuestring);
        bounded_copy(dst->method, sizeof(dst->method), method->valuestring);
        bounded_copy(dst->capability, sizeof(dst->capability), cap->valuestring);
        dst->rights = 0;
        cJSON_ArrayForEach(r, rights) {
            if (!cJSON_IsString(r) || !r->valuestring) {
                free(m);
                cJSON_Delete(root);
                return NULL;
            }
            dst->rights |= parse_right(r->valuestring);
        }
        m->row_n++;
    }
    cJSON_Delete(root);
    return m;
}

void nl_effect_map_free(NlEffectMap *m) {
    free(m);
}

int nl_effect_map_has_layer(const NlEffectMap *m, const char *layer) {
    int i;
    if (!m || !layer) return 0;
    for (i = 0; i < m->layer_n; i++) {
        if (strcmp(m->layers[i], layer) == 0) return 1;
    }
    return 0;
}

int nl_effect_inventory_from_rows(const NlEffectMap *map,
                                  const char **effects, int n,
                                  NlEffectInventory *out) {
    int i, j;
    if (!map || !out) return NL_POLICY_ERR;
    if (n < 0 || (n > 0 && !effects)) return NL_POLICY_ERR;
    memset(out, 0, sizeof(*out));
    for (i = 0; i < n; i++) {
        if (!effects[i]) return NL_POLICY_ERR;
        if (!add_unique16(out->effects, &out->effect_n, NL_POLICY_MAX_ROWS, effects[i]))
            return NL_POLICY_ERR;
        for (j = 0; j < map->row_n; j++) {
            if (strcmp(map->rows[j].effect, effects[i]) != 0) continue;
            if (!add_unique32(out->traps, &out->trap_n, NL_POLICY_MAX_ROWS, map->rows[j].trap))
                return NL_POLICY_ERR;
            if (!add_unique(out->methods, &out->method_n, NL_POLICY_MAX_ROWS,
                            sizeof(out->methods[0]), map->rows[j].method))
                return NL_POLICY_ERR;
            if (!add_unique64(out->cap_ids, &out->cap_n, NL_POLICY_MAX_ROWS,
                              map->rows[j].capability))
                return NL_POLICY_ERR;
            out->required_rights |= map->rows[j].rights;
        }
    }
    return NL_POLICY_OK;
}

static void add_string_array(cJSON *obj, const char *key,
                             const char names[][NL_POLICY_MAX_NAME],
                             int n, size_t elemsz) {
    cJSON *arr = cJSON_CreateArray();
    int i;
    (void)elemsz;
    for (i = 0; i < n; i++)
        cJSON_AddItemToArray(arr, cJSON_CreateString(names[i]));
    cJSON_AddItemToObject(obj, key, arr);
}

char *nl_effect_inventory_json(const NlEffectInventory *inv) {
    cJSON *obj;
    char *out;
    int i;
    if (!inv) return NULL;
    obj = cJSON_CreateObject();
    if (!obj) return NULL;
    cJSON_AddNumberToObject(obj, "nsi_version", 0);
    {
        cJSON *arr = cJSON_CreateArray();
        for (i = 0; i < inv->effect_n; i++)
            cJSON_AddItemToArray(arr, cJSON_CreateString(inv->effects[i]));
        cJSON_AddItemToObject(obj, "effects", arr);
    }
    {
        cJSON *arr = cJSON_CreateArray();
        for (i = 0; i < inv->trap_n; i++)
            cJSON_AddItemToArray(arr, cJSON_CreateString(inv->traps[i]));
        cJSON_AddItemToObject(obj, "traps", arr);
    }
    add_string_array(obj, "methods", inv->methods, inv->method_n,
                     sizeof(inv->methods[0]));
    {
        cJSON *arr = cJSON_CreateArray();
        for (i = 0; i < inv->cap_n; i++)
            cJSON_AddItemToArray(arr, cJSON_CreateString(inv->cap_ids[i]));
        cJSON_AddItemToObject(obj, "capabilities", arr);
    }
    cJSON_AddNumberToObject(obj, "required_rights", inv->required_rights);
    out = cJSON_PrintUnformatted(obj);
    cJSON_Delete(obj);
    return out;
}

static int cap_granted(const char **granted, int n, const char *need) {
    int i;
    if (!need || !need[0]) return 1;
    for (i = 0; i < n; i++) {
        if (granted[i] && strcmp(granted[i], need) == 0) return 1;
    }
    return 0;
}

int nl_deploy_unused_count(const NlEffectInventory *inv,
                           const char **granted_caps, int granted_n) {
    int i, unused = 0;
    if (!inv) return -1;
    for (i = 0; i < granted_n; i++) {
        int j, found = 0;
        if (!granted_caps || !granted_caps[i] || !granted_caps[i][0]) continue;
        for (j = 0; j < inv->cap_n; j++) {
            if (strcmp(inv->cap_ids[j], granted_caps[i]) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) unused++;
    }
    return unused;
}

int nl_deploy_check(const NlEffectInventory *inv, uint32_t granted_rights,
                    const char **granted_caps, int granted_n, int override) {
    int i;
    if (!inv) return NL_POLICY_ERR;
    if ((inv->required_rights & granted_rights) != inv->required_rights) {
        return override ? NL_POLICY_OK : NL_POLICY_ERR_UNCOVERED;
    }
    for (i = 0; i < inv->cap_n; i++) {
        if (!cap_granted(granted_caps, granted_n, inv->cap_ids[i]))
            return override ? NL_POLICY_OK : NL_POLICY_ERR_UNCOVERED;
    }
    return NL_POLICY_OK;
}

char *nl_deploy_manifest_json(const NlEffectInventory *inv,
                              uint32_t granted_rights,
                              const char **granted_caps, int granted_n,
                              int override, const char *override_reason) {
    cJSON *obj;
    char *invj;
    char *out;
    cJSON *inv_parsed;
    int i;
    int unused;
    int check;
    if (!inv) return NULL;
    unused = nl_deploy_unused_count(inv, granted_caps, granted_n);
    check = nl_deploy_check(inv, granted_rights, granted_caps, granted_n, override);
    obj = cJSON_CreateObject();
    if (!obj) return NULL;
    cJSON_AddNumberToObject(obj, "nsi_version", 0);
    invj = nl_effect_inventory_json(inv);
    inv_parsed = invj ? cJSON_Parse(invj) : NULL;
    free(invj);
    if (inv_parsed)
        cJSON_AddItemToObject(obj, "inventory", inv_parsed);
    cJSON_AddNumberToObject(obj, "granted_rights", granted_rights);
    {
        cJSON *arr = cJSON_CreateArray();
        for (i = 0; i < granted_n; i++) {
            if (granted_caps && granted_caps[i])
                cJSON_AddItemToArray(arr, cJSON_CreateString(granted_caps[i]));
        }
        cJSON_AddItemToObject(obj, "granted_capabilities", arr);
    }
    cJSON_AddNumberToObject(obj, "unused_grants", unused);
    cJSON_AddBoolToObject(obj, "override", override ? 1 : 0);
    cJSON_AddStringToObject(obj, "override_reason",
                            override_reason ? override_reason : "");
    cJSON_AddBoolToObject(obj, "source_declarations_widened", 0);
    cJSON_AddBoolToObject(obj, "accepted", check == NL_POLICY_OK ? 1 : 0);
    out = cJSON_PrintUnformatted(obj);
    cJSON_Delete(obj);
    return out;
}

int nl_deploy_source_widened(const NlEffectInventory *before,
                             const NlEffectInventory *after) {
    int i;
    if (!before || !after) return 1;
    if (after->effect_n != before->effect_n) return 1;
    for (i = 0; i < before->effect_n; i++) {
        if (strcmp(before->effects[i], after->effects[i]) != 0) return 1;
    }
    return 0;
}
