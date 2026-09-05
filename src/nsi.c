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

void nl_nsi_free(NlNsi *nsi) {
    if (!nsi) return;
    free(nsi->iface.id);
    free(nsi->iface.name);
    free_named(nsi->methods, nsi->method_count);
    free_named(nsi->types, nsi->type_count);
    free_named(nsi->errors, nsi->error_count);
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

static bool ids_unique(const NlNsi *nsi) {
    size_t n;
    size_t i;
    size_t j;
    size_t k = 0;
    const char **ids;
    bool ok = true;

    n = 1u + nsi->method_count + nsi->type_count + nsi->error_count +
        nsi->capability_count;
    ids = malloc(n * sizeof(*ids));
    if (!ids) return false;
    ids[k++] = nsi->iface.id;
    for (i = 0; i < nsi->method_count; i++) ids[k++] = nsi->methods[i].id;
    for (i = 0; i < nsi->type_count; i++) ids[k++] = nsi->types[i].id;
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
    if (!starts_with(nsi->iface.id, "nsi:")) return false;
    if (strchr(nsi->iface.id, '#') != NULL) return false;
    for (i = 0; i < nsi->method_count; i++) {
        if (!fragment_of_interface(nsi->iface.id, nsi->methods[i].id)) return false;
    }
    for (i = 0; i < nsi->type_count; i++) {
        if (!fragment_of_interface(nsi->iface.id, nsi->types[i].id)) return false;
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
        !parse_named_array(cJSON_GetObjectItemCaseSensitive(json, "methods"),
                           &nsi->methods, &nsi->method_count) ||
        !parse_named_array(cJSON_GetObjectItemCaseSensitive(json, "types"),
                           &nsi->types, &nsi->type_count) ||
        !parse_named_array(cJSON_GetObjectItemCaseSensitive(json, "errors"),
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
