#include "nsi_manifest.h"

#include "cJSON.h"
#include "utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int allowed_isolation(const char *s) {
    return s && (strcmp(s, "in-process") == 0 ||
                 strcmp(s, "local-process") == 0 ||
                 strcmp(s, "mock") == 0 ||
                 strcmp(s, "remote") == 0);
}

static int allowed_restart(const char *s) {
    return s && (strcmp(s, "never") == 0 ||
                 strcmp(s, "on-failure") == 0 ||
                 strcmp(s, "always") == 0 ||
                 strcmp(s, "fail-request") == 0);
}

static int allowed_adapter(const char *s) {
    return s && (strcmp(s, "inproc") == 0 ||
                 strcmp(s, "mock") == 0 ||
                 strcmp(s, "local") == 0 ||
                 strcmp(s, "python") == 0 ||
                 strcmp(s, "remote") == 0);
}

static char *xstrdup(const char *s) {
    size_t n;
    char *d;
    if (!s) return NULL;
    n = strlen(s);
    d = malloc(n + 1);
    if (!d) return NULL;
    memcpy(d, s, n + 1);
    return d;
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

static int sibling_module_json(const char *manifest_path, char *out, size_t n) {
    size_t i;
    size_t slash = 0;
    if (!manifest_path || !out || n < 12) return 0;
    for (i = 0; manifest_path[i]; i++) {
        if (manifest_path[i] == '/') slash = i;
    }
    if (slash + 1 + 11 >= n) return 0;
    memcpy(out, manifest_path, slash + 1);
    memcpy(out + slash + 1, "module.json", 12);
    return 1;
}

void nl_nsi_manifest_free(NlNsiManifest *m) {
    size_t i;
    if (!m) return;
    free(m->interface_id);
    free(m->interface_version);
    free(m->schema_path);
    if (m->required_capabilities) {
        for (i = 0; i < m->cap_count; i++)
            free(m->required_capabilities[i]);
        free(m->required_capabilities);
    }
    free(m->isolation);
    free(m->restart);
    free(m->adapter);
    free(m->build_path);
    free(m);
}

static int keys_allowed(const cJSON *obj, const char *const *allowed, size_t n) {
    const cJSON *child;
    if (!obj || !cJSON_IsObject(obj)) return 0;
    cJSON_ArrayForEach(child, obj) {
        size_t i;
        int ok = 0;
        if (!child->string) return 0;
        for (i = 0; i < n; i++) {
            if (strcmp(child->string, allowed[i]) == 0) {
                ok = 1;
                break;
            }
        }
        if (!ok) return 0;
    }
    return 1;
}

NlNsiManifest *nl_nsi_manifest_load(const char *manifest_path) {
    static const char *const nsi_keys[] = {
        "interface_id", "interface_version", "schema",
        "required_capabilities", "isolation", "resource_budgets",
        "restart", "adapter"
    };
    static const char *const budget_keys[] = { "queue", "memory_bytes" };
    size_t n = 0;
    char *raw;
    cJSON *root;
    cJSON *nsi;
    cJSON *build_probe;
    NlNsiManifest *m;
    char build[1024];
    if (!manifest_path) return NULL;
    raw = read_file(manifest_path, &n);
    if (!raw) return NULL;
    if (!nl_utf8_validate(raw, n, NULL)) { free(raw); return NULL; }
    root = cJSON_Parse(raw);
    free(raw);
    if (!root) return NULL;
    build_probe = cJSON_GetObjectItemCaseSensitive(root, "c_sources");
    if (build_probe) {
        cJSON_Delete(root);
        return NULL;
    }
    nsi = cJSON_GetObjectItemCaseSensitive(root, "nsi");
    if (!nsi || !cJSON_IsObject(nsi) || !keys_allowed(nsi, nsi_keys, 8)) {
        cJSON_Delete(root);
        return NULL;
    }
    m = calloc(1, sizeof(*m));
    if (!m) { cJSON_Delete(root); return NULL; }
    {
        cJSON *id = cJSON_GetObjectItemCaseSensitive(nsi, "interface_id");
        cJSON *ver = cJSON_GetObjectItemCaseSensitive(nsi, "interface_version");
        cJSON *schema = cJSON_GetObjectItemCaseSensitive(nsi, "schema");
        cJSON *iso = cJSON_GetObjectItemCaseSensitive(nsi, "isolation");
        cJSON *rst = cJSON_GetObjectItemCaseSensitive(nsi, "restart");
        cJSON *ad = cJSON_GetObjectItemCaseSensitive(nsi, "adapter");
        cJSON *caps = cJSON_GetObjectItemCaseSensitive(nsi, "required_capabilities");
        cJSON *bud = cJSON_GetObjectItemCaseSensitive(nsi, "resource_budgets");
        if (!id || !cJSON_IsString(id) || !id->valuestring ||
            !ver || !cJSON_IsString(ver) || !ver->valuestring ||
            !schema || !cJSON_IsString(schema) || !schema->valuestring ||
            !iso || !cJSON_IsString(iso) || !allowed_isolation(iso->valuestring) ||
            !rst || !cJSON_IsString(rst) || !allowed_restart(rst->valuestring) ||
            !ad || !cJSON_IsString(ad) || !allowed_adapter(ad->valuestring)) {
            nl_nsi_manifest_free(m);
            cJSON_Delete(root);
            return NULL;
        }
        m->interface_id = xstrdup(id->valuestring);
        m->interface_version = xstrdup(ver->valuestring);
        m->schema_path = xstrdup(schema->valuestring);
        m->isolation = xstrdup(iso->valuestring);
        m->restart = xstrdup(rst->valuestring);
        m->adapter = xstrdup(ad->valuestring);
        if (caps) {
            int i;
            int cn;
            if (!cJSON_IsArray(caps)) {
                nl_nsi_manifest_free(m);
                cJSON_Delete(root);
                return NULL;
            }
            cn = cJSON_GetArraySize(caps);
            if (cn < 0) {
                nl_nsi_manifest_free(m);
                cJSON_Delete(root);
                return NULL;
            }
            if (cn > 0) {
                m->required_capabilities = calloc((size_t)cn, sizeof(char *));
                if (!m->required_capabilities) {
                    nl_nsi_manifest_free(m);
                    cJSON_Delete(root);
                    return NULL;
                }
                for (i = 0; i < cn; i++) {
                    cJSON *it = cJSON_GetArrayItem(caps, i);
                    if (!it || !cJSON_IsString(it) || !it->valuestring) {
                        nl_nsi_manifest_free(m);
                        cJSON_Delete(root);
                        return NULL;
                    }
                    m->required_capabilities[i] = xstrdup(it->valuestring);
                    m->cap_count++;
                }
            }
        }
        if (bud) {
            cJSON *q;
            cJSON *mem;
            if (!cJSON_IsObject(bud) || !keys_allowed(bud, budget_keys, 2)) {
                nl_nsi_manifest_free(m);
                cJSON_Delete(root);
                return NULL;
            }
            q = cJSON_GetObjectItemCaseSensitive(bud, "queue");
            mem = cJSON_GetObjectItemCaseSensitive(bud, "memory_bytes");
            if (!q || !cJSON_IsNumber(q) || !mem || !cJSON_IsNumber(mem)) {
                nl_nsi_manifest_free(m);
                cJSON_Delete(root);
                return NULL;
            }
            m->queue_budget = (int)q->valuedouble;
            m->memory_bytes = (int)mem->valuedouble;
        }
    }
    if (sibling_module_json(manifest_path, build, sizeof(build))) {
        FILE *fp = fopen(build, "rb");
        if (fp) {
            fclose(fp);
            m->build_path = xstrdup(build);
        }
    }
    cJSON_Delete(root);
    return m;
}

const char *nl_nsi_manifest_interface_id(const NlNsiManifest *m) {
    return m && m->interface_id ? m->interface_id : "";
}
