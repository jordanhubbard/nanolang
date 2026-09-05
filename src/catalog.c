#define _POSIX_C_SOURCE 200809L

#include "catalog.h"
#include "diag_id.h"
#include "utf8.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *id;
    char *text;
} NlCatalogEntry;

static NlCatalogEntry *g_entries = NULL;
static size_t g_count = 0;
static int g_missing = 0;
static char g_lang[NL_BCP47_SUBTAG];

void nl_catalog_clear(void) {
    size_t i;
    for (i = 0; i < g_count; i++) {
        free(g_entries[i].id);
        free(g_entries[i].text);
    }
    free(g_entries);
    g_entries = NULL;
    g_count = 0;
    g_missing = 0;
    g_lang[0] = '\0';
}

bool nl_catalog_load_path(const char *path) {
    FILE *fp;
    long size;
    char *buf;
    cJSON *json;
    cJSON *child;
    size_t n;
    const char *slash;
    const char *dot;
    size_t lang_len;

    nl_catalog_clear();
    if (!path) return false;

    fp = fopen(path, "rb");
    if (!fp) return false;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return false; }
    size = ftell(fp);
    if (size < 0) { fclose(fp); return false; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return false; }

    buf = malloc((size_t)size + 1);
    if (!buf) { fclose(fp); return false; }
    if (fread(buf, 1, (size_t)size, fp) != (size_t)size) {
        free(buf);
        fclose(fp);
        return false;
    }
    buf[size] = '\0';
    fclose(fp);

    if (!nl_utf8_validate(buf, (size_t)size, NULL)) {
        free(buf);
        return false;
    }

    json = cJSON_Parse(buf);
    free(buf);
    if (!json || !cJSON_IsObject(json)) {
        if (json) cJSON_Delete(json);
        return false;
    }

    n = 0;
    cJSON_ArrayForEach(child, json) {
        if (cJSON_IsString(child) && child->string) n++;
    }

    if (n > 0) {
        g_entries = calloc(n, sizeof(*g_entries));
        if (!g_entries) {
            cJSON_Delete(json);
            return false;
        }
    }

    cJSON_ArrayForEach(child, json) {
        if (!cJSON_IsString(child) || !child->string) continue;
        if (!nl_utf8_ok_cstr(child->valuestring)) continue;
        g_entries[g_count].id = strdup(child->string);
        g_entries[g_count].text = strdup(child->valuestring);
        if (!g_entries[g_count].id || !g_entries[g_count].text) {
            nl_catalog_clear();
            cJSON_Delete(json);
            return false;
        }
        g_count++;
    }

    cJSON_Delete(json);

    slash = strrchr(path, '/');
    slash = slash ? slash + 1 : path;
    dot = strchr(slash, '.');
    lang_len = dot ? (size_t)(dot - slash) : strlen(slash);
    if (lang_len >= sizeof g_lang) lang_len = sizeof g_lang - 1;
    memcpy(g_lang, slash, lang_len);
    g_lang[lang_len] = '\0';
    return true;
}

bool nl_catalog_load_locale(const NlResolvedLocale *loc, const char *dir) {
    char chain[NL_BCP47_FALLBACK_MAX][NL_BCP47_TAG];
    char path[1024];
    int n, i;

    if (!dir || dir[0] == '\0') return false;
    if (loc) {
        n = nl_bcp47_fallback_chain(&loc->locale, chain, NL_BCP47_FALLBACK_MAX);
        for (i = 0; i < n; i++) {
            snprintf(path, sizeof path, "%s/%s.json", dir, chain[i]);
            if (nl_catalog_load_path(path)) return true;
        }
    }
    snprintf(path, sizeof path, "%s/en.json", dir);
    return nl_catalog_load_path(path);
}

const char *nl_catalog_get(const char *id) {
    size_t i;
    if (!id) return NULL;
    for (i = 0; i < g_count; i++) {
        if (strcmp(g_entries[i].id, id) == 0) return g_entries[i].text;
    }
    return NULL;
}

const char *nl_catalog_text(const char *id) {
    const char *hit;
    const char *en;
    if (!id) return "";
    hit = nl_catalog_get(id);
    if (hit) return hit;
    if (g_count > 0) g_missing++;
    en = nl_diag_en(id);
    return en ? en : id;
}

int nl_catalog_missing_count(void) {
    return g_missing;
}

const char *nl_catalog_lang(void) {
    return g_lang;
}

typedef enum {
    PL_ZERO = 0, PL_ONE, PL_TWO, PL_FEW, PL_MANY, PL_OTHER
} PlCat;

static PlCat plural_cat(const char *lang, long n) {
    if (lang && strcmp(lang, "zh") == 0) return PL_OTHER;
    if (lang && strcmp(lang, "ar") == 0) {
        if (n == 0) return PL_ZERO;
        if (n == 1) return PL_ONE;
        if (n == 2) return PL_TWO;
        if (n >= 3 && n <= 10) return PL_FEW;
        if (n >= 11 && n <= 99) return PL_MANY;
        return PL_OTHER;
    }
    if (lang && (strcmp(lang, "fr") == 0 || strcmp(lang, "hi") == 0)) {
        return (n == 0 || n == 1) ? PL_ONE : PL_OTHER;
    }
    return (n == 1) ? PL_ONE : PL_OTHER;
}

static const char *pl_name(PlCat c) {
    switch (c) {
        case PL_ZERO: return "zero";
        case PL_ONE: return "one";
        case PL_TWO: return "two";
        case PL_FEW: return "few";
        case PL_MANY: return "many";
        default: return "other";
    }
}

static const char *arg_at(const char **args, int idx) {
    int i = 0;
    if (!args || idx < 0) return "";
    while (args[i] && i < idx) i++;
    return (args[i] && i == idx) ? args[i] : "";
}

static int append_str(char *out, size_t out_n, size_t *o, const char *s) {
    size_t len;
    if (!s) s = "";
    len = strlen(s);
    if (*o + len >= out_n) return 0;
    memcpy(out + *o, s, len);
    *o += len;
    return 1;
}

static int find_plural_arm(const char *body, const char *want, const char **start, const char **end) {
    const char *p = body;
    while (*p) {
        const char *name_end;
        const char *brace;
        int depth;
        name_end = p;
        while (*name_end && *name_end != '{') name_end++;
        if (*name_end != '{') return 0;
        if ((size_t)(name_end - p) == strlen(want) && strncmp(p, want, (size_t)(name_end - p)) == 0) {
            brace = name_end;
            depth = 0;
            do {
                if (*brace == '{') depth++;
                else if (*brace == '}') depth--;
                brace++;
            } while (*brace && depth > 0);
            *start = name_end + 1;
            *end = brace - 1;
            return 1;
        }
        brace = name_end;
        depth = 0;
        do {
            if (*brace == '{') depth++;
            else if (*brace == '}') depth--;
            brace++;
        } while (*brace && depth > 0);
        p = brace;
    }
    return 0;
}

static void quote_wrap(const char *lang, const char *inner, char *buf, size_t n) {
    if (lang && strcmp(lang, "zh") == 0) {
        snprintf(buf, n, "「%s」", inner);
    } else if (lang && strcmp(lang, "fr") == 0) {
        snprintf(buf, n, "« %s »", inner);
    } else if (lang && strcmp(lang, "ar") == 0) {
        snprintf(buf, n, "«%s»", inner);
    } else if (lang && strcmp(lang, "es") == 0) {
        snprintf(buf, n, "«%s»", inner);
    } else if (lang && strcmp(lang, "hi") == 0) {
        snprintf(buf, n, "“%s”", inner);
    } else {
        snprintf(buf, n, "“%s”", inner);
    }
}

static void list_join(const char *lang, const char *piped, char *buf, size_t n) {
    char tmp[512];
    char *parts[16];
    int count = 0;
    char *save;
    char *tok;
    size_t o = 0;
    int i;

    snprintf(tmp, sizeof tmp, "%s", piped ? piped : "");
    tok = strtok_r(tmp, "|", &save);
    while (tok && count < 16) {
        parts[count++] = tok;
        tok = strtok_r(NULL, "|", &save);
    }
    buf[0] = '\0';
    if (count == 0) return;
    if (count == 1) {
        snprintf(buf, n, "%s", parts[0]);
        return;
    }
    if (lang && strcmp(lang, "zh") == 0) {
        for (i = 0; i < count; i++) {
            int w = snprintf(buf + o, n - o, "%s%s", i ? "、" : "", parts[i]);
            if (w < 0) break;
            o += (size_t)w;
            if (o >= n) break;
        }
        return;
    }
    if (count == 2) {
        const char *conj = (lang && strcmp(lang, "fr") == 0) ? " et "
                         : (lang && strcmp(lang, "es") == 0) ? " y "
                         : (lang && strcmp(lang, "ar") == 0) ? " و "
                         : (lang && strcmp(lang, "hi") == 0) ? " और "
                         : " and ";
        snprintf(buf, n, "%s%s%s", parts[0], conj, parts[1]);
        return;
    }
    for (i = 0; i < count; i++) {
        const char *sep = "";
        if (i == count - 1) {
            sep = (lang && strcmp(lang, "fr") == 0) ? " et "
                : (lang && strcmp(lang, "es") == 0) ? " y "
                : " and ";
        } else if (i > 0) {
            sep = ", ";
        }
        int w = snprintf(buf + o, n - o, "%s%s", sep, parts[i]);
        if (w < 0) break;
        o += (size_t)w;
        if (o >= n) break;
    }
}

int nl_catalog_format(char *out, size_t out_n, const char *pattern,
                      const char *lang, const char **args) {
    size_t i = 0;
    size_t o = 0;
    if (!out || out_n == 0) return 0;
    out[0] = '\0';
    if (!pattern) return 1;
    while (pattern[i] && o + 1 < out_n) {
        if (pattern[i] == '{' && pattern[i + 1] == '{') {
            out[o++] = '{';
            i += 2;
            continue;
        }
        if (pattern[i] == '{' ) {
            const char *start = pattern + i + 1;
            const char *end = strchr(start, '}');
            int idx = 0;
            const char *comma;
            const char *val;
            char inner[256];
            char rendered[512];
            size_t field_len;
            const char *close;

            if (!end) {
                if (!append_str(out, out_n, &o, pattern + i)) break;
                break;
            }
            /* plural arms contain nested braces; find matching close for the field */
            if (strncmp(start, "0,plural,", 9) == 0 ||
                (start[0] >= '0' && start[0] <= '9' && strstr(start, ",plural,"))) {
                int depth = 1;
                close = start;
                while (*close && depth > 0) {
                    if (*close == '{') depth++;
                    else if (*close == '}') depth--;
                    if (depth > 0) close++;
                }
                end = close;
            }

            field_len = (size_t)(end - start);
            if (field_len >= sizeof inner) field_len = sizeof inner - 1;
            memcpy(inner, start, field_len);
            inner[field_len] = '\0';

            idx = atoi(inner);
            val = arg_at(args, idx);
            comma = strchr(inner, ',');
            rendered[0] = '\0';
            if (!comma) {
                snprintf(rendered, sizeof rendered, "%s", val);
            } else if (strstr(inner, ",plural,")) {
                long nval = strtol(val, NULL, 10);
                PlCat cat = plural_cat(lang, nval);
                const char *body = strstr(inner, ",plural,");
                const char *arm_s = NULL;
                const char *arm_e = NULL;
                body += 8;
                if (!find_plural_arm(body, pl_name(cat), &arm_s, &arm_e)) {
                    find_plural_arm(body, "other", &arm_s, &arm_e);
                }
                if (arm_s && arm_e && arm_e >= arm_s) {
                    size_t alen = (size_t)(arm_e - arm_s);
                    if (alen >= sizeof rendered) alen = sizeof rendered - 1;
                    memcpy(rendered, arm_s, alen);
                    rendered[alen] = '\0';
                }
            } else if (strstr(inner, ",number")) {
                snprintf(rendered, sizeof rendered, "%s", val);
            } else if (strstr(inner, ",date")) {
                snprintf(rendered, sizeof rendered, "%s", val);
            } else if (strstr(inner, ",list")) {
                list_join(lang, val, rendered, sizeof rendered);
            } else if (strstr(inner, ",quote")) {
                quote_wrap(lang, val, rendered, sizeof rendered);
            } else {
                snprintf(rendered, sizeof rendered, "%s", val);
            }
            if (!append_str(out, out_n, &o, rendered)) break;
            i = (size_t)(end - pattern) + 1;
            continue;
        }
        out[o++] = pattern[i++];
    }
    out[o] = '\0';
    return 1;
}
