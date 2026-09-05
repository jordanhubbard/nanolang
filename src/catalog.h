#ifndef NL_CATALOG_H
#define NL_CATALOG_H

#include "locale.h"
#include <stddef.h>
#include <stdbool.h>

/* UTF-8 JSON catalogs: object of id -> message string.
 * English is the fallback. Missing keys are counted, not fatal. */

bool nl_catalog_load_path(const char *path);
bool nl_catalog_load_locale(const NlResolvedLocale *loc, const char *dir);
void nl_catalog_clear(void);

/* Catalog text if present and valid UTF-8. NULL if missing. */
const char *nl_catalog_get(const char *id);

/* Catalog, else nl_diag_en, else id. Never NULL if id is non-NULL. */
const char *nl_catalog_text(const char *id);

int nl_catalog_missing_count(void);
const char *nl_catalog_lang(void);

/* Format a catalog pattern. args is a NULL-terminated list of UTF-8 strings.
 * Supports {n}, {n,plural,one{...}other{...}}, {n,number}, {n,date},
 * {n,list} (pipe-separated items in args[n]), and {n,quote}. */
int nl_catalog_format(char *out, size_t out_n, const char *pattern,
                      const char *lang, const char **args);

#endif
