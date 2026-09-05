#ifndef NL_LOCALE_H
#define NL_LOCALE_H

#include "bcp47.h"
#include <stddef.h>
#include <stdbool.h>

/* Process locale: BCP 47 tag plus how I chose it.
 * Encoding and collation stay on NlLocale, not in the tag. */

#define NL_LOCALE_SOURCE 24

typedef struct {
    NlLocale locale;
    char source[NL_LOCALE_SOURCE];
} NlResolvedLocale;

/* Convert a POSIX LANG/LC_ALL value (en_US.UTF-8, C, POSIX) to a BCP 47
 * tag. Does not guess script from region. Returns false if empty. */
bool nl_posix_locale_to_bcp47(const char *posix, char *out, size_t out_sz);

/* Resolve in order: cli_tag, NANO_LOCALE, LC_ALL, LANG, then en.
 * cli_tag and NANO_LOCALE must be BCP 47 or the call fails.
 * POSIX env values that do not convert are skipped. */
bool nl_locale_resolve(const char *cli_tag, NlResolvedLocale *out);

/* Language-neutral field dump for --print-locale and tests. */
void nl_locale_format(const NlResolvedLocale *r, char *out, size_t out_sz);

#endif
