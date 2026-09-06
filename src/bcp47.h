#ifndef NL_BCP47_H
#define NL_BCP47_H

#include <stddef.h>
#include <stdbool.h>

/* BCP 47 language tag pieces. Encoding and collation are not BCP 47
 * subtags; I keep them beside the tag so locale is not one setting. */

#define NL_BCP47_SUBTAG 16
#define NL_BCP47_TAG 64
#define NL_BCP47_FALLBACK_MAX 8

typedef enum {
    NL_TEXT_LTR = 0,
    NL_TEXT_RTL = 1
} NlTextDirection;

typedef enum {
    NL_ENCODING_UTF8 = 0,
    NL_ENCODING_BINARY = 1
} NlTextEncoding;

typedef enum {
    NL_COLLATION_UNICODE = 0,
    NL_COLLATION_BYTE = 1
} NlCollation;

typedef struct {
    char language[NL_BCP47_SUBTAG];
    char script[NL_BCP47_SUBTAG];
    char region[NL_BCP47_SUBTAG];
    char variant[NL_BCP47_SUBTAG];
    char tag[NL_BCP47_TAG];
    NlTextDirection direction;
    NlTextEncoding encoding;
    NlCollation collation;
} NlLocale;

bool nl_bcp47_parse(const char *tag, NlLocale *out);
int nl_bcp47_fallback_chain(const NlLocale *loc, char out[][NL_BCP47_TAG],
                            int max_out);
const char *nl_bcp47_encoding_name(NlTextEncoding encoding);
const char *nl_bcp47_collation_name(NlCollation collation);
const char *nl_bcp47_direction_name(NlTextDirection direction);

#endif
