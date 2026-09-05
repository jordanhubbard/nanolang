#ifndef NL_UTF8_H
#define NL_UTF8_H

#include <stddef.h>
#include <stdbool.h>

/* Strict UTF-8 (RFC 3629): reject overlong encodings, surrogates,
 * truncated sequences, and code points above U+10FFFF.
 * Empty input is valid. Binary payloads that are not UTF-8 fail. */

bool nl_utf8_validate(const char *data, size_t n, size_t *error_offset);

/* NUL-terminated. NULL is not valid. Empty is valid. */
bool nl_utf8_ok_cstr(const char *s);

/* If s is valid UTF-8, return s. NULL becomes "". Invalid becomes
 * a stable ASCII marker so JSON/TOON/log output stays UTF-8. */
const char *nl_utf8_cstr_or_marker(const char *s);

/* ASCII classifiers. Independent of LC_CTYPE. Negative EOF is not a letter. */
int nl_ascii_isspace(int c);
int nl_ascii_isdigit(int c);
int nl_ascii_isalpha(int c);
int nl_ascii_isalnum(int c);
int nl_ascii_isupper(int c);
int nl_ascii_islower(int c);
int nl_ascii_isprint(int c);
int nl_ascii_toupper(int c);
int nl_ascii_tolower(int c);

/* Copy s into out as UTF-8 safe for logs and terminals: invalid UTF-8
 * becomes the ASCII marker; ANSI CSI/OSC and bidi overrides are dropped.
 * Always NUL-terminates when out_n > 0. */
void nl_utf8_sanitize_log(const char *s, char *out, size_t out_n);

#endif
