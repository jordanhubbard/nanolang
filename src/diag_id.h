#ifndef NL_DIAG_ID_H
#define NL_DIAG_ID_H

/* Stable diagnostic identifiers. English prose is a lookup, not the identity.
 * Pipeline ids cover main.c. Lexer events use L0003–L0008. Parser events use
 * P0001. Typechecker titles that go through emit_context_error are E001–E034.
 * Log events use LOG01–LOG04. CAT01 is a missing catalog key. */

#define NL_DIAG_IO_OPEN      "CIO01"
#define NL_DIAG_LEX_FAILED   "CLEX01"
#define NL_DIAG_PARSE_FAILED "CPARSE01"
#define NL_DIAG_IMPORT_FAILED "CIMPORT01"
#define NL_DIAG_TYPE_FAILED  "CTYPE01"
#define NL_DIAG_MOD_COMPILE  "CMOD01"
#define NL_DIAG_SHADOW_FAILED "CSHADOW01"
#define NL_DIAG_TRANS_FAILED "CTRANS01"
#define NL_DIAG_C_TEMP       "CC01"
#define NL_DIAG_CC_CMD       "CCC02"
#define NL_DIAG_CC_FAILED    "CCC01"
#define NL_DIAG_SRC_UTF8     "CSRC01"
#define NL_DIAG_CAT_MISSING  "CAT01"
#define NL_DIAG_LEX_BYTE     "L0003"
#define NL_DIAG_LEX_STRING   "L0004"
#define NL_DIAG_LEX_FSTRING  "L0005"
#define NL_DIAG_LEX_FBRACE   "L0006"
#define NL_DIAG_LEX_CHAR     "L0007"
#define NL_DIAG_LEX_ESC      "L0008"
#define NL_DIAG_PARSE_EVENT  "P0001"
#define NL_DIAG_PARSE_TOKENS "P0002"
#define NL_DIAG_LOG          "LOG01"
#define NL_DIAG_LOG_ENTER    "LOG02"
#define NL_DIAG_LOG_EXIT     "LOG03"
#define NL_DIAG_LOG_EVENT    "LOG04"

/* English until a catalog supplies another language. NULL if unknown. */
const char *nl_diag_en(const char *id);

#endif
