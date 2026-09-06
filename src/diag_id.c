#include "diag_id.h"

#include <stddef.h>
#include <string.h>

typedef struct {
    const char *id;
    const char *en;
} NlDiagEntry;

static const NlDiagEntry k_diags[] = {
    { NL_DIAG_IO_OPEN,       "Could not open input file" },
    { NL_DIAG_LEX_FAILED,    "Lexing failed" },
    { NL_DIAG_PARSE_FAILED,  "Parsing failed" },
    { NL_DIAG_IMPORT_FAILED, "Module loading failed" },
    { NL_DIAG_TYPE_FAILED,   "Type checking failed" },
    { NL_DIAG_MOD_COMPILE,   "Failed to compile imported modules" },
    { NL_DIAG_SHADOW_FAILED, "Shadow tests failed" },
    { NL_DIAG_TRANS_FAILED,  "Transpilation failed" },
    { NL_DIAG_C_TEMP,        "Could not create temporary C file" },
    { NL_DIAG_CC_CMD,        "C compile command too long" },
    { NL_DIAG_CC_FAILED,     "C compilation failed" },
    { NL_DIAG_SRC_UTF8,      "Source is not valid UTF-8" },
    { NL_DIAG_CAT_MISSING,   "Missing catalog key" },
    { NL_DIAG_LEX_BYTE,      "Unknown byte; identifiers are ASCII" },
    { NL_DIAG_LEX_STRING,    "Unterminated string" },
    { NL_DIAG_LEX_FSTRING,   "Unterminated f-string" },
    { NL_DIAG_LEX_FBRACE,    "Unclosed '{' in f-string" },
    { NL_DIAG_LEX_CHAR,      "Unterminated character literal" },
    { NL_DIAG_LEX_ESC,       "Incomplete escape sequence" },
    { NL_DIAG_PARSE_EVENT,   "Parsing error" },
    { NL_DIAG_PARSE_TOKENS,  "Invalid token array" },
    { NL_DIAG_LOG,           "log" },
    { NL_DIAG_LOG_ENTER,     "trace enter" },
    { NL_DIAG_LOG_EXIT,      "trace exit" },
    { NL_DIAG_LOG_EVENT,     "trace event" },
};

const char *nl_diag_en(const char *id) {
    size_t i;
    if (!id) return NULL;
    for (i = 0; i < sizeof k_diags / sizeof k_diags[0]; i++) {
        if (strcmp(k_diags[i].id, id) == 0) return k_diags[i].en;
    }
    return NULL;
}
