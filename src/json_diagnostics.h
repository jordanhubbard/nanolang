/* json_diagnostics.h - Structured JSON error output for LLM agents */

#ifndef JSON_DIAGNOSTICS_H
#define JSON_DIAGNOSTICS_H

#include <stdbool.h>

/* Diagnostic severity levels */
typedef enum {
    DIAG_ERROR,
    DIAG_WARNING,
    DIAG_INFO,
    DIAG_HINT
} DiagnosticSeverity;

/* Diagnostic structure */
typedef struct {
    DiagnosticSeverity severity;
    char *code;        /* Error code: E0001, W0001, etc. */
    char *message;     /* Human-readable message */
    char *file;        /* Source file */
    int line;          /* Line number (1-based) */
    int column;        /* Column number (1-based) */
    char *suggestion;  /* Optional fix suggestion */
} Diagnostic;

/* Global diagnostic state */
extern bool g_json_output_enabled;
extern Diagnostic *g_diagnostics;
extern int g_diagnostic_count;
extern int g_diagnostic_capacity;

/* Initialize JSON diagnostic system */
void json_diagnostics_init(void);

/* Enable JSON output mode */
void json_diagnostics_enable(void);

/* Add a diagnostic */
void json_diagnostics_add(DiagnosticSeverity severity, const char *code,
                          const char *message, const char *file,
                          int line, int column, const char *suggestion);

/* Output all diagnostics as JSON */
void json_diagnostics_output(void);

/* Cleanup */
void json_diagnostics_cleanup(void);

/* Convenience wrappers */
void json_error(const char *code, const char *message, const char *file,
                int line, int column, const char *suggestion);

void json_warning(const char *code, const char *message, const char *file,
                  int line, int column, const char *suggestion);

#endif /* JSON_DIAGNOSTICS_H */
