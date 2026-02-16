#ifndef DIAGNOSTICS_FWD_H
#define DIAGNOSTICS_FWD_H

#include <stdint.h>

/* Forward declarations for diagnostics module functions */
/* Note: These functions are compiled separately in obj/nano_modules/diagnostics.o */

/* Function declarations */
CompilerSourceLocation diagnostics__diag_location(const char* file, int64_t line, int64_t column);
CompilerSourceLocation diagnostics__diag_location_empty(void);
CompilerDiagnostic diagnostics__diag_parser_error(const char* code, const char* message, CompilerSourceLocation location);
void diagnostics__diag_list_add(void* list, CompilerDiagnostic diag);

#endif /* DIAGNOSTICS_FWD_H */
