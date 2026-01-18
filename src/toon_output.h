/**
 * @file toon_output.h
 * @brief TOON format output for LLM-friendly diagnostics
 *
 * Provides Token-Oriented Object Notation output as an alternative
 * to JSON for reduced token consumption in LLM agent workflows.
 * TOON format uses ~40% fewer tokens than equivalent JSON.
 *
 * See: https://toonformat.dev/
 */

#ifndef TOON_OUTPUT_H
#define TOON_OUTPUT_H

#include <stdbool.h>
#include <stdio.h>

/**
 * @brief Enable TOON diagnostic output
 * @note Mutually exclusive with JSON output
 */
void toon_diagnostics_enable(void);

/**
 * @brief Check if TOON output is enabled
 * @return true if TOON output mode is active
 */
bool toon_diagnostics_enabled(void);

/**
 * @brief Add a diagnostic entry for TOON output
 * @param severity Severity level ("error", "warning", "info", "hint")
 * @param code Diagnostic code (e.g., "E001")
 * @param message Human-readable message
 * @param file Source file path
 * @param line Line number (1-indexed)
 * @param column Column number (1-indexed)
 */
void toon_diagnostics_add(const char *severity, const char *code,
                          const char *message, const char *file,
                          int line, int column);

/**
 * @brief Output accumulated diagnostics in TOON format
 * @param fp File pointer to write to (e.g., stdout or file)
 *
 * Writes diagnostics in TOON format. Call after compilation completes.
 * Requires prior toon_diagnostics_enable().
 */
void toon_diagnostics_output(FILE *fp);

/**
 * @brief Output accumulated diagnostics to a file path
 * @param path File path to write TOON output to
 * @return true on success, false on file open failure
 */
bool toon_diagnostics_output_to_file(const char *path);

/**
 * @brief Free all TOON diagnostic resources
 */
void toon_diagnostics_cleanup(void);

#endif /* TOON_OUTPUT_H */
