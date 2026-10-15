#ifndef NANO_COMPILER_SUPPORT_H
#define NANO_COMPILER_SUPPORT_H
/* I return an invocation-local snapshot. The next call replaces it.
 * Empty means no usable manifest-backed shared artifact was built. */
const char *nlc_module_artifact(const char *source_path);
/* I snapshot ordered name=decimal-value lines from the module's headers.
 * The next query replaces this borrowed buffer; empty means no values. */
const char *nlc_module_header_constants(const char *source_path);
#endif
