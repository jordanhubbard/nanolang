#ifndef NANO_COMPILER_SUPPORT_H
#define NANO_COMPILER_SUPPORT_H
/* I return an invocation-local snapshot. The next call replaces it.
 * Empty means no usable manifest-backed shared artifact was built. */
const char *nlc_module_artifact(const char *source_path);
#endif
