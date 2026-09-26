#ifndef NANO_COMPILER_SUPPORT_H
#define NANO_COMPILER_SUPPORT_H
#include <stdint.h>
/* I export my facade while my embedded builder keeps private definitions. */
#define NANO_COMPILER_SUPPORT_EXPORT __attribute__((visibility("default")))
/* I return an invocation-local snapshot. The next call replaces it.
 * Empty means no usable manifest-backed shared artifact was built. */
NANO_COMPILER_SUPPORT_EXPORT const char *nlc_module_artifact(const char *source_path);
/* I expose an invocation-immutable copied data path and the compiled ABI,
 * never source authority. I select environment overrides on the first call. */
NANO_COMPILER_SUPPORT_EXPORT const char *nlc_runtime_root(void);
NANO_COMPILER_SUPPORT_EXPORT int64_t nlc_native_array_abi(void);
#endif
