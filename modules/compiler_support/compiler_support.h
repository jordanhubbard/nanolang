#ifndef NANO_COMPILER_SUPPORT_H
#define NANO_COMPILER_SUPPORT_H
#include <stdint.h>
/* I return an invocation-local snapshot. The next call replaces it.
 * Empty means no usable manifest-backed shared artifact was built. */
const char *nlc_module_artifact(const char *source_path);
/* I expose an invocation-immutable copied data path and the compiled ABI,
 * never source authority. I select environment overrides on the first call. */
const char *nlc_runtime_root(void);
int64_t nlc_native_array_abi(void);
#endif
