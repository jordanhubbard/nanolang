/* sign.c — Ed25519 binary signing (stub)
 *
 * Full implementation lives in feat/compiler-advanced (PR #44).
 * This stub allows main to build until that PR is merged.
 */
#include "nanolang.h"

int sign_binary(const char *path, const char *key_path) {
    (void)path;
    (void)key_path;
    /* no-op stub — signing not yet available */
    return 0;
}
