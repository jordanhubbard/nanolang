/*
 * Nano Scheme — 4.6 laboratory frontend.
 *
 * I compile a bounded Scheme subset to verified NanoISA. This is a C host
 * compiler, like src/forth/, not a NanoLang language feature and not a
 * product Scheme. Continuations, macros, and the full language are out of
 * scope. Authority: docs/SCHEME.md, docs/NANOISA_FRONTEND.md.
 */

#ifndef NANOLANG_SCHEME_H
#define NANOLANG_SCHEME_H

#include "nanoisa/frontend.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/value.h"
#include "nanovm/vm.h"

#include <stddef.h>
#include <stdint.h>

#define NL_SCHEME_ERR_SIZE 256

typedef struct NlScheme NlScheme;

NvmModule *nl_scheme_compile(const char *src, const char *path,
                             char *err, size_t errlen);

NlFrontendResult nl_scheme_accept(const NvmModule *mod, const char *path);

VmResult nl_scheme_execute(const NvmModule *mod, NanoValue *out,
                           uint32_t *max_frame_depth, char *err, size_t errlen);

int nl_scheme_eval_i64(const char *src, int64_t *out, char *err, size_t errlen);

NlScheme *nl_scheme_open(void);
void nl_scheme_close(NlScheme *session);
int nl_scheme_eval(NlScheme *session, const char *src, NanoValue *out,
                   char *err, size_t errlen);
int nl_scheme_eval_i64_session(NlScheme *session, const char *src, int64_t *out,
                               char *err, size_t errlen);

#endif
