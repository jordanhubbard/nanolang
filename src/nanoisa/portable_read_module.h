#ifndef NANOISA_PORTABLE_READ_MODULE_H
#define NANOISA_PORTABLE_READ_MODULE_H
#include "portable_read_host.h"

/* I bind one generated native module instance while it is inactive. I copy the
 * callback/context pair; the caller owns its context until unbind or disposal.
 * NULL revokes authority. A binding grants no source or import admission. */
NprStatus npr_module_bind(const NprHostBinding *);
/* I retain the first host failure for the current generated entry. Managed
 * failures remain in nano_try_entry's high word. Calls are serialized. */
uint32_t npr_module_host_status(void);
/* I expose these to generated LLVM only. The argument remains a borrowed root;
 * success returns an independently owned STRING root, failure returns zero. */
void npr_module_reset(void);
uint64_t npr_module_read_text(uint64_t argument);
#endif
