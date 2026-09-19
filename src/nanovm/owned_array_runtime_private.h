#ifndef NANOVM_OWNED_ARRAY_RUNTIME_PRIVATE_H
#define NANOVM_OWNED_ARRAY_RUNTIME_PRIVATE_H
#ifdef NANO_OWNED_ARRAY_PRIVATE_RUNTIME
#include "vm.h"
/* I require an initialized idle standalone VM, immutable module and no caller
 * operands. I prepare fresh authority and preserve *out on failure. Success
 * returns only INT/BOOL/U8; the caller still owns and frees its VM. */
VmResult vm_execute_owned_array_private(VmState *,NanoValue *out);
#endif
#endif
