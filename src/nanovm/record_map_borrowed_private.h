#ifndef NANOVM_RECORD_MAP_BORROWED_PRIVATE_H
#define NANOVM_RECORD_MAP_BORROWED_PRIVATE_H
#include "vm.h"
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
/* Qualification-only module-taking entry. It owns a fresh proof/preparation;
 * callers cannot supply a trusted context. Module/caches and managed caller
 * storage must remain exclusively stable throughout this synchronous call.
 * Success leaves the INT entry result on the actual stack; caller consumes it
 * before another invocation. Failures unwind selected activation owners, not
 * mutations already made through globals/aliases. Public selectors stay closed. */
VmResult vm_record_map_borrowed_execute_private(const NvmModule *,VmState *);
#endif
#endif
