#ifndef NANOISA_PASSIVE_H
#define NANOISA_PASSIVE_H
#include "nvm_format.h"
/* I validate version-1 scalar eligibility records against authoritative code.
 * The wire format is documented in docs/NANOISA_PASSIVE.md. */
bool nvm_passive_valid(const NvmModule *module);
#endif
