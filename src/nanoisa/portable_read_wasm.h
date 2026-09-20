#ifndef NANOISA_PORTABLE_READ_WASM_H
#define NANOISA_PORTABLE_READ_WASM_H
#include "portable_read_managed.h"
/* I borrow argument ownership and publish a copied STRING only with two OK
 * statuses. One static workspace belongs to one serialized wasm32 instance.
 * An engine trap invalidates that instance; normal returns release the latch.
 * This private adapter grants no NanoISA import/profile execution authority. */
NprManagedResult npr_wasm_read_managed(NmsRuntime *, NmsHandle argument);
#endif
