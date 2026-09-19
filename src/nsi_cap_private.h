#ifndef NL_NSI_CAP_PRIVATE_H
#define NL_NSI_CAP_PRIVATE_H

#include "nsi_cap.h"

/* I serve only contexts that privately own the entire table. Existing public
 * users keep their current lifetime policy. No external restart/revoke/borrow
 * operation may race these single-threaded private commits. */
#define NL_CAP_PRIVATE_SLOTS 64u
#define NL_CAP_PRIVATE_ERR_GENERATION 8

int nl_cap_private_mint(NlCapTable *table, const char *type_id,
                       const char *service_id, uint32_t rights, NlCap *out);
/* I publish a new token and retire the old slot only after mint succeeds.
 * Failure retains the source and leaves *out unchanged; aliasing out/src works. */
int nl_cap_private_transfer(NlCapTable *table, const NlCap *src, NlCap *out);
/* I retire only an exact live token, invalidate its Forth bindings, and preserve
 * the monotone generation counter. The host association must be detached first;
 * a rejected token has no table effect except the existing audit record. */
int nl_cap_private_consume(NlCapTable *table, const NlCap *token);

#endif
