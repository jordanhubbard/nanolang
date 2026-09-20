#ifndef NL_NSI_INTERNAL_H
#define NL_NSI_INTERNAL_H
#include "nsi.h"
#include "cJSON.h"
/* I borrow a decoded tree and return an independent owned NSI or NULL.
 * NULL retains the legacy malformed-input/allocation ambiguity. This helper
 * does not establish strict byte consumption, duplicate-key or extent facts. */
NlNsi *nl_nsi_decode_object(cJSON *json);
#endif
