#ifndef NL_NSI_FILE_VALUES_INTERNAL_H
#define NL_NSI_FILE_VALUES_INTERNAL_H
#include "nsi_file_values.h"
/* Pure source-private lifetime validation. No host I/O, minting, mutation or
 * raw capability exposure. These declarations are not installed APIs. */
NlFileValueStatus nl_file_value_validate(NlFileValues *,const NlFileValue *,bool open_result);
NlFileValueStatus nl_file_value_borrow_validate(NlFileValues *,const NlFileValueBorrow *);
NlFileValueStatus nl_file_values_live_slots(NlFileValues *,uint64_t *owners,uint64_t *borrowed);
#endif
