/* I require explicit carrier draining before the core's final sweep. */
#define nl_file_values_destroy dispatch_core_destroy_impl
#include "file_cyclic_runtime_values.c"
#undef nl_file_values_destroy
unsigned dispatch_core_drains;
NlFileValuesFinish nl_file_values_destroy(NlFileValues *s,NlFileValueStatus status){
 if(s){uint64_t owners=UINT64_MAX,borrows=UINT64_MAX;
  if(nl_file_values_live_slots(s,&owners,&borrows)!=NL_FILE_VALUE_OK || owners || borrows)abort();
  dispatch_core_drains++;
 }
 return dispatch_core_destroy_impl(s,status);
}
