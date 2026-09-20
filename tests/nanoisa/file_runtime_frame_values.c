/* Instrumented owning TU only. The build applies the same allocation hooks as
 * the real core. I set a valid handle/slot pair near exhaustion; I do not mock
 * the move, generation check, destructor or service. Linked mode uses the
 * separately compiled unchanged production TU without this fixture helper. */
#include "../../src/nsi_file_values.c"
void file_frame_set_generation(NlFileValues *s,NlFileValue *v,uint64_t generation) {
    FvSlot *slot=NULL;
    if(fv_resolve(s,v,&slot)!=NL_FILE_VALUE_OK || !generation)abort();
    slot->generation=generation;v->generation=generation;
}
