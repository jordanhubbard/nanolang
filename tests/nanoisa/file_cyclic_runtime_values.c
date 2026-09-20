/* I include the real core only to seed boundary counters in the fixture. */
#include "file_runtime_frame_values.c"
void file_cyclic_epoch(NlFileValues *s,NlFileValueBorrow *b,uint64_t epoch){
 FvSlot *slot=NULL;if(fv_borrow(s,b,&slot)!=NL_FILE_VALUE_OK || !epoch)abort();
 slot->borrow_epoch=epoch;b->epoch=epoch;
}
void file_cyclic_retire_empty(NlFileValues *s){
 if(fv_context(s)!=NL_FILE_VALUE_OK)abort();
 for(unsigned i=0;i<NL_FILE_VALUE_SLOTS;i++){if(s->slots[i].kind!=FV_EMPTY)abort();s->slots[i].generation=UINT64_MAX;s->slots[i].borrow_epoch=UINT64_MAX;}
}
