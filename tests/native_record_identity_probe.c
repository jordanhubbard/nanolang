/* I probe canonical native ownership, not bytecode AGG_SET admission. */
#ifndef NREC_CANONICAL
#error I require the explicitly selected canonical native representation.
#endif
#include <assert.h>
#undef main
int main(void) {
    assert(nrec_empty_slot(&(nrec_t){0}));
    assert(!nrec_empty_slot(&(nrec_t){.n=1}));
    assert(!nrec_empty_slot(&(nrec_t){.f=(int64_t *)(uintptr_t)1}));
    nmap_collection_budget=1; /* I force collection before the first record-local store. */
    assert(nl_main()==0);nmap_collect();
    assert(!nrec_owned_head && !narr_owners && !nagg_live_bytes);
    nroot_frame roots={0};roots.prev=nroot_head;nroot_head=&roots;
    nrec_t record=nrec_create(1,0,0);
    nroot_add(&roots.live,4,&record);
    nrec_owned *before_failure=nrec_owned_head;size_t before_bytes=nagg_live_bytes;
    fault_armed=1;fail_record_allocation=1;
    if(!setjmp(record_fault)) { (void)nrec_create(1,0,0);exit(98); }
    fail_record_allocation=0;fault_armed=0;
    assert(nrec_owned_head==before_failure && nagg_live_bytes==before_bytes);
    nrec_t recovered=nrec_create(1,0,0);
    assert(recovered.identity && recovered.identity!=record.identity);
    nmap_collect();assert(nrec_owned_head==before_failure && !before_failure->next);

    record.k[0]=3;record.a[0]=narr_push(narr_new(),7);
    const nrec_t *snapshot=nrec_snapshot(record);
    nrec_t alias=*snapshot;
    assert(snapshot==record.identity && alias.identity==record.identity);
    size_t owned=nagg_live_bytes;
    for(unsigned i=0;i<100;i++)assert(nrec_snapshot(alias)==snapshot);
    assert(nagg_live_bytes==owned);
    narr_t retained=record.a[0];nroot_add(&roots.live,3,retained);
    record.a[0]=narr_push(narr_new(),9);
    assert(alias.a[0]==record.a[0] && snapshot->a[0]==record.a[0]);
    nmap_collect();assert(retained->data[0]==7 && alias.a[0]->data[0]==9);
    owned=nagg_live_bytes;
    nroot_reset(&roots.live);nroot_add(&roots.live,4,&alias);nmap_collect();
    assert(nagg_live_bytes<owned && alias.a[0]->data[0]==9);
    assert(narr_owners && !narr_owners->next);
    nrec_t parent=nrec_create(1,0,0);parent.k[0]=4;parent.rec[0]=nrec_snapshot(alias);
    nroot_reset(&roots.live);nroot_add(&roots.live,4,&parent);nmap_collect();
    assert(parent.rec[0]->a[0]->data[0]==9);
    assert(nrec_owned_head && nrec_owned_head->next && !nrec_owned_head->next->next);
    nroot_head=roots.prev;nroot_destroy(&roots.live);nmap_collect();
    assert(!nrec_owned_head && !narr_owners && !nagg_live_bytes);
    return 0;
}
