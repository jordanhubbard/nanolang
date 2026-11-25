/* I inspect mutable authority in a diagnostic TU; I add no product API. */
#include "../src/nanoisa/affine_state.c"
#include <stdio.h>
#include <inttypes.h>

static void owner_number(uint32_t n) { printf("%" PRIu32 ",",n); }
void nvm2c_affine_owner_state_snapshot(const NvmAffineState *s) {
    owner_number(s->region);owner_number(s->caller_bound);owner_number(s->handler);
    owner_number(s->handler_first);owner_number(s->handler_key);
    owner_number(s->resume_type.tag);owner_number(s->resume_type.layout);
    owner_number((uint32_t)s->invocation);owner_number((uint32_t)(s->invocation>>32));putchar('/');
    for(uint16_t i=0;i<s->facts->count;i++) {
        owner_number(s->live[i]);owner_number(s->variants[i]);putchar(';');
    }
    putchar('/');
    for(uint32_t i=0;i<s->ref_count;i++) {
        const Reference *r=&s->refs[i];const NvmReferencePlace *p=&r->place;
        owner_number(r->live);owner_number(r->region);owner_number(r->parent);
        owner_number((uint32_t)p->invocation);owner_number((uint32_t)(p->invocation>>32));
        owner_number(p->local);owner_number(p->root_layout);owner_number(p->referent_layout);
        owner_number(p->mode);owner_number(p->field_count);
        for(uint16_t field=0;field<p->field_count;field++)owner_number(p->fields[field]);
        putchar(';');
    }
    putchar('/');
    for(uint16_t i=0;i<s->origin_count;i++) {
        owner_number((uint32_t)s->origins[i].invocation);owner_number((uint32_t)(s->origins[i].invocation>>32));
        owner_number(s->origins[i].local);owner_number(s->origins[i].layout);putchar(';');
    }
}
bool nvm2c_affine_owner_state_region(NvmAffineState *s,uint32_t region) {
    if(!s)return false;
    for(uint32_t i=0;i<s->ref_count;i++)if(s->refs[i].live)return false;
    s->region=region;return true;
}
