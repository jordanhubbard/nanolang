#ifndef NANOISA_AFFINE_STATE_H
#define NANOISA_AFFINE_STATE_H
#include "ownership_contracts.h"
#include "reference_places.h"

/* I check local-normalized transitions, not bytecode/stack/CFG execution.
 * My opaque state owns its facts; failed transitions leave it unchanged.
 * Reference slots are verifier identities, not runtime pointers. Entry
 * reference parameters occupy their corresponding local-numbered slots.
 * I accept only scalar/complete-record declarations. Joins compare clones
 * of one analysis; symbolic invocation 1 does not prove caller alias facts. */
typedef struct NvmAffineState NvmAffineState;
NvmAffineState *nvm_affine_state_create(const NvmModule *module,
                                       uint32_t function, uint32_t references);
NvmAffineState *nvm_affine_state_clone(const NvmAffineState *state);
void nvm_affine_state_free(NvmAffineState *state);
bool nvm_affine_state_equal(const NvmAffineState *a, const NvmAffineState *b);
bool nvm_affine_scalar_define(NvmAffineState *state, uint16_t local);
bool nvm_affine_move(NvmAffineState *state, uint16_t source, uint16_t destination);
bool nvm_affine_pack(NvmAffineState *state, uint16_t destination,
                      const uint16_t *fields, uint16_t count);
bool nvm_affine_unpack(NvmAffineState *state, uint16_t source,
                        const uint16_t *fields, uint16_t count);
bool nvm_affine_owner_access(const NvmAffineState *state, uint16_t root,
                              const uint16_t *path, uint16_t count, bool write);
bool nvm_affine_region_begin(NvmAffineState *state);
bool nvm_affine_region_end(NvmAffineState *state);
bool nvm_affine_borrow(NvmAffineState *state, uint32_t reference,
                        uint16_t root, const uint16_t *path, uint16_t count,
                        NvmReferenceMode mode);
bool nvm_affine_reborrow(NvmAffineState *state, uint32_t reference,
                          uint32_t parent, NvmReferenceMode mode);
bool nvm_affine_reference_access(const NvmAffineState *state, uint32_t reference,
                                  uint16_t field, bool write);
bool nvm_affine_reference_field(const NvmAffineState *state, uint32_t reference,
                                  uint16_t field, bool write, uint8_t *tag);
/* I inspect obligations without changing them. An owned result must have the
 * exact declared type; UINT16_MAX means no returned local. All other live
 * resource locals and all nested call regions prevent exit. */
bool nvm_affine_can_exit(const NvmAffineState *state, uint16_t result);
/* I expose only checked live-local declarations and permitted scalar field
 * observations to the bytecode analysis; callers cannot mutate my facts. */
bool nvm_affine_local_info(const NvmAffineState *state, uint16_t local,
                            uint8_t *tag, uint8_t *mode);
bool nvm_affine_scalar_field(const NvmAffineState *state, uint16_t local,
                              uint16_t field, uint8_t *tag);
bool nvm_affine_can_exit_scalar(const NvmAffineState *state, uint8_t tag);
typedef struct { uint8_t tag; uint32_t layout; } NvmAffineType;
/* These transfer APIs exchange an exact record token with the bytecode stack.
 * The stack analysis must prohibit duplication, loss and incompatible joins. */
bool nvm_affine_take_local(NvmAffineState *state, uint16_t local, NvmAffineType *type);
bool nvm_affine_put_local(NvmAffineState *state, uint16_t local, NvmAffineType type);
bool nvm_affine_local_type(const NvmAffineState *state, uint16_t local, NvmAffineType *type);
bool nvm_affine_record_fields(const NvmAffineState *state, uint32_t layout,
                               NvmAffineType *fields, uint16_t capacity, uint16_t *count);
bool nvm_affine_can_exit_type(const NvmAffineState *state, NvmAffineType type);
#endif
