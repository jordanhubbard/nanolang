/* Design-only sizeof probe. No target or emitter implementation. */
#include "union_field_snapshot_private.h"
#include <stdio.h>
typedef struct {
 const NvmUnionFieldImage *image;
 const NvmModule *module;
 uint32_t record_ordinal,record_layout,union_ordinal,union_layout;
 uint8_t payload_variant,empty_variant;
 uint8_t reserved[6];
} ProposedUnionTargetContext;
typedef struct {
 NvmUnionFieldOrigin origins[64];
 NvmUnionFieldExecutionInstruction instruction;
 NvmRecordArrayExecutionFunction function;
 NvmHeader header;
 NvmRecordArrayExecutionCounts counts;
 NvmUnionFieldProofCounts proof;
 NvmRecordValueOrigins value;
 NvmDeclarationLayout layouts[2];
 NvmUnionVariantFact variants[2];
 NvmV2LayoutField fields[2];
 uint64_t record_mask,variant_masks[2],work;
 uint32_t function_index,instruction_index,origin_index,global_index;
 uint32_t parameter_index,byte_index,byte_length,instruction_total;
 uint32_t opcode_index,field_index,variant_index,layout_index;
 uint8_t unsupported,parameter_tag;
 uint8_t reserved[6];
 ProposedUnionTargetContext staged;
} ProposedUnionTargetScratch;
#define SHOW(T) printf("%s=%zu\n",#T,sizeof(T))
int main(void) {
 SHOW(NvmUnionFieldOrigin); SHOW(NvmUnionFieldExecutionInstruction);
 SHOW(NvmRecordArrayExecutionFunction); SHOW(NvmHeader);
 SHOW(NvmRecordArrayExecutionCounts); SHOW(NvmUnionFieldProofCounts);
 SHOW(NvmRecordValueOrigins); SHOW(NvmDeclarationLayout);
 SHOW(NvmUnionVariantFact); SHOW(NvmV2LayoutField);
 SHOW(ProposedUnionTargetContext); SHOW(ProposedUnionTargetScratch);
 return 0;
}
