/* I share immutable target descriptors, not a reusable admission certificate. */
#ifndef MIXED_COUNTED_CATALOG_H
#define MIXED_COUNTED_CATALOG_H
#include "../../src/nanoisa/managed_strings.h"
static const unsigned char mc_literal_bytes[]={0x61,0,0xc3,0xa9};
static const NmsView mc_literals[]={{mc_literal_bytes,4}};
/* Global zero is an unused union; records zero/one have equal shapes. */
static const NmsRecordDescriptor mc_records[]={{1,1},{2,1},{3,1}};
#endif
