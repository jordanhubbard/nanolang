/* I expose counters only to my separately built diagnostic provider closure. */
#ifndef NANOLANG_EVALUATOR_TIMING_PRIVATE_H
#define NANOLANG_EVALUATOR_TIMING_PRIVATE_H
#ifdef NANO_EVALUATOR_LIFETIME_TIMING
#include <stdint.h>
enum NanoLifetimeCounter {
    LIFE_GRAPH_ALLOC, LIFE_CLONE_NODES, LIFE_CLONE_NS,
    LIFE_RETIREMENTS, LIFE_ROOTS, LIFE_LOOKUPS, LIFE_LOOKUP_HITS, LIFE_LOOKUP_NS,
    LIFE_INDEX_CALLS, LIFE_INDEX_PROBES, LIFE_INDEX_HITS,
    LIFE_VIEW_CALLS, LIFE_VIEW_NS, LIFE_VIEW_COPIES, LIFE_VIEW_WRAP_ALLOC,
    LIFE_CHECKED_METADATA_ALLOC, LIFE_LEGACY_METADATA_ALLOC,
    LIFE_COUNTER_COUNT
};
void nano_lifetime_add(enum NanoLifetimeCounter, uint64_t);
uint64_t nano_lifetime_now(void);
void nano_lifetime_elapsed(enum NanoLifetimeCounter, uint64_t);
void nano_evaluator_lifetime_marker(const char *, const char *);
#define LIFE_COUNT(counter) nano_lifetime_add(counter, 1)
#else
#define LIFE_COUNT(counter) ((void)0)
#endif
#endif
