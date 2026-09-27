#include "evaluator_collection_alloc.h"
/* Evaluator collections retain one owner until Environment destruction.
 * Only actual fresh-allocation sites call registration, once per allocation.
 * The public low-level constructors and caller-owned inputs remain separate. */
struct EnvCollectionAllocation {
    void *allocation;
    void (*destroy)(void *);
    struct EnvCollectionAllocation *next;
};

bool env_register_new_collection(Environment *env, void *allocation, void (*destroy)(void *)) {
    if (!env || !allocation || !destroy) return false;
    struct EnvCollectionAllocation *entry = NANO_COLLECTION_CALLOC(1, sizeof *entry);
    if (!entry) return false;
    entry->allocation = allocation;
    entry->destroy = destroy;
    entry->next = env->collection_allocations;
    env->collection_allocations = entry;
    return true;
}

bool env_detach_collection(Environment *env, void *allocation) {
    if (!env || !allocation) return false;
    struct EnvCollectionAllocation **link = &env->collection_allocations;
    while (*link) {
        struct EnvCollectionAllocation *entry = *link;
        if (entry->allocation == allocation) {
            *link = entry->next;
            free(entry);
            return true;
        }
        link = &entry->next;
    }
    return false;
}

static void env_collection_storage_free(Environment *env) {
    while (env->collection_allocations) {
        struct EnvCollectionAllocation *entry = env->collection_allocations;
        env->collection_allocations = entry->next;
        entry->destroy(entry->allocation);
        free(entry);
    }
}
