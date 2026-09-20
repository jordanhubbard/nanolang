/* I include the exact owning source under only allocator hooks. This is a
 * separate fixture object, never linked together with the real owning object. */
#include "file_host_grant_internal.h"
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

static size_t attempts, live;
static int fail_next, reenter_next;
static void *grant_malloc(size_t size) {
    void *result;
    ++attempts;
    if (reenter_next) {
        reenter_next = 0;
        assert(nvm_file_host_grant_create_temporary_files(NULL) == NVM_FILE_HOST_BUSY);
        assert(nvm_file_host_enter_query() == NVM_FILE_HOST_BUSY);
    }
    if (fail_next) { fail_next = 0; return NULL; }
    result = malloc(size);
    if (result) ++live;
    return result;
}
static void grant_free(void *pointer) {
    assert(pointer && live);
    --live;
    free(pointer);
}
#define malloc grant_malloc
#define free grant_free
#include "../../src/nanoisa/file_host_grant.c"
#undef free
#undef malloc

size_t file_host_test_attempts(void) { return attempts; }
size_t file_host_test_live(void) { return live; }
void file_host_test_fail(void) { fail_next = 1; }
void file_host_test_reenter(void) { reenter_next = 1; }
/* I mutate only my test-owned opaque object while no other thread uses it,
 * restore it before return, and check the complete refusal/recovery sequence. */
void file_host_test_mismatch(NvmFileHostGrant *grant) {
    NvmFileHostGrant original = *grant;
    static const int other_runtime = 0;
    unsigned field;
    for (field = 0; field < 4; ++field) {
        NvmFileHostGrant *same = grant;
        switch (field) {
        case 0: grant->runtime_identity = &other_runtime; break;
        case 1: ++grant->abi; break;
        case 2: ++grant->catalog; break;
        default: ++grant->policy; break;
        }
        assert(nvm_file_host_enter(grant, NVM_FILE_HOST_ABI, NVM_FILE_HOST_CATALOG)
               == NVM_FILE_HOST_UNRESOLVED);
        assert(nvm_file_host_grant_revoke(grant) == NVM_FILE_HOST_UNRESOLVED);
        assert(nvm_file_host_grant_destroy(&same) == NVM_FILE_HOST_UNRESOLVED);
        assert(same == grant);
        *grant = original;
        assert(nvm_file_host_enter(grant, NVM_FILE_HOST_ABI, NVM_FILE_HOST_CATALOG)
               == NVM_FILE_HOST_OK);
        nvm_file_host_leave();
    }
}
