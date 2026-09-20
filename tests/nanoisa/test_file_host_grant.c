#include "file_host_grant_internal.h"
#include <assert.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>

NvmFileHostStatus file_host_peer_enter(NvmFileHostGrant *grant);
#ifdef FILE_HOST_INSTRUMENTED
size_t file_host_test_attempts(void);
size_t file_host_test_live(void);
void file_host_test_fail(void);
void file_host_test_reenter(void);
void file_host_test_mismatch(NvmFileHostGrant *);
#endif

static void *contender(void *opaque) {
    NvmFileHostGrant *grant = opaque, *out = grant, *same = grant;
    assert(nvm_file_host_enter_query() == NVM_FILE_HOST_BUSY);
    assert(file_host_peer_enter(grant) == NVM_FILE_HOST_BUSY);
    assert(nvm_file_host_enter(NULL, 0, 0) == NVM_FILE_HOST_BUSY);
    assert(nvm_file_host_grant_create_temporary_files(&out) == NVM_FILE_HOST_BUSY);
    assert(out == grant);
    assert(nvm_file_host_grant_create_temporary_files(NULL) == NVM_FILE_HOST_BUSY);
    assert(nvm_file_host_grant_revoke(grant) == NVM_FILE_HOST_BUSY);
    assert(nvm_file_host_grant_revoke(NULL) == NVM_FILE_HOST_BUSY);
    assert(nvm_file_host_grant_destroy(&same) == NVM_FILE_HOST_BUSY);
    assert(same == grant);
    assert(nvm_file_host_grant_destroy(NULL) == NVM_FILE_HOST_BUSY);
    return NULL;
}

int main(void) {
    NvmFileHostGrant *first = NULL, *second = NULL, *empty = NULL;
    pthread_t thread;
    unsigned round;
    assert(NVM_FILE_HOST_OK == 0 && NVM_FILE_HOST_BUSY == 5);
    assert(nvm_file_host_grant_create_temporary_files(NULL) == NVM_FILE_HOST_INVALID);
    assert(nvm_file_host_grant_revoke(NULL) == NVM_FILE_HOST_INVALID);
    assert(nvm_file_host_grant_destroy(NULL) == NVM_FILE_HOST_INVALID);
    assert(nvm_file_host_grant_destroy(&empty) == NVM_FILE_HOST_OK && !empty);
    assert(nvm_file_host_enter(NULL, NVM_FILE_HOST_ABI, NVM_FILE_HOST_CATALOG)
           == NVM_FILE_HOST_INVALID);
    assert(nvm_file_host_grant_create_temporary_files(&first) == NVM_FILE_HOST_OK);
    assert(nvm_file_host_grant_create_temporary_files(&second) == NVM_FILE_HOST_OK);
    assert(first && second && first != second);
    assert(nvm_file_host_enter(first, 0, NVM_FILE_HOST_CATALOG) == NVM_FILE_HOST_UNRESOLVED);
    assert(nvm_file_host_enter(first, NVM_FILE_HOST_ABI, 0) == NVM_FILE_HOST_UNRESOLVED);
    assert(file_host_peer_enter(first) == NVM_FILE_HOST_OK);
    for (round = 0; round < 32; ++round) {
#ifdef FILE_HOST_INSTRUMENTED
        size_t before = file_host_test_attempts();
#endif
        assert(nvm_file_host_enter(first, NVM_FILE_HOST_ABI, NVM_FILE_HOST_CATALOG)
               == NVM_FILE_HOST_OK);
        /* Same-thread reentry and a different C99 TU both see the owning gate. */
        assert(contender(second) == NULL);
        /* The main thread retains the gate until the contender is joined;
         * success cannot depend on a timing race or a repeated retry. */
        assert(pthread_create(&thread, NULL, contender, second) == 0);
        assert(pthread_join(thread, NULL) == 0);
        assert(file_host_peer_enter(first) == NVM_FILE_HOST_BUSY);
        nvm_file_host_leave();
        assert(file_host_peer_enter(second) == NVM_FILE_HOST_OK);
#ifdef FILE_HOST_INSTRUMENTED
        assert(file_host_test_attempts() == before);
#endif
    }
    assert(nvm_file_host_enter_query() == NVM_FILE_HOST_OK);
    assert(contender(first) == NULL);
    nvm_file_host_leave();
#ifdef FILE_HOST_INSTRUMENTED
    {
        NvmFileHostGrant *sentinel = first, *reentrant = NULL;
        size_t before = file_host_test_attempts();
        file_host_test_fail();
        assert(nvm_file_host_grant_create_temporary_files(&sentinel) == NVM_FILE_HOST_MEMORY);
        assert(sentinel == first && file_host_test_attempts() == before + 1);
        assert(file_host_test_live() == 2);
        assert(file_host_peer_enter(first) == NVM_FILE_HOST_OK);
        file_host_test_reenter();
        assert(nvm_file_host_grant_create_temporary_files(&reentrant) == NVM_FILE_HOST_OK);
        assert(nvm_file_host_grant_destroy(&reentrant) == NVM_FILE_HOST_OK && !reentrant);
        file_host_test_mismatch(first);
    }
#endif
    assert(nvm_file_host_grant_revoke(first) == NVM_FILE_HOST_OK);
    assert(nvm_file_host_grant_revoke(first) == NVM_FILE_HOST_OK);
    assert(file_host_peer_enter(first) == NVM_FILE_HOST_STATE);
    assert(file_host_peer_enter(second) == NVM_FILE_HOST_OK);
    assert(nvm_file_host_grant_destroy(&first) == NVM_FILE_HOST_OK && !first);
    assert(nvm_file_host_grant_destroy(&first) == NVM_FILE_HOST_OK);
    assert(nvm_file_host_grant_destroy(&second) == NVM_FILE_HOST_OK && !second);
    for (round = 0; round < 128; ++round) {
        assert(nvm_file_host_grant_create_temporary_files(&first) == NVM_FILE_HOST_OK);
        assert(file_host_peer_enter(first) == NVM_FILE_HOST_OK);
        assert(nvm_file_host_grant_destroy(&first) == NVM_FILE_HOST_OK && !first);
    }
#ifdef FILE_HOST_INSTRUMENTED
    assert(file_host_test_live() == 0);
    printf("I checked instrumented grant lifecycle, faults and shared gate; attempts=%zu, live=%zu.\n",
           file_host_test_attempts(), file_host_test_live());
#else
    puts("I checked linked grant lifecycle and shared gate; no service execution.");
#endif
    return 0;
}
