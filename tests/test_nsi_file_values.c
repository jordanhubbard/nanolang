/* I reuse the qualified adapter's real-host interception without executing its
 * former main. Only the explicitly selected new controls run below. */
#define main previous_private_file_fixture_main
#include "test_nsi_file.c"
#undef main
#define calloc checked_calloc
#define free checked_free
#include "../src/nsi_file_values.c"
#undef calloc
#undef free
#include "test_nsi_file_values_cases.h"

static void qv_faults(void) {
    unsigned successes=0;
    for(long limit=0;limit<4;limit++) {
        NlFileValues *s=NULL;allocation_budget=limit;
        NlFileValueStatus status=nl_file_values_create(&s);allocation_budget=-1;
        if(status==NL_FILE_VALUE_OK){successes++;qv_clean(s);}else CHECK(status==NL_FILE_VALUE_MEMORY && !s);
        empty();
    }
    CHECK(successes==1);
    NlFileValues *s=qv_create();NlFileValue result={0};NlFileOpenView view;
    fail_open=1;unsigned opens=host_opens;
    CHECK(nl_file_values_temp(s,&result)==NL_FILE_VALUE_OK);fail_open=0;
    CHECK(host_opens==opens && nl_file_open_view(s,&result,&view)==NL_FILE_VALUE_OK && !view.ok);
    CHECK(view.error.status==NL_FILE_IO && view.error.host_errno==EACCES);
    NlFileValue moved={0};CHECK(nl_file_value_move(s,&result,&moved)==NL_FILE_VALUE_OK && qv_empty(result));
    CHECK(nl_file_open_take_ok(s,&moved,&result)==NL_FILE_VALUE_TYPE && qv_empty(result));
    NlFileResult error;CHECK(nl_file_open_take_error(s,&moved,&error)==NL_FILE_VALUE_OK && qv_empty(moved));
    CHECK(error.status==NL_FILE_IO && error.host_errno==EACCES);
    allocation_budget=0;unsigned attempts=allocations;NlFileValue file=qv_file(s);
    CHECK(allocations==attempts);allocation_budget=-1;
    NlFileValueBorrow borrow={0};CHECK(nl_file_value_borrow(s,&file,&borrow)==NL_FILE_VALUE_OK);
    NlFileScalarResult out;unsigned before=io_calls;
    CHECK(nl_file_value_write_byte(s,&borrow,INT64_MAX,&out)==NL_FILE_VALUE_OK && !out.ok && io_calls==before);
    CHECK(nl_file_value_write_byte(s,&borrow,INT64_MIN,&out)==NL_FILE_VALUE_OK && !out.ok && io_calls==before);
    FvSlot *slot=&s->slots[file.slot];NlCapSlot *cap=&s->service->caps->slots[slot->token.cap.slot];uint32_t rights=cap->rights;
    cap->rights=NL_CAP_READ;CHECK(nl_file_value_write_byte(s,&borrow,3,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.status==NL_FILE_RIGHTS && io_calls==before);cap->rights=rights;
    partial_write=1;CHECK(nl_file_value_write_byte(s,&borrow,97,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.host_errno==ENOSPC && out.detail.bytes==1);partial_write=0;forced_error=NULL;
    fail_seek=1;CHECK(nl_file_value_rewind(s,&borrow,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.host_errno==ESPIPE);fail_seek=0;
    before=io_calls;CHECK(nl_file_value_read_byte(s,&borrow,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.status==NL_FILE_DIRECTION && io_calls==before);
    CHECK(nl_file_value_rewind(s,&borrow,&out)==NL_FILE_VALUE_OK && out.ok);
    partial_read=1;CHECK(nl_file_value_read_byte(s,&borrow,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.host_errno==EIO && out.detail.bytes==1);partial_read=0;forced_error=NULL;
    fail_seek=1;CHECK(nl_file_value_rewind(s,&borrow,&out)==NL_FILE_VALUE_OK && !out.ok);fail_seek=0;
    before=io_calls;CHECK(nl_file_value_write_byte(s,&borrow,5,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.status==NL_FILE_DIRECTION && io_calls==before);
    CHECK(nl_file_value_end_borrow(s,&borrow)==NL_FILE_VALUE_OK);
    int fd=fileno(s->service->files[slot->token.cap.slot].stream);CHECK(nl_file_value_drop(s,&file)==NL_FILE_VALUE_OK);
    errno=0;CHECK(fcntl(fd,F_GETFD)==-1 && errno==EBADF);qv_clean(s);empty();
}
static void qv_limits_and_cleanup(void) {
    NlFileValues *s=qv_create();NlFileValue file=qv_file(s),out={0};FvSlot *slot=&s->slots[file.slot];
    slot->generation=UINT64_MAX;file.generation=UINT64_MAX;
    CHECK(nl_file_value_move(s,&file,&out)==NL_FILE_VALUE_LIMIT && qv_empty(out));
    CHECK(nl_file_value_drop(s,&file)==NL_FILE_VALUE_OK);file=qv_file(s);CHECK(file.slot!=0);
    slot=&s->slots[file.slot];slot->borrow_epoch=UINT64_MAX;NlFileValueBorrow borrow={0};
    CHECK(nl_file_value_borrow(s,&file,&borrow)==NL_FILE_VALUE_LIMIT && !borrow.epoch);
    unsigned retired=file.slot;CHECK(nl_file_value_drop(s,&file)==NL_FILE_VALUE_OK);file=qv_file(s);CHECK(file.slot!=retired);qv_clean(s);empty();
    uint64_t saved=fv_identity;fv_identity=UINT64_MAX;s=NULL;unsigned allocations_before=allocations;
    CHECK(nl_file_values_create(&s)==NL_FILE_VALUE_LIMIT && !s && allocations==allocations_before);fv_identity=saved;
    saved=file_context_counter;file_context_counter=UINT64_MAX;CHECK(nl_file_values_create(&s)==NL_FILE_VALUE_LIMIT && !s);file_context_counter=saved;empty();
    s=qv_create();s->service->caps->next_generation=UINT32_MAX;errors(ENOSPC,0);
    CHECK(nl_file_values_temp(s,&out)==NL_FILE_VALUE_OK);NlFileOpenView view;
    CHECK(nl_file_open_view(s,&out,&view)==NL_FILE_VALUE_OK && !view.ok && view.error.status==NL_FILE_LIMIT && view.error.cleanup_failed && view.error.cleanup_errno==ENOSPC);
    CHECK(nl_file_value_drop(s,&out)==NL_FILE_VALUE_OK);NlFileValuesFinish report=nl_file_values_destroy(s,NL_FILE_VALUE_MEMORY);
    CHECK(report.execution==NL_FILE_VALUE_MEMORY && report.cleanup_failures==1 && report.first_cleanup.cleanup_errno==ENOSPC);errors(0,0);empty();
    s=qv_create();NlFileValue a=qv_file(s),b=qv_file(s),c=qv_file(s);borrow=(NlFileValueBorrow){0};
    CHECK(nl_file_value_borrow(s,&c,&borrow)==NL_FILE_VALUE_OK);errors(EIO,ENOSPC);close_errors[2]=EPIPE;
    NlFileScalarResult close;CHECK(nl_file_value_close(s,&a,&close)==NL_FILE_VALUE_OK && !close.ok && close.detail.consumed && close.detail.host_errno==EIO && qv_empty(a));
    CHECK(nl_file_value_drop(s,&b)==NL_FILE_VALUE_OK);report=nl_file_values_finish(s,NL_FILE_VALUE_TYPE);
    CHECK(report.execution==NL_FILE_VALUE_TYPE && report.cleanup_failures==3 && report.first_cleanup.host_errno==EIO && report.next_cleanup.host_errno==ENOSPC);
    CHECK(live_files==0 && live_allocations==1);unsigned closes=host_closes;
    NlFileValuesFinish again=nl_file_values_finish(s,NL_FILE_VALUE_STATUS_COUNT);
    CHECK(again.execution==report.execution && again.cleanup_failures==3 && host_closes==closes);
    CHECK(nl_file_values_destroy(s,NL_FILE_VALUE_OK).cleanup_failures==3);CHECK(host_closes==closes);errors(0,0);empty();
    s=qv_create();a=qv_file(s);slot=&s->slots[a.slot];uint64_t secret=slot->token.cap.secret;slot->token.cap.secret^=1;
    close.value=999;CHECK(nl_file_value_close(s,&a,&close)==NL_FILE_VALUE_STATE && close.value==999 && !qv_empty(a));slot->token.cap.secret=secret;
    report=nl_file_values_destroy(s,NL_FILE_VALUE_STATE);CHECK(report.execution==NL_FILE_VALUE_STATE && report.cleanup_failures==1 && report.first_cleanup.status==NL_FILE_TOKEN);empty();
}
int main(void) {
    errors(0,0);qv_lifetime();qv_capacity();empty();qv_faults();qv_limits_and_cleanup();
    printf("PASS %u File value checks; %u real opens and %u real closes; no live resources\n",checks,host_opens,host_closes);return 0;
}
