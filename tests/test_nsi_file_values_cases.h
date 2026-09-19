#include "../src/nsi_file_values.h"
#include <fcntl.h>
#include <unistd.h>

static int qv_empty(NlFileValue v) {return !v.invocation && !v.generation && !v.slot;}
static int qv_equal(NlFileValue a,NlFileValue b) {
    return a.invocation==b.invocation && a.generation==b.generation && a.slot==b.slot;
}
static NlFileValues *qv_create(void) {
    NlFileValues *s=NULL;CHECK(nl_file_values_create(&s)==NL_FILE_VALUE_OK);CHECK(s);return s;
}
static NlFileValue qv_file(NlFileValues *s) {
    NlFileValue result={0},file={0};NlFileOpenView view;
    CHECK(nl_file_values_temp(s,&result)==NL_FILE_VALUE_OK);
    CHECK(nl_file_open_view(s,&result,&view)==NL_FILE_VALUE_OK && view.ok);
    NlFileValue stale=result;
    CHECK(nl_file_open_take_ok(s,&result,&file)==NL_FILE_VALUE_OK && qv_empty(result));
    CHECK(nl_file_value_drop(s,&stale)==NL_FILE_VALUE_STALE);return file;
}
static void qv_clean(NlFileValues *s) {
    NlFileValuesFinish r=nl_file_values_destroy(s,NL_FILE_VALUE_OK);
    CHECK(r.execution==NL_FILE_VALUE_OK && !r.cleanup_failures);
}
static void qv_lifetime(void) {
    FILE *sentinel=tmpfile();CHECK(sentinel);int sentinel_fd=fileno(sentinel);
    NlFileValues *s=qv_create(),*other=qv_create();NlFileValue file=qv_file(s),old=file;
    NlFileValue next={0};CHECK(nl_file_value_move(s,&file,&next)==NL_FILE_VALUE_OK);
    CHECK(qv_empty(file) && next.generation>old.generation && next.slot==old.slot);
    CHECK(nl_file_value_drop(s,&old)==NL_FILE_VALUE_STALE);
    CHECK(nl_file_value_drop(other,&next)==NL_FILE_VALUE_STALE);
    CHECK(nl_file_value_move(s,&next,&next)==NL_FILE_VALUE_ARGUMENT);
    NlFileValue occupied=qv_file(s),saved=occupied;
    CHECK(nl_file_value_move(s,&next,&occupied)==NL_FILE_VALUE_ARGUMENT && qv_equal(saved,occupied));
    CHECK(nl_file_value_drop(s,&occupied)==NL_FILE_VALUE_OK);
    NlFileValueBorrow borrow={0},second={0};
    CHECK(nl_file_value_borrow(s,&next,&borrow)==NL_FILE_VALUE_OK);
    CHECK(nl_file_value_borrow(s,&next,&second)==NL_FILE_VALUE_BORROWED && !second.epoch);
    CHECK(nl_file_value_move(s,&next,&file)==NL_FILE_VALUE_BORROWED && qv_empty(file));
    CHECK(nl_file_value_drop(s,&next)==NL_FILE_VALUE_BORROWED);
    NlFileScalarResult out={0};out.value=987;
    CHECK(nl_file_value_close(s,&next,&out)==NL_FILE_VALUE_BORROWED && out.value==987);
    CHECK(nl_file_value_write_byte(s,&borrow,-1,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.status==NL_FILE_ARGUMENT);
    CHECK(nl_file_value_write_byte(s,&borrow,256,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.status==NL_FILE_ARGUMENT);
    CHECK(nl_file_value_write_byte(s,&borrow,0,&out)==NL_FILE_VALUE_OK && out.ok && out.value==1);
    CHECK(nl_file_value_write_byte(s,&borrow,255,&out)==NL_FILE_VALUE_OK && out.ok && out.value==1);
    CHECK(nl_file_value_read_byte(s,&borrow,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.status==NL_FILE_DIRECTION);
    CHECK(nl_file_value_rewind(s,&borrow,&out)==NL_FILE_VALUE_OK && out.ok && !out.value);
    CHECK(nl_file_value_read_byte(s,&borrow,&out)==NL_FILE_VALUE_OK && out.ok && out.value==0 && !out.eof);
    CHECK(nl_file_value_read_byte(s,&borrow,&out)==NL_FILE_VALUE_OK && out.ok && out.value==255 && !out.eof);
    CHECK(nl_file_value_read_byte(s,&borrow,&out)==NL_FILE_VALUE_OK && out.ok && out.value==0 && out.eof);
    CHECK(nl_file_value_write_byte(s,&borrow,1,&out)==NL_FILE_VALUE_OK && !out.ok && out.detail.status==NL_FILE_DIRECTION);
    CHECK(nl_file_value_rewind(s,&borrow,&out)==NL_FILE_VALUE_OK && out.ok);
    CHECK(nl_file_value_write_byte(s,&borrow,1,&out)==NL_FILE_VALUE_OK && out.ok);
    NlFileValueBorrow stale_borrow=borrow;
    CHECK(nl_file_value_end_borrow(s,&borrow)==NL_FILE_VALUE_OK && !borrow.epoch);
    out.value=456;CHECK(nl_file_value_read_byte(s,&stale_borrow,&out)==NL_FILE_VALUE_STALE && out.value==456);
    CHECK(nl_file_value_borrow(s,&next,&borrow)==NL_FILE_VALUE_OK && borrow.epoch>stale_borrow.epoch);
    CHECK(nl_file_value_end_borrow(s,&stale_borrow)==NL_FILE_VALUE_STALE);
    CHECK(nl_file_value_end_borrow(s,&borrow)==NL_FILE_VALUE_OK);
    old=next;CHECK(nl_file_value_close(s,&next,&out)==NL_FILE_VALUE_OK && out.ok && out.detail.consumed && qv_empty(next));
    CHECK(nl_file_value_close(s,&old,&out)==NL_FILE_VALUE_STALE);
    CHECK(nl_file_value_drop(s,&next)==NL_FILE_VALUE_OK);
    for(unsigned n=0;n<192;n++) {
        NlFileValue v=qv_file(s);CHECK(nl_file_value_drop(s,&old)==NL_FILE_VALUE_STALE);
        old=v;CHECK(nl_file_value_drop(s,&v)==NL_FILE_VALUE_OK);
    }
    NlFileValue unhandled={0};CHECK(nl_file_values_temp(s,&unhandled)==NL_FILE_VALUE_OK);
    CHECK(nl_file_value_drop(s,&unhandled)==NL_FILE_VALUE_OK && qv_empty(unhandled));
    NlFileValue live=qv_file(s);CHECK(nl_file_value_borrow(s,&live,&borrow)==NL_FILE_VALUE_OK);
    NlFileValuesFinish finished=nl_file_values_finish(s,NL_FILE_VALUE_TYPE);
    CHECK(finished.execution==NL_FILE_VALUE_TYPE && !finished.cleanup_failures);
    CHECK(nl_file_values_finish(s,NL_FILE_VALUE_OK).execution==NL_FILE_VALUE_TYPE);
    CHECK(nl_file_value_end_borrow(s,&borrow)==NL_FILE_VALUE_DISPOSED);
    CHECK(nl_file_values_temp(s,&unhandled)==NL_FILE_VALUE_DISPOSED && qv_empty(unhandled));
    CHECK(nl_file_values_destroy(s,NL_FILE_VALUE_OK).execution==NL_FILE_VALUE_TYPE);qv_clean(other);
    s=qv_create();CHECK(nl_file_value_drop(s,&live)==NL_FILE_VALUE_STALE);qv_clean(s);
    CHECK(fcntl(sentinel_fd,F_GETFD)>=0);CHECK(fputs("sentinel",sentinel)>=0);CHECK(fflush(sentinel)==0);CHECK(fclose(sentinel)==0);
}
static void qv_capacity(void) {
    NlFileValues *s=qv_create();NlFileValue results[NL_FILE_VALUE_SLOTS]={0};
    for(unsigned i=0;i<NL_FILE_VALUE_SLOTS;i++)CHECK(nl_file_values_temp(s,&results[i])==NL_FILE_VALUE_OK);
    NlFileValue out={0};CHECK(nl_file_values_temp(s,&out)==NL_FILE_VALUE_LIMIT && qv_empty(out));
    NlFileValue stale=results[0];CHECK(nl_file_value_drop(s,&results[0])==NL_FILE_VALUE_OK);
    CHECK(nl_file_values_temp(s,&out)==NL_FILE_VALUE_OK && out.slot==stale.slot && out.generation>stale.generation);
    CHECK(nl_file_value_drop(s,&stale)==NL_FILE_VALUE_STALE);qv_clean(s);
}
