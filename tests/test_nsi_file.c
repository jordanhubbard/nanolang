/* Private, deterministic qualification. I do not change production entry points. */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
static long allocation_budget = -1;
static unsigned allocations, live_allocations, host_opens, host_closes, live_files, io_calls;
static FILE *tracked[128];
static int fail_open, fail_seek, partial_read, partial_write;
static FILE *forced_error;
static int close_errors[4], close_error_index;
static void *checked_calloc(size_t n, size_t size) {
    allocations++;
    if (allocation_budget == 0) return NULL;
    if (allocation_budget > 0) allocation_budget--;
    void *p = calloc(n,size);
    if (p) live_allocations++;
    return p;
}
static void checked_free(void *p) {
    if (p) { CHECK(live_allocations > 0); live_allocations--; }
    free(p);
    errno = ERANGE; /* A later free cannot replace a saved close error. */
}
static FILE *checked_tmpfile(void) {
    if (fail_open) { errno=EACCES; return NULL; }
    FILE *p=tmpfile();
    if (!p) return NULL;
    unsigned i=0;while(i<128 && tracked[i])i++;
    CHECK(i<128);tracked[i]=p;live_files++;host_opens++;return p;
}
static int checked_fclose(FILE *p) {
    unsigned i=0;while(i<128 && tracked[i]!=p)i++;
    CHECK(i<128 && live_files>0);
    tracked[i]=NULL;live_files--;host_closes++;
    int rc=fclose(p);int saved=errno;
    int injected=close_error_index<4?close_errors[close_error_index++]:0;
    if(injected){errno=injected;return EOF;}
    errno=saved;return rc;
}
static size_t checked_fread(void *p,size_t size,size_t n,FILE *f) {
    io_calls++;
    size_t result=fread(p,size,partial_read && n>2?2:n,f);
    if(partial_read){forced_error=f;errno=EIO;}
    return result;
}
static size_t checked_fwrite(const void *p,size_t size,size_t n,FILE *f) {
    io_calls++;
    size_t result=fwrite(p,size,partial_write && n>3?3:n,f);
    if(partial_write){forced_error=f;errno=ENOSPC;}
    return result;
}
static int checked_ferror(FILE *f) { return f==forced_error?1:ferror(f); }
static int checked_fseek(FILE *f,long offset,int whence) {
    io_calls++;
    if(fail_seek){errno=ESPIPE;return -1;}
    return fseek(f,offset,whence);
}
/* System headers are already read: SDK stdio macros must not replace hooks. */
#undef fread
#undef fwrite
#undef ferror
#undef fseek
#define calloc checked_calloc
#define free checked_free
#define tmpfile checked_tmpfile
#define fclose checked_fclose
#define fread checked_fread
#define fwrite checked_fwrite
#define ferror checked_ferror
#define fseek checked_fseek
#include "../src/nsi_cap.c"
#include "../src/nsi_file.c"
#undef calloc
#undef free
#undef tmpfile
#undef fclose
#undef fread
#undef fwrite
#undef ferror
#undef fseek

static void errors(int first,int second) {
    memset(close_errors,0,sizeof close_errors);close_errors[0]=first;close_errors[1]=second;close_error_index=0;
}
static NlFileService *create(void) {
    NlFileService *s=NULL;CHECK(nl_file_service_create(&s).status==NL_FILE_OK);CHECK(s);return s;
}
static NlFileToken acquire(NlFileService *s,uint32_t rights) {
    NlFileToken t={0};CHECK(nl_file_acquire_temp(s,rights,&t).status==NL_FILE_OK);return t;
}
static void empty(void) { CHECK(live_files==0);CHECK(live_allocations==0);CHECK(host_opens==host_closes); }
static void reject_untouched(NlFileResult r,NlFileStatus status,unsigned before) {
    CHECK(r.status==status && !r.consumed && r.bytes==0);CHECK(io_calls==before);
}
static void io_and_lifetimes(void) {
    FILE *sentinel=tmpfile();CHECK(sentinel);int sentinel_fd=fileno(sentinel);
    NlFileService *s=create();NlFileToken a=acquire(s,FILE_RIGHTS);
    int file_fd=fileno(s->files[a.cap.slot].stream);
    const unsigned char bytes[]={1,0,2,3,255,4};unsigned char buffer[12];memset(buffer,0x5a,sizeof buffer);
    NlFileResult r=nl_file_write(s,&a,bytes,sizeof bytes);CHECK(r.status==NL_FILE_OK && r.bytes==sizeof bytes);
    unsigned before=io_calls;reject_untouched(nl_file_read(s,&a,buffer,1),NL_FILE_DIRECTION,before);CHECK(buffer[0]==0x5a);
    fail_seek=1;r=nl_file_rewind(s,&a);CHECK(r.status==NL_FILE_IO && r.host_errno==ESPIPE);fail_seek=0;
    before=io_calls;reject_untouched(nl_file_read(s,&a,buffer,1),NL_FILE_DIRECTION,before);
    CHECK(nl_file_read(s,&a,NULL,0).status==NL_FILE_OK);CHECK(io_calls==before);
    CHECK(nl_file_rewind(s,&a).status==NL_FILE_OK);
    r=nl_file_read(s,&a,buffer,sizeof buffer);CHECK(r.status==NL_FILE_OK && r.bytes==sizeof bytes && r.eof);CHECK(!memcmp(buffer,bytes,sizeof bytes));
    before=io_calls;reject_untouched(nl_file_write(s,&a,bytes,1),NL_FILE_DIRECTION,before);
    fail_seek=1;CHECK(nl_file_rewind(s,&a).status==NL_FILE_IO);fail_seek=0;
    before=io_calls;reject_untouched(nl_file_write(s,&a,bytes,1),NL_FILE_DIRECTION,before);
    NlFileToken stale=a;unsigned closes=host_closes;
    r=nl_file_transfer(s,&a,&a);CHECK(r.status==NL_FILE_OK && r.consumed);CHECK(host_closes==closes);CHECK(!file_same_cap(a.cap,stale.cap));
    before=io_calls;reject_untouched(nl_file_write(s,&a,bytes,1),NL_FILE_DIRECTION,before);
    CHECK(nl_file_consume_close(s,&stale).status==NL_FILE_TOKEN);CHECK(host_closes==closes);
    CHECK(nl_file_rewind(s,&a).status==NL_FILE_OK);CHECK(nl_file_write(s,&a,bytes,1).status==NL_FILE_OK);
    r=nl_file_consume_close(s,&a);CHECK(r.status==NL_FILE_OK && r.consumed);CHECK(host_closes==closes+1);
    errno=0;CHECK(fcntl(file_fd,F_GETFD)==-1 && errno==EBADF);
    CHECK(nl_file_consume_close(s,&a).status==NL_FILE_TOKEN);CHECK(host_closes==closes+1);
    NlFileToken b=acquire(s,FILE_RIGHTS);closes=host_closes;
    CHECK(nl_file_consume_close(s,&stale).status==NL_FILE_TOKEN);CHECK(host_closes==closes);
    CHECK(nl_file_write(s,&b,bytes,sizeof bytes).status==NL_FILE_OK);
    CHECK(fcntl(sentinel_fd,F_GETFD)>=0);CHECK(fputs("sentinel",sentinel)>=0);CHECK(fflush(sentinel)==0);
    CHECK(nl_file_service_dispose(s).status==NL_FILE_OK);CHECK(live_files==0 && live_allocations==2);
    CHECK(nl_file_write(s,&b,bytes,1).status==NL_FILE_DISPOSED);CHECK(nl_file_service_dispose(s).status==NL_FILE_OK);
    CHECK(nl_file_service_destroy(s).status==NL_FILE_OK);CHECK(fclose(sentinel)==0);empty();
}
static void identities_and_rights(void) {
    NlFileService *s=create(),*other=create();NlFileToken read_only=acquire(s,NL_CAP_READ),write_only=acquire(s,NL_CAP_WRITE);
    unsigned before=io_calls;char b='x';
    reject_untouched(nl_file_write(s,&read_only,&b,1),NL_FILE_RIGHTS,before);
    reject_untouched(nl_file_read(s,&write_only,&b,1),NL_FILE_RIGHTS,before);
    reject_untouched(nl_file_rewind(s,&write_only),NL_FILE_RIGHTS,before);
    reject_untouched(nl_file_read(other,&read_only,&b,1),NL_FILE_TOKEN,before);
    NlFileToken sentinel=write_only,out=sentinel;CHECK(nl_file_transfer(s,&read_only,&out).status==NL_FILE_RIGHTS);CHECK(!memcmp(&out,&sentinel,sizeof out));
    NlCapSlot *slot=&s->caps->slots[read_only.cap.slot];char old=slot->type_id[0];slot->type_id[0]='X';CHECK(nl_file_consume_close(s,&read_only).status==NL_FILE_TOKEN);slot->type_id[0]=old;
    old=slot->service_id[0];slot->service_id[0]='X';CHECK(nl_file_consume_close(s,&read_only).status==NL_FILE_TOKEN);slot->service_id[0]=old;
    NlFileToken wrong=read_only;wrong.cap.generation++;CHECK(nl_file_consume_close(s,&wrong).status==NL_FILE_TOKEN);
    CHECK(nl_file_service_destroy(other).status==NL_FILE_OK);
    CHECK(nl_file_consume_close(s,&read_only).status==NL_FILE_OK);CHECK(nl_file_consume_close(s,&write_only).status==NL_FILE_OK);
    /* I simulate a stale token at reused context storage without relying on malloc's reuse choice. */
    uint64_t prior=s->identity;s->identity=++file_context_counter;NlFileToken fresh=acquire(s,FILE_RIGHTS);
    wrong=fresh;wrong.context_id=prior;before=host_closes;CHECK(nl_file_consume_close(s,&wrong).status==NL_FILE_TOKEN);CHECK(host_closes==before);
    CHECK(nl_file_service_destroy(s).status==NL_FILE_OK);empty();
}
static void capacity_and_generation(void) {
    NlFileService *s=create();NlFileToken all[NL_CAP_PRIVATE_SLOTS];
    for(unsigned i=0;i<NL_CAP_PRIVATE_SLOTS;i++)all[i]=acquire(s,FILE_RIGHTS);
    NlFileToken output=all[1],saved=output;unsigned opens=host_opens;
    CHECK(nl_file_acquire_temp(s,FILE_RIGHTS,&output).status==NL_FILE_CAPACITY);CHECK(host_opens==opens);CHECK(!memcmp(&output,&saved,sizeof output));
    NlFileToken original=all[0];CHECK(nl_file_transfer(s,&all[0],&all[0]).status==NL_FILE_CAPACITY);CHECK(!memcmp(&all[0],&original,sizeof original));
    CHECK(nl_file_transfer(s,&all[0],&output).status==NL_FILE_CAPACITY);CHECK(!memcmp(&output,&saved,sizeof output));CHECK(nl_file_write(s,&all[0],"a",1).status==NL_FILE_OK);
    CHECK(nl_file_consume_close(s,&all[1]).status==NL_FILE_OK);
    CHECK(nl_file_transfer(s,&all[0],&output).status==NL_FILE_OK);CHECK(nl_file_consume_close(s,&all[0]).status==NL_FILE_TOKEN);
    CHECK(nl_file_service_destroy(s).status==NL_FILE_OK);empty();
    s=create();NlFileToken previous={0};
    for(unsigned i=0;i<320;i++) {NlFileToken t=acquire(s,FILE_RIGHTS);if(i){CHECK(t.cap.slot==previous.cap.slot);CHECK(t.cap.generation>previous.cap.generation);CHECK(nl_file_consume_close(s,&previous).status==NL_FILE_TOKEN);}CHECK(nl_file_consume_close(s,&t).status==NL_FILE_OK);previous=t;}
    NlFileToken a=acquire(s,FILE_RIGHTS);s->caps->next_generation=UINT32_MAX;output=a;saved=output;
    opens=host_opens;unsigned closes=host_closes;
    errors(EIO,0);NlFileResult r=nl_file_acquire_temp(s,FILE_RIGHTS,&output);
    CHECK(r.status==NL_FILE_LIMIT && r.cleanup_failed && r.cleanup_errno==EIO && !r.consumed);CHECK(host_opens==opens+1 && host_closes==closes+1);CHECK(!memcmp(&output,&saved,sizeof output));errors(0,0);
    CHECK(nl_file_transfer(s,&a,&output).status==NL_FILE_LIMIT);CHECK(!memcmp(&output,&saved,sizeof output));CHECK(nl_file_write(s,&a,"x",1).status==NL_FILE_OK);
    CHECK(nl_file_service_destroy(s).status==NL_FILE_OK);empty();
    uint64_t old=file_context_counter;file_context_counter=UINT64_MAX;NlFileService *unchanged=(NlFileService *)(uintptr_t)1;CHECK(nl_file_service_create(&unchanged).status==NL_FILE_LIMIT && unchanged==(NlFileService *)(uintptr_t)1);file_context_counter=old;
}
static void faults(void) {
    unsigned succeeded=0;
    for(long budget=0;budget<3;budget++) {allocation_budget=budget;NlFileService *s=(NlFileService *)(uintptr_t)1;NlFileResult r=nl_file_service_create(&s);allocation_budget=-1;if(r.status==NL_FILE_OK){succeeded++;CHECK(nl_file_service_destroy(s).status==NL_FILE_OK);}else CHECK(r.status==NL_FILE_MEMORY && s==(NlFileService *)(uintptr_t)1);empty();}
    CHECK(succeeded==1);
    NlFileService *s=create();NlFileToken t={0},saved=t;fail_open=1;NlFileResult r=nl_file_acquire_temp(s,FILE_RIGHTS,&t);fail_open=0;CHECK(r.status==NL_FILE_IO && r.host_errno==EACCES && !r.consumed);CHECK(!memcmp(&t,&saved,sizeof t));
    t=acquire(s,FILE_RIGHTS);partial_write=1;r=nl_file_write(s,&t,"abcdef",6);partial_write=0;forced_error=NULL;CHECK(r.status==NL_FILE_IO && r.bytes==3 && r.host_errno==ENOSPC && !r.consumed);
    CHECK(nl_file_rewind(s,&t).status==NL_FILE_OK);char buffer[8]={0};partial_read=1;r=nl_file_read(s,&t,buffer,8);partial_read=0;forced_error=NULL;CHECK(r.status==NL_FILE_IO && r.bytes==2 && r.host_errno==EIO && !r.consumed);CHECK(!memcmp(buffer,"ab",2));
    errors(ENOSPC,0);r=nl_file_consume_close(s,&t);CHECK(r.status==NL_FILE_IO && r.host_errno==ENOSPC && r.consumed);CHECK(nl_file_consume_close(s,&t).status==NL_FILE_TOKEN);errors(0,0);
    NlFileToken a=acquire(s,FILE_RIGHTS),b=acquire(s,FILE_RIGHTS);(void)a;(void)b;
    errors(EIO,ENOSPC);r=nl_file_service_destroy(s);CHECK(r.status==NL_FILE_IO && r.host_errno==EIO && r.consumed);errors(0,0);empty();
    s=create();t=acquire(s,FILE_RIGHTS);unsigned before=io_calls;
    reject_untouched(nl_file_read(s,&t,NULL,1),NL_FILE_ARGUMENT,before);reject_untouched(nl_file_write(s,&t,NULL,1),NL_FILE_ARGUMENT,before);
    reject_untouched(nl_file_read(s,&t,buffer,SIZE_MAX),NL_FILE_ARGUMENT,before);
    CHECK(nl_file_service_destroy(s).status==NL_FILE_OK);empty();
}
static void private_cap_lifecycle(void) {
    NlCapTable *t=nl_cap_table_create();CHECK(t);NlCap a,b;
    CHECK(nl_cap_private_mint(t,"t","s",NL_CAP_READ|NL_CAP_TRANSFER,&a)==NL_CAP_OK);
    uint64_t cell=0;CHECK(nl_cap_forth_bind(t,&a,&cell)==NL_CAP_OK);
    b=a;CHECK(nl_cap_private_transfer(t,&a,&a)==NL_CAP_OK);CHECK(nl_cap_check(t,&b,0)!=NL_CAP_OK);
    CHECK(nl_cap_forth_lookup(t,cell,&b)!=NL_CAP_OK);
    CHECK(nl_cap_private_consume(t,&a)==NL_CAP_OK);CHECK(nl_cap_private_consume(t,&a)!=NL_CAP_OK);
    nl_cap_table_destroy(t);empty();
}
int main(void) {
    errors(0,0);io_and_lifetimes();identities_and_rights();capacity_and_generation();faults();private_cap_lifecycle();
    printf("PASS %u checks; %u real opens and %u real closes; %u allocator calls; no live resources\n",checks,host_opens,host_closes,allocations);return 0;
}
