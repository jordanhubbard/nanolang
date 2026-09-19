#include "../src/nsi_file.h"
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while (0)
int main(void) {
    NlFileService *service=NULL;
    CHECK(nl_file_service_create(&service).status==NL_FILE_OK);
    NlFileToken token={0},old;
    CHECK(nl_file_acquire_temp(service,NL_CAP_READ|NL_CAP_WRITE|NL_CAP_TRANSFER,&token).status==NL_FILE_OK);
    const unsigned char value[]={0,1,2,255,0};unsigned char copy[9]={0};
    NlFileResult r=nl_file_write(service,&token,value,sizeof value);CHECK(r.status==NL_FILE_OK && r.bytes==sizeof value);
    CHECK(nl_file_read(service,&token,copy,sizeof copy).status==NL_FILE_DIRECTION);
    CHECK(nl_file_rewind(service,&token).status==NL_FILE_OK);
    r=nl_file_read(service,&token,copy,sizeof copy);CHECK(r.status==NL_FILE_OK && r.eof && r.bytes==sizeof value && !memcmp(value,copy,sizeof value));
    CHECK(nl_file_write(service,&token,value,1).status==NL_FILE_DIRECTION);
    old=token;CHECK(nl_file_transfer(service,&token,&token).status==NL_FILE_OK);
    CHECK(nl_file_consume_close(service,&old).status==NL_FILE_TOKEN);
    CHECK(nl_file_consume_close(service,&token).status==NL_FILE_OK);
    CHECK(nl_file_consume_close(service,&token).status==NL_FILE_TOKEN);
    CHECK(nl_file_service_dispose(service).status==NL_FILE_OK);
    CHECK(nl_file_acquire_temp(service,NL_CAP_READ,&token).status==NL_FILE_DISPOSED);
    CHECK(nl_file_service_destroy(service).status==NL_FILE_OK);
    puts("PASS ordinary linked private local-file adapter");return 0;
}
