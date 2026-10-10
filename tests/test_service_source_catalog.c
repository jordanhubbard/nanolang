/* I compare File's legacy view and every query with the selected File API,
 * then exercise exact TCP identity, domains, boundaries and output retention. */
#include "nanoisa/file_source_plan.h"
#include "nanoisa/service_source_catalog.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    assert(nl_service_source_catalog_id(NULL) == 0);
    assert(nl_service_source_catalog_id("nsi:nanolang/filesystem") == 1);
    assert(nl_service_source_catalog_id("nsi:nanolang/net") == 2);
    assert(nl_service_source_catalog_id("nsi:nanolang/net#Socket") == 0);
    assert(nl_service_source_catalog_id("nsi:nanolang/netx") == 0);
    for(int64_t k=-1;k<4;k++) for(int64_t i=-1;i<11;i++)
        for(int64_t f=-1;f<12;f++) for(int64_t j=-1;j<13;j++) {
            assert(!strcmp(nl_file_source_catalog_string(k,i,f,j),
                           nl_service_source_catalog_string(1,k,i,f,j)));
            assert(nl_file_source_catalog_number(k,i,f,j)==
                   nl_service_source_catalog_number(1,k,i,f,j));
        }
    char old[32768],text[32768];size_t old_size=0,needed=0;
    assert(nl_file_source_catalog_view(old,sizeof old,&old_size));
    assert(nl_service_source_catalog_view(1,text,sizeof text,&needed));
    assert(needed==old_size && !memcmp(old,text,needed));
    int64_t bad[]={INT64_MIN,-1,0,3,INT64_MAX};
    for(size_t i=0;i<sizeof bad/sizeof bad[0];i++) {
        assert(nl_service_source_catalog_count(bad[i],1)==-1);
        assert(nl_service_source_catalog_count(bad[i],2)==-1);
        assert(!strcmp(nl_service_source_catalog_string(bad[i],0,0,0,0),""));
        assert(!strcmp(nl_service_source_catalog_string(bad[i],1,0,1,0),""));
        assert(nl_service_source_catalog_number(bad[i],1,0,0,0)==-1);
        needed=73;memset(text,0xa5,sizeof text);
        assert(!nl_service_source_catalog_view(bad[i],text,sizeof text,&needed));
        assert(needed==73);
        for(size_t b=0;b<sizeof text;b++) assert((unsigned char)text[b]==0xa5);
    }
    assert(nl_service_source_catalog_count(2,1)==9);
    assert(nl_service_source_catalog_count(2,2)==5);
    assert(nl_service_source_catalog_count(2,0)==-1);
    assert(!strcmp(nl_service_source_catalog_string(2,1,0,0,0),"nsi:nanolang/net#Conn"));
    assert(!strcmp(nl_service_source_catalog_string(2,1,8,1,0),"Endpoint"));
    assert(!strcmp(nl_service_source_catalog_string(2,1,9,1,0),""));
    assert(nl_service_source_catalog_number(2,1,1,1,0)==11);
    assert(nl_service_source_catalog_number(2,1,8,1,0)==7);
    const int64_t domains[]={4,2,2,2,2,3,2};
    for(int64_t j=0;j<7;j++)assert(nl_service_source_catalog_number(2,1,8,2,j)==domains[j]);
    assert(nl_service_source_catalog_number(2,1,8,2,7)==-1);
    assert(nl_service_source_catalog_number(2,2,0,3,0)==0);
    assert(nl_service_source_catalog_number(2,2,1,3,0)==1);
    assert(nl_service_source_catalog_number(2,2,4,3,0)==2);
    assert(!strcmp(nl_service_source_catalog_string(2,2,0,7,0),"nsi:nanolang/net#Conn"));
    assert(!strcmp(nl_service_source_catalog_string(2,2,0,7,1),""));
    for(int64_t catalog=1;catalog<=2;catalog++) {
        needed=0;assert(nl_service_source_catalog_view(catalog,NULL,0,&needed));
        size_t capacity=needed,unchanged=73;
        memset(text,0xa5,sizeof text);
        assert(!nl_service_source_catalog_view(catalog,text,capacity-1,&unchanged));
        assert(unchanged==73);
        for(size_t b=0;b<sizeof text;b++)assert((unsigned char)text[b]==0xa5);
        assert(nl_service_source_catalog_view(catalog,text,capacity,&needed));
        assert(needed==capacity && strlen(text)+1==needed);
        printf("CAT%lld:%s\n",(long long)catalog,text);
    }
    return 0;
}
