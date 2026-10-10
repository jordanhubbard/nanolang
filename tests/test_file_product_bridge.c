#include "runtime/service_product.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
static int calls;
static bool expected_tcp,expected_websocket;
int nl_service_publish(const uint8_t *bytes,size_t size,const NlServiceShadow *shadows,
                       size_t count,const NlServiceProductOptions *options) {
    ++calls;
    assert(size==3 && bytes[0]==0 && bytes[1]==128 && bytes[2]==255);
    assert(count==1 && shadows[0].size==2 && shadows[0].bytes[1]==1);
    assert(!strcmp(shadows[0].origin,"import.nano") && !strcmp(shadows[0].name,"test"));
    assert(options->emit_nvm && options->allow_temporary_files && !options->run);
    assert(options->allow_tcp_connections==expected_tcp);
    assert((options->websocket!=NULL)==expected_websocket);
    if(expected_websocket) {
        assert(options->websocket->revision==1 && options->websocket->allow_connections && options->websocket->allow_lookup);
        assert(options->websocket->max_timeout_ms==60000 && !strcmp(options->websocket->resolver_helper,"/resolver"));
    }
    return 7;
}
static long remaining=-1;
static void *checked_malloc(size_t size) {
    if(remaining==0)return NULL;if(remaining>0)--remaining;return malloc(size);
}
static void *checked_calloc(size_t count,size_t size) {
    if(remaining==0)return NULL;if(remaining>0)--remaining;return calloc(count,size);
}
static void *checked_realloc(void *p,size_t size) {
    if(remaining==0)return NULL;if(remaining>0)--remaining;return realloc(p,size);
}
#define malloc checked_malloc
#define calloc checked_calloc
#define realloc checked_realloc
#include "../modules/file_product/file_product.c"
#undef malloc
#undef calloc
#undef realloc
int main(void) {
    assert(!nl_file_product_new("",".",0));
    assert(!nl_file_product_new("out",".",8));
    for(long failure=0;failure<10;failure++) {
        remaining=failure;calls=0;
        NlFileProduct *p=nl_file_product_new("out",".",3);
        if(p) {
            bool staged=nl_file_product_append(p,"0080ff")==0 &&
                nl_file_product_seal(p,"root.nano","")==0 &&
                nl_file_product_append(p,"0001")==0 &&
                nl_file_product_seal(p,"import.nano","test")==0;
            assert(nl_file_product_publish(p)==(staged?7:1));
            assert(calls==(staged?1:0));
            assert(nl_file_product_publish(p)==1);
            assert(nl_file_product_append(p,"00")==1);
        }
        nl_file_product_free(p);
    }
    remaining=-1;
    expected_tcp=true;
    NlFileProduct *tcp=nl_file_product_new("out",".",7);assert(tcp);
    assert(nl_file_product_append(tcp,"0080ff")==0 && nl_file_product_seal(tcp,"root.nano","")==0);
    assert(nl_file_product_append(tcp,"0001")==0 && nl_file_product_seal(tcp,"import.nano","test")==0);
    assert(nl_file_product_publish(tcp)==7);nl_file_product_free(tcp);
    expected_tcp=false;expected_websocket=true;
    NlFileProduct *websocket=nl_file_product_new("out",".",3);assert(websocket);
    char resolver[]="/resolver";
    assert(nl_file_product_websocket(websocket,3,resolver)==0);resolver[1]='X';
    assert(nl_file_product_append(websocket,"0080ff")==0 && nl_file_product_seal(websocket,"root.nano","")==0);
    assert(nl_file_product_append(websocket,"0001")==0 && nl_file_product_seal(websocket,"import.nano","test")==0);
    assert(nl_file_product_publish(websocket)==7);nl_file_product_free(websocket);expected_websocket=false;
    const char *bad_helpers[]={"relative","/ungranted",""};
    const int bad_flags[]={3,1,2};
    for(unsigned i=0;i<3;i++) {
        websocket=nl_file_product_new("out",".",1);assert(websocket);
        assert(nl_file_product_websocket(websocket,bad_flags[i],bad_helpers[i])==1);
        assert(!nl_file_product_valid(websocket));nl_file_product_free(websocket);
    }
    websocket=nl_file_product_new("out",".",1);assert(websocket);
    remaining=0;assert(nl_file_product_websocket(websocket,3,"/resolver")==1);remaining=-1;
    assert(!nl_file_product_valid(websocket));nl_file_product_free(websocket);
    websocket=nl_file_product_new("out",".",1);assert(websocket);
    assert(nl_file_product_websocket(websocket,0,"")==0);
    assert(nl_file_product_websocket(websocket,1,"")==1);nl_file_product_free(websocket);
    const char *bad[]={"","0","gg","FF"};
    for(size_t i=0;i<sizeof bad/sizeof bad[0];i++) {
        NlFileProduct *p=nl_file_product_new("out",".",1);assert(p);
        assert(nl_file_product_append(p,"00")==0);
        assert(nl_file_product_append(p,bad[i])==1);
        assert(!nl_file_product_valid(p) && nl_file_product_publish(p)==1);
        nl_file_product_free(p);
    }
    NlFileProduct *p=nl_file_product_new("out",".",0);assert(p);
    p->total=PRODUCT_BYTES;
    assert(nl_file_product_append(p,"00")==1);nl_file_product_free(p);
    p=nl_file_product_new("out",".",0);assert(p);
    assert(nl_file_product_append(p,"00")==0);
    assert(nl_file_product_seal(p,"root","not-main")==1);
    assert(nl_file_product_publish(p)==1);nl_file_product_free(p);
    puts("I pass byte-only File publication staging and allocation controls.");
    return 0;
}
