/* I qualify owned parsing only; no generated adapter or provider is invoked. */
#ifdef NDEBUG
#error I require active ownership assertions
#endif
#include "module_builder.h"
#include "cJSON.h"
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
static size_t calls,live,fail_at=SIZE_MAX;
static bool persistent;
typedef union { max_align_t alignment; size_t magic; } Header;
static void *observed_malloc(size_t n) {
    size_t at=calls++;
    if(at==fail_at||(persistent&&at>fail_at))return NULL;
    if(n>SIZE_MAX-sizeof(Header))return NULL;
    Header *h=malloc(sizeof *h+n);if(!h)return NULL;
    h->magic=0xabcdefu;live++;return h+1;
}
static void observed_free(void *p) {
    if(!p)return;
    Header *h=(Header *)p-1;assert(h->magic==0xabcdefu&&live);live--;h->magic=0;free(h);
}
#define malloc observed_malloc
#include "../src/module_sdk_abi.inc"
#undef malloc
static const char valid[]=
"{\"typed_abi\":{\"version\":1,\"target\":\"test-host\",\"types\":["
"{\"semantic\":\"int\",\"c_type\":\"int64_t\",\"kind\":\"int\"},"
"{\"semantic\":\"string\",\"c_type\":\"BorrowedString\",\"kind\":\"string\"},"
"{\"semantic\":\"Handle\",\"c_type\":\"HandlePtr\",\"kind\":\"opaque\"},"
"{\"semantic\":\"Packet\",\"c_type\":\"Packet\",\"kind\":\"record\",\"fields\":[\"value\",\"owner\"]},"
"{\"semantic\":\"(int,string)\",\"c_type\":\"Pair\",\"kind\":\"tuple\",\"fields\":[\"first\",\"second\"]},"
"{\"semantic\":\"Choice\",\"c_type\":\"Choice\",\"kind\":\"union\",\"tag\":\"kind\",\"variants\":[{\"tag\":\"CHOICE_A\",\"fields\":[\"data.a\"]},{\"tag\":\"CHOICE_B\",\"fields\":[]}]},"
"{\"semantic\":\"Mode\",\"c_type\":\"Mode\",\"kind\":\"enum\",\"variants\":[\"MODE_A\",\"MODE_B\"]},"
"{\"semantic\":\"array<Packet>\",\"c_type\":\"DynArray\",\"kind\":\"array\",\"abi\":\"dyn_array_v2\",\"element\":3},"
"{\"semantic\":\"fn(int)->int\",\"c_type\":\"Callback\",\"kind\":\"function\"}"
"],\"functions\":[{\"name\":\"read\",\"symbol\":\"provider_read\",\"parameters\":[2,7],\"result\":4},{\"name\":\"release\",\"symbol\":\"provider_release\",\"parameters\":[2],\"result\":null}]}}";
static void refusal(const char *text) {
    cJSON *j=cJSON_Parse(text);assert(j);ModuleBuildMetadata m={0};size_t baseline=live;
    assert(!module_parse_typed_abi(j,&m)&&!m.typed_abi&&live==baseline);cJSON_Delete(j);
}
int main(void) {
    cJSON_Hooks hooks={observed_malloc,observed_free};cJSON_InitHooks(&hooks);
    cJSON *j=cJSON_Parse(valid);assert(j);ModuleBuildMetadata m={0};size_t baseline=live;calls=0;
    assert(module_parse_typed_abi(j,&m)&&m.typed_abi);size_t measured=calls;
    char *saved=m.typed_abi;assert(!module_parse_typed_abi(j,&m)&&m.typed_abi==saved);
    cJSON_Delete(j);assert(strstr(saved,"provider_read"));
    cJSON *round=cJSON_CreateObject();cJSON *schema=cJSON_Parse(saved);assert(round&&schema&&cJSON_AddItemToObject(round,"typed_abi",schema));
    ModuleBuildMetadata again={0};assert(module_parse_typed_abi(round,&again)&&!strcmp(saved,again.typed_abi));
    observed_free(again.typed_abi);observed_free(saved);cJSON_Delete(round);assert(live==0);
    j=cJSON_Parse(valid);assert(j);baseline=live;
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<measured;i++) {
        calls=0;fail_at=i;persistent=mode!=0;m=(ModuleBuildMetadata){0};
        assert(!module_parse_typed_abi(j,&m)&&!m.typed_abi&&live==baseline);
        fail_at=SIZE_MAX;persistent=false;m=(ModuleBuildMetadata){0};
        assert(module_parse_typed_abi(j,&m));observed_free(m.typed_abi);assert(live==baseline);
    }
    cJSON_Delete(j);assert(live==0);
    j=cJSON_Parse("{\"name\":\"old\",\"unknown_old_field\":true}");m=(ModuleBuildMetadata){0};calls=0;
    assert(module_parse_typed_abi(j,&m)&&!m.typed_abi&&!calls);cJSON_Delete(j);
    const char *bad[]={
        "{\"typed_abi\":null}","{\"typed_abi\":{},\"typed_abi\":{}}",
        "{\"typed_abi\":{\"version\":1,\"version\":1,\"target\":\"x\",\"types\":[],\"functions\":[]}}",
        "{\"typed_abi\":{\"version\":2,\"target\":\"x\",\"types\":[],\"functions\":[]}}",
        "{\"typed_abi\":{\"version\":1,\"target\":\"x\",\"types\":[],\"functions\":[],\"extra\":0}}",
        "{\"typed_abi\":{\"version\":1,\"target\":\"x\",\"types\":[{\"semantic\":\"int\",\"c_type\":\"int;bad\",\"kind\":\"int\"}],\"functions\":[]}}",
        "{\"typed_abi\":{\"version\":1,\"target\":\"x\",\"types\":[{\"semantic\":\"A\",\"c_type\":\"A\",\"kind\":\"array\",\"abi\":\"dyn_array_v2\",\"element\":1}],\"functions\":[]}}",
        "{\"typed_abi\":{\"version\":1,\"target\":\"x\",\"types\":[{\"semantic\":\"A\",\"c_type\":\"A\",\"kind\":\"record\",\"fields\":[\"a..b\"]}],\"functions\":[]}}",
        "{\"typed_abi\":{\"version\":1,\"target\":\"x\",\"types\":[{\"semantic\":\"A\",\"c_type\":\"A\",\"kind\":\"record\",\"fields\":[\"a\",\"a\"]}],\"functions\":[]}}",
        "{\"typed_abi\":{\"version\":1,\"target\":\"x\",\"types\":[],\"functions\":[{\"name\":\"f\",\"symbol\":\"f\",\"parameters\":[0],\"result\":null}]}}"
    };
    for(size_t i=0;i<sizeof bad/sizeof bad[0];i++)refusal(bad[i]);
    assert(live==0);cJSON_InitHooks(NULL);
    puts("I passed owned typed ABI parsing, roundtrip and allocation rollback without admission.");return 0;
}
