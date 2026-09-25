/* I check my freestanding memory ABI through real Wasm links and engines. */
#include "../../src/nanoisa/managed_strings.c"
#define CHECK(x) do { if(!(x))return __LINE__; } while(0)
int nms_memory_tests(void) {
    unsigned char source[521],destination[525];
    for(size_t i=0;i<sizeof source;i++)source[i]=(unsigned char)(i*37u+11u);
    for(size_t i=0;i<sizeof destination;i++)destination[i]=0xa5;
    CHECK(memcpy(destination+1,source+2,517)==destination+1);
    for(size_t i=0;i<sizeof destination;i++) {
        unsigned char expected=i>=1 && i<518?source[i+1]:0xa5;
        CHECK(destination[i]==expected);
    }
    CHECK(memset(destination+3,0x1d3,511)==destination+3);
    for(size_t i=0;i<sizeof destination;i++) {
        unsigned char expected=i>=3 && i<514?0xd3:
            i>=1 && i<518?source[i+1]:0xa5;
        CHECK(destination[i]==expected);
    }
    CHECK(memset(destination+2,-1,1)==destination+2);
    CHECK(destination[2]==0xff && destination[1]==source[2]);
    CHECK(memset(destination+2,0,0)==destination+2 && destination[2]==0xff);
    CHECK(memcpy(destination+2,source,0)==destination+2 && destination[2]==0xff);
    for(size_t i=0;i<sizeof source;i++)CHECK(source[i]==(unsigned char)(i*37u+11u));
    CHECK(nms_reserved_entry("memcpy") && nms_reserved_entry("memset"));
    CHECK(!nms_reserved_entry("memcpy_other") && !nms_reserved_entry("memset_other"));
    return 0;
}
