/* I report actual private/public C decoding and canonical numeric field bytes. */
#include "../src/nanoisa/ownership_layouts_private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv) {
    if(argc<2)return 2;
    int profile=atoi(argv[1]);
    if(profile<0 || profile>3)return 2;
    for(int i=2;i<argc;i++) {
        FILE *f=fopen(argv[i],"rb");
        if(!f || fseek(f,0,SEEK_END))return 2;
        long length=ftell(f);
        if(length<0 || fseek(f,0,SEEK_SET))return 2;
        unsigned char *bytes=malloc(length?(size_t)length:1);
        if(!bytes)return 2;
        if(fread(bytes,1,(size_t)length,f)!=(size_t)length || fclose(f))return 2;
        NvmV2Layout sentinel={0};
        NvmV2Layouts out={.items=&sentinel,.count=12345};
        bool allocation_failed=true;
        NvmV2Result status;
        if(profile==0) {out=(NvmV2Layouts){0};status=nvm_v2_layouts_decode(bytes,(size_t)length,&out);}
        else if(profile==1)status=nvm_ownership_layouts_private_decode(bytes,(size_t)length,&out);
        else if(profile==2)status=nvm_ownership_mixed_layouts_private_decode_detailed(bytes,(size_t)length,&out,&allocation_failed);
        else status=nvm_ownership_map_layouts_private_decode_detailed(bytes,(size_t)length,&out,&allocation_failed);
        if(profile>=2 && allocation_failed)return 3; /* No allocation denial in this oracle. */
        if(profile && status!=NVM_V2_OK && (out.items!=&sentinel || out.count!=12345))return 4;
        uint32_t count=0;size_t size=0;
        unsigned char *encoded=NULL;
        if(status==NVM_V2_OK) {
            count=out.count;size=nvm_v2_layouts_encoded_size(&out);
            encoded=malloc(size?size:1);if(!encoded)return 2;
            if(nvm_v2_layouts_encode(&out,encoded,size)!=NVM_V2_OK)return 2;
        }
        /* I compare both mixed entry points, including failure-atomic outputs. */
        if(profile==2) {
            NvmV2Layouts other={.items=&sentinel,.count=12345};
            NvmV2Result second=nvm_ownership_mixed_layouts_private_decode(bytes,(size_t)length,&other);
            if(second!=status)return 5;
            if(second==NVM_V2_OK) {
                size_t n=nvm_v2_layouts_encoded_size(&other);
                unsigned char *copy=malloc(n?n:1);if(!copy)return 2;
                if(nvm_v2_layouts_encode(&other,copy,n)!=NVM_V2_OK || n!=size || memcmp(copy,encoded,n))return 5;
                free(copy);nvm_v2_layouts_free(&other);
            } else if(other.items!=&sentinel || other.count!=12345)return 5;
        }
        printf("%u:%u:%zu:",(unsigned)status,(unsigned)count,size);
        for(size_t j=0;j<size;j++)printf("%02x",(unsigned)encoded[j]);
        putchar('\n');
        if(status==NVM_V2_OK)nvm_v2_layouts_free(&out);
        free(encoded);free(bytes);
    }
    return 0;
}
