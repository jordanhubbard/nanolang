/* I expose the five real raw codecs and keep pointer controls separate. */
#include "../src/nanoisa/service_bindings.h"
#include "../src/nanoisa/service_file_nominal.h"
#include "../src/nanoisa/service_socket_nominal.h"
#include "../src/nanoisa/service_websocket_nominal.h"
#include "../src/nanoisa/service_multi_nominal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef union {
    NvmServiceBindings legacy;
    NvmFileNominalBindings file;
    NvmSocketNominalBindings tcp;
    NvmWebSocketNominalBindings ws;
    NvmMultiNominalBindings multi;
} Value;
static unsigned methods(unsigned p) { return p==4?4:5; }
static unsigned types(unsigned p) { return p==1?0:p==2?8:p==3?9:7; }
static NvmServiceResult check(unsigned p,const Value *v) {
    switch(p) {
        case 1:return nvm_service_bindings_check(v?&v->legacy:NULL);
        case 2:return nvm_file_nominal_check(v?&v->file:NULL);
        case 3:return nvm_socket_nominal_check(v?&v->tcp:NULL);
        case 4:return nvm_websocket_nominal_check(v?&v->ws:NULL);
        case 5:return nvm_multi_nominal_check(v?&v->multi:NULL);
        default:return NVM_SERVICE_ARGUMENT;
    }
}
static NvmServiceResult decode(unsigned p,const uint8_t *b,size_t n,Value *v) {
    switch(p) {
        case 1:return nvm_service_bindings_decode(b,n,v?&v->legacy:NULL);
        case 2:return nvm_file_nominal_decode(b,n,v?&v->file:NULL);
        case 3:return nvm_socket_nominal_decode(b,n,v?&v->tcp:NULL);
        case 4:return nvm_websocket_nominal_decode(b,n,v?&v->ws:NULL);
        case 5:return nvm_multi_nominal_decode(b,n,v?&v->multi:NULL);
        default:return NVM_SERVICE_ARGUMENT;
    }
}
static NvmServiceResult encode(unsigned p,const Value *v,uint8_t *b,size_t n,size_t *size) {
    switch(p) {
        case 1:return nvm_service_bindings_encode(v?&v->legacy:NULL,b,n,size);
        case 2:return nvm_file_nominal_encode(v?&v->file:NULL,b,n,size);
        case 3:return nvm_socket_nominal_encode(v?&v->tcp:NULL,b,n,size);
        case 4:return nvm_websocket_nominal_encode(v?&v->ws:NULL,b,n,size);
        case 5:return nvm_multi_nominal_encode(v?&v->multi:NULL,b,n,size);
        default:return NVM_SERVICE_ARGUMENT;
    }
}
static uint32_t *imports(Value *v,unsigned p,unsigned row) {
    switch(p) {case 1:return v->legacy.imports;case 2:return v->file.imports;
        case 3:return v->tcp.imports;case 4:return v->ws.imports;default:return v->multi.instances[row].imports;}
}
static uint32_t *layouts(Value *v,unsigned p,unsigned row) {
    switch(p) {case 1:return NULL;case 2:return v->file.layouts;
        case 3:return v->tcp.layouts;case 4:return v->ws.layouts;default:return v->multi.instances[row].layouts;}
}
static uint32_t u32(const uint8_t *b) {return b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;}
static unsigned carrier(unsigned p,const uint8_t *b,size_t n,Value *v) {
    if(p<1 || p>5 || n<4)return 99;
    uint32_t count=u32(b);
    if(p==5) {
        v->multi.count=count;
        if(count==0 || count==65)return n==4?0:99;
    } else if(count!=1)return 99;
    if(count>64 || n!=4+(size_t)count*60)return 99;
    for(unsigned i=0;i<count;i++) {
        const uint8_t *row=b+4+60*i;
        if(p==5)v->multi.instances[i].catalog=u32(row);
        for(unsigned j=0;j<(p==5?5:methods(p));j++)imports(v,p,i)[j]=u32(row+4+4*j);
        for(unsigned j=0;j<(p==5?9:types(p));j++)layouts(v,p,i)[j]=u32(row+24+4*j);
    }
    return 0;
}
static void report(unsigned p,Value *v) {
    NvmServiceResult status=check(p,v);
    uint8_t output[NVM_MULTI_NOMINAL_MAX_BYTES],prior[sizeof output];
    memset(output,0xa5,sizeof output);memcpy(prior,output,sizeof output);
    size_t size=123456;
    if(status!=NVM_SERVICE_OK) {
        assert(encode(p,v,output,sizeof output,&size)==status);
        assert(size==123456 && !memcmp(output,prior,sizeof output));
        assert(encode(p,v,NULL,0,&size)==status && size==123456);
        printf("%u\n",(unsigned)status);return;
    }
    assert(encode(p,v,NULL,0,&size)==NVM_SERVICE_OK);
    assert(size && size<=sizeof output);
    size_t required=size;size=123456;
    assert(encode(p,v,output,required-1,&size)==NVM_SERVICE_SIZE);
    assert(size==123456 && !memcmp(output,prior,sizeof output));
    assert(encode(p,v,output,sizeof output,NULL)==NVM_SERVICE_ARGUMENT);
    assert(!memcmp(output,prior,sizeof output));
    assert(encode(p,v,output,sizeof output,&size)==NVM_SERVICE_OK && size==required);
    unsigned count=p==5?v->multi.count:1;
    printf("0|%u,",count);
    for(unsigned i=0;i<count;i++) {
        printf("%u,",p==5?v->multi.instances[i].catalog:p==1?1:p-1);
        for(unsigned j=0;j<(p==5?5:methods(p));j++)printf("%u,",imports(v,p,i)[j]);
        for(unsigned j=0;j<(p==5?9:types(p));j++)printf("%u,",layouts(v,p,i)[j]);
    }
    putchar('|');for(size_t i=0;i<size;i++)printf("%02x",output[i]);putchar('\n');
}
int main(int argc,char **argv) {
    if(argc<3)return 2;
    unsigned profile=(unsigned)strtoul(argv[1],NULL,10);
    bool values=!strcmp(argv[2],"value");
    for(int i=3;i<argc;i++) {
        FILE *f=fopen(argv[i],"rb");if(!f || fseek(f,0,SEEK_END))return 2;
        long n=ftell(f);if(n<0 || fseek(f,0,SEEK_SET))return 2;
        uint8_t *bytes=malloc(n?(size_t)n:1);if(!bytes)return 2;
        if(fread(bytes,1,(size_t)n,f)!=(size_t)n || fclose(f))return 2;
        Value value,prior;memset(&value,0xa5,sizeof value);memcpy(&prior,&value,sizeof value);
        unsigned status;
        if(values)status=carrier(profile,bytes,(size_t)n,&value);
        else {
            assert(check(profile,NULL)==NVM_SERVICE_ARGUMENT);
            assert(decode(profile,NULL,(size_t)n,&value)==NVM_SERVICE_ARGUMENT);
            assert(decode(profile,bytes,(size_t)n,NULL)==NVM_SERVICE_ARGUMENT);
            assert(!memcmp(&value,&prior,sizeof value));
            status=decode(profile,bytes,(size_t)n,&value);
            if(status)assert(!memcmp(&value,&prior,sizeof value));
        }
        if(status)printf("%u\n",status);else report(profile,&value);
        free(bytes);
    }
    return 0;
}
