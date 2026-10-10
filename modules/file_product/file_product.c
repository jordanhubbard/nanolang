#include "file_product.h"
#include "../../src/runtime/service_product.h"
#include <stdlib.h>
#include <string.h>

#define PRODUCT_BYTES (64u*1024u*1024u)
#define PRODUCT_SHADOWS 64u
struct NlFileProduct {
    char *output,*root;
    uint8_t *pending,*main;
    size_t used,capacity,total,main_size,count;
    NlServiceShadow shadows[PRODUCT_SHADOWS];
    bool failed,finished;
    int64_t flags;
};
static bool text_ok(const char *s,bool empty) {
    return s && (empty || *s) && strlen(s)<=4096;
}
static char *copy_text(const char *s) {
    size_t size=strlen(s)+1;char *copy=malloc(size);
    if(copy)memcpy(copy,s,size);return copy;
}
NlFileProduct *nl_file_product_new(const char *output,const char *root,int64_t flags) {
    if(!text_ok(output,false) || !text_ok(root,false) || flags<0 || flags>7)return NULL;
    NlFileProduct *p=calloc(1,sizeof *p);if(!p)return NULL;
    p->output=copy_text(output);p->root=copy_text(root);p->flags=flags;
    if(!p->output || !p->root){nl_file_product_free(p);return NULL;}
    return p;
}
int64_t nl_file_product_valid(NlFileProduct *p){return p && !p->failed && !p->finished;}
static int hex(unsigned char c) {
    if(c>='0' && c<='9')return c-'0';
    if(c>='a' && c<='f')return c-'a'+10;
    return -1;
}
int64_t nl_file_product_append(NlFileProduct *p,const char *text) {
    if(!nl_file_product_valid(p))return 1;
    size_t size=text?strlen(text):0;
    if(!size || size>8192 || (size&1) || size/2>PRODUCT_BYTES-p->total)goto failed;
    for(size_t i=0;i<size;i++)if(hex((unsigned char)text[i])<0)goto failed;
    size_t needed=p->used+size/2;
    if(needed>p->capacity) {
        size_t capacity=p->capacity?p->capacity:4096;
        while(capacity<needed)capacity*=2;
        uint8_t *bytes=realloc(p->pending,capacity);if(!bytes)goto failed;
        p->pending=bytes;p->capacity=capacity;
    }
    for(size_t i=0;i<size;i+=2)p->pending[p->used++]=(uint8_t)(16*hex(text[i])+hex(text[i+1]));
    p->total+=size/2;return 0;
failed:
    p->failed=true;return 1;
}
int64_t nl_file_product_seal(NlFileProduct *p,const char *origin,const char *name) {
    if(!nl_file_product_valid(p))return 1;
    if(!p->used || !text_ok(origin,false) || !text_ok(name,true))goto failed;
    if(!p->main) {
        if(*name)goto failed;
        p->main=p->pending;p->main_size=p->used;
    } else {
        if(!*name || p->count==PRODUCT_SHADOWS)goto failed;
        char *owned_origin=copy_text(origin),*owned_name=copy_text(name);
        if(!owned_origin || !owned_name){free(owned_origin);free(owned_name);goto failed;}
        p->shadows[p->count++]=(NlServiceShadow){p->pending,p->used,owned_origin,owned_name};
    }
    p->pending=NULL;p->used=0;p->capacity=0;return 0;
failed:
    p->failed=true;return 1;
}
int64_t nl_file_product_publish(NlFileProduct *p) {
    if(!nl_file_product_valid(p))return 1;
    p->finished=true;
    if(!p->main || p->used)return 1;
    const char *cc=getenv("NANO_CC");if(!cc || !*cc)cc=getenv("CC");
    NlServiceProductOptions options={p->output,p->root,cc,getenv("NANO_CFLAGS"),getenv("NANO_LDFLAGS"),
        (p->flags&1)!=0,(p->flags&2)!=0,false,(p->flags&4)!=0};
    return nl_service_publish(p->main,p->main_size,p->shadows,p->count,&options);
}
void nl_file_product_free(NlFileProduct *p) {
    if(!p)return;
    free(p->pending);free(p->main);free(p->output);free(p->root);
    for(size_t i=0;i<p->count;i++) {
        free((void *)p->shadows[i].bytes);free((void *)p->shadows[i].origin);free((void *)p->shadows[i].name);
    }
    free(p);
}
