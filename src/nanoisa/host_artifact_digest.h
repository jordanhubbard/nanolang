#ifndef NANO_HOST_ARTIFACT_DIGEST_H
#define NANO_HOST_ARTIFACT_DIGEST_H
/* I bind deployed host artifacts to SHA-256 without adding generated link flags. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

typedef struct { uint32_t h[8]; uint64_t bytes; size_t used; unsigned char block[64]; } NanoHostDigest;
static inline uint32_t nano_host_rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
static inline void nano_host_digest_block(NanoHostDigest *s, const unsigned char *p) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t w[64];
    for (unsigned i=0; i<16; ++i) w[i] = (uint32_t)p[4*i]<<24 | (uint32_t)p[4*i+1]<<16 | (uint32_t)p[4*i+2]<<8 | p[4*i+3];
    for (unsigned i=16; i<64; ++i) {
        uint32_t x=w[i-15], y=w[i-2];
        w[i]=w[i-16]+(nano_host_rotr(x,7)^nano_host_rotr(x,18)^(x>>3))+w[i-7]+(nano_host_rotr(y,17)^nano_host_rotr(y,19)^(y>>10));
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for (unsigned i=0; i<64; ++i) {
        uint32_t t=h+(nano_host_rotr(e,6)^nano_host_rotr(e,11)^nano_host_rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
        uint32_t u=(nano_host_rotr(a,2)^nano_host_rotr(a,13)^nano_host_rotr(a,22))+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static inline void nano_host_digest_init(NanoHostDigest *s) {
    *s=(NanoHostDigest){{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19},0,0,{0}};
}
static inline void nano_host_digest_add(NanoHostDigest *s, const unsigned char *p, size_t n) {
    s->bytes+=n;
    while (n) {
        size_t take=64-s->used;if (take>n) take=n;
        memcpy(s->block+s->used,p,take);s->used+=take;p+=take;n-=take;
        if (s->used==64) { nano_host_digest_block(s,s->block);s->used=0; }
    }
}
static inline void nano_host_digest_finish(NanoHostDigest *s, char hex[65]) {
    uint64_t bits=s->bytes*8;s->block[s->used++]=0x80;
    if (s->used>56) { memset(s->block+s->used,0,64-s->used);nano_host_digest_block(s,s->block);s->used=0; }
    memset(s->block+s->used,0,56-s->used);
    for (unsigned i=0;i<8;++i) s->block[63-i]=(unsigned char)(bits>>(8*i));
    nano_host_digest_block(s,s->block);
    static const char digits[]="0123456789abcdef";
    for (unsigned i=0;i<32;++i) { unsigned byte=(s->h[i/4]>>(24-8*(i%4)))&255;hex[2*i]=digits[byte>>4];hex[2*i+1]=digits[byte&15]; }
    hex[64]=0;
}
static inline int nano_host_digest_file(const char *path, char hex[65]) {
    FILE *file=fopen(path,"rb");if (!file) return 0;
    NanoHostDigest digest;nano_host_digest_init(&digest);
    unsigned char bytes[16384];size_t n;
    while ((n=fread(bytes,1,sizeof bytes,file))!=0) nano_host_digest_add(&digest,bytes,n);
    int ok=!ferror(file);if (fclose(file)) ok=0;
    if (ok) nano_host_digest_finish(&digest,hex);
    return ok;
}
#endif
