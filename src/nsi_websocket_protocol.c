#include "nsi_websocket_protocol.h"
#include "utf8.h"
#include <openssl/evp.h>
#include <stdlib.h>
#include <string.h>

struct NlWsDecoder {
    uint8_t header[10], control[125];
    uint8_t *message;
    size_t capacity, length, frame_length, frame_read;
    unsigned header_length, header_needed;
    uint8_t opcode, message_opcode;
    bool final, fragmented, reset_message;
    NlWsStatus terminal;
};
static bool ws_overlap(const void *a,size_t na,const void *b,size_t nb) {
    if (!na || !nb) return false;
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<=y ? y-x<na : x-y<nb;
}
static bool ws_close_valid(const uint8_t *bytes,size_t size) {
    if (!size) return true;
    if (size<2) return false;
    unsigned code=((unsigned)bytes[0]<<8)|bytes[1];
    if (!((code>=1000 && code<=1014 && code!=1004 && code!=1005 && code!=1006) ||
          (code>=3000 && code<=4999))) return false;
    return nl_utf8_validate((const char *)bytes+2,size-2,NULL);
}
NlWsStatus nl_ws_decoder_create(NlWsDecoder **out) {
    if (!out || *out) return NL_WS_ARGUMENT;
    NlWsDecoder *d=calloc(1,sizeof(*d));
    if (!d) return NL_WS_MEMORY;
    d->header_needed=2;
    *out=d;
    return NL_WS_OK;
}
void nl_ws_decoder_destroy(NlWsDecoder *d) {
    if (d) { free(d->message); free(d); }
}
static NlWsStatus ws_header(NlWsDecoder *d) {
    uint8_t a=d->header[0], b=d->header[1], op=a&15;
    bool control=(op&8)!=0;
    if ((a&0x70) || (b&0x80) || (op!=0 && op!=1 && op!=2 && op!=8 && op!=9 && op!=10) ||
        (control && (!(a&0x80) || (b&127)>125))) return NL_WS_PROTOCOL;
    uint64_t length=b&127;
    if (length==126) {
        length=((uint64_t)d->header[2]<<8)|d->header[3];
        if (length<126) return NL_WS_PROTOCOL;
    } else if (length==127) {
        if (d->header[2]&0x80) return NL_WS_PROTOCOL;
        length=0;
        for (unsigned i=2;i<10;i++) length=(length<<8)|d->header[i];
        if (length<=65535) return NL_WS_PROTOCOL;
    }
    if (length>NL_WS_MESSAGE_MAX) return NL_WS_LIMIT;
    if (!control) {
        if ((op==0 && !d->fragmented) || (op!=0 && d->fragmented)) return NL_WS_PROTOCOL;
        if (op) { d->message_opcode=op; d->length=0; }
        if (length>NL_WS_MESSAGE_MAX-d->length) return NL_WS_LIMIT;
        size_t needed=d->length+(size_t)length;
        if (needed>d->capacity) {
            size_t capacity=d->capacity ? d->capacity : 256;
            while (capacity<needed) capacity*=2;
            uint8_t *next=realloc(d->message,capacity);
            if (!next) return NL_WS_MEMORY;
            d->message=next;d->capacity=capacity;
        }
    }
    d->opcode=op;d->final=(a&0x80)!=0;
    d->frame_length=(size_t)length;d->frame_read=0;
    return NL_WS_OK;
}
static NlWsStatus ws_event(NlWsDecoder *d,NlWsEvent *out) {
    bool control=(d->opcode&8)!=0;
    d->header_length=0;d->header_needed=2;
    if (control) {
        if (d->opcode==NL_WS_CLOSE) {
            if (!ws_close_valid(d->control,d->frame_length)) return NL_WS_PROTOCOL;
            d->terminal=NL_WS_CLOSED;
        }
        *out=(NlWsEvent){(NlWsOpcode)d->opcode,d->control,d->frame_length};
        return NL_WS_EVENT;
    }
    d->fragmented=!d->final;
    if (!d->final) return NL_WS_MORE;
    if (d->message_opcode==NL_WS_TEXT && !nl_utf8_validate((const char *)d->message,d->length,NULL))
        return NL_WS_PROTOCOL;
    *out=(NlWsEvent){(NlWsOpcode)d->message_opcode,d->message,d->length};
    d->reset_message=true;
    return NL_WS_EVENT;
}
NlWsStatus nl_ws_decoder_feed(NlWsDecoder *d,const uint8_t *bytes,size_t size,
                             size_t *consumed,NlWsEvent *out) {
    if (!d || !consumed || !out || (size && !bytes) || size>NL_WS_FEED_MAX ||
        ws_overlap(bytes,size,consumed,sizeof(*consumed)) || ws_overlap(bytes,size,out,sizeof(*out)) ||
        ws_overlap(consumed,sizeof(*consumed),out,sizeof(*out))) return NL_WS_ARGUMENT;
    *consumed=0;
    if (d->terminal) return d->terminal;
    if (d->reset_message) {d->length=0;d->reset_message=false;}
    while (*consumed<size) {
        if (d->header_length<d->header_needed) {
            d->header[d->header_length++]=bytes[(*consumed)++];
            if (d->header_length==2) {
                unsigned length=d->header[1]&127;
                /* I refuse invalid basic headers without waiting for an extension. */
                uint8_t op=d->header[0]&15;
                if ((d->header[0]&0x70) || (d->header[1]&0x80) ||
                    (op!=0 && op!=1 && op!=2 && op!=8 && op!=9 && op!=10) ||
                    ((op&8) && (!(d->header[0]&0x80) || length>125)))
                    return d->terminal=NL_WS_PROTOCOL;
                d->header_needed=length==126 ? 4 : length==127 ? 10 : 2;
            }
            if (d->header_length<d->header_needed) continue;
            NlWsStatus status=ws_header(d);
            if (status!=NL_WS_OK) return d->terminal=status;
        }
        size_t count=d->frame_length-d->frame_read;
        if (count>size-*consumed) count=size-*consumed;
        if (count) {
            if (d->opcode&8) memcpy(d->control+d->frame_read,bytes+*consumed,count);
            else {memcpy(d->message+d->length,bytes+*consumed,count);d->length+=count;}
            d->frame_read+=count;*consumed+=count;
        }
        if (d->frame_read==d->frame_length) {
            NlWsStatus status=ws_event(d,out);
            if (status==NL_WS_EVENT) return status;
            if (status!=NL_WS_MORE) return d->terminal=status;
        }
    }
    return NL_WS_MORE;
}
NlWsStatus nl_ws_decoder_eof(NlWsDecoder *d) {
    if (!d) return NL_WS_ARGUMENT;
    if (d->terminal) return d->terminal;
    return d->terminal=NL_WS_PROTOCOL;
}
NlWsStatus nl_ws_client_frame(NlWsOpcode opcode,const uint8_t *bytes,size_t size,
                             const uint8_t mask[4],uint8_t *out,size_t capacity,size_t *written) {
    if (!mask || !out || !written || (size && !bytes) ||
        (opcode!=NL_WS_TEXT && opcode!=NL_WS_BINARY && opcode!=NL_WS_CLOSE && opcode!=NL_WS_PING && opcode!=NL_WS_PONG))
        return NL_WS_ARGUMENT;
    if (size>NL_WS_MESSAGE_MAX) return NL_WS_LIMIT;
    if (((opcode&8) && size>125) || (opcode==NL_WS_CLOSE && !ws_close_valid(bytes,size)) ||
        (opcode==NL_WS_TEXT && !nl_utf8_validate((const char *)bytes,size,NULL))) return NL_WS_PROTOCOL;
    size_t prefix=size<126 ? 2 : size<=65535 ? 4 : 10, total=prefix+4+size;
    if (capacity<total) return NL_WS_LIMIT;
    if (ws_overlap(bytes,size,out,total) || ws_overlap(mask,4,out,total) ||
        ws_overlap(written,sizeof(*written),out,total) || ws_overlap(written,sizeof(*written),bytes,size) ||
        ws_overlap(written,sizeof(*written),mask,4)) return NL_WS_ARGUMENT;
    out[0]=(uint8_t)(0x80|opcode);
    out[1]=(uint8_t)(0x80|(size<126 ? size : size<=65535 ? 126 : 127));
    if (prefix>2) for (size_t i=2;i<prefix;i++) out[i]=(uint8_t)((uint64_t)size>>(8*(prefix-1-i)));
    memcpy(out+prefix,mask,4);
    for (size_t i=0;i<size;i++) out[prefix+4+i]=bytes[i]^mask[i%4];
    *written=total;
    return NL_WS_OK;
}
static void ws_base64(const uint8_t *bytes,size_t size,char *out) {
    static const char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t at=0;
    for (size_t i=0;i<size;i+=3) {
        size_t left=size-i;
        uint32_t word=(uint32_t)bytes[i]<<16;
        if (left>1) word|=(uint32_t)bytes[i+1]<<8;
        if (left>2) word|=bytes[i+2];
        out[at++]=alphabet[word>>18];out[at++]=alphabet[(word>>12)&63];
        out[at++]=left>1 ? alphabet[(word>>6)&63] : '=';
        out[at++]=left>2 ? alphabet[word&63] : '=';
    }
    out[at]=0;
}
NlWsStatus nl_ws_handshake_key(const uint8_t nonce[16],char key[25],char accept[29]) {
    if (!nonce || !key || !accept || ws_overlap(nonce,16,key,25) ||
        ws_overlap(nonce,16,accept,29) || ws_overlap(key,25,accept,29)) return NL_WS_ARGUMENT;
    char value[61], encoded[29];uint8_t digest[EVP_MAX_MD_SIZE];unsigned length=0;
    ws_base64(nonce,16,value);
    memcpy(value+24,"258EAFA5-E914-47DA-95CA-C5AB0DC85B11",36);
    if (!EVP_Digest(value,60,digest,&length,EVP_sha1(),NULL) || length!=20) return NL_WS_CRYPTO;
    ws_base64(digest,20,encoded);
    memcpy(key,value,24);key[24]=0;memcpy(accept,encoded,29);
    return NL_WS_OK;
}
static bool ws_case_equal(const char *p,size_t n,const char *word) {
    if (strlen(word)!=n) return false;
    for (size_t i=0;i<n;i++) {
        unsigned char c=(unsigned char)p[i];
        if (c>='A' && c<='Z') c=(unsigned char)(c+('a'-'A'));
        if (c!=(unsigned char)word[i]) return false;
    }
    return true;
}
static bool ws_token_char(unsigned char c) {
    return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') ||
        (c && strchr("!#$%&'*+-.^_`|~",c));
}
static bool ws_connection_tokens(const char *p,size_t n,bool *upgrade) {
    size_t i=0;bool any=false;
    while (i<n) {
        while (i<n && (p[i]==' ' || p[i]=='\t')) i++;
        size_t start=i;
        while (i<n && ws_token_char((unsigned char)p[i])) i++;
        if (i==start) return false;
        any=true;
        if (ws_case_equal(p+start,i-start,"upgrade")) *upgrade=true;
        while (i<n && (p[i]==' ' || p[i]=='\t')) i++;
        if (i<n && (p[i++]!=',' || i==n)) return false;
    }
    return any;
}
NlWsStatus nl_ws_upgrade_validate(const char *bytes,size_t size,const char accept[29],size_t *consumed) {
    if (!bytes || !accept || !consumed || ws_overlap(bytes,size,consumed,sizeof(*consumed)) ||
        ws_overlap(accept,29,consumed,sizeof(*consumed))) return NL_WS_ARGUMENT;
    size_t limit=size<NL_WS_HEADER_MAX ? size : NL_WS_HEADER_MAX, end=0;
    for (size_t i=0;i<limit;i++) {
        unsigned char c=(unsigned char)bytes[i];
        if (!c || (c<32 && c!='\r' && c!='\n' && c!='\t') || c==127) return NL_WS_PROTOCOL;
        if (i>=3 && !memcmp(bytes+i-3,"\r\n\r\n",4)) {end=i+1;break;}
    }
    if (!end) return size>=NL_WS_HEADER_MAX ? NL_WS_LIMIT : NL_WS_MORE;
    size_t line=0;
    while (line+1<end && !(bytes[line]=='\r' && bytes[line+1]=='\n')) line++;
    if (line<13 || memcmp(bytes,"HTTP/1.1 101 ",13)) return NL_WS_PROTOCOL;
    for (size_t i=13;i<line;i++) if (bytes[i]=='\r' || bytes[i]=='\n') return NL_WS_PROTOCOL;
    bool upgraded=false,connection=false,accepted=false;
    size_t at=line+2;
    while (at<end-2) {
        size_t stop=at;
        while (stop+1<end && !(bytes[stop]=='\r' && bytes[stop+1]=='\n')) stop++;
        size_t colon=at;
        while (colon<stop && ws_token_char((unsigned char)bytes[colon])) colon++;
        if (colon==at || colon>=stop || bytes[colon]!=':') return NL_WS_PROTOCOL;
        size_t start=colon+1, finish=stop;
        while (start<finish && (bytes[start]==' ' || bytes[start]=='\t')) start++;
        while (finish>start && (bytes[finish-1]==' ' || bytes[finish-1]=='\t')) finish--;
        for (size_t i=start;i<finish;i++) if (bytes[i]=='\r' || bytes[i]=='\n') return NL_WS_PROTOCOL;
        if (ws_case_equal(bytes+at,colon-at,"upgrade")) {
            if (upgraded || !ws_case_equal(bytes+start,finish-start,"websocket")) return NL_WS_PROTOCOL;
            upgraded=true;
        } else if (ws_case_equal(bytes+at,colon-at,"connection")) {
            if (!ws_connection_tokens(bytes+start,finish-start,&connection)) return NL_WS_PROTOCOL;
        } else if (ws_case_equal(bytes+at,colon-at,"sec-websocket-accept")) {
            if (accepted || finish-start!=28 || memcmp(bytes+start,accept,28) || accept[28]) return NL_WS_PROTOCOL;
            accepted=true;
        } else if (ws_case_equal(bytes+at,colon-at,"sec-websocket-extensions") ||
                   ws_case_equal(bytes+at,colon-at,"sec-websocket-protocol")) return NL_WS_PROTOCOL;
        at=stop+2;
    }
    if (!upgraded || !connection || !accepted) return NL_WS_PROTOCOL;
    *consumed=end;
    return NL_WS_OK;
}
