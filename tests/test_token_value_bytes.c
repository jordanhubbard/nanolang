/* I exercise token metadata and copies without executing source programs. */
#include "nanolang.h"
#include "string_literal_decode.h"
#include "runtime/list_LexerToken.h"
#include "runtime/list_token.h"
#include "runtime/token_helpers.h"
#include <assert.h>
#include <inttypes.h>

static List_LexerToken *bridge_input;
List_LexerToken *nl_tokenize(const char *source) { (void)source; return bridge_input; }
static int fail_copy;
static int fail_array;
static void *copy_array(size_t size) { return fail_array ? NULL : malloc(size); }
static char *copy_string(const char *s) {
    if (fail_copy > 0 && --fail_copy == 0) return NULL;
    return strdup(s);
}
#define malloc copy_array
#define strdup copy_string
#include "lexer_bridge.c"
#undef strdup
#undef malloc

static void decoder_edges(void) {
    const char *raw[] = {"", "abc", "a\\0suffix", "\\n\\t\\r\\0\\\\\\'\\\"",
        "\\u0000\\x41\\q", "é", "x\\", "\\\\0", "\\q\\n"};
    const char *decoded[] = {"", "abc", "a\0suffix", "\n\t\r\0\\'\"",
        "\\u0000\\x41\\q", "é", "x\\", "\\0", "\\q\n"};
    const size_t lengths[] = {0,3,8,7,12,2,2,2,3};
    for (size_t i=0;i<sizeof(raw)/sizeof(raw[0]);i++) {
        assert(nl_string_literal_value_bytes(raw[i])==lengths[i]);
        char *value=nl_decode_string_literal(raw[i]); assert(value);
        assert(memcmp(value,decoded[i],lengths[i]+1)==0); free(value);
    }
    /* Every possible one-byte escape, including unknown and high bytes. */
    for (unsigned code=1;code<256;code++) {
        char raw_byte[]={'\\',(char)code,'s','u','f','f','i','x',0};
        char byte=0; bool known=nl_string_escape_byte((char)code,&byte);
        char *out=nl_decode_string_literal(raw_byte); assert(out);
        size_t width=known?1:2;
        assert(nl_string_literal_value_bytes(raw_byte)==width+6);
        if (known) assert(out[0]==byte);
        else { assert(out[0]=='\\'); assert((unsigned char)out[1]==code); }
        assert(memcmp(out+width,"suffix",7)==0);free(out);
    }
}

static void lists_and_bridge(void) {
    Token value={.token_type=TOKEN_STRING,.value="a\\0suffix",.line=7,.column=9,
                 .value_bytes=INT64_C(1099511627776)};
    /* A bridge preserves supplied metadata; it must not repair/recompute it. */
    List_LexerToken *list=nl_list_LexerToken_new();
    for (int i=0;i<33;i++) nl_list_LexerToken_push(list,value);
    nl_list_LexerToken_insert(list,1,value);nl_list_LexerToken_set(list,2,value);
    assert(nl_list_LexerToken_remove(list,1).value_bytes==value.value_bytes);
    assert(nl_list_LexerToken_pop(list).value_bytes==value.value_bytes);
    assert(token_get_value_bytes(list,1)==value.value_bytes);
    assert(token_get_type(list,1)==value.token_type && token_get_line(list,1)==7);
    assert(token_get_column(list,1)==9 && strcmp(token_get_value(list,1),value.value)==0);
    nl_list_LexerToken_clear(list);nl_list_LexerToken_free(list);
    List_Token *legacy=nl_list_Token_new();
    nl_list_Token_push(legacy,value);nl_list_Token_insert(legacy,0,value);
    nl_list_Token_set(legacy,1,value);
    assert(nl_list_Token_remove(legacy,0).value_bytes==value.value_bytes);
    assert(nl_list_Token_pop(legacy).value_bytes==value.value_bytes);
    nl_list_Token_free(legacy);
    int count=-1;
    bridge_input=NULL; assert(tokenize_nano("unused",&count)==NULL && count==0);
    bridge_input=nl_list_LexerToken_new(); count=-1;
    assert(tokenize_nano("unused",&count)==NULL && count==0);
    bridge_input=nl_list_LexerToken_new();nl_list_LexerToken_push(bridge_input,value);
    fail_array=1;count=-1;
    assert(tokenize_nano("unused",&count)==NULL && count==0);fail_array=0;
    for (int failure=0;failure<=2;failure++) {
        bridge_input=nl_list_LexerToken_new();
        nl_list_LexerToken_push(bridge_input,value);nl_list_LexerToken_push(bridge_input,value);
        nl_list_LexerToken_insert(bridge_input,0,value);
        assert(nl_list_LexerToken_remove(bridge_input,0).value_bytes==value.value_bytes);
        nl_list_LexerToken_set(bridge_input,0,value);
        fail_copy=failure;int count=-1;Token *copy=tokenize_nano("unused",&count);
        if (failure) { assert(!copy && count==0); }
        else {
            assert(copy && count==2 && copy[0].value_bytes==value.value_bytes);
            assert(copy[0].value!=value.value && strcmp(copy[0].value,value.value)==0);
            assert(copy[1].line==7 && copy[1].column==9);free_tokens(copy,count);
        }
    }
}

int main(void) {
    decoder_edges();lists_and_bridge();
    const char *source="name 42 \"a\\0suffix\" \"\\u0000\" \"é\" not ( )";
    const int types[]={TOKEN_IDENTIFIER,TOKEN_NUMBER,TOKEN_STRING,TOKEN_STRING,
                      TOKEN_STRING,TOKEN_NOT,TOKEN_LPAREN,TOKEN_RPAREN,TOKEN_EOF};
    const int64_t bytes[]={4,2,8,6,2,3,0,0,0};
    int count=0;Token *tokens=tokenize(source,&count);assert(tokens && count==9);
    for(int i=0;i<count;i++) {
        assert(tokens[i].token_type==types[i] && tokens[i].value_bytes==bytes[i]);
        printf("%d:%d:%" PRId64 "\n",i,tokens[i].token_type,tokens[i].value_bytes);
    }
    assert(strcmp(tokens[2].value,"a\\0suffix")==0);free_tokens(tokens,count);
    tokens=tokenize("f\"a\\0suffix{42}é\"",&count);assert(tokens);
    int strings=0;
    for(int i=0;i<count;i++) if(tokens[i].token_type==TOKEN_STRING) {
        assert(tokens[i].value_bytes==(strings==0?8:2));strings++;
    }
    assert(strings==2);free_tokens(tokens,count);
    return 0;
}
