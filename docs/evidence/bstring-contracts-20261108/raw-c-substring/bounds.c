#include "runtime/nl_string.h"
#include <assert.h>
#include <stdint.h>
int main(void) {
 nl_string_t *source = nl_string_new("abc");
 assert(source);
 nl_string_t *tail = nl_string_substring(source, 1, SIZE_MAX);
 assert(tail && tail->length == 2);
 assert(tail->data[0] == 'b' && tail->data[1] == 'c');
 nl_string_free(tail); nl_string_free(source);
 return 0;
}
