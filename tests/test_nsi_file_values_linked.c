#include <stdio.h>
#include <stdlib.h>
static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}} while(0)
#include "test_nsi_file_values_cases.h"
int main(void) {
    qv_lifetime();qv_capacity();printf("PASS %u ordinary linked File value checks\n",checks);return 0;
}
