/* I export immutable catalog data, never validate a module for NanoLang. */
#include "../src/nanoisa/service_source_catalog.h"
#include <stdio.h>
int main(void) {
    for(int catalog=1;catalog<=3;catalog++) {
        char text[32768];size_t size=0;
        if(!nl_service_source_catalog_view(catalog,text,sizeof text,&size) || !size)return 2;
        puts(text);
    }
    return 0;
}
