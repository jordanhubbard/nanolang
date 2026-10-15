#include "src/nanoisa/service_source_catalog.h"
#include <stdio.h>
int main(void){char view[32768];size_t size=0;for(int i=1;i<=3;i++){if(!nl_service_source_catalog_view(i,view,sizeof view,&size))return 1;puts(view);}return 0;}
