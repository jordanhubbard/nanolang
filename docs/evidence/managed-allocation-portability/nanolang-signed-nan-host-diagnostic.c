#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main(void){uint64_t a[]={0x7ff8000000000001ULL,0xfff8000000000001ULL,0x7ff0000000000001ULL,0xfff0000000000001ULL,0x7ff0000000000000ULL,0xfff0000000000000ULL};for(unsigned i=0;i<6;i++){double x;memcpy(&x,a+i,8);printf("%016llx [%g] [%a]\n",(unsigned long long)a[i],x,x);}return 0;}
