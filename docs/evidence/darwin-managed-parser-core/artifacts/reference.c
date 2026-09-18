#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <fenv.h>
#include "inputs.h"
int main(void){if(sizeof(double)!=8||DBL_MANT_DIG!=53||fegetround()!=FE_TONEAREST)return 1;
for(unsigned i=0;i<sizeof inputs/sizeof inputs[0];i++){double value=strtod((const char*)inputs[i].text,0);uint64_t bits;memcpy(&bits,&value,8);printf("%016llx\n",(unsigned long long)bits);}return 0;}
