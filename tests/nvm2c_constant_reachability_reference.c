/* I call the actual unchanged conservative helper, including its fixed limits. */
#include "../src/nanovm/vm_decode.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../src/nanoisa/nvm2c_reachability.h"
static uint32_t word(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
int main(int argc,char **argv){
    if(sizeof(Nvm2cConstant)!=16)return 3;
    for(int arg=1;arg<argc;arg++){
        FILE *file=fopen(argv[arg],"rb");if(!file)return 2;
        if(fseek(file,0,SEEK_END))return 2;
        long size=ftell(file);if(size<0 || fseek(file,0,SEEK_SET))return 2;
        unsigned char *p=malloc(size?(size_t)size:1);if(!p)return 2;
        if(fread(p,1,(size_t)size,file)!=(size_t)size || fclose(file))return 2;
        NvmModule m={0};int valid=0;uint32_t current=0,offset=0;
        if(size>=16){
            uint32_t count=word(p),foreign=word(p+4);size_t header=16+(size_t)count*20+(size_t)foreign*8;
            if(header<=(size_t)size && (size_t)size-header<=UINT32_MAX){
                m.functions=calloc(count?count:1,sizeof *m.functions);m.imports=calloc(foreign?foreign:1,sizeof *m.imports);
                if(!m.functions || !m.imports)return 2;
                valid=1;m.function_count=count;m.import_count=foreign;current=word(p+8);offset=word(p+12);
                m.code=p+header;m.code_size=(uint32_t)((size_t)size-header);
                for(uint32_t i=0;i<count;i++){
                    const unsigned char *f=p+16+(size_t)i*20;uint32_t arity=word(f+8),locals=word(f+12),results=word(f+16);
                    if(arity>65535 || locals>65535 || results>255){valid=0;break;}
                    m.functions[i].code_offset=word(f);m.functions[i].code_length=word(f+4);
                    m.functions[i].arity=(uint16_t)arity;m.functions[i].local_count=(uint16_t)locals;m.functions[i].result_count=(uint8_t)results;
                }
                for(uint32_t i=0;i<foreign;i++){
                    const unsigned char *f=p+16+(size_t)count*20+(size_t)i*8;uint32_t parameters=word(f),tag=word(f+4);
                    if(parameters>65535 || tag>255){valid=0;break;}
                    m.imports[i].param_count=(uint16_t)parameters;m.imports[i].return_type=(uint8_t)tag;
                }
            }
        }
        printf("%d\n",valid?nvm2c_constant_unreachable(&m,current,offset):0);
        free(m.functions);free(m.imports);free(p);
    }
    return 0;
}
