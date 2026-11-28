/* I call the actual shared C graph checker on my bounded synthetic carrier. */
#include "../src/nanoisa/affine_bytecode.h"
#include "nvm2c_affine_model_reader.h"
#include <stdio.h>
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *file=fopen(argv[i],"rb");
        if(!file){puts("D");continue;}
        if(fseek(file,0,SEEK_END)){fclose(file);puts("D");continue;}
        long size=ftell(file);
        if(size<12 || fseek(file,0,SEEK_SET)){fclose(file);puts("D");continue;}
        unsigned char *bytes=malloc((size_t)size);
        if(!bytes){fclose(file);return 2;}
        bool loaded=fread(bytes,1,(size_t)size,file)==(size_t)size;
        fclose(file);NvmModule model={0};
        if(!loaded || read32(bytes+4)>1 || read32(bytes+8)>1 ||
           !load_model(bytes+12,(size_t)size-12,&model)){free(bytes);puts("D");continue;}
        model.header.entry_point=read32(bytes);
        NvmFunctionEntry *functions=model.functions;
        if(!read32(bytes+8))model.functions=NULL;
        bool result=nvm_affine_value_call_graph(read32(bytes+4)?&model:NULL);
        model.functions=functions;free_model(&model);free(bytes);
        puts(result?"1":"0");
    }
    return 0;
}
