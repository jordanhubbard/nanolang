/* I invoke the actual ordinary shared function body without changing it. */
#include "../src/nanoisa/verifier.c"
static uint32_t word(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static bool pair(unsigned char *bytes,size_t size,size_t *cursor,uint32_t *a,uint32_t *b) {
    if(*cursor>size || size-*cursor<8)return false;
    *a=word(bytes+*cursor);*b=word(bytes+*cursor+4);*cursor+=8;return true;
}
int main(int argc,char **argv) {
    for(int arg=1;arg<argc;arg++) {
        FILE *file=fopen(argv[arg],"rb");
        if(!file || fseek(file,0,SEEK_END))return 2;
        long length=ftell(file);
        if(length<0 || fseek(file,0,SEEK_SET))return 2;
        size_t size=(size_t)length;unsigned char *bytes=malloc(size?size:1);
        if(!bytes || fread(bytes,1,size,file)!=size || fclose(file))return 2;
        NvmModule module={0},*linked=NULL;const NvmModule **links=NULL;
        uint32_t link_count=0,current=0;size_t cursor=40;bool valid=false;
        if(size>=40) {
            uint32_t start=word(bytes),span=word(bytes+4),count=word(bytes+8);current=word(bytes+12);
            uint32_t foreign=word(bytes+16);link_count=word(bytes+20);module.string_count=word(bytes+24);
            module.struct_count=word(bytes+28);module.enum_count=word(bytes+32);module.union_count=word(bytes+36);
            if(count && count<=65536 && current<count && foreign<=65536 && link_count<=64 && module.string_count<=65536 &&
               (uint64_t)count*20+(uint64_t)foreign*8<=size-cursor) {
                module.functions=calloc(count,sizeof *module.functions);module.function_count=count;
                module.imports=calloc(foreign?foreign:1,sizeof *module.imports);module.import_count=foreign;
                module.strings=calloc(module.string_count?module.string_count:1,sizeof *module.strings);
                module.string_lengths=calloc(module.string_count?module.string_count:1,sizeof *module.string_lengths);
                linked=calloc(link_count?link_count:1,sizeof *linked);links=calloc(link_count?link_count:1,sizeof *links);
                if(!module.functions || !module.imports || !module.strings || !module.string_lengths || !linked || !links)return 2;
                valid=true;
                for(uint32_t i=0;i<module.string_count;i++)module.strings[i]="";
                for(uint32_t i=0;i<count;i++) {
                    uint32_t arity=word(bytes+cursor),results=word(bytes+cursor+4),tag=word(bytes+cursor+8);
                    uint32_t locals=word(bytes+cursor+12),upvalues=word(bytes+cursor+16);cursor+=20;
                    if(arity>UINT16_MAX || results>UINT8_MAX || tag>UINT8_MAX || locals>UINT16_MAX || upvalues>UINT16_MAX)valid=false;
                    module.functions[i]=(NvmFunctionEntry){.arity=(uint16_t)arity,.result_count=(uint8_t)results,.result_tag=(uint8_t)tag,.local_count=(uint16_t)locals,.upvalue_count=(uint16_t)upvalues};
                }
                for(uint32_t i=0;i<foreign;i++) {
                    uint32_t params=word(bytes+cursor),tag=word(bytes+cursor+4);cursor+=8;
                    if(params>UINT16_MAX || tag>UINT8_MAX)valid=false;
                    module.imports[i].param_count=(uint16_t)params;module.imports[i].return_type=(uint8_t)tag;
                }
                for(uint32_t i=0;valid && i<link_count;i++) {
                    uint32_t resolved,functions;
                    if(!pair(bytes,size,&cursor,&resolved,&functions) || resolved>1 || functions>65536 || (uint64_t)functions*8>size-cursor){valid=false;break;}
                    linked[i].functions=calloc(functions?functions:1,sizeof *linked[i].functions);linked[i].function_count=functions;
                    if(!linked[i].functions)return 2;
                    if(resolved)links[i]=&linked[i];
                    for(uint32_t j=0;j<functions;j++) {
                        uint32_t arity,results;if(!pair(bytes,size,&cursor,&arity,&results))return 2;
                        if(arity>UINT16_MAX || results>UINT8_MAX)valid=false;
                        linked[i].functions[j].arity=(uint16_t)arity;linked[i].functions[j].result_count=(uint8_t)results;
                    }
                }
                if(cursor>size || size-cursor>UINT32_MAX || start>size-cursor || span>size-cursor-start)valid=false;
                module.code=bytes+cursor;module.code_size=(uint32_t)(size-cursor);
                module.functions[current].code_offset=start;module.functions[current].code_length=span;
            }
        }
        VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
        if(valid)valid=vm_decode_function(&module,current,&decoded,error);
        vm_decoded_function_free(&decoded);
        if(!valid)puts("D");
        else {
            uint16_t maximum=0xabcd;
            NvmVerifyResult result=verify_function_body(&module,current,links,link_count,&maximum,NULL,NULL);
            NvmVerifyResult no_max=verify_function_body(&module,current,links,link_count,NULL,NULL,NULL);
            if(result.ok!=no_max.ok || strcmp(result.error_msg,no_max.error_msg))return 3;
            if(result.ok)printf("1:%u\n",maximum);
            else {
                if(maximum!=0xabcd)return 3;
                if(!strncmp(result.error_msg,"I refuse reference lifetime and ownership instruction dataflow: ",sizeof("I refuse reference lifetime and ownership instruction dataflow: ")-1) ||
                   !strcmp(result.error_msg,"I require owned-transfer execution semantics before execution")) {
                    puts("A");fprintf(stderr,"%s\t%s\n",argv[arg],result.error_msg);
                } else printf("0:%s\n",result.error_msg);
            }
        }
        if(linked)for(uint32_t i=0;i<link_count;i++)free(linked[i].functions);
        free(linked);free(links);free(module.functions);free(module.imports);free(module.strings);free(module.string_lengths);free(bytes);
    }
    return 0;
}
