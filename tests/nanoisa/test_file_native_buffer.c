/* I inspect private emission extent arithmetic, not a claimed constructible
 * 128MiB valid module. No service or dispatcher runs in these unit controls. */
#define nvm2c_file_private_emit file_buffer_unused_emitter
#include "../../src/nanoisa/nvm2c_file_private.c"
#undef nvm2c_file_private_emit
int file_native_buffer_checks(void){
 char sentinel='x';Fne b={.text=&sentinel,.used=NVM_FILE_NATIVE_OUTPUT_BYTES-1u,
                          .capacity=NVM_FILE_NATIVE_OUTPUT_BYTES,.status=NVM_FILE_RUNTIME_OK};
 fn_text(&b,"%s","x");
 if(b.status!=NVM_FILE_RUNTIME_LIMIT || b.text!=&sentinel || sentinel!='x')return 1;
 b=(Fne){0};fn_text(&b,"%s:%u","root",7u);
 if(b.status!=NVM_FILE_RUNTIME_OK || strcmp(b.text,"root:7")){free(b.text);return 2;}
 size_t used=b.used;char *original=b.text;b.status=NVM_FILE_RUNTIME_MEMORY;fn_text(&b,"ignored");
 if(b.text!=original || b.used!=used || b.status!=NVM_FILE_RUNTIME_MEMORY){free(b.text);return 3;}
 free(b.text);return 0;
}
