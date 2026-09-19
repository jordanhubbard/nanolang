/* I qualify the separately linked private adapter with an actual CUDA GPU. */
#include "../src/nsi_gpu.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1;} } while(0)
int main(void) {
    NlGpuService *s=NULL;NlGpuResult r=nl_gpu_service_create(0,&s);
    CHECK(r.status==NL_GPU_OK && s);NlGpuDevice d;CHECK(nl_gpu_service_device(s,&d).status==NL_GPU_OK);
    CHECK(d.cuda_gpu && d.driver_version>0 && d.name[0]);
    printf("actual backend=CUDA class=GPU ordinal=%d driver=%d name=%s uuid=",d.ordinal,d.driver_version,d.name);
    for(unsigned i=0;i<16;i++){printf("%02x",d.uuid[i]);}
    puts("");
    unsigned char input[257],output[257];for(unsigned i=0;i<257;i++)input[i]=(unsigned char)(i*73u);
    NlGpuToken a,b,moved;CHECK(nl_gpu_buffer_allocate(s,257,NL_CAP_READ|NL_CAP_WRITE|NL_CAP_TRANSFER,&a).status==NL_GPU_OK);
    CHECK(nl_gpu_buffer_allocate(s,257,NL_CAP_READ|NL_CAP_WRITE,&b).status==NL_GPU_OK);
    CHECK(nl_gpu_buffer_write(s,&a,0,input,257).status==NL_GPU_OK);
    CHECK(nl_gpu_buffer_write(s,&b,0,input,257).status==NL_GPU_OK);
    memset(output,0xaa,257);CHECK(nl_gpu_buffer_read(s,&a,0,output,257).status==NL_GPU_OK);CHECK(!memcmp(input,output,257));
    CHECK(nl_gpu_buffer_transfer(s,&a,&moved).status==NL_GPU_OK);
    CHECK(nl_gpu_buffer_close(s,&a).status==NL_GPU_TOKEN);
    CHECK(nl_gpu_buffer_close(s,&moved).status==NL_GPU_OK);
    memset(output,0,257);CHECK(nl_gpu_buffer_read(s,&b,0,output,257).status==NL_GPU_OK);CHECK(!memcmp(input,output,257));
    CHECK(nl_gpu_buffer_read(s,&b,257,NULL,0).status==NL_GPU_OK);
    CHECK(nl_gpu_buffer_read(s,&b,257,output,1).status==NL_GPU_ARGUMENT);
    r=nl_gpu_service_destroy(s);CHECK(r.status==NL_GPU_OK && r.context_destroyed && !r.library_retained);
    CHECK(nl_gpu_service_diagnostics().records==0);puts("PASS linked real GPU buffer lifecycle");return 0;
}
