/* I compare repeated record mutation on one VM instance with a fresh instance. */
#include "../../modules/nanoisa/nanoisa.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanovm/vm.h"
int g_argc=0;
char **g_argv=NULL;
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    NanoisaErr error;NvmModule *m=nanoisa_load_file(argv[1],&error);
    if(!m)return 3;
    if(!nvm_verify_profile(m,NVM_PROFILE_CLOSED_MANAGED_STRINGS).ok){nvm_module_free(m);return 4;}
    for(int instance=0;instance<2;instance++) {
        VmState vm;vm_init(&vm,m);
        for(int call=1;call<4;call++) {
            VmResult status=vm_execute(&vm);
            NanoValue result=vm_get_result(&vm);
            if(status!=VM_OK || result.tag!=TAG_INT || result.as.i64!=call) {
                vm_destroy(&vm);nvm_module_free(m);return 5;
            }
        }
        vm_destroy(&vm);
    }
    nvm_module_free(m);return 0;
}
