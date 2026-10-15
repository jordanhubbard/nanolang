"""I exercise the public VM and installed generated-native WebSocket boundaries."""
import os
from tests import test_websocket_dispatch as dispatch

class WebSocketPublic(dispatch.WebSocketDispatch):
    fixture_source = "tests/nanoisa/test_websocket_public.c"
    dispatch_sources = ["src/nanovm/websocket_indirect_public_vm.c", "src/nanoisa/websocket_indirect_public_native.c"]
    extra_sources = ["src/nanoisa/file_host_grant.c", "src/nanoisa/websocket_host_grant.c", "src/nanoisa/websocket_indirect_public_abi.c"]
    include_flags = ["-I" + os.environ["WEBSOCKET_PUBLIC_TEST_PREFIX"] + "/include"]
    policy_expression = "runtime_policy(c,&policy)"

    def driver_source(self, private_source):
        return '''#include <nanolang/websocket/nanoisa/websocket_indirect_native_public.h>
#include <stdlib.h>
#include <assert.h>
#include "tests/nanoisa/websocket_dispatch_host.h"
NvmWebSocketIndirectExecutionReport nvm_websocket_indirect_program_test(NvmWebSocketHostGrant *,const NvmWebSocketIndirectOptions *,NvmWebSocketScalar *);
int main(int argc,char **argv){if(argc!=4)return 2;
 NvmWebSocketIndirectOptions options={1,strtoull(argv[1],NULL,10)};
 NvmWebSocketHostPolicy policy={1,atoi(argv[2])!=0,atoi(argv[2])!=2,2000,getenv("NANOLANG_RESOLVER")};
 NvmWebSocketHostGrant *grant=NULL;
 if(atoi(argv[2])>=0)assert(nvm_websocket_host_grant_create(&policy,&grant)==NVM_WEBSOCKET_HOST_OK);
 fail_close=atoi(argv[3])!=0;NvmWebSocketRuntimeView out={.fields=99,.values={12345}};
 NvmWebSocketScalar scalar={TAG_INT,12345};
 NvmWebSocketIndirectExecutionReport r=nvm_websocket_indirect_program_test(grant,&options,&scalar);
 if(r.runtime.status==NVM_WEBSOCKET_RUNTIME_OK){out.fields=1;out.values[0]=scalar.value;}
 else assert(scalar.tag==TAG_INT && scalar.value==12345);
 assert(nvm_websocket_host_grant_destroy(&grant)==NVM_WEBSOCKET_HOST_OK);
 report(r,out);return 0;}
'''

    def test_installed_consumer(self):
        compiler, flags, ordinary, ldflags, providers, fixture = self.build_fixture()
        grant_test = self.artifacts / "grant-boundaries"
        self.command("grant-boundaries-build", [*compiler, *flags, "tests/nanoisa/test_websocket_public_grant.c",
            *self.dispatch_sources, *providers, *ordinary, *ldflags, "-pthread", "-o", str(grant_test)])
        self.command("grant-boundaries", [str(grant_test)])
        prefix = os.environ["WEBSOCKET_PUBLIC_TEST_PREFIX"]
        driver = self.artifacts / "installed.c"
        driver.write_text('''#include <nanolang/websocket/nanoisa/websocket_indirect_native_public.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
NvmWebSocketIndirectExecutionReport nvm_websocket_indirect_program_test(NvmWebSocketHostGrant *,const NvmWebSocketIndirectOptions *,NvmWebSocketScalar *);
int main(int argc,char **argv){
 (void)argv;NvmWebSocketHostPolicy policy={1,argc>1,false,2000,NULL};NvmWebSocketHostGrant *grant=NULL;
 assert(nvm_websocket_host_grant_create(&policy,&grant)==NVM_WEBSOCKET_HOST_OK);
 assert(nvm_websocket_host_enter_query()==NVM_WEBSOCKET_HOST_OK);
 void *poison=(void *)(uintptr_t)1;
 assert(nvm_websocket_indirect_program_test(poison,poison,poison).runtime.status==NVM_WEBSOCKET_RUNTIME_BUSY);
 nvm_websocket_host_leave();
 NvmWebSocketIndirectOptions options={1,100000};NvmWebSocketScalar out={TAG_INT,12345};
 NvmWebSocketIndirectExecutionReport result=nvm_websocket_indirect_program_test(grant,&options,&out);
 assert(result.runtime.status==NVM_WEBSOCKET_RUNTIME_OK && out.tag==TAG_INT && out.value==2);
 assert(nvm_websocket_host_grant_revoke(grant)==NVM_WEBSOCKET_HOST_OK);out.value=12345;
 result=nvm_websocket_indirect_program_test(grant,&options,&out);
 assert(result.runtime.status==NVM_WEBSOCKET_RUNTIME_STATE && out.value==12345);
 assert(nvm_websocket_host_grant_destroy(&grant)==NVM_WEBSOCKET_HOST_OK);
 result=nvm_websocket_indirect_program_test(NULL,&options,&out);
 assert(result.runtime.status==NVM_WEBSOCKET_RUNTIME_INVALID && out.value==12345);
 return 0;}
''')
        vm_source = self.artifacts / "installed-vm.c"
        vm_source.write_text('''#include <nanolang/websocket/nanoisa/websocket_indirect_public.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
int main(int argc,char **argv){
 assert(argc==2);FILE *f=fopen(argv[1],"rb");assert(f);assert(!fseek(f,0,SEEK_END));
 long size=ftell(f);assert(size>0);rewind(f);unsigned char *bytes=malloc((size_t)size);assert(bytes);
 assert(fread(bytes,1,(size_t)size,f)==(size_t)size);assert(!fclose(f));
 NvmWebSocketHostPolicy policy={1,false,false,2000,NULL};NvmWebSocketHostGrant *grant=NULL;
 assert(nvm_websocket_host_grant_create(&policy,&grant)==NVM_WEBSOCKET_HOST_OK);
 NvmWebSocketIndirectOptions options={1,100000};NvmWebSocketScalar out={TAG_INT,12345};
 NvmWebSocketIndirectExecutionReport report=nvm_websocket_execute_indirect_bytes(grant,bytes,(size_t)size,&options,&out);
 assert(report.runtime.status==NVM_WEBSOCKET_RUNTIME_OK && out.tag==TAG_INT && out.value==2);
 char *text=NULL,error[256];
 assert(nvm2c_emit_websocket_indirect_bytes(bytes,(size_t)size,"installed",&text,error,sizeof error)==NVM_WEBSOCKET_RUNTIME_OK);
 assert(text && strstr(text,"nvm_websocket_indirect_program_installed"));free(text);
 text=(void *)bytes;
 assert(nvm2c_emit_websocket_indirect_bytes(bytes,(size_t)size,"bad-name",&text,error,sizeof error)==NVM_WEBSOCKET_RUNTIME_INVALID);
 assert(text==(void *)bytes);
 assert(nvm_websocket_host_grant_destroy(&grant)==NVM_WEBSOCKET_HOST_OK);free(bytes);return 0;}
''')
        installed_vm = self.artifacts / "installed-vm"
        self.command("installed-vm-build", [*compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
            "-I" + prefix + "/include", str(vm_source), prefix + "/lib/libnano_websocket_runtime.a", *ldflags, "-o", str(installed_vm)])
        import json
        for case in range(3):
            self.command("installed-emit-" + str(case), [str(fixture), "emit", str(self.artifacts), "1", str(case)])
            self.command("installed-vm-" + str(case), [str(installed_vm), str(self.artifacts / f"case-{case}.nvm")])
            source = self.artifacts / f"case-{case}.c"
            native = self.artifacts / f"installed-{case}"
            self.command(native.name + "-build", [*compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
                "-I" + prefix + "/include", str(source), str(driver), prefix + "/lib/libnano_websocket_runtime.a", *ldflags, "-o", str(native)])
            self.command(native.name, [str(native)])
            if case == 1:
                self.command(native.name + "-lookup-denied", [str(native), "lookup-denied"])
                denied = json.loads(self.command(native.name + "-vm-lookup-denied", [str(fixture), "vm", str(self.artifacts / f"case-{case}.nvm"), "100000", "2", "0"]))
                self.assertEqual((denied["status"], denied["value"], denied["opens"]), (0, 2, 0))
            symbols = self.command(native.name + "-symbols", ["nm", str(native)])
            self.assertNotRegex(symbols, r"\b_?(nvm_websocket_execute_indirect_bytes|nvm2c_emit_websocket_indirect_bytes|vm_execute|vm_core_execute)\b")
            result = json.loads(self.command(native.name + "-vm", [str(fixture), "vm", str(self.artifacts / f"case-{case}.nvm"), "100000", "0", "0"]))
            self.assertEqual((result["status"], result["value"], result["opens"], result["closes"]), (0, 2, 0, 0))
