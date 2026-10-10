"""I qualify explicit TCP grants around the same VM/native network corpus."""
import json
import os
import shlex
import shutil
from pathlib import Path
from tests import test_socket_dispatch as private

class SocketPublic(private.SocketDispatch):
    fixture_source = "tests/nanoisa/test_socket_public.c"
    dispatch_sources = ["src/nanovm/socket_indirect_public_vm.c", "src/nanoisa/socket_indirect_public_native.c"]
    extra_sources = ["src/nanoisa/file_host_grant.c", "src/nanoisa/socket_host_grant.c", "src/nanoisa/socket_indirect_public_abi.c"]
    prefix = Path(os.environ.get("SOCKET_PUBLIC_TEST_PREFIX", private.ROOT / "obj/socket-public-test-install"))
    include_flags = ["-I" + str(prefix / "include"), "-pthread"]

    def test_vm_and_generated_native_tcp(self):
        super().test_vm_and_generated_native_tcp()
        consumer = self.artifacts / "installed-consumer"
        consumer.mkdir()
        compiler = shlex.split(os.environ.get("NANO_SOCKET_DISPATCH_CC", "cc"))
        ldflags = shlex.split(os.environ.get("SOCKET_DISPATCH_LDFLAGS", "-lm -lcrypto"))
        for case, expected in ((0, 77), (2, 4)):
            generated = consumer / f"program-{case}.c"
            shutil.copyfile(self.artifacts / f"case-{case}.c", generated)
            driver = consumer / f"driver-{case}.c"
            driver.write_text('''#include <nanolang/socket/nanoisa/socket_indirect_public.h>
extern NvmSocketIndirectExecutionReport nvm_socket_indirect_program_test(NvmSocketHostGrant *,const NvmSocketIndirectOptions *,NvmSocketScalar *);
int main(void){NvmSocketHostGrant *grant=0;NvmSocketIndirectOptions options={1,100000};NvmSocketScalar out={0};
 if(nvm_socket_host_grant_create_tcp_connections(&grant)!=NVM_SOCKET_HOST_OK)return 1;
 NvmSocketIndirectExecutionReport r=nvm_socket_indirect_program_test(grant,&options,&out);
 if(nvm_socket_host_grant_destroy(&grant)!=NVM_SOCKET_HOST_OK || grant)return 2;
 return !(r.runtime.status==NVM_SOCKET_RUNTIME_OK && !r.runtime.cleanup.cleanup_failures && out.tag==TAG_INT && out.value==''' + str(expected) + ");}\n")
            exe = consumer / f"program-{case}"
            self.command(f"installed-{case}-build", [*compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                         "-I" + str(self.prefix / "include"), str(generated), str(driver),
                         str(self.prefix / "lib/libnano_socket_runtime.a"), *ldflags, "-o", str(exe)], cwd=consumer)
            self.command(f"installed-{case}-run", [str(exe)], cwd=consumer)
            symbols = self.command(f"installed-{case}-symbols", ["nm", str(exe)], cwd=consumer)
            self.assertNotRegex(symbols, r"\b_?(nvm_socket_execute_indirect_bytes|nvm2c_emit_socket_indirect_bytes|vm_execute)\b")
        # I also resolve the public VM and emitter from the installed archive.
        wire = (self.artifacts / "case-2.nvm").read_bytes()
        driver = consumer / "vm-emitter.c"
        driver.write_text('''#include <nanolang/socket/nanoisa/socket_indirect_native_public.h>
#include <stdlib.h>
static const unsigned char wire[]={''' + ",".join(map(str, wire)) + '''};
int main(void){
 NvmSocketHostGrant *grant=0;NvmSocketIndirectOptions options={1,100000};NvmSocketScalar out={0};
 char *generated=0;char diagnostic[256]={0};
 if(nvm2c_emit_socket_indirect_bytes(wire,sizeof wire,"installed",&generated,diagnostic,sizeof diagnostic)!=NVM_SOCKET_RUNTIME_OK)return 1;
 free(generated);
 if(nvm_socket_host_grant_create_tcp_connections(&grant)!=NVM_SOCKET_HOST_OK)return 2;
 NvmSocketIndirectExecutionReport r=nvm_socket_execute_indirect_bytes(grant,wire,sizeof wire,&options,&out);
 if(nvm_socket_host_grant_destroy(&grant)!=NVM_SOCKET_HOST_OK || grant)return 3;
 return !(r.runtime.status==NVM_SOCKET_RUNTIME_OK && !r.runtime.cleanup.cleanup_failures && out.tag==TAG_INT && out.value==4);
}
''')
        exe = consumer / "vm-emitter"
        self.command("installed-vm-emitter-build", [*compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                     "-I" + str(self.prefix / "include"), str(driver),
                     str(self.prefix / "lib/libnano_socket_runtime.a"), *ldflags, "-o", str(exe)], cwd=consumer)
        self.command("installed-vm-emitter-run", [str(exe)], cwd=consumer)

    def check_public_boundaries(self, native, fixture, wire, case):
        for mode, expected in ((1, 1), (2, 5), (3, 11)):
            for label, args in (("native", [str(native), "100000", "0", str(mode)]),
                                ("vm", [str(fixture), "vm", str(wire), "100000", "0", str(mode)])):
                report = json.loads(self.command(native.name + f"-grant-{mode}-" + label, args))
                self.assertEqual(report["status"], expected)
                self.assertEqual((report["acquired"], report["steps"], report["opens"], report["closes"], report["fields"], report["value"]), (0, 0, 0, 0, 99, 12345))

    def driver_source(self):
        return '''#include <nanolang/socket/nanoisa/socket_indirect_native_public.h>
#include <nanolang/socket/nanoisa/socket_host_grant_internal.h>
#include <stdlib.h>
static void public_native_busy(void);
#define SOCKET_DISPATCH_ON_OPEN() public_native_busy()
#include "tests/nanoisa/socket_dispatch_host.h"
extern NvmSocketIndirectExecutionReport nvm_socket_indirect_program_test(NvmSocketHostGrant *,const NvmSocketIndirectOptions *,NvmSocketScalar *);
static void public_native_busy(void){
 NvmSocketIndirectExecutionReport r=nvm_socket_indirect_program_test((void *)1,(void *)1,(void *)1);
 if(r.runtime.status!=NVM_SOCKET_RUNTIME_BUSY || r.instruction_limit || r.runtime.acquired)abort();
 if(nvm_socket_host_grant_revoke((void *)1)!=NVM_SOCKET_HOST_BUSY)abort();
}
int main(int argc,char **argv){
 if(argc!=3 && argc!=4)return 2;
 NvmSocketHostGrant *grant=NULL;
 if(nvm_socket_host_grant_create_tcp_connections(&grant)!=NVM_SOCKET_HOST_OK)return 3;
 int mode=argc==4?atoi(argv[3]):0;
 if(mode==2 && nvm_socket_host_grant_revoke(grant)!=NVM_SOCKET_HOST_OK)return 4;
 if(mode==3 && nvm_socket_host_enter_query()!=NVM_SOCKET_HOST_OK)return 5;
 NvmSocketIndirectOptions options={1,strtoull(argv[1],NULL,10)};
 dispatch_close_fault=atoi(argv[2])!=0;
 NvmSocketScalar scalar={TAG_INT,12345};
 NvmSocketRuntimeView out={.fields=99,.values={12345}};
 NvmSocketIndirectExecutionReport r=nvm_socket_indirect_program_test(mode==1?NULL:grant,&options,&scalar);
 if(r.runtime.status==NVM_SOCKET_RUNTIME_OK){out.fields=1;out.values[0]=scalar.value;}
 else if(scalar.tag!=TAG_INT || scalar.value!=12345)abort();
 if(mode==3)nvm_socket_host_leave();
 if(nvm_socket_host_grant_destroy(&grant)!=NVM_SOCKET_HOST_OK || grant)return 6;
 dispatch_report(r,out);return 0;}
'''
