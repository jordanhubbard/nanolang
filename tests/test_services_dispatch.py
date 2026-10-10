"""I run the same checked mixed program through VM and independent generated C."""
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import threading
import unittest
from tests import test_socket_dispatch as support

class ServicesDispatch(unittest.TestCase):
    command = support.SocketDispatch.command

    def test_mixed_vm_and_native(self):
        self.artifacts = Path(tempfile.mkdtemp(prefix='nano-services-dispatch-'))
        print(f'I retain mixed dispatch evidence at {self.artifacts}', flush=True)
        compiler = shlex.split(os.environ.get('NANO_SOCKET_DISPATCH_CC', 'cc'))
        flags = ['-std=c11', '-D_DEFAULT_SOURCE', '-Wall', '-Wextra', '-Werror', '-g', '-O1', '-I.',
                 '-DNVM_SERVICES_INDIRECT_VM_PRIVATE', '-DNVM_SERVICES_INDIRECT_NATIVE_PRIVATE',
                 '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        sources = ['nanoisa/services_nominal', 'nanoisa/services_flow', 'nanoisa/services_runtime',
                   'nsi_services_values', 'nsi_file_values', 'nsi_file', 'nsi_socket_values', 'nsi_socket',
                   'nsi_cap', 'nsi_file_plan', 'nsi_socket_plan', 'nsi_websocket_values',
                   'nsi_websocket_transport', 'nsi_websocket_protocol', 'nsi_socket_resolver', 'utf8']
        flags += shlex.split(subprocess.check_output(['pkg-config', '--cflags', 'libcrypto'], text=True))
        ordinary = shlex.split(os.environ['SOCKET_DISPATCH_OBJECTS'])
        ordinary = [p for p in ordinary if not any(p.endswith('/'+s+'.o') for s in sources)]
        ldflags = shlex.split(os.environ.get('SOCKET_DISPATCH_LDFLAGS', '-lm -lcrypto'))
        providers = []
        for source in sources:
            obj = self.artifacts / (Path(source).name+'.o')
            self.command(obj.stem+'-build', [*compiler, *flags, '-c', 'src/'+source+'.c', '-o', str(obj)])
            providers.append(str(obj))
        fixture = self.artifacts/'fixture'
        fixture_objects = [p for p in shlex.split(os.environ['SERVICES_VM_OBJECTS']) if p not in ordinary and not any(p.endswith('/'+s+'.o') for s in sources)]
        self.command('fixture-build', [*compiler, *flags, 'tests/nanoisa/test_services_dispatch.c',
            'src/nanovm/services_vm_indirect_private.c', 'src/nanoisa/nvm2c_services_indirect_private.c',
            *providers, *ordinary, *fixture_objects, *ldflags, '-o', str(fixture)])
        allocated = self.artifacts/'services_runtime-allocation.o'
        self.command('runtime-allocation-build', [*compiler,*flags,'-include','tests/nanoisa/service_alloc_hooks.h',
            '-Dcalloc=service_test_calloc','-c','src/nanoisa/services_runtime.c','-o',str(allocated)])
        allocated_providers=[str(allocated) if Path(p).name=='services_runtime.o' else p for p in providers]
        for source in ('nsi_services_values','nsi_file_values','nsi_file','nsi_socket_values','nsi_socket','nsi_cap'):
            obj=self.artifacts/(source+'-allocation.o')
            self.command(source+'-allocation-build', [*compiler,*flags,'-include','tests/nanoisa/service_alloc_hooks.h',
                '-Dcalloc=service_test_calloc','-c','src/'+source+'.c','-o',str(obj)])
            allocated_providers=[str(obj) if Path(p).name==source+'.o' else p for p in allocated_providers]
        fault_fixture=self.artifacts/'fixture-allocation'
        self.command('fixture-allocation-build', [*compiler,*flags,'-DSERVICE_ALLOC_TEST','tests/nanoisa/test_services_dispatch.c',
            'src/nanovm/services_vm_indirect_private.c','src/nanoisa/nvm2c_services_indirect_private.c',
            *allocated_providers,*ordinary,*fixture_objects,*ldflags,'-o',str(fault_fixture)])
        native_cache = {}
        for ipv6 in (False, True):
            server = socket.socket(socket.AF_INET6 if ipv6 else socket.AF_INET, socket.SOCK_STREAM)
            server.bind(('::1' if ipv6 else '127.0.0.1', 0));server.listen(16);server.settimeout(.1)
            port = server.getsockname()[1];stop = threading.Event();received=[];failures=[]
            self.command(f'{int(ipv6)}-allocation', [str(fault_fixture),'3',str(port),str(int(ipv6)),str(self.artifacts/f'allocation-{ipv6}.c'),'0'])
            def serve():
                while not stop.is_set():
                    try: peer, _ = server.accept()
                    except socket.timeout: continue
                    except OSError: break
                    try:
                        with peer:
                            peer.settimeout(5);data=peer.recv(1)
                            if data:
                                received.append(data);peer.sendall(b'\0')
                                if peer.recv(1):failures.append('I expected closure after the byte')
                    except Exception as error: failures.append(str(error))
            thread=threading.Thread(target=serve,daemon=True);thread.start()
            try:
                for mode,fuel in [(0,100000),(1,100000),(2,100000),(3,100000),(5,100000),(3,0),(3,45)]:
                    label=f'{int(ipv6)}-{mode}-{fuel}';generated=self.artifacts/(label+'.c')
                    output=self.command(label+'-vm',[str(fixture),str(mode),str(port),str(int(ipv6)),str(generated),str(fuel)])
                    print(output.strip(),flush=True)
                    text=generated.read_text();self.assertIn('nvm_services_runtime_indirect_enter',text)
                    self.assertNotIn('nvm_services_vm_indirect_execute',text)
                    expected=3 if fuel<100 else 9 if mode&4 else 0
                    driver=self.artifacts/(label+'-driver.c')
                    driver.write_text('''#include "src/nanoisa/nvm2c_services_indirect_private.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv){
 if(argc!=3)return 2;unsigned expected=(unsigned)strtoul(argv[2],NULL,10);
 NvmServicesIndirectOptions options={1,strtoull(argv[1],NULL,10)};NvmServicesRuntimeView out={.values={999}};
 NvmServicesIndirectExecutionReport r=nvm_services_native_indirect_execute(&options,&out);
 if(r.runtime.status!=expected || r.runtime.cleanup.cleanup_failures || r.runtime.cleanup.count!=3 ||
    out.values[0]!=(expected?999:42)){fprintf(stderr,"status=%u core=%u site=%u:%u value=%lld\\n",r.runtime.status,r.runtime.core_status,r.runtime.function,r.runtime.instruction,(long long)out.values[0]);return 1;}
 puts("PASS mixed generated native");return 0;}
''')
                    key=(ipv6,mode)
                    if key in native_cache:
                        native,prior_text=native_cache[key];self.assertEqual(text,prior_text)
                    else:
                        native=self.artifacts/(label+'-native')
                        self.command(label+'-native-build',[*compiler,*flags,str(generated),str(driver),*providers,*ordinary,*ldflags,'-o',str(native)])
                        native_cache[key]=(native,text)
                    self.command(label+'-native',[str(native),str(fuel),str(expected)])
                    if mode==0 and not ipv6:
                        probe=self.artifacts/'file-host-probe.o'
                        self.command('file-host-probe-build', [*compiler,*flags,'-Dtmpfile=services_test_tmpfile',
                            '-c','src/nsi_file.c','-o',str(probe)])
                        probe_providers=[str(probe) if Path(p).name=='nsi_file.o' else p for p in providers]
                        changes=[('wrong-instance',text.replace('runtime_service(c,0,reference,src,dst)',
                            'runtime_service(c,10,reference,src,dst)',1),6,3),
                            ('wrong-map',text.replace('ordinal!=0)return false;','ordinal!=999)return false;',1),2,0)]
                        for name,changed,status,count in changes:
                            self.assertNotEqual(changed,text)
                            bad=self.artifacts/(name+'.c');bad.write_text(changed)
                            bad_driver=self.artifacts/(name+'-driver.c')
                            bad_driver.write_text(driver.read_text().replace('r.runtime.cleanup.count!=3',f'r.runtime.cleanup.count!={count}')
                                .replace('int main(', 'static unsigned services_tmpfiles;\nFILE *services_test_tmpfile(void){services_tmpfiles++;return tmpfile();}\nint main(')
                                .replace(' puts("PASS mixed generated native");',' if(services_tmpfiles)return 3;\n puts("PASS mixed generated native");'))
                            exe=self.artifacts/name
                            self.command(name+'-build',[*compiler,*flags,str(bad),str(bad_driver),*probe_providers,*ordinary,*ldflags,'-o',str(exe)])
                            self.command(name+'-run',[str(exe),str(fuel),str(status)])
                    if mode==0:
                        symbols=self.command(label+'-symbols',['nm',str(native)])
                        self.assertNotRegex(symbols,r'\b_?(nvm_services_vm_indirect_execute|nvm2c_services_indirect_private_emit|vm_execute|vm_core_execute)\b')
                self.assertEqual(len(received),8);self.assertTrue(all(v==b'\0' for v in received));self.assertEqual(failures,[])
            finally:
                stop.set();server.close();thread.join(6);self.assertFalse(thread.is_alive())
