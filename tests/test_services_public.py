"""I qualify per-instance public grants and installed mixed service consumers."""
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import tempfile
import threading
import unittest
from tests import test_socket_dispatch as support

ROOT=Path(__file__).resolve().parents[1]

class ServicesPublic(unittest.TestCase):
    command=support.SocketDispatch.command

    def validate(self, report, mode, fuel, trap=False, hooked=True):
        status={1:1,2:5,3:11,4:5,5:5,6:2,7:2,8:5,9:9}.get(mode,3 if fuel<100 else 9 if trap else 0)
        self.assertEqual(report['status'],status,report)
        self.assertEqual(report['value'],42 if status==0 else 999)
        if mode in range(1,9):
            self.assertEqual((report['acquired'],report['steps'],report['files'],report['sockets']),(0,0,0,0))
        elif hooked and fuel>=100:self.assertEqual((report['files'],report['sockets']),(2,1))
        if mode==9:self.assertGreater(report['cleanup'],0)
        else:self.assertEqual(report['cleanup'],0)

    def test_grants_and_installed_consumers(self):
        self.artifacts=Path(tempfile.mkdtemp(prefix='nano-services-public-'))
        print(f'I retain mixed public evidence at {self.artifacts}',flush=True)
        prefix=Path(os.environ['SERVICES_PUBLIC_TEST_PREFIX'])
        compiler=shlex.split(os.environ.get('NANO_SOCKET_DISPATCH_CC','cc'))
        flags=['-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-g','-O1','-I.',
            '-I'+str(prefix/'include'),'-fsanitize=address,undefined','-fno-omit-frame-pointer']
        ldflags=shlex.split(os.environ.get('SOCKET_DISPATCH_LDFLAGS','-lm -lcrypto'))
        sources=['nanoisa/services_nominal','nanoisa/services_flow','nanoisa/services_runtime',
            'nsi_services_values','nsi_file_values','nsi_file','nsi_socket_values','nsi_socket','nsi_cap',
            'nsi_file_plan','nsi_socket_plan','nanoisa/file_host_grant','nanoisa/services_host_grant',
            'nanoisa/services_indirect_public_abi']
        ordinary=[p for p in shlex.split(os.environ['SOCKET_DISPATCH_OBJECTS']) if not any(p.endswith('/'+s+'.o') for s in sources)]
        fixture_objects=[p for p in shlex.split(os.environ['SERVICES_VM_OBJECTS']) if p not in ordinary and not any(p.endswith('/'+s+'.o') for s in sources)]
        providers=[]
        for source in sources:
            obj=self.artifacts/(Path(source).name+'.o');hooks=[]
            if source in ('nsi_file','nsi_socket'):
                hooks=['-include','tests/nanoisa/services_public_hooks.h']
                hooks+=['-Dtmpfile=services_public_tmpfile','-Dfclose=services_public_fclose'] if source=='nsi_file' else ['-Dsocket=services_public_socket','-Dclose=services_public_close']
            self.command(obj.stem+'-build',[*compiler,*flags,*hooks,'-c','src/'+source+'.c','-o',str(obj)]);providers.append(str(obj))
        fixture=self.artifacts/'fixture'
        self.command('fixture-build',[*compiler,*flags,'tests/nanoisa/test_services_public.c',
            'src/nanovm/services_indirect_public_vm.c','src/nanoisa/services_indirect_public_native.c',
            *providers,*ordinary,*fixture_objects,*ldflags,'-o',str(fixture)])
        allocated=self.artifacts/'services_host_grant-allocation.o'
        self.command('grant-allocation-build',[*compiler,*flags,'-include','tests/nanoisa/service_alloc_hooks.h',
            '-Dcalloc=service_test_calloc','-c','src/nanoisa/services_host_grant.c','-o',str(allocated)])
        allocation_providers=[str(allocated) if Path(p).name=='services_host_grant.o' else p for p in providers]
        fault=self.artifacts/'fixture-allocation'
        self.command('fixture-allocation-build',[*compiler,*flags,'-DSERVICE_ALLOC_TEST','tests/nanoisa/test_services_public.c',
            'src/nanovm/services_indirect_public_vm.c','src/nanoisa/services_indirect_public_native.c',
            *allocation_providers,*ordinary,*fixture_objects,*ldflags,'-o',str(fault)])
        report=json.loads(self.command('grant-allocation',[str(fault),'3','1','0',str(self.artifacts/'allocation.c'),str(self.artifacts/'allocation.nvm'),'0','0']))
        self.validate(report,0,0)
        consumer=self.artifacts/'installed';consumer.mkdir()
        host=(ROOT/'tests/nanoisa/services_public_host.h').read_text()
        (consumer/'host.h').write_text(host)
        for ipv6 in (False,True):
            server=socket.socket(socket.AF_INET6 if ipv6 else socket.AF_INET,socket.SOCK_STREAM)
            server.bind(('::1' if ipv6 else '127.0.0.1',0));server.listen(16);server.settimeout(.1)
            port=server.getsockname()[1];stop=threading.Event();received=[];failures=[]
            def serve():
                while not stop.is_set():
                    try: peer,_=server.accept()
                    except socket.timeout:continue
                    except OSError:break
                    try:
                        with peer:
                            peer.settimeout(5);data=peer.recv(1)
                            if data:received.append(data);peer.sendall(b'\0');peer.recv(1)
                    except OSError as error:failures.append(str(error))
            thread=threading.Thread(target=serve,daemon=True);thread.start()
            try:
                for program in (0,3,5):
                    label=f'{int(ipv6)}-{program}';generated=self.artifacts/(label+'.c');wire=self.artifacts/(label+'.nvm')
                    output=self.command(label+'-vm',[str(fixture),str(program),str(port),str(int(ipv6)),str(generated),str(wire),'0','100000'])
                    reference=json.loads(output);self.validate(reference,0,100000,program==5)
                    driver=self.artifacts/(label+'-driver.c')
                    driver.write_text('#include "tests/nanoisa/services_public_host.h"\n'+
                        'extern NvmServicesIndirectExecutionReport nvm_services_indirect_program_test(NvmServicesHostGrant *,const NvmServicesIndirectOptions *,NvmServicesScalar *);\n'+
                        'int main(int argc,char **argv){if(argc!=3)return 2;return public_run(nvm_services_indirect_program_test,(unsigned)strtoul(argv[1],NULL,10),strtoull(argv[2],NULL,10));}\n')
                    native=self.artifacts/(label+'-native')
                    self.command(label+'-native-build',[*compiler,*flags,str(generated),str(driver),*providers,*ordinary,*ldflags,'-o',str(native)])
                    modes=[(0,100000)] if program!=3 else [(0,100000),*( (m,100000) for m in range(1,10)),(0,0),(0,45)]
                    original=generated.read_bytes()
                    for mode,fuel in modes:
                        actual=json.loads(self.command(label+f'-native-{mode}-{fuel}',[str(native),str(mode),str(fuel)]))
                        self.validate(actual,mode,fuel,program==5)
                        if program==3:
                            vm=json.loads(self.command(label+f'-vm-{mode}-{fuel}',[str(fixture),str(program),str(port),str(int(ipv6)),str(generated),str(wire),str(mode),str(fuel)]))
                            self.validate(vm,mode,fuel)
                            for field in ('status','acquired','cleanup','value','tag','files','sockets'):self.assertEqual(actual[field],vm[field])
                            self.assertEqual(original,generated.read_bytes())
                    if program==3:
                        # I link a C99 consumer outside the checkout, using only installed files.
                        cg=consumer/(label+'.c');shutil.copyfile(generated,cg)
                        cd=consumer/(label+'-driver.c');cd.write_text(driver.read_text().replace('tests/nanoisa/services_public_host.h','host.h'))
                        exe=consumer/(label+'-native')
                        installed_flags=['-std=c99','-Wall','-Wextra','-Werror','-O2','-I'+str(prefix/'include')]
                        self.command(label+'-installed-build',[*compiler,*installed_flags,str(cg),str(cd),str(prefix/'lib/libnano_services_runtime.a'),*ldflags,'-o',str(exe)],cwd=consumer)
                        for mode in (0,1,2,3,4,5,6,7,8):
                            r=json.loads(self.command(label+f'-installed-{mode}',[str(exe),str(mode),'100000'],cwd=consumer));self.validate(r,mode,100000,hooked=False)
                        symbols=self.command(label+'-installed-symbols',['nm',str(exe)],cwd=consumer)
                        self.assertNotRegex(symbols,r'\b_?(nvm_services_execute_indirect_bytes|nvm2c_emit_services_indirect_bytes|vm_execute)\b')
                        vd=consumer/(label+'-vm.c');vd.write_text('#include "host.h"\nstatic const unsigned char wire[]={'+','.join(map(str,wire.read_bytes()))+'''};
static NvmServicesIndirectExecutionReport execute(NvmServicesHostGrant *g,const NvmServicesIndirectOptions *o,NvmServicesScalar *v){return nvm_services_execute_indirect_bytes(g,wire,sizeof wire,o,v);}
int main(void){char *code=NULL,why[256];if(nvm2c_emit_services_indirect_bytes(wire,sizeof wire,"installed",&code,why,sizeof why)!=NVM_SERVICES_RUNTIME_OK)return 1;free(code);return public_run(execute,0,100000);}
''')
                        ve=consumer/(label+'-vm');self.command(label+'-installed-vm-build',[*compiler,*installed_flags,str(vd),str(prefix/'lib/libnano_services_runtime.a'),*ldflags,'-o',str(ve)],cwd=consumer)
                        r=json.loads(self.command(label+'-installed-vm',[str(ve)],cwd=consumer));self.validate(r,0,100000,hooked=False)
                self.assertTrue(received);self.assertTrue(all(v==b'\0' for v in received));self.assertEqual(failures,[])
            finally:
                stop.set();server.close();thread.join(6);self.assertFalse(thread.is_alive())
