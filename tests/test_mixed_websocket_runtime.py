"""I compare checked mixed WebSocket execution in VM and independent native C."""
import json
import os
from pathlib import Path
import shlex
import shutil
import unittest
from tests import test_service_bindings_module as support
from tests import test_websocket_dispatch as peers

ROOT = Path(__file__).resolve().parents[1]

class MixedWebSocketRuntime(unittest.TestCase):
    command = support.ServiceModule.command
    server = peers.WebSocketDispatch.server
    bad_reply = False

    @classmethod
    def setUpClass(cls):
        support.ServiceModule.setUpClass.__func__(cls)
        cls.prepared = False

    def prepare(self, real=False, allocated=False):
        if not self.prepared:
            prefix = self.artifacts / 'installed'
            self.command('install', ['make', '-f', 'Makefile.gnu', 'install-services-public-runtime',
                f'PREFIX={prefix}', f'CC={shlex.join(self.compiler)}'])
            type(self).prefix = self.artifacts / 'relocated'
            shutil.move(prefix, self.prefix)
            sources = ['nanoisa/services_runtime', 'nanoisa/services_host_grant',
                'nsi_services_values', 'nsi_websocket_values', 'nsi_file_values', 'nsi_file',
                'nsi_socket_values', 'nsi_socket', 'nsi_cap']
            type(self).providers = []
            for source in sources:
                obj = self.artifacts / (Path(source).name+'.o')
                self.command(obj.stem+'-build', [*self.compiler, *self.flags,
                    '-c', 'src/'+source+'.c', '-o', str(obj)])
                self.providers.append(str(obj))
            type(self).ordinary = [p for p in self.objects if not any(p.endswith('/'+s+'.o') for s in sources)]
            type(self).prepared = True
        if allocated:
            type(self).allocated_providers = []
            for old in self.providers:
                name = Path(old).stem
                source = 'src/nanoisa/'+name+'.c' if name in ('services_runtime','services_host_grant') else 'src/'+name+'.c'
                obj = self.artifacts / (name+'-allocated.o')
                self.command(obj.stem+'-build', [*self.compiler, *self.flags,
                    '-include', 'tests/nanoisa/mixed_websocket_runtime_alloc.h',
                    '-Dmalloc=mixed_runtime_malloc', '-Dcalloc=mixed_runtime_calloc', '-c', source, '-o', str(obj)])
                self.allocated_providers.append(str(obj))
        fixture = self.artifacts / ('fixture-allocated' if allocated else 'fixture-real' if real else 'fixture-controlled')
        flags = ['-DMIXED_RUNTIME_REAL_TRANSPORT'] if real else []
        self.command(fixture.name+'-build', [*self.compiler, *self.flags, *flags,
            'tests/nanoisa/test_mixed_websocket_runtime.c', *(self.allocated_providers if allocated else self.providers), *self.ordinary,
            'lib/libnano_services_runtime.a', *self.linkflags, '-o', str(fixture)])
        return fixture

    def native(self, graph, real=False, allocated=False, installed=False):
        label = f'{int(real)}-{graph}'
        driver = self.artifacts / (label+'-driver.c')
        header = self.artifacts / 'host.h'
        header.write_text((ROOT/'tests/nanoisa/mixed_websocket_runtime_host.h').read_text().replace(
            '#include "../../src/nanoisa/services_indirect_public.h"',
            '#include <nanolang/services/nanoisa/services_indirect_public.h>'))
        driver.write_text('''#include "host.h"
extern NvmServicesIndirectExecutionReport nvm_services_indirect_program_mixed_test(NvmServicesHostGrant *,const NvmServicesIndirectOptions *,NvmServicesScalar *);
int main(int argc,char **argv){
 MIXED_CHECK(argc==4);mixed_mode=(unsigned)strtoul(argv[2],NULL,10);
 NvmServicesHostGrant *g=mixed_grant((unsigned)strtoul(argv[1],NULL,10),mixed_mode);
 NvmServicesIndirectOptions options={1,strtoull(argv[3],NULL,10)};NvmServicesScalar out={TAG_INT,12345};
 mixed_budget_start();
 NvmServicesIndirectExecutionReport r=nvm_services_indirect_program_mixed_test(g,&options,&out);
 MIXED_CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK && !g);mixed_report(r,out);return 0;
}
''')
        exe = self.artifacts / (label+('-installed' if installed else '-allocated' if allocated else '-native'))
        flags = ['-std=c99', '-Wall', '-Wextra', '-Werror', '-g', '-O1',
                 '-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if installed else self.flags
        self.command(exe.name+'-build', [*self.compiler, *flags,
            *(['-DMIXED_RUNTIME_REAL_TRANSPORT'] if real else []), '-I'+str(self.prefix/'include'),
            str(self.artifacts/(label+'.c')), str(driver), *([] if installed else self.allocated_providers if allocated else self.providers),
            str(self.prefix/'lib/libnano_services_runtime.a'), *self.linkflags, '-o', str(exe)])
        return exe

    def compare(self, fixture, native, graph, mode=0, fuel=100000, real=False):
        label = f'{int(real)}-{graph}'; shape = graph % 3
        vm = json.loads(self.command(label+f'-vm-{mode}-{fuel}',
            [str(fixture), 'vm', str(self.artifacts/(label+'.nvm')), str(shape), str(mode), str(fuel)]))
        compiled = json.loads(self.command(label+f'-native-{mode}-{fuel}',
            [str(native), str(shape), str(mode), str(fuel)]))
        self.assertEqual(vm, compiled)
        status = 5 if mode in (1, 3) else 2 if mode == 2 else 3 if fuel < 50 else 10 if mode == 7 else 0
        self.assertEqual(vm['status'], status, vm)
        self.assertEqual(vm['value'], 12345 if status else 0, vm)
        if mode in (1, 2, 3):
            self.assertEqual((vm['acquired'], vm['steps'], vm['connects']), (0, 0, 0))
        elif fuel < 50:
            self.assertEqual((vm['acquired'], vm['steps']), (1, fuel))
            self.assertEqual(vm['closes'], vm['connects'])
            self.assertEqual(vm['cleanup'], 0)
        elif not real:
            count = (1, 2, 5)[shape]
            self.assertEqual((vm['connects'], vm['sends'], vm['receives'], vm['closes']), (count,)*4)
            self.assertEqual(vm['instances'], ([1,0,0,0,0], [0,0,1,1,0], [1]*5)[shape])
            self.assertEqual(vm['cleanup'], count if mode == 7 else 0)
        return vm

    def test_controlled_vm_native_lifecycles(self):
        fixture = self.prepare()
        for graph in range(24):
            label = f'0-{graph}'
            self.command(label+'-emit', [str(fixture), 'emit', str(self.artifacts/(label+'.nvm')),
                str(self.artifacts/(label+'.c')), str(graph), '0'])
            native = self.native(graph)
            self.compare(fixture, native, graph)
            if graph >= 21:
                for mode in (1,2,3,5,6,7):
                    self.compare(fixture, native, graph, mode)
                for fuel in (0,3,7,16,28,40):
                    self.compare(fixture, native, graph, fuel=fuel)

    def test_real_transport_invalid_url(self):
        fixture = self.prepare(real=True)
        for graph in (21,22,23):
            label = f'1-{graph}'
            self.command(label+'-emit', [str(fixture), 'emit', str(self.artifacts/(label+'.nvm')),
                str(self.artifacts/(label+'.c')), str(graph), '1'])
            native = self.native(graph, real=True)
            self.compare(fixture, native, graph, real=True)

    def test_allocation_failures_match(self):
        fixture = self.prepare(allocated=True)
        graph = 23;label = '0-23'
        self.command('allocated-emit', [str(fixture), 'emit', str(self.artifacts/(label+'.nvm')),
            str(self.artifacts/(label+'.c')), str(graph), '0'])
        native = self.native(graph, allocated=True)
        for budget in range(128):
            extra = {'NANOLANG_TEST_RUNTIME_BUDGET': str(budget)}
            vm = json.loads(self.command(f'allocated-vm-{budget}', [str(fixture), 'vm',
                str(self.artifacts/(label+'.nvm')), '2', '0', '100000'], extra=extra))
            compiled = json.loads(self.command(f'allocated-native-{budget}',
                [str(native), '2', '0', '100000'], extra=extra))
            self.assertEqual(vm, compiled)
            self.assertEqual(vm['closes'], vm['connects'])
            self.assertEqual(vm['cleanup'], 0)
            if not vm['faults']:
                self.assertEqual((vm['status'], vm['value'], vm['receives']), (0, 0, 5))
                self.assertGreater(budget, 10)
                print(f'I checked {budget} mixed runtime allocation failure prefixes.', flush=True)
                break
            self.assertEqual(vm['status'], 4, vm)
            self.assertEqual(vm['value'], 12345)
        else:
            self.fail('I did not reach an allocation-complete mixed invocation')

    def test_relocated_archive_execution(self):
        fixture = self.prepare()
        for graph in (22,23):
            label = f'0-{graph}'
            self.command(label+'-installed-emit', [str(fixture), 'emit', str(self.artifacts/(label+'.nvm')),
                str(self.artifacts/(label+'.c')), str(graph), '0'])
            native = self.native(graph, installed=True)
            self.compare(fixture, native, graph)
            self.compare(fixture, native, graph, mode=1)

    def test_negative_close_consumes_each_instance(self):
        fixture = self.prepare()
        for graph in (24,25,26):
            label = f'0-{graph}'
            self.command(label+'-negative-close-emit', [str(fixture), 'emit', str(self.artifacts/(label+'.nvm')),
                str(self.artifacts/(label+'.c')), str(graph), '2'])
            native = self.native(graph)
            self.compare(fixture, native, graph, mode=8)

    def test_required_real_peer_lifecycles(self):
        # I keep this gate required when the host refuses local listeners.
        port, messages = self.server()
        fixture = self.prepare(real=True)
        for graph in (21,22,23):
            label = f'1-{graph}'
            self.command(label+'-live-emit', [str(fixture), 'emit', str(self.artifacts/(label+'.nvm')),
                str(self.artifacts/(label+'.c')), str(graph), str(port)])
            native = self.native(graph, real=True)
            self.compare(fixture, native, graph, real=True)
        self.assertEqual(messages, [b'a\0b']*16)

    def test_live_graphs_prepare(self):
        fixture = self.prepare(real=True)
        for graph in (21,22,23):
            label = f'1-{graph}'
            self.command(label+'-live-prepare', [str(fixture), 'emit', str(self.artifacts/(label+'.nvm')),
                str(self.artifacts/(label+'.c')), str(graph), '9'])
            self.native(graph, real=True)

    def test_generated_c99_portability(self):
        fixture = self.prepare();graph = 22;label = '0-22'
        self.command('portable-emit', [str(fixture), 'emit', str(self.artifacts/(label+'.nvm')),
            str(self.artifacts/(label+'.c')), str(graph), '0'])
        self.native(graph, installed=True)
        compiler = shlex.split(os.environ.get('NANO_MIXED_C99_CC', shlex.join(self.compiler)))
        exe = self.artifacts / 'portable-c99'
        self.command('portable-c99-build', [*compiler, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
            '-I'+str(self.prefix/'include'), str(self.artifacts/(label+'.c')),
            str(self.artifacts/(label+'-driver.c')), str(self.prefix/'lib/libnano_services_runtime.a'),
            *self.linkflags, '-o', str(exe)])
        self.compare(fixture, exe, graph)
