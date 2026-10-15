"""I qualify mixed/repeated nominal identities and retained module boundaries."""
import shlex
import shutil
import subprocess
import unittest
from tests import test_service_bindings_module as support

class ServicesFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        support.ServiceModule.setUpClass.__func__(cls)

    command = support.ServiceModule.command

    def qualify(self, instrument, websocket=False):
        name = ('websocket-' if websocket else '') + ('services-allocation' if instrument else 'services-linked')
        objects = list(self.objects)
        for source in ('nvm_format', 'nvm_v2_convert', 'nvm_v2_module',
                       'service_bindings_module', 'service_multi_nominal',
                       'service_multi_nominal_plan', 'services_nominal', 'services_flow', *(['services_host_grant'] if websocket else []), 'retained_layouts', 'nvm_v2_layouts'):
            objects = [p for p in objects if not p.endswith(f'/nanoisa/{source}.o')]
            obj = self.artifacts / f'{name}-{source}.o'
            hooks = (['-include', 'tests/nanoisa/service_alloc_hooks.h',
                      '-Dmalloc=service_test_malloc', '-Dcalloc=service_test_calloc',
                      '-Drealloc=service_test_realloc'] if instrument else [])
            self.command(f'{name}-{source}-build', [*self.compiler, *self.flags, *hooks,
                '-c', f'src/nanoisa/{source}.c', '-o', str(obj)])
            objects.insert(0, str(obj))
        exe = self.artifacts / name
        self.command(f'{name}-build', [*self.compiler, *self.flags,
            *(['-DSERVICE_ALLOC_TEST'] if instrument else []),
            'tests/nanoisa/test_mixed_websocket_flow.c' if websocket else 'tests/nanoisa/test_services_flow.c',
            *objects, *(['lib/libnano_services_runtime.a'] if websocket else []), *self.linkflags, '-o', str(exe)])
        output = self.command(f'{name}-run', [str(exe)], extra={})
        self.assertIn('PASS', output)
        print(output.strip(), flush=True)

    def test_linked_mixed_and_repeated_catalogs(self):
        self.qualify(False)

    def test_allocation_failures(self):
        self.qualify(True)

    def test_websocket_mixed_flow_and_runtime_refusal(self):
        self.qualify(False, True)

    def test_websocket_allocation_failures(self):
        self.qualify(True, True)

    def test_installed_websocket_grants(self):
        prefix = self.artifacts / 'grants-installed'
        self.command('grant-install', ['make', '-f', 'Makefile.gnu', 'install-services-public-runtime',
            f'PREFIX={prefix}', f'CC={shlex.join(self.compiler)}'])
        relocated = self.artifacts / 'grants-relocated'
        shutil.move(prefix, relocated)
        source = self.artifacts / 'grant-consumer.c'
        source.write_text(r'''#include <nanolang/services/nanoisa/services_host_grant.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1; } } while(0)
int main(void) {
    char path[]="/resolver/copied";
    NvmServicesHostConfig configs[4]={
        {.revision=NVM_SERVICES_HOST_POLICY_REVISION,.catalog=NVM_SERVICES_HOST_FILE,.allowed=true},
        {.revision=NVM_SERVICES_HOST_POLICY_REVISION,.catalog=NVM_SERVICES_HOST_TCP,.allowed=true},
        {.revision=NVM_SERVICES_HOST_POLICY_REVISION,.catalog=NVM_SERVICES_HOST_WEBSOCKET,.allowed=true,
         .allow_lookup=true,.max_timeout_ms=17,.resolver_helper=path},
        {.revision=NVM_SERVICES_HOST_POLICY_REVISION,.catalog=NVM_SERVICES_HOST_WEBSOCKET,.allowed=true,
         .max_timeout_ms=29}};
    NvmServicesHostGrant *g=NULL;
    CHECK(nvm_services_host_grant_create_config(configs,4,&g)==NVM_SERVICES_HOST_OK && g);
    memset(configs,0,sizeof configs);memset(path,0,sizeof path);
    CHECK(nvm_services_host_grant_revoke_instance(g,2)==NVM_SERVICES_HOST_OK);
    CHECK(nvm_services_host_grant_revoke_instance(g,4)==NVM_SERVICES_HOST_INVALID);
    CHECK(nvm_services_host_grant_revoke(g)==NVM_SERVICES_HOST_OK);
    CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK && !g);
    NvmServicesHostPolicy legacy={NVM_SERVICES_HOST_WEBSOCKET,true};
    CHECK(nvm_services_host_grant_create(&legacy,1,&g)==NVM_SERVICES_HOST_INVALID && !g);
    puts("PASS relocated C99 mixed WebSocket grant consumer");return 0;
}
''')
        crypto = shlex.split(subprocess.check_output(['pkg-config', '--libs', 'libcrypto'], text=True))
        exe = self.artifacts / 'grant-consumer'
        self.command('grant-consumer-build', [*self.compiler, '-std=c99', '-Wall', '-Wextra', '-Werror',
            '-fsanitize=address,undefined', '-I'+str(relocated / 'include'), str(source),
            str(relocated / 'lib/libnano_services_runtime.a'), *crypto, '-o', str(exe)])
        self.assertIn('PASS', self.command('grant-consumer-run', [str(exe)]))
