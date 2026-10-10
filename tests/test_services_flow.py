"""I qualify mixed/repeated nominal identities and retained module boundaries."""
import unittest
from tests import test_service_bindings_module as support

class ServicesFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        support.ServiceModule.setUpClass.__func__(cls)

    command = support.ServiceModule.command

    def qualify(self, instrument):
        name = 'services-allocation' if instrument else 'services-linked'
        objects = list(self.objects)
        for source in ('nvm_format', 'nvm_v2_convert', 'nvm_v2_module',
                       'service_bindings_module', 'service_multi_nominal',
                       'service_multi_nominal_plan', 'services_nominal', 'services_flow', 'retained_layouts', 'nvm_v2_layouts'):
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
            'tests/nanoisa/test_services_flow.c', *objects, *self.linkflags, '-o', str(exe)])
        output = self.command(f'{name}-run', [str(exe)], extra={})
        self.assertIn('PASS', output)
        print(output.strip(), flush=True)

    def test_linked_mixed_and_repeated_catalogs(self):
        self.qualify(False)

    def test_allocation_failures(self):
        self.qualify(True)
