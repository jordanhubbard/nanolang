"""I qualify TCP documents through the same complete-byte and failure corpus."""
import unittest
from tests import test_nsi_file_binding

class SocketBindingPlan(test_nsi_file_binding.FileBindingPlan):
    kind='socket'

    def test_catalog_isolation(self):
        exe=self.work/'catalog-isolation'
        self.command('isolation-build', [*self.cc,*self.flags,
            'tests/test_nsi_binding_isolation.c','src/nsi_file_binding.c',
            'src/nsi_file_plan.c',*self.objects['linked'],*self.links,'-o',exe])
        out,_=self.command('isolation-run',[exe,'tests/fixtures/nsi_file_plan.json',
            'tests/fixtures/nsi_socket_plan.json'])
        self.assertIn(b'PASS simultaneous File/Socket plans',out)
        print(out.decode().strip(),flush=True)

if __name__=='__main__':
    unittest.main()
