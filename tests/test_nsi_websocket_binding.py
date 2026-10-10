"""I qualify owned WebSocket documents through the strict binding corpus."""
from tests import test_nsi_file_binding

class WebSocketBindingPlan(test_nsi_file_binding.FileBindingPlan):
    kind = 'websocket'

    def test_immutable_source_snapshots(self):
        import shutil
        from tests.test_nsi_file_binding import ROOT
        folder = self.work / 'snapshot'
        folder.mkdir()
        for name, kind in [('websocket', 'websocket'), ('file', 'file'), ('socket', 'socket')]:
            shutil.copyfile(ROOT / f'tests/fixtures/nsi_{kind}_plan.json', folder / (name + '.json'))
        executable = self.work / 'snapshot-test'
        self.command('snapshot-build', [*self.cc, *self.flags, 'tests/test_websocket_source_snapshot.c',
            'src/nanoisa/file_source_snapshot.c', 'src/nsi_file_binding.c', 'src/nsi_file_plan.c',
            'src/nsi_socket_binding.c', 'src/nsi_socket_plan.c', *self.objects['linked'], *self.links, '-o', executable])
        output, _ = self.command('snapshot-run', [executable, folder / 'binding.nano'])
        self.assertIn(b'PASS immutable WebSocket source snapshots', output)
