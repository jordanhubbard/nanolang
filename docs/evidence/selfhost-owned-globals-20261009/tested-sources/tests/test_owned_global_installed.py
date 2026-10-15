"""I retain the same global source contract through both installed Nano stages."""
from tests.test_owned_global_source import OwnedGlobalSource as _GlobalSources


class _InstalledGlobals(_GlobalSources):
    def source_route(self, source, accepted, diagnostics=None):
        # I assert the actual independent checker/lowerer boundary for each stage.
        expected = {
            'test_wrong_global_assignment_type_refused':
                'Assignment to counter: I expected a value of type `int`, but found `bool`',
            'test_same_shape_wrong_union_assignment_refused':
                'Assignment to selected: I expected a value of type `First`, but found `Second`',
            'test_same_shape_wrong_union_initializer_refused':
                'I require the exact union constructor declaration',
        }.get(self._testMethodName)
        if expected:
            diagnostics = {'nanoc_c': expected}
        super().source_route(source, accepted, diagnostics)

    def test_failed_global_shadow_preserves_output(self):
        self.source('let counter: int = 1', 'assert (== counter 1)',
                    'fn read() -> int { return counter }\n'
                    'shadow read { assert (== (read) 99) }\n',
                    accepted=False, diagnostic='I could not execute the module: Assertion failed')


class OwnedGlobalsStage1(_InstalledGlobals):
    source_compiler = 'nanoc_stage1'


class OwnedGlobalsStage2(_InstalledGlobals):
    source_compiler = 'nanoc_stage2'


del _GlobalSources, _InstalledGlobals
