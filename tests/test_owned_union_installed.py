"""I preserve the original selected-owner sources through both installed stages."""
from tests.test_owned_union_c_source import _SourceRoute
from tests.test_selected_variant_ownership import SelectedVariantOwnership as _Selected
from tests.test_generic_selected_ownership import GenericSelectedOwnership as _Generic


class SelectedUnionStage1(_SourceRoute, _Selected):
    source_compiler = 'nanoc_stage1'


class SelectedUnionStage2(_SourceRoute, _Selected):
    source_compiler = 'nanoc_stage2'


class _GenericRoute(_SourceRoute):
    def check(self, source, accepted):
        self.source_route(source, accepted)


class GenericSelectedUnionStage1(_GenericRoute, _Generic):
    source_compiler = 'nanoc_stage1'


class GenericSelectedUnionStage2(_GenericRoute, _Generic):
    source_compiler = 'nanoc_stage2'


del _Selected, _Generic
