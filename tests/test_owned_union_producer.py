"""I retain every original selected-owner source through raw Nano production."""
from tests.test_owned_global_producer import _ProducerRoute
from tests.test_selected_variant_ownership import SelectedVariantOwnership as _Selected
from tests.test_generic_selected_ownership import GenericSelectedOwnership as _Generic


class _UnionRoute(_ProducerRoute):
    producer_refusals = {
        'test_duplicate_selected_field_consumption': 'named live exact whole owner argument',
        'test_guarded_owned_match_remains_rejected': 'exhaustive unguarded owned match',
        'test_ignored_resource_binding': 'explicit source owner consumption',
        'test_ignored_selected_payload': 'explicit source owner consumption',
        'test_incompatible_outer_join': 'matching ownership on every reaching edge',
        'test_loop_cannot_reconsume_outer_scrutinee': 'matching ownership on every reaching edge',
        'test_original_scrutinee_after_match': 'not a copied owner/reference',
        'test_partial_resource_field_move': 'resource constructor, exact owner move or destructive pattern',
        'test_unresolved_selected_field': 'explicit source owner consumption',
        'test_use_payload_after_destructure': 'live resource place',
        'test_wildcard_cannot_hide_owned_payload': 'complete named scalar union match coverage',
        'test_drop_selected_field_rejected': 'explicit source owner consumption',
        'test_drop_union_rejected': 'explicit source owner consumption',
        'test_duplicate_consume_rejected': 'not a copied owner/reference',
        'test_guarded_generic_match_remains_rejected': 'exhaustive unguarded owned match',
        'test_incomplete_match_rejected': 'complete named scalar union match coverage',
        'test_join_mismatch_rejected': 'matching ownership on every reaching edge',
        'test_match_then_reuse_rejected': 'not a copied owner/reference',
        'test_partial_move_rejected': 'resource constructor, exact owner move or destructive pattern',
        'test_resource_collection_still_rejected': 'exact concrete scalar union signatures',
        'test_unresolved_tuple_still_rejected': 'exact concrete scalar union signatures',
    }


class SelectedUnionProducer(_UnionRoute, _Selected):
    pass


class GenericSelectedUnionProducer(_UnionRoute, _Generic):
    def check(self, source, accepted):
        self.source_route(source, accepted)


del _Selected, _Generic
