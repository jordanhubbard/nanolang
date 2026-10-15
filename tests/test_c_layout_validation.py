"""I apply the same layout contract through both C frontend drivers."""
from tests import test_selfhost_layout_validation as contract


class CLayoutValidation(contract.SelfhostLayoutValidation):
    native_compiler = contract.ROOT / 'bin/nanoc_c'
    nvm_compiler = contract.ROOT / 'bin/nano_virt'
