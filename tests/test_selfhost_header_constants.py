"""I run the same header contract through both installed self-hosted stages."""
import unittest
from tests import test_header_constant_functions as headers


class Stage1HeaderConstants(headers.HeaderConstantValues):
    compiler = 'nanoc_stage1'


class Stage2HeaderConstants(headers.HeaderConstantValues):
    compiler = 'nanoc_stage2'


if __name__ == '__main__':
    unittest.main()
