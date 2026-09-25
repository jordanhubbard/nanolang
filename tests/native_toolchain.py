"""I retain explicit compiler commands and native runtime instrumentation in tests."""
import os
import shlex


def native_cc():
    compiler = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or
                           os.environ.get('NANO_CC') or os.environ.get('CC') or 'cc')
    if not compiler:
        raise ValueError('I require a nonempty native compiler command.')
    return compiler


def native_link_flags():
    return shlex.split(os.environ.get('NANO_ARTIFACT_LDFLAGS',
                                     os.environ.get('NANO_LDFLAGS',
                                                    os.environ.get('LDFLAGS', ''))))
