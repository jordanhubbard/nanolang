"""I preserve selected compiler and link flags for direct managed probes."""
import os
import shlex


def compiler_command(compiler):
    if compiler == 'clang':
        return shlex.split(os.environ.get('NMA_TEST_CLANG', 'clang'))
    return shlex.split(os.environ.get('NMA_TEST_CC', os.environ.get('CC', compiler)))


def compile_flags():
    return shlex.split(os.environ.get('NMA_TEST_CFLAGS', os.environ.get('CFLAGS', '')))


def link_flags():
    return shlex.split(os.environ.get('NMA_TEST_LDFLAGS', os.environ.get('LDFLAGS', '')))
