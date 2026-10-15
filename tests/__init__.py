"""I own the AddressSanitizer leak-detection policy for every process a test spawns.

Support differs by OS: Linux ASan enables leak detection by default, while Apple's ASan runtime aborts with
"detect_leaks is not supported on this platform" when it is requested. Fixtures therefore do not name
detect_leaks=1; I apply the platform default at the subprocess boundary. On Darwin I set detect_leaks=0
unless a fixture chose a value, and I turn an explicit detect_leaks=1 off. A fixture that needs leak
detection to be off everywhere still says detect_leaks=0. Set NANO_FORCE_DETECT_LEAKS=1 to leave the
options untouched, for example with a Homebrew LLVM runtime that does support leak detection.
"""
import os
import re
import subprocess
import sys

if sys.platform == 'darwin' and os.environ.get('NANO_FORCE_DETECT_LEAKS') != '1':
    _LEAKS_ON = re.compile(r'(?<![\w-])detect_leaks=1(?!\d)')
    _LEAKS_SET = re.compile(r'(?<![\w-])detect_leaks=')
    _original_init = subprocess.Popen.__init__

    def _platform_leak_policy_init(self, *args, **kwargs):
        env = dict(os.environ if kwargs.get('env') is None else kwargs['env'])
        options = env.get('ASAN_OPTIONS', '')
        if _LEAKS_ON.search(options):
            options = _LEAKS_ON.sub('detect_leaks=0', options)
        elif not _LEAKS_SET.search(options):
            options = ':'.join(part for part in (options, 'detect_leaks=0') if part)
        env['ASAN_OPTIONS'] = options
        kwargs['env'] = env
        _original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _platform_leak_policy_init
