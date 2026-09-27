"""I keep sanitizer runtime options inside each host's supported boundary."""
import sys


def asan_options(*options):
    detect_leaks = "0" if sys.platform == "darwin" else "1"
    return ":".join(("detect_leaks=" + detect_leaks, *options))
