"""I budget native builds explicitly and reap failed process groups."""
import os
import signal
import subprocess


def native_compile_timeout():
    """I share the existing host-build budget without changing guest limits."""
    try:
        timeout = int(os.environ.get('CMD_TIMEOUT', '600'))
    except ValueError as error:
        raise ValueError('I require a positive integer native compilation timeout.') from error
    if timeout < 1:
        raise ValueError('I require a positive integer native compilation timeout.')
    return timeout


def run_captured(args, *, timeout, build=False, **kwargs):
    """I capture output and kill the whole session on timeout or interruption."""
    budget = native_compile_timeout() if build else timeout
    command = list(map(str, args))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=budget)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=10)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
