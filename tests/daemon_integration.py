"""I compare successful standalone and daemon runs without touching ambient daemons."""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "tests/selfhost" / f"test_{name}.nano" for name in (
    "arithmetic_ops", "comparison_ops", "function_calls", "if_else",
    "let_set", "logical_ops", "recursion", "while_loops")]


def stop(process):
    # I signal only a process group I created, never a name or a PID file.
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def execute(command, env, timeout, log):
    with log.open("wb") as output:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"I timed out running {command[0]}") from error
        finally:
            stop(process)
    if status:
        raise RuntimeError(f"I observed exit status {status}: {' '.join(map(str, command))}")
    with log.open("rb") as output:
        return output.read(os.fstat(output.fileno()).st_size)


def ready(process, path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("I observed daemon death during startup")
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(min(timeout, 0.1))
            try:
                client.connect(str(path))
                client.sendall(bytes([1, 2, 0, 0, 0, 0, 0, 0]))
                response = b""
                while len(response) < 8:
                    part = client.recv(8 - len(response))
                    if not part:
                        raise RuntimeError("I lost my daemon during its readiness ping")
                    response += part
                if response != bytes([1, 0x13, 0, 0, 0, 0, 0, 0]):
                    raise RuntimeError("I received an invalid daemon readiness response")
                return
            except OSError:
                pass
        time.sleep(0.02)
    raise RuntimeError("I timed out waiting for my daemon socket")


def compare(compiler, vm, daemon, sources, timeout=60, startup_timeout=5):
    if not sources:
        raise RuntimeError("I refuse an empty daemon corpus")
    with tempfile.TemporaryDirectory(prefix="nano-vmd-", dir="/tmp") as directory:
        work = Path(directory)
        env = dict(os.environ, NANOVMD_SOCKET=str(work / "vm.sock"),
                   NANOVMD_NO_AUTOSTART="1")
        process = None
        try:
            artifacts = []
            for index, source in enumerate(sources):
                artifact = work / f"{index}.nvm"
                execute([str(compiler), str(source), "-o", str(artifact), "--emit-nvm"],
                        env, timeout, work / f"{index}.compile.log")
                if not artifact.is_file() or not artifact.stat().st_size:
                    raise RuntimeError(f"I did not receive bytecode for {source}")
                expected = execute([str(vm), str(artifact)], env, timeout,
                                   work / f"{index}.standalone.log")
                artifacts.append((source, artifact, expected))
            with (work / "daemon.log").open("wb") as output:
                process = subprocess.Popen([str(daemon), "--foreground", "--no-timeout"],
                                           env=env, cwd=ROOT, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                ready(process, work / "vm.sock", startup_timeout)
                for index, (source, artifact, expected) in enumerate(artifacts):
                    if process.poll() is not None:
                        raise RuntimeError("I observed daemon death before execution")
                    actual = execute([str(vm), "--daemon", str(artifact)], env,
                                     timeout, work / f"{index}.daemon.log")
                    if process.poll() is not None:
                        raise RuntimeError("I observed daemon death during execution")
                    if actual != expected:
                        raise RuntimeError(f"I found differing output for {source}")
                    print(f"I passed {source.name}")
                if process.poll() is not None:
                    raise RuntimeError("I observed daemon death after execution")
        except (OSError, RuntimeError):
            for log in sorted(work.glob("*.log")):
                with log.open("rb") as stream:
                    stream.seek(max(0, os.fstat(stream.fileno()).st_size - 65536))
                    print(f"{log.name}:\n{stream.read(65536).decode(errors='replace')}",
                          file=sys.stderr)
            raise
        finally:
            if process is not None:
                stop(process)
    print(f"{len(sources)} selected, {len(sources)} passed, 0 skipped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "bin/nano_virt")
    parser.add_argument("--vm", type=Path, default=ROOT / "bin/nano_vm")
    parser.add_argument("--daemon", type=Path, default=ROOT / "bin/nano_vmd")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("sources", type=Path, nargs="*")
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error("I require a timeout in (0, 600] seconds")
    try:
        compare(args.compiler.resolve(), args.vm.resolve(), args.daemon.resolve(),
                [p.resolve() for p in args.sources] if args.sources else SOURCES,
                args.timeout)
        return 0
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
