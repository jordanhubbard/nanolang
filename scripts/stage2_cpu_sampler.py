#!/usr/bin/env python3
"""I retain sampled native CPU call stacks as diagnostic evidence."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


class Sampler:
    def __init__(self, perf=None, event="cpu-clock:u"):
        self.event = event
        self.perf = Path(perf or shutil.which('perf')).resolve(strict=True)
        with self.perf.open('rb') as stream:
            assert stream.read(4) == b'\x7fELF'

    def start(self, pid, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        # I require the observed root to own its newly created process group.
        assert os.getpgid(pid) == pid
        record = {'qualification': False, 'target_pid': pid,
                  'perf': str(self.perf), 'perf_sha256_before': digest(self.perf),
                  'sampling_changes_timing': True, 'sample_frequency_hz': 49,
                  'event': self.event, 'stack_bytes': 8192}
        record['perf_version'] = subprocess.check_output([self.perf, 'version'], text=True, timeout=5)
        command = [str(self.perf), 'record', '-e', self.event, '-F', '49',
                   '--call-graph', 'dwarf,8192', '-p', str(pid), '-o', str(directory/'perf.data')]
        record['argv'] = command
        streams = [(directory/name).open('wb') for name in ['record.stdout', 'record.stderr']]
        try:
            process = subprocess.Popen(command, stdout=streams[0], stderr=streams[1],
                                       start_new_session=True)
        finally:
            for stream in streams: stream.close()
        save(directory/'record.json', record)
        return process, directory, record

    def finish(self, session):
        process, directory, record = session
        try:
            if process.poll() is None:
                assert os.getpgid(process.pid) == process.pid
                os.killpg(process.pid, signal.SIGINT)
                record['interrupt_sent'] = True
            try:
                record['exit_code'] = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                record['exit_code'] = process.wait(timeout=5)
                record['forced_stop'] = True
            record['perf_sha256_after'] = digest(self.perf)
            assert record['perf_sha256_before'] == record['perf_sha256_after']
            permitted = [0, -signal.SIGINT, 128+signal.SIGINT] if record.get('interrupt_sent') else [0]
            assert record['exit_code'] in permitted and not record.get('forced_stop')
            data = directory/'perf.data'
            assert data.is_file() and data.stat().st_size > 0
            record['data_sha256'] = digest(data)
            record['data_bytes'] = data.stat().st_size
            for label, options in [('report', ['report', '--stdio', '--header', '--no-children',
                                              '--sort', 'dso,symbol', '--percent-limit', '0.1']),
                                   ('script', ['script']), ('buildids', ['buildid-list'])]:
                argv = [str(self.perf), *options, '-i', str(data)]
                with (directory/(label+'.stdout')).open('wb') as out, (directory/(label+'.stderr')).open('wb') as err:
                    done = subprocess.run(argv, stdout=out, stderr=err, timeout=120)
                record[label] = {'argv': argv, 'exit_code': done.returncode,
                                 'stdout_sha256': digest(directory/(label+'.stdout')),
                                 'stderr_sha256': digest(directory/(label+'.stderr'))}
                assert done.returncode == 0
            script = (directory/'script.stdout').read_text(errors='replace')
            # Setup success and an empty perf header do not establish a CPU sample.
            record['actual_samples'] = len(re.findall(r'(?m)^.*\b\d+\.\d+:\s+\d+\s+\S+:\s*$', script))
            assert record['actual_samples'] > 0
            record['valid'] = True
        finally:
            save(directory/'record.json', record)
        return record


def calibrate(directory, event="cpu-clock:u", perf=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source, binary = directory/'busy.c', directory/'busy'
    source.write_text('''#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <time.h>
static volatile uint64_t result;
__attribute__((noinline)) static void nano_stage2_sample_control(void) {
    struct timespec begin, now;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    do {
        for (uint64_t i=0;i<100000;i++) result=result*33+i;
        clock_gettime(CLOCK_MONOTONIC, &now);
    } while ((now.tv_sec-begin.tv_sec)+(now.tv_nsec-begin.tv_nsec)/1e9 < 2.0);
}
int main(void) { nano_stage2_sample_control(); return 0; }
''')
    cc = Path(shutil.which('cc')).resolve(strict=True)
    before = {'compiler': digest(cc), 'source': digest(source)}
    argv = [str(cc), '-std=c99', '-O1', '-g', '-fno-omit-frame-pointer',
            '-Wall', '-Wextra', '-Werror', str(source), '-o', str(binary)]
    with (directory/'build.stdout').open('wb') as out, (directory/'build.stderr').open('wb') as err:
        subprocess.run(argv, stdout=out, stderr=err, timeout=30, check=True)
    before['binary'] = digest(binary)
    sampler = Sampler(perf=perf, event=event)
    process = subprocess.Popen([str(binary)], start_new_session=True)
    session = None
    try:
        session = sampler.start(process.pid, directory/'sample')
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        if session is not None:
            record = sampler.finish(session)
    report = (directory/'sample/report.stdout').read_text(errors='replace')
    assert 'nano_stage2_sample_control' in report
    after = {'compiler': digest(cc), 'source': digest(source), 'binary': digest(binary)}
    assert before == after
    save(directory/'control.json', {'qualification': False, 'valid': True,
                                    'before': before, 'after': after, 'build_argv': argv,
                                    'sample_count': record['actual_samples']})
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibrate', type=Path, required=True)
    parser.add_argument('--event', default='cpu-clock:u')
    parser.add_argument('--perf', type=Path)
    args = parser.parse_args()
    calibrate(args.calibrate, args.event, args.perf)
