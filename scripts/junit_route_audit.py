#!/usr/bin/env python3
"""I retain every original JUnit route result before returning its real failure."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def read_report(path):
    root = ET.parse(path).getroot()
    cases = []
    for node in root.iter('testcase'):
        outcomes = [kind for kind in ('failure', 'error', 'skipped') if node.find(kind) is not None]
        if len(outcomes) > 1:
            raise ValueError('I refuse contradictory JUnit outcomes.')
        cases.append({'name': node.attrib['name'], 'class': node.attrib['classname'],
                      'status': outcomes[0] if outcomes else 'passed',
                      'diagnostics': [item.attrib.get('message', '') for item in node if item.tag in outcomes]})
    totals = {kind: sum(case['status'] == kind for case in cases)
              for kind in ('passed', 'failure', 'error', 'skipped')}
    if (int(root.attrib['tests']) != len(cases) or
            int(root.attrib.get('failures', 0)) != totals['failure'] or
            int(root.attrib.get('errors', 0)) != totals['error'] or
            int(root.attrib.get('skipped', 0)) != totals['skipped']):
        raise ValueError('I refuse JUnit totals that differ from the actual case outcomes.')
    if not cases:
        raise ValueError('I require an executed JUnit corpus.')
    return {'total': len(cases), **totals, 'cases': cases}


def move_outputs(destination):
    """I preserve flat runner-owned files so no route can reuse another's output."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / '.test_output').iterdir():
        if path.is_file() and path.name.startswith(('nl_', 'app_', 'unit_')) and path.name.endswith(
                ('.compile.log', '.run.log', '.out', '.out.nvm')):
            shutil.move(str(path), destination / path.name)


def main():
    output = ROOT / '.test_output/junit-route-audit'
    output.mkdir(parents=True, exist_ok=True)
    routes = output / 'routes'
    routes.mkdir()  # I refuse to overwrite a prior route run.
    move_outputs(output / 'prior-runner-files')
    result = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'native_lifecycle_admitted': False, 'all_original_routes_passed': False, 'routes': []}
    failed = False
    route_selection = None
    for name, compiler, backend in [('seed-native', 'nanoc_c', 'c'),
                                    ('seed-vm', 'nano_virt', 'vm'),
                                    ('stage1', 'nanoc_stage1', 'c'),
                                    ('stage2', 'nanoc_stage2', 'c')]:
        destination = routes / name
        destination.mkdir()
        env = dict(os.environ, NANOLANG_COMPILER=str(ROOT / 'bin' / compiler), NANOLANG_BACKEND=backend)
        command = [sys.executable, 'scripts/ci_qualification_storage.py', 'run', '--label', 'junit-' + name,
                   '--output', str(output / 'storage'), '--', 'bash', 'tests/run_all_tests.sh',
                   '--format=junit', '--output=' + str(destination / 'results.xml')]
        with (destination / 'runner.log').open('w') as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'name': name, 'compiler': compiler, 'backend': backend, 'command': command,
               'returncode': process.returncode}
        try:
            row['report'] = read_report(destination / 'results.xml')
            selection = [(case['class'], case['name']) for case in row['report']['cases']]
            if route_selection is None:
                route_selection = selection
            row['same_original_selection'] = selection == route_selection
            row['passed'] = (process.returncode == 0 and row['report']['passed'] == row['report']['total']
                             and row['same_original_selection'])
        except (ET.ParseError, OSError, ValueError, KeyError) as error:
            row.update(passed=False, report_error=type(error).__name__ + ': ' + str(error))
        move_outputs(destination / 'retained-output')
        result['routes'].append(row)
        failed |= not row['passed']
        (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: value for key, value in row.items() if key != 'report'}), flush=True)
    result['all_original_routes_passed'] = not failed
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
