#!/usr/bin/env python3
"""I qualify one private version-token candidate and restore the checkout afterward."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ci_source_capture_linux as capture

OUT = ROOT / '.test_output/source-capture-diagnostic/as246-candidate'


def command(argv, label, timeout, *, cwd=None, output=None, env=None):
    destination = OUT if output is None else Path(output)
    stdout_path, stderr_path = (destination / (label + suffix) for suffix in ('.stdout', '.stderr'))
    # I retain bytes as the native process writes them, including controller loss.
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        process = subprocess.Popen(list(map(str, argv)), cwd=ROOT if cwd is None else cwd,
            env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            if isinstance(error, subprocess.TimeoutExpired):
                error.output, error.stderr = stdout_path.read_bytes(), stderr_path.read_bytes()
            raise
    result = subprocess.CompletedProcess(list(map(str, argv)), process.returncode,
                                        stdout_path.read_bytes(), stderr_path.read_bytes())
    if result.returncode:
        raise RuntimeError(f'I retain {label} refusal with exit {result.returncode}.')
    return result


def fixed_assembler_candidate(source, test, previous):
    directory = Path(os.environ['NANO_FIXED_AS246_DIRECTORY']).resolve(strict=True)
    metadata = json.loads((directory / 'build.json').read_text())
    expected = (directory / 'bin/as').resolve(strict=True)
    if (not metadata.get('completed') or metadata.get('source_sha256') !=
            'd75a94f4d73e7a4086f7513e67e439e8fcdcbb726ffe63f4661744e6256b2cf2' or
            metadata.get('fixed_source_sha256') !=
            '6ffe0ed82d02c8f92198b4c2b5200a8093b7eea0a07a96bf951eecabf021b036' or
            metadata.get('assembler_sha256') != hashlib.sha256(expected.read_bytes()).hexdigest()):
        raise RuntimeError('I require the complete fixed-source/native assembler qualification.')
    if (metadata.get('gas_expected_passes', 0) < 1000 or len(metadata.get('witnesses', [])) != 8 or
            any(item['original_source_identity'] != item['expected'] for item in metadata['witnesses'])):
        raise RuntimeError('I require all native regression and source-identity outcomes.')
    # Artifact downloads preserve bytes but can remove the executable bit.
    expected.chmod(0o700)
    compiler = str(Path(shutil.which('cc')).resolve(strict=True))
    if (compiler != previous['compiler'] or hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
            != previous['tool_sha256'][compiler]):
        raise RuntimeError('I require the unchanged original C compiler image.')
    command([compiler, '-print-prog-name=as'], 'fixed-selected-assembler', 30)
    selected = Path((OUT / 'fixed-selected-assembler.stdout').read_text().strip()).resolve(strict=True)
    if selected != expected:
        raise RuntimeError('I require the actual original compiler to select the fixed assembler.')
    command([str(selected), '--version'], 'fixed-assembler-version', 30)
    version = (OUT / 'fixed-assembler-version.stdout').read_text()
    if (version != metadata['assembler_version'] or
            'GNU Binutils for NanoLang retained capture r1' not in version.splitlines()[0]):
        raise RuntimeError('I require the exact separately identified fixed assembler.')
    banner = version.splitlines()[0] + '\n'
    signature = 'static bool module_assembler_version_supported(const char *version) {\n'
    if source.count(signature) != 1:
        raise RuntimeError('I refuse an unexpected assembler-admission function.')
    source = source.replace(signature, signature +
        '    /* I qualify only this identified source repair, not generic GNU2.46. */\n' +
        '    if (version && !strncmp(version, ' + json.dumps(banner) + ', ' +
        str(len(banner)) + ')) return true;\n')
    end = '"GNU assembler (GNU Binutils for Ubuntu) 2.42\\nCopyright text\\n"]'
    if test.count(end) != 1:
        raise RuntimeError('I require the unchanged original assembler fixtures.')
    test = test.replace(end, end[:-1] + ', ' + json.dumps(banner + 'Copyright text\n') + ']')
    start = 'rejected = ["GNU assembler (GNU Binutils) 2.41\\n",'
    refused = [banner.replace('capture r1', 'capture r2'), banner.rstrip('\n'),
               'GNU assembler (GNU Binutils) 2.46.0.20260210\n',
               'GNU assembler (GNU Binutils for Ubuntu) 2.46\n']
    if test.count(start) != 1:
        raise RuntimeError('I require the unchanged original assembler refusal fixtures.')
    test = test.replace(start, 'rejected = [' + ', '.join(map(json.dumps, refused)) +
                        ', "GNU assembler (GNU Binutils) 2.41\\n",')
    return source, test, {'compiler': compiler, 'assembler': str(selected),
        'assembler_sha256': metadata['assembler_sha256'], 'version': version,
        'source_sha256': metadata['source_sha256'],
        'fixed_source_sha256': metadata['fixed_source_sha256']}



def original_userguide(record):
    env = dict(os.environ)
    inherited = env.pop('NANO_SHADOW_TIMEOUT_SECONDS', None)
    record['ordinary_userguide_environment'] = {
        'inherited_shadow_override': inherited, 'shadow_override': None,
        'scope': 'I require the ordinary ten-second guest default.'}
    command(['make', '-f', 'Makefile.gnu', 'userguide-check'], 'original-userguide', 1800,
            env=env)
    record['original_userguide_passed'] = True


def main():
    if sys.argv[1:] not in ([], ['--native-debug-only'], ['--clang-2118-only'],
                           ['--clang-2118-complete'], ['--response-only'],
                           ['--fixed-as246-complete'], ['--fixed-clang-complete']):
        raise ValueError('I require an explicit supported diagnostic mode.')
    native_debug_only = sys.argv[1:] == ['--native-debug-only']
    clang_only = sys.argv[1:] == ['--clang-2118-only']
    fixed_clang_complete = sys.argv[1:] == ['--fixed-clang-complete']
    clang_complete = sys.argv[1:] == ['--clang-2118-complete'] or fixed_clang_complete
    fixed_complete = sys.argv[1:] == ['--fixed-as246-complete'] or fixed_clang_complete
    full_corpus = clang_complete or fixed_complete
    clang_candidate = clang_only or full_corpus
    response_only = sys.argv[1:] == ['--response-only']
    OUT.mkdir(parents=True, exist_ok=False)
    previous = json.loads((capture.OUT / 'observation.json').read_text())
    if not previous['assembler_version'].splitlines()[0].endswith(') 2.46'):
        raise RuntimeError('I require the actual GNU assembler 2.46 host.')
    source = ROOT / 'src/module_builder.c'
    test = ROOT / 'tests/test_source_snapshots.py'
    original = {path: path.read_bytes() for path in (source, test)}
    record = {'source_pin': previous['commit'], 'product_admission_changed': False,
              'native_debug_only': native_debug_only, 'clang_2118_only': clang_only,
              'clang_2118_complete': clang_complete,
              'fixed_as246_complete': fixed_complete and not fixed_clang_complete,
              'fixed_clang_complete': fixed_clang_complete,
              'response_only': response_only,
              'limits': ['I qualify a private checkout delta. Promotion and complete '
                         'source/snippet/platform acceptance remain separate. '
                         'Original unittest temporary products retain their original cleanup.']}
    try:
        text = original[source].decode()
        token = 'static const char *versions[] = {" 2.40", " 2.42"};'
        if text.count(token) != 1:
            raise RuntimeError('I refuse an unexpected production version table.')
        source.write_text(text if full_corpus else
                          text.replace(token, token.replace('" 2.42"', '" 2.42", " 2.46"')))
        text = original[test].decode()
        old = '"GNU assembler (GNU Binutils for Ubuntu) 2.42\\nCopyright text\\n"]'
        new = old[:-1] + ',\n                    "GNU assembler (GNU Binutils for Ubuntu) 2.46\\nCopyright text\\n"]'
        if text.count(old) != 1:
            raise RuntimeError('I refuse an unexpected version-control fixture.')
        text = text.replace(old, new)
        old = 'rejected = ["GNU assembler (GNU Binutils) 2.41\\n",'
        new = ('rejected = ["GNU assembler (GNU Binutils) 2.46.1\\n", '
               '"GNU assembler (GNU Binutils) 2.460\\n",\n'
               '                    "GNU assembler (GNU Binutils) 2.47\\n", '
               '"GNU assembler (GNU Binutils) 2.46",\n'
               '                    "GNU ld (GNU Binutils) 2.46\\n", '
               '"unrelated banner\\nGNU assembler (GNU Binutils) 2.46\\n",\n'
               '                    "GNU assembler (GNU Binutils) 2.41\\n",')
        if text.count(old) != 1:
            raise RuntimeError('I refuse an unexpected refusal-control fixture.')
        test.write_text(original[test].decode() if full_corpus else text.replace(old, new))
        if fixed_complete:
            revised_source, revised_test, record['fixed_backend'] = fixed_assembler_candidate(
                source.read_text(), test.read_text(), previous)
            source.write_text(revised_source)
            test.write_text(revised_test)
        if clang_candidate:
            clang = shutil.which('clang')
            if not clang:
                raise RuntimeError('I require the actual selected Ubuntu Clang 21.1.8 driver.')
            command([clang, '--version'], 'selected-clang-version', 30)
            version = (OUT / 'selected-clang-version.stdout').read_text()
            if not version.startswith('Ubuntu clang version 21.1.8 (6ubuntu1)\n'):
                raise RuntimeError('I refuse an unqualified selected Clang identity.')
            record['selected_clang'] = {'path': str(Path(clang).resolve()),
                'sha256': hashlib.sha256(Path(clang).read_bytes()).hexdigest(),
                'version': version}
            if clang_complete and Path(shutil.which('cc')).resolve() != Path(clang).resolve():
                raise RuntimeError('I require cc and clang to select the same actual qualification driver.')
            qualified_backend = os.environ.get('NANO_QUALIFIED_AS242_DIRECTORY')
            if qualified_backend:
                expected = (Path(qualified_backend) / 'bin/as').resolve(strict=True)
                command([clang, '-print-prog-name=as'], 'selected-assembler', 30)
                selected = Path((OUT / 'selected-assembler.stdout').read_text().strip()).resolve(strict=True)
                if selected != expected:
                    raise RuntimeError('I require the actual Clang query to select my qualified backend.')
                command([str(selected), '--version'], 'selected-assembler-version', 30)
                version = (OUT / 'selected-assembler-version.stdout').read_text()
                command([str(ROOT / 'obj/test_module_generation_probe'), 'assembler-version', version],
                        'qualified-assembler-admission', 30)
                backend_record = json.loads((Path(qualified_backend) / 'build.json').read_text())
                actual_hash = hashlib.sha256(selected.read_bytes()).hexdigest()
                if (backend_record.get('assembler_sha256') != actual_hash or
                        backend_record.get('source_sha256') !=
                        'f6e4d41fd5fc778b06b7891457b3620da5ecea1006c6a4a41ae998109f85a800'):
                    raise RuntimeError('I require exact qualified backend source and image identities.')
                record['qualified_backend'] = {'path': str(selected), 'sha256': actual_hash,
                    'version': version, 'source_sha256': backend_record['source_sha256'],
                    'selection': 'private COMPILER_PATH; original CI defaults unchanged'}
            text = source.read_text()
            old = '? "Ubuntu clang version 18.1.3 (1ubuntu1)" : NULL;'
            new = ('? "Ubuntu clang version 18.1.3 (1ubuntu1)" : '
                   '!strncmp(report, "Ubuntu clang version 21.1.8 (6ubuntu1)\\n", 39)\n'
                   '        ? "Ubuntu clang version 21.1.8 (6ubuntu1)" : NULL;')
            if text.count(old) != 1:
                raise RuntimeError('I refuse an unexpected Clang report table.')
            source.write_text(text.replace(old, new))
            text = test.read_text()
            old = '        accepted_reports.extend((apple15 + command, ubuntu18 + command))'
            new = ('        ubuntu21 = debian.replace("Debian clang version 14.0.6", '
                   '"Ubuntu clang version 21.1.8 (6ubuntu1)")\n'
                   '        accepted_reports.extend((apple15 + command, ubuntu18 + command, ubuntu21 + command))')
            if text.count(old) != 1:
                raise RuntimeError('I refuse an unexpected Clang report control.')
            text = text.replace(old, new)
            old = '        reports = accepted_reports + [apple15.replace'
            new = ('        reports = accepted_reports + [ubuntu21.replace("21.1.8", "21.1.80") + command,\n'
                   '                   ubuntu21.replace("(6ubuntu1)", "(6ubuntu2)") + command,\n'
                   '                   ubuntu21 + command + command,\n'
                   '                   apple15.replace')
            if text.count(old) != 1:
                raise RuntimeError('I refuse an unexpected Clang refusal control.')
            test.write_text(text.replace(old, new))
        patch = subprocess.check_output(['git', 'diff', '--', str(source), str(test)], cwd=ROOT)
        (OUT / 'candidate.patch').write_bytes(patch)
        record['candidate_patch_sha256'] = hashlib.sha256(patch).hexdigest()
        if fixed_clang_complete:
            from scripts.ci_fixed_clang_inputs import require_candidate, native_closure
            prior = Path(os.environ['NANO_FIXED_CLANG_PRIOR']).resolve(strict=True)
            record['fixed_clang_prior_candidate'] = require_candidate(prior, source, test, patch)
        for path in original:
            shutil.copy2(path, OUT / path.name)
        command(['make', '-f', 'Makefile.gnu', '-j2',
                 'obj/test_module_generation_probe', 'bin/nano_as_capture.so'], 'build', 600)
        for name in ('obj/test_module_generation_probe', 'bin/nano_as_capture.so'):
            shutil.copy2(ROOT / name, OUT / Path(name).name)
        if fixed_clang_complete:
            record['native_closure_before'] = native_closure(record, ROOT, OUT, 'before')
        runner = '''import json,sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_source_snapshots')
discovered=suite.countTestCases()
result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(discovered=discovered,ran=result.testsRun,errors=len(result.errors),
 failures=len(result.failures),skipped=[(case.id(),reason) for case,reason in result.skipped])
open(sys.argv[1],'w').write(json.dumps(report,indent=2)+'\\n')
sys.exit(0 if result.wasSuccessful() and discovered==128 and result.testsRun==128 else 1)
'''
        if native_debug_only:
            methods = ['test_assembler_instruction_and_location_provenance',
                       'test_equals_path_native_debug_identity',
                       'test_integrated_instruction_and_location_provenance',
                       'test_integrated_native_unit_debug_identity']
            selection = ['tests.test_source_snapshots.SourceSnapshots.' + name for name in methods]
            runner = runner.replace("unittest.defaultTestLoader.loadTestsFromName('tests.test_source_snapshots')",
                                    'unittest.defaultTestLoader.loadTestsFromNames(' + repr(selection) + ')')
            runner = runner.replace('discovered==128 and result.testsRun==128',
                                    'discovered==4 and result.testsRun==4')
        if clang_only:
            methods = ['test_selected_clang_assembler_report_boundaries',
                       'test_integrated_native_unit_post_capture_reads',
                       'test_integrated_native_unit_failure_recovery',
                       'test_integrated_native_unit_debug_identity',
                       'test_integrated_units_preserve_c_assembler_search',
                       'test_integrated_instruction_and_location_provenance',
                       'test_equals_path_post_capture_reads',
                       'test_selected_tool_fifo_failure_recovery']
            selection = ['tests.test_source_snapshots.SourceSnapshots.' + name for name in methods]
            runner = runner.replace("unittest.defaultTestLoader.loadTestsFromName('tests.test_source_snapshots')",
                                    'unittest.defaultTestLoader.loadTestsFromNames(' + repr(selection) + ')')
            runner = runner.replace('discovered==128 and result.testsRun==128',
                                    'discovered==8 and result.testsRun==8')
        if response_only:
            methods = ['test_response_file_restored_arguments',
                       'test_link_driver_response_arguments_are_retained',
                       'test_large_response_arguments_are_retained',
                       'test_response_words_match_the_real_compiler',
                       'test_response_transport_outlives_build_info_and_rejects_changed_sidecars',
                       'test_response_rebuild_errors_and_recovery',
                       'test_response_unsupported_fragments_keep_the_original_path',
                       'test_response_driver_mode_declines_the_whole_argument_capture']
            selection = ['tests.test_source_snapshots.SourceSnapshots.' + name for name in methods]
            selection.append('tests.test_response_driver_grammar.ResponseDriverGrammar.'
                             'test_response_quoted_backslashes_follow_selected_driver')
            runner = runner.replace("unittest.defaultTestLoader.loadTestsFromName('tests.test_source_snapshots')",
                                    'unittest.defaultTestLoader.loadTestsFromNames(' + repr(selection) + ')')
            runner = runner.replace('discovered==128 and result.testsRun==128',
                                    'discovered==9 and result.testsRun==9')
        command([sys.executable, '-c', runner, str(OUT / 'tests.json')], 'original-tests', 1500)
        report = json.loads((OUT / 'tests.json').read_text())
        if full_corpus:
            # I preserve the original 128-method allocation and budget the
            # additional response matrix separately; neither corpus is reduced.
            response_runner = runner.replace('tests.test_source_snapshots',
                                             'tests.test_response_driver_grammar')
            response_runner = response_runner.replace('discovered==128 and result.testsRun==128',
                                                       'discovered==1 and result.testsRun==1')
            command([sys.executable, '-c', response_runner, str(OUT / 'response-tests.json')],
                    'response-matrix', 600)
            response_report = json.loads((OUT / 'response-tests.json').read_text())
            record['complete_corpus'] = {'source': report, 'response': response_report,
                'separate_invocations': True, 'total_methods': report['ran'] + response_report['ran']}
            report = dict(report, skipped=report['skipped'] + response_report['skipped'])
        if (clang_only or native_debug_only or response_only) and report['skipped']:
            raise RuntimeError('I require every selected native diagnostic to execute.')
        if full_corpus:
            required = ('test_selected_clang_assembler_report_boundaries',
                        'test_integrated_native_unit_post_capture_reads',
                        'test_integrated_native_unit_failure_recovery',
                        'test_integrated_native_unit_debug_identity',
                        'test_integrated_units_preserve_c_assembler_search',
                        'test_integrated_instruction_and_location_provenance',
                        'test_equals_path_post_capture_reads',
                        'test_selected_tool_fifo_failure_recovery',
                        'test_response_quoted_backslashes_follow_selected_driver')
            if any(case.endswith('.' + name) for case, _ in report['skipped'] for name in required):
                raise RuntimeError('I require all selected Clang/response controls to execute.')
            if fixed_clang_complete:
                external = ('test_clang_external_assembler_cache_restored_inputs',
                    'test_clang_external_assembler_requires_read_capture',
                    'test_external_assembler_cache_nested_changes_and_recovery',
                    'test_linux_clang_external_macros_replay_captured_reads')
                if any(case.endswith('.' + name) for case, _ in report['skipped'] for name in external):
                    raise RuntimeError('I require the original external Clang/r1 replay controls to execute.')
        if response_only:
            record['response_grammar_observation_passed'] = True
            record['private_candidate_qualification_passed'] = False
            return
        if clang_candidate and (not fixed_complete or fixed_clang_complete):
            comparisons = []
            for layout in ('source', 'cache', 'all'):
                for macro, explicit in ((False, False), (True, False), (True, True)):
                    label = 'clang-equals-' + layout + '-' + str(int(macro)) + str(int(explicit))
                    argv = [sys.executable, '-m', 'tests.characterize_assembler_debug', clang,
                            '--integrated', '--module-alias', '--require-physical-debug',
                            '--equals-paths', layout]
                    if macro:
                        argv += ['--macro-read', '--nested-read', '--instruction-macro']
                    if explicit:
                        argv += ['--explicit-locations']
                    command(argv, label, 180)
                    observed = json.loads((OUT / (label + '.stdout')).read_text())
                    if len(observed['cases']) != 4 or any(case['published_unit_aliases']
                            for case in observed['cases']):
                        raise RuntimeError('I require four native debug cases without private unit aliases.')
                    comparisons.append({'label': label, 'cases': len(observed['cases'])})
            record['clang_native_equals_comparisons'] = comparisons
            record['clang_2118_observation_passed'] = True
            if clang_complete:
                capture.OUT = OUT / 'cold-compiler-support'
                capture.main()
                cold = json.loads((capture.OUT / 'observation.json').read_text())
                if cold.get('returncode') != 0 or cold['compiler'] != record['selected_clang']['path']:
                    raise RuntimeError('I require actual selected-Clang cold compiler support publication.')
                record['clang_2118_complete_observation_passed'] = True
                record['gnu_assembler_admission_changed'] = False
            if fixed_clang_complete:
                original_userguide(record)
                record['ordinary_userguide_environment']['scope'] = (
                    'I qualify the explicit Clang/r1 pair at the ten-second default; '
                    'the original GCC guide result remains separate.')
                record['native_closure_after'] = native_closure(record, ROOT, OUT, 'after')
                if record['native_closure_before'] != record['native_closure_after']:
                    raise RuntimeError('I refuse changed native images or dependencies during qualification.')
                record['fixed_clang_profile_qualification_passed'] = True
            record['private_candidate_qualification_passed'] = False
            return
        if native_debug_only:
            record['native_debug_observation_passed'] = True
            record['private_candidate_qualification_passed'] = False
            return
        report = json.loads((OUT / 'tests.json').read_text())
        required = ('test_gcc_macro_argument_replays_captured_reads',
                    'test_gcc_missing_capture_helper_preserves_generation_and_recovers',
                    'test_gcc_replay_cleanup_failure_and_tool_selection',
                    'test_gcc_retained_object_reproducibility_and_external_inputs',
                    'test_gcc_assembler_restoration_uses_captured_inputs',
                    'test_gcc_incomplete_assembler_capture_refuses_cold_output',
                    'test_gcc_validation_cleanup_and_cold_failure',
                    'test_ordinary_capture_failure_refuses_live_cold_output',
                    'test_admitted_capture_phase_failure_and_recovery',
                    'test_supported_assembler_version_line')
        if any(case.endswith('.' + name) for case, _ in report['skipped'] for name in required):
            raise RuntimeError('I require the original GNU replay controls to execute.')
        capture.OUT = OUT / 'cold-compiler-support'
        capture.main()
        cold = json.loads((capture.OUT / 'observation.json').read_text())
        if cold.get('returncode') != 0 or cold['supported_version_decision'] != 0:
            raise RuntimeError('I require actual cold compiler support capture and publication.')
        if fixed_complete:
            original_userguide(record)
            record['fixed_identified_profile_qualification_passed'] = True
            record['generic_gnu_246_admission_changed'] = False
        record['private_candidate_qualification_passed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        matrix = ROOT / '.test_output/response-driver-grammar-matrix.json'
        if (response_only or full_corpus) and matrix.is_file():
            shutil.copy2(matrix, OUT / matrix.name)
        for path, raw in original.items():
            path.write_bytes(raw)
        record['checkout_sources_restored'] = all(path.read_bytes() == raw for path, raw in original.items())
        (OUT / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
