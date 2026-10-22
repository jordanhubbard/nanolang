#!/usr/bin/env python3
"""I select the hosted Darwin capture toolchain qualified by actual replay."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

DEVELOPER = Path('/Applications/Xcode_27.2_beta_2.app/Contents/Developer')
SDK = Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk')
SDK_VERSION = '27.0'  # Exact -target-sdk-version in the retained successful reports.
PINS = {
    'clang': 'cb5fd7fc6cdf43f60a06a6038e49958d6364c73c6a5b05a507d45dbd36f2eef6',
    'as': '06770364632b454317238e79d670c094e27301c7322c3a488323bd42e4d0bcbc',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrapper_source(compiler):
    # A symlink named cc changes diagnostic spelling; exec preserves clang's
    # actual basename without modifying any argument or backend choice.
    return ('#!'+sys.executable+'\nimport os,sys\ncompiler='+repr(str(compiler))+
            '\nos.execv(compiler, [compiler, *sys.argv[1:]])\n')


def select_sdk(qualified_sdk, environment):
    # A generic --sdk macosx lookup follows DEVELOPER_DIR and can select a
    # different SDK from the one whose capture/replay was qualified.
    if not qualified_sdk.is_dir():
        raise ValueError('I require the exact qualified macOS SDK directory.')
    sdk = subprocess.check_output(['xcrun', '--sdk', str(qualified_sdk), '--show-sdk-path'],
                                  env=environment, text=True, timeout=30).strip()
    if sdk != str(qualified_sdk):
        raise ValueError('I require the exact qualified macOS SDK path: '+repr(sdk))
    # Xcode's SDK catalog need not contain the separately installed CLT SDK.
    # Read the version from the exact selected SDK, not that other catalog.
    settings = qualified_sdk/'SDKSettings.json'
    data = json.loads(settings.read_text())
    if data.get('Version') != SDK_VERSION:
        raise ValueError('I require the qualified macOS SDK version.')
    return {'path': sdk, 'resolved': str(qualified_sdk.resolve()),
            'version': data['Version'], 'settings_sha256': digest(settings)}


def select(output, developer=DEVELOPER, pins=PINS, qualified_sdk=SDK):
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, DEVELOPER_DIR=str(developer))
    tools = developer/'Toolchains/XcodeDefault.xctoolchain/usr/bin'
    identities = {}
    for name, expected in pins.items():
        path = (tools/name).resolve(strict=True)
        actual = digest(path)
        if actual != expected:
            raise ValueError('I require the qualified capture '+name+' identity; requalify changed tools.')
        identities[name] = {'path': str(path), 'sha256': actual}
    sdk_record = select_sdk(qualified_sdk, environment)
    sdk, sdk_version = sdk_record['path'], sdk_record['version']
    environment['SDKROOT'] = sdk
    compiler = Path(identities['clang']['path'])
    version = subprocess.check_output([compiler, '--version'], env=environment,
                                     text=True, timeout=30)
    if version.splitlines()[0] != 'Apple clang version 21.0.0 (clang-2100.3.34.2)':
        raise ValueError('I require the qualified capture compiler report.')
    selected = output/'selected-tools'
    selected.mkdir()
    wrappers = {}
    for name in ('cc', 'clang'):
        path = selected/name
        path.write_text(wrapper_source(compiler)); path.chmod(0o700)
        wrappers[name] = {'path': str(path), 'sha256': digest(path)}
    record = {'developer_dir': str(developer), 'sdk_root': sdk, 'sdk_version': sdk_version,
              'sdk_resolved': sdk_record['resolved'], 'sdk_settings_sha256': sdk_record['settings_sha256'],
              'tools': identities, 'wrappers': wrappers, 'version': version}
    (output/'capture-toolchain.json').write_text(json.dumps(record, indent=2)+'\n')
    with open(os.environ['GITHUB_ENV'], 'a') as stream:
        stream.write('DEVELOPER_DIR='+str(developer)+'\nSDKROOT='+sdk+'\n')
    with open(os.environ['GITHUB_PATH'], 'a') as stream:
        stream.write(str(selected)+'\n')
    return record


if __name__ == '__main__':
    select(Path('.test_output/retirement').resolve())
