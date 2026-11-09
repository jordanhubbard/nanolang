"""I verify only the three reported public-source checksum occurrences."""
import hashlib
import json
import subprocess

EVIDENCE = '3c3f1edae92125dd44cf392ddc0f160a2868e129'
SOURCE = 'b897da5d404d6b148d2734b3e4673ebcc8d81b3f'
REPORT = 'docs/evidence/collection-aot-20261109/corrected-gcc-source-deadline.json'
PATH = 'src_nano/compiler/token_value_bytes.nano'


def verify():
    lines = subprocess.check_output(['git', 'show', EVIDENCE+':'+REPORT], text=True).splitlines()
    blob = subprocess.check_output(['git', 'show', SOURCE+':'+PATH])
    expected = hashlib.sha256(blob).hexdigest()
    results = []
    for line in (50, 119, 191):
        key, value = lines[line-1].strip().rstrip(',').split(': ', 1)
        key, value = json.loads(key), json.loads(value)
        assert key == PATH or key.endswith('/'+PATH)
        assert value == expected
        results.append({'line': line, 'source_path': PATH, 'full_git_blob_sha256_matches': True})
    return {'check_run': 129026331943, 'incident': 37472714, 'evidence_commit': EVIDENCE,
            'source_commit': SOURCE, 'occurrences': results,
            'scope': 'I verify these three fields as the same public Git source-content checksum. I do not classify unrelated findings, clear a security verdict, exclude a detector or perform a supported disposition.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
