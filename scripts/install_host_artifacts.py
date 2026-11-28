#!/usr/bin/env python3
"""I deploy the exact host-library bytes declared inside a selected native image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile

PREFIX = b'NANOLANG_HOST_V1:'


def section_bytes(image):
    """I read bounded native sections, never ordinary executable strings."""
    def region(offset, size):
        if offset < 0 or size < 0 or offset > len(image) or size > len(image)-offset:
            raise ValueError('I refuse an out-of-bounds native host section.')
        return image[offset:offset+size]
    def number(order, format_, offset):
        return struct.unpack(order+format_, region(offset, struct.calcsize(format_)))[0]
    if image.startswith(b'\x7fELF'):
        if len(image) < 64 or image[4] != 2 or image[5] not in (1, 2):
            raise ValueError('I require a supported 64-bit ELF native image.')
        order = '<' if image[5] == 1 else '>'
        start, stride, count, strings = (number(order, fmt, offset) for fmt, offset in
                                       [('Q', 40), ('H', 58), ('H', 60), ('H', 62)])
        if stride < 64 or not count or strings >= count:
            raise ValueError('I require an ordinary bounded ELF section table.')
        region(start, stride*count)
        entry = start+strings*stride
        names = region(number(order, 'Q', entry+24), number(order, 'Q', entry+32))
        for index in range(count):
            entry = start+index*stride
            name = number(order, 'I', entry)
            if name >= len(names):
                raise ValueError('I refuse an invalid ELF section name.')
            end = names.find(b'\0', name)
            if end < 0:
                raise ValueError('I refuse an unterminated ELF section name.')
            if names[name:end] == b'.nanohost':
                yield region(number(order, 'Q', entry+24), number(order, 'Q', entry+32))
    elif image[:4] in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf'):
        order = '<' if image[:4] == b'\xcf\xfa\xed\xfe' else '>'
        count, size = number(order, 'I', 16), number(order, 'I', 20)
        region(32, size)
        position = 32
        for _ in range(count):
            command, length = number(order, 'I', position), number(order, 'I', position+4)
            if length < 8 or position+length > 32+size:
                raise ValueError('I refuse an invalid Mach-O load command.')
            if command == 0x19:  # I read LC_SEGMENT_64 section declarations.
                if length < 72:
                    raise ValueError('I refuse a truncated Mach-O segment.')
                sections = number(order, 'I', position+64)
                if sections > (length-72)//80:
                    raise ValueError('I refuse a truncated Mach-O section table.')
                for index in range(sections):
                    entry = position+72+80*index
                    if region(entry, 16).split(b'\0')[0] == b'__nanohost':
                        yield region(number(order, 'I', entry+48), number(order, 'Q', entry+40))
            position += length
        if position != 32+size:
            raise ValueError('I refuse inconsistent Mach-O load-command bounds.')
    else:
        raise ValueError('I require a supported ELF or Mach-O native image.')


def declared_artifacts(image):
    records = {}
    for section in section_bytes(image):
        if section and section[-1] != 0:
            raise ValueError('I refuse an unterminated native host-artifact section.')
        for record in section.split(b'\0'):
            if not record:  # I permit native section alignment padding only.
                continue
            if record.startswith(b'NANOLANG_HOST_ABSENT_V1:'):
                raise ValueError('I refuse deployment of an unbound native host artifact.')
            parsed = re.fullmatch(re.escape(PREFIX)+rb'([0-9a-f]{64}):(.*)', record, re.DOTALL)
            if not parsed:
                raise ValueError('I refuse a malformed native host-artifact declaration.')
            digest = parsed[1].decode('ascii')
            raw = parsed[2]
            if b'\x00' in raw or not raw.startswith(b'/') or len(raw) >= 4096:
                raise ValueError('I require a bounded absolute host-artifact source path.')
            records.setdefault(digest, set()).add(os.fsdecode(raw))
    return records


def atomic_write(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.nano-host-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            os.fchmod(stream.fileno(), mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install(binary, prefix, name):
    image = binary.read_bytes()
    declarations = declared_artifacts(image)
    selected = {}
    for digest, paths in declarations.items():
        for path in sorted(paths):
            source = Path(path)
            if not source.is_file():
                raise ValueError(f'I require a regular host-artifact source: {path}')
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'I refuse changed host-artifact bytes: {path}')
            selected[digest] = data
    # I validate the entire selected input set before publishing any library.
    for digest, data in selected.items():
        atomic_write(prefix/'lib/nanolang/hosts'/digest, data, 0o755)
    receipt = dict(version=1, executable_sha256=hashlib.sha256(image).hexdigest(),
                   artifacts=[dict(sha256=digest, bytes=len(selected[digest]), sources=sorted(paths))
                              for digest, paths in sorted(declarations.items())])
    atomic_write(prefix/'share/nanolang/host-closures'/(name+'.json'),
                 (json.dumps(receipt, indent=2)+'\n').encode(), 0o644)
    return receipt


def uninstall(prefix, name):
    directory = prefix/'share/nanolang/host-closures'
    receipt = directory/(name+'.json')
    if not receipt.exists():
        return
    def hashes(path):
        record = json.loads(path.read_text())
        values = {entry['sha256'] for entry in record['artifacts']}
        if record['version'] != 1 or any(not re.fullmatch('[0-9a-f]{64}', value) for value in values):
            raise ValueError('I refuse an invalid installed host-closure receipt.')
        return values
    owned = hashes(receipt)
    shared = set().union(*(hashes(p) for p in directory.glob('*.json') if p != receipt))
    receipt.unlink()
    for digest in owned-shared:
        (prefix/'lib/nanolang/hosts'/digest).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--name', default='nanoc')
    parser.add_argument('--uninstall', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', args.name):
        parser.error('I require a plain installed executable name.')
    try:
        if args.uninstall:
            uninstall(args.prefix, args.name)
        elif args.binary:
            result = install(args.binary, args.prefix, args.name)
            print(f"I install {len(result['artifacts'])} exact native host artifacts.")
        else:
            parser.error('I require --binary or --uninstall.')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'I could not deploy my native host closure: {error}\n')


if __name__ == '__main__':
    main()
