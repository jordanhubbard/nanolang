"""I run real Wasmtime calls and separately labeled trusted-hook fault probes."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import wasmtime as wt

if not __debug__:
    raise RuntimeError("I require active fixture assertions")
library, wasm_path, vectors_path = sys.argv[1:]
specification = importlib.util.spec_from_file_location("private_read_host", library)
host = importlib.util.module_from_spec(specification)
specification.loader.exec_module(host)
wasm = Path(wasm_path).read_bytes()
spec = json.loads(Path(vectors_path).read_text())
checks = real_vectors = modeled = 0


def eq(actual, expected):
    global checks
    checks += 1
    assert actual == expected, (actual, expected)


def bad(action, kind=Exception):
    global checks
    checks += 1
    try:
        action()
    except kind:
        return
    raise AssertionError("I expected this call to refuse")


def make(paths):
    return host.create_read_text_instance(wasm, paths)


def start(api, path, fill=1):
    for index, byte in enumerate(path):
        api.call("put", index, byte)
    api.call("init", len(path), fill)


def done(api):
    eq(api.call("roots_ok"), 1)
    eq(api.call("finish"), 1)
    eq(api.close(), 0)
    eq(api.close(), 0)
    bad(lambda: api.call("pages"))


small = next(v for v in spec["vectors"] if v["name"] == "small")
big = next(v for v in spec["vectors"] if v["name"] == "exact")
for vector in spec["vectors"]:
    path = bytes.fromhex(vector["pathHex"])
    api = make([path])
    start(api, path)
    before = api.call("pages")
    eq(api.call("read", 0, 0), vector["status"] << 8)
    real_vectors += 1
    if not vector["status"]:
        eq(api.call("length", 0), vector["length"])
        eq(api.call("hash", 0) & 0xffffffff, vector["hash"])
    if vector["name"] == "exact":
        eq(api.call("pages") > before, True)
    eq(api.report()["closeAttempted"], vector["name"] != "missing")
    done(api)

original = bytes.fromhex(small["pathHex"])
copy = bytearray(original)
rows = [("deny" + str(i)).encode() for i in range(63)] + [copy]
api = make(rows)
copy[:] = b"!" * len(copy)
rows[:] = [b"changed"] * 64
start(api, original)
eq(api.call("read", 0, 0), 0)
second = b"second"
Path(wasm_path + ".python-copy-second.bin").write_bytes(second)
try:
    with open(original, "wb") as stream:
        stream.write(second)
    eq(api.call("read", 1, 0), 0)
finally:
    with open(original, "wb") as stream:
        stream.write(b"copied")
second_hash = 2166136261
for byte in second:
    second_hash = ((second_hash ^ byte) * 16777619) & 0xffffffff
eq(api.call("hash", 0) & 0xffffffff, small["hash"])
eq(api.call("hash", 1) & 0xffffffff, second_hash)
other = make([])
start(other, original)
eq(other.call("read", 0, 0), 256)
done(other)
done(api)
bad(lambda: make([original] * 65))
bad(lambda: make([b"a\0b"]))
bad(lambda: make([b""]))
for mode, status in ((1, 6), (2, 6), (3, 5), (4, 6), (5, 1)):
    api = make([original])
    start(api, original)
    eq(api.call("read", 0, mode), status)
    eq(api.report()["opened"], False)
    done(api)
for size in (0, 4096, 4097):
    path = b"q" * size
    api = make([path] if size == 4096 else [])
    start(api, path)
    observed = []
    original_open = os.open

    def counted_open(path, *args):
        observed.append(len(path))
        return original_open(path, *args)

    os.open = counted_open
    try:
        eq(api.call("read", 0, 0), 512 if size == 4097 else 256 if size == 0 else 0)
    finally:
        os.open = original_open
    eq(observed, [4096] if size == 4096 else [])
    done(api)
api = make([original])
start(api, original + b"\0ignored")
eq(api.call("read", 0, 0), 0)
eq(api.call("hash", 0) & 0xffffffff, small["hash"])
done(api)
probe = make([original])
start(probe, original, 8)
if probe.call("testing"):
    probe.call("budget", 100)
    eq(probe.call("read", 0, 0), 0)
    measured = 100 - probe.call("remaining")
    eq(measured, 3)
    probe.call("budget", 0xffffffff)
    done(probe)
    for point in range(measured):
        api = make([original])
        start(api, original, 8)
        api.call("budget", point)
        eq(api.call("read", 0, 0), 3)
        eq(api.call("roots_ok"), 1)
        api.call("budget", 0xffffffff)
        eq(api.call("read", 0, 0), 0)
        done(api)
        modeled += 1
else:
    done(probe)
for exhaust in (False, True):
    path = bytes.fromhex((big if exhaust else small)["pathHex"])
    api = make([path])
    start(api, path)
    pages = api.call("pages")
    eq(api.call("grow", 1024 - pages if exhaust else 1), pages)
    eq(api.call("read", 0, 0), 3 if exhaust else 0)
    done(api)


def raw_setup(api, path):
    base = api.call("raw_base") & 0xffffffff
    for i, byte in enumerate(path):
        api.call("raw_put", i, byte)
    for i in range(4):
        api.call("raw_put", 7001 + i, 0 if i else 91)
    return base, base + 7001, [base, len(path), base + 5000, 16, base + 7001]


def raw_length(api):
    return sum(api.call("raw_get", 7001 + i) << (8 * i) for i in range(4))


api = make([original])
base, out, args = raw_setup(api, original)
size = api.call("pages") * 65536
refusals = [[0xffffffff, 2, args[2], 16, out], [size, 1, args[2], 16, out],
            [base, len(original), base, 16, out], [out, 4, args[2], 16, out],
            [base, len(original), out, 4, out], [base, len(original), args[2], 1048577, out],
            [base, 4097, args[2], 16, out], [base, len(original), args[2], 16, size - 3], [base, len(original), 0xffffffff, 16, out],
            [base, len(original), args[2], 16, 0xffffffff]]
observed = []
original_open = os.open


def counted_open(path, *args):
    observed.append(path)
    return original_open(path, *args)


os.open = counted_open
try:
    for values in refusals:
        eq(api.call("raw_call", *values), 4)
        eq(raw_length(api), 91)
    eq(observed, [])
    eq(api.call("raw_call", *args), 0)
    eq(raw_length(api), small["length"])
    eq(observed, [original])
finally:
    os.open = original_open
eq(api.close(), 0)

for vector in (small, next(v for v in spec["vectors"] if v["name"] == "empty")):
    path = bytes.fromhex(vector["pathHex"])
    api = make([path])
    _, _, values = raw_setup(api, path)
    values[2], values[3] = api.call("pages") * 65536, 0
    eq(api.call("raw_call", *values), 2 if vector["length"] else 0)
    eq(raw_length(api), 91 if vector["length"] else 0)
    eq(api.close(), 0)

for mode in ("progress", "close", "limit-close", "memory-read", "memory-close", "grow", "active", "publication", "alloc0", "alloc1", "alloc2"):
    api = make([original])
    base, out, args = raw_setup(api, original)
    saved_open, saved_read, saved_close = os.open, os.readv, os.close
    saved_memory_read, saved_memory_write = wt.Memory.read, wt.Memory.write
    state = dict(opened=0, closed=0, reads=0, fd=-1, allocations=0, active=0)
    captured = {}

    def capture_read(memory, caller, *args):
        captured.update(memory=memory, caller=caller)
        return saved_memory_read(memory, caller, *args)

    def injected_open(*args):
        state["opened"] += 1
        fd = state["fd"] = saved_open(*args)
        if mode == "grow":
            captured["memory"].grow(captured["caller"], 1)
        if mode == "active":
            eq(api.close(), 4)
            bad(lambda: api.call("grow", 1))
            state["active"] += 1
        return fd

    def injected_read(fd, buffers):
        state["reads"] += 1
        if mode == "memory-read":
            raise MemoryError("I inject read allocation refusal")
        if mode == "progress":
            if state["reads"] > 1:
                raise OSError(5, "I inject an error after real progress")
            return saved_read(fd, [buffers[0][:1]])
        return saved_read(fd, buffers)

    def injected_close(fd):
        state["closed"] += 1
        saved_close(fd)
        if mode == "memory-close":
            raise MemoryError("I inject close allocation refusal")
        if mode in ("close", "limit-close"):
            raise OSError(5, "I inject an error after real close")

    def allocation_point():
        point = state["allocations"]
        state["allocations"] += 1
        if mode.startswith("alloc") and point == int(mode[-1]):
            raise MemoryError("I inject a direct host allocation refusal")

    def injected_bytes(*args):
        allocation_point()
        return bytes(*args)

    def injected_bytearray(*args):
        allocation_point()
        return bytearray(*args)

    def injected_write(memory, caller, value, start=None):
        if mode == "publication" and start == out:
            raise MemoryError("I inject length publication allocation refusal")
        return saved_memory_write(memory, caller, value, start)

    wt.Memory.read, wt.Memory.write = capture_read, injected_write
    os.open, os.readv, os.close = injected_open, injected_read, injected_close
    host.bytes, host.bytearray = injected_bytes, injected_bytearray
    try:
        values = list(args)
        if mode == "limit-close":
            values[3] = 1
        result = api.call("raw_call", *values)
    finally:
        os.open, os.readv, os.close = saved_open, saved_read, saved_close
        wt.Memory.read, wt.Memory.write = saved_memory_read, saved_memory_write
        del host.bytes, host.bytearray
    expected = 2 if mode == "limit-close" else 4 if mode == "grow" else 3 if mode.startswith(("memory", "alloc")) or mode == "publication" else 0
    eq(result, expected)
    eq(raw_length(api), 91 if expected else 0 if mode in ("progress", "close") else small["length"])
    eq(state["opened"], 0 if mode.startswith("alloc") else 1)
    eq(state["closed"], state["opened"])
    if state["fd"] >= 0:
        bad(lambda: os.fstat(state["fd"]))
    if mode == "active":
        eq(state["active"], 1)
    if mode == "progress":
        eq(api.report()["bytesRead"], 1)
    if mode == "limit-close":
        eq(api.report()["closeError"], True)
    if mode == "publication":
        eq(api.call("raw_get", 5000), ord("c"))
    eq(api.call("raw_call", *args), 0)
    eq(raw_length(api), small["length"])
    eq(api.close(), 0)
    modeled += 1


def leb(n):
    data = bytearray()
    while True:
        b, n = n & 127, n >> 7
        data.append(b | (128 if n else 0))
        if not n:
            return bytes(data)


def name(s):
    data = s.encode()
    return leb(len(data)) + data


def section(kind, data):
    return bytes([kind]) + leb(len(data)) + bytes(data)


def module_bytes(parts):
    return b"\0asm\1\0\0\0" + b"".join(parts)


type_part = section(1, [1, 0x60, 5, 127, 127, 127, 127, 127, 1, 127])
import_part = section(2, b"\1" + name("nanolang_host_v1") + name("read_text") + b"\0\0")
memory_part = section(5, b"\1\1\x20" + leb(1024))
export_part = section(7, b"\1" + name("memory") + b"\2\0")
minimal = module_bytes([type_part, import_part, memory_part, export_part])
eq(host.create_read_text_instance(minimal, []).close(), 0)
malformed = [module_bytes(parts) for parts in (
    [type_part, type_part, import_part, memory_part, export_part],
    [import_part, type_part, memory_part, export_part],
    [type_part, import_part, memory_part, export_part, section(8, [0])],
    [type_part, import_part, section(5, b"\1\3\x20" + leb(1024)), export_part],
    [section(1, [1, 0x60, 0, 1, 127]), import_part, memory_part, export_part],
    [type_part, section(2, b"\2" + import_part[3:]), memory_part, export_part],
    [type_part, import_part, section(5, b"\1\1\x1f" + leb(1024)), export_part],
    [type_part, import_part, memory_part],
    [type_part, import_part, section(3, [1, 0]), memory_part, export_part, section(10, [1, 2, 0, 255])])]
malformed.append(minimal + bytes([0, 128]))
malformed.extend([
    module_bytes([type_part, section(2, b"\1" + name("wrong") + name("read_text") + b"\0\0"), memory_part, export_part]),
    module_bytes([type_part, import_part, section(5, b"\1\5\x20" + leb(1024)), export_part]),
    module_bytes([section(1, leb(4097)), import_part, memory_part, export_part]),
    module_bytes([type_part, import_part, section(3, leb(65536)), memory_part, export_part]),
    module_bytes([type_part, import_part, memory_part, export_part] + [section(0, [0])] * 61),
    minimal + bytes([0, 255, 255, 255, 255, 16])])
for i, data in enumerate(malformed):
    Path(wasm_path + f".python-negative-{i}.wasm").write_bytes(data)
observed = []
original_open = os.open
os.open = counted_open
try:
    for i, data in enumerate(malformed):
        bad(lambda data=data: host.create_read_text_instance(data, []), wt.WasmtimeError if i in (1,8) else Exception)
    eq(observed, [])
finally:
    os.open = original_open

# These imports are fixture replacements, not the real reader acceptance above.
for mode in ("status", "unset", "oversize", "nul", "reenter"):
    engine = wt.Engine()
    store = wt.Store(engine)
    module = wt.Module(engine, wasm)
    linker = wt.Linker(engine)
    state = dict(calls=0)
    instance = None

    def callback(caller, p, n, d, cap, out):
        state["calls"] += 1
        memory = caller.get("memory")
        if mode == "reenter":
            eq(instance.exports(caller)["read"](caller, 1, 0), 1024)
            memory.write(caller, bytearray(4), out)
            return 0
        if mode == "status":
            return 99
        if mode == "unset":
            return 0
        memory.write(caller, bytearray((cap + 1 if mode == "oversize" else 1).to_bytes(4, "little")), out)
        if mode == "nul":
            memory.write(caller, bytearray(1), d)
        return 0

    linker.define_func("nanolang_host_v1", "read_text", wt.FuncType([wt.ValType.i32()] * 5, [wt.ValType.i32()]), callback, access_caller=True)
    instance = linker.instantiate(store, module)

    class Raw:
        def call(self, name, *args):
            return instance.exports(store)[name](store, *args)

    raw = Raw()
    start(raw, original)
    eq(raw.call("read", 0, 0), 0 if mode == "reenter" else 1024)
    eq(state["calls"], 1)
    eq(raw.call("roots_ok"), 1)
    eq(raw.call("finish"), 1)
    instance = None
    linker.close()
    store.close()
    module.close()
    engine.close()
    modeled += 1
api = make([])
start(api, original)
bad(lambda: api.call("trap"))
eq(api.report()["terminal"], True)
bad(lambda: api.call("pages"))
eq(api.close(), 0)
print(json.dumps(dict(engine="Wasmtime43", checks=checks, realVectors=real_vectors, modeled=modeled,
                     scope="private direct Wasm adapter; no NanoISA admission")))
