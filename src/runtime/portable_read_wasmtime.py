"""I provide a private Wasmtime43 read import, not NanoISA execution authority.

My embedding application and Python hooks are trusted. I own one Store/Instance,
serialize calls, and never expose a guest pointer as a native filesystem path.
"""
from importlib.metadata import version
import os
import wasmtime as wt

NPR_OK, NPR_DENIED, NPR_LIMIT, NPR_MEMORY, NPR_INVALID = range(5)
_PATH_LIMIT, _TEXT_LIMIT = 4096, 1048576
_MODULE_LIMIT, _TYPE_LIMIT, _FUNCTION_LIMIT = 16 * 1024 * 1024, 4096, 65536
_TYPE_ARITY_LIMIT, _SECTION_LIMIT = 64, 64
_NAMESPACE, _NAME = "nanolang_host_v1", "read_text"


def _refuse():
    raise ValueError("I refuse this private read-text module envelope")


def _bytes_copy(value, limit):
    if not isinstance(value, (bytes, bytearray)) or len(value) > limit:
        _refuse()
    return bytes(value)


class _Cursor:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, count):
        if count < 0 or count > len(self.data) - self.pos:
            _refuse()
        result = self.data[self.pos:self.pos + count]
        self.pos += count
        return result

    def u8(self):
        return self.take(1)[0]

    def u32(self):
        result = 0
        for i in range(5):
            b = self.u8()
            if i == 4 and b & 0xf0:
                _refuse()
            result |= (b & 0x7f) << (7 * i)
            if not b & 0x80:
                return result
        _refuse()

    def name(self):
        count = self.u32()
        if count > _PATH_LIMIT:
            _refuse()
        return self.take(count).decode("utf-8", errors="strict")

    def done(self):
        if self.pos != len(self.data):
            _refuse()


def _value_types(cursor):
    count = cursor.u32()
    if count > _TYPE_ARITY_LIMIT:
        _refuse()
    result = []
    for _ in range(count):
        tag = cursor.u8()
        if tag not in (0x7f, 0x7e, 0x7d, 0x7c):
            _refuse()
        result.append(tag)
    return result


def _module_envelope(data):
    cursor = _Cursor(data)
    if cursor.take(8) != b"\0asm\1\0\0\0":
        _refuse()
    sections, functions = 0, 0
    types = import_type = exports = None
    memory, seen = False, set()
    while cursor.pos < len(data):
        sections += 1
        if sections > _SECTION_LIMIT:
            _refuse()
        kind = cursor.u8()
        part = _Cursor(cursor.take(cursor.u32()))
        if kind > 12 or kind == 8 or (kind and kind in seen):
            _refuse()
        if kind:
            seen.add(kind)
        if kind == 0:
            part.name()
            part.pos = len(part.data)
        elif kind == 1:
            count = part.u32()
            if count > _TYPE_LIMIT:
                _refuse()
            types = []
            for _ in range(count):
                if part.u8() != 0x60:
                    _refuse()
                types.append((_value_types(part), _value_types(part)))
        elif kind == 2:
            if (part.u32() != 1 or part.name() != _NAMESPACE or
                    part.name() != _NAME or part.u8() != 0):
                _refuse()
            import_type = part.u32()
        elif kind == 3:
            functions = part.u32()
            if functions >= _FUNCTION_LIMIT:
                _refuse()
            for _ in range(functions):
                type_index = part.u32()
                if types is None or type_index >= len(types):
                    _refuse()
        elif kind == 5:
            if (part.u32() != 1 or part.u32() != 1 or
                    part.u32() != 32 or part.u32() != 1024):
                _refuse()
            memory = True
        elif kind == 7:
            count = part.u32()
            if count > _FUNCTION_LIMIT:
                _refuse()
            exports = {}
            for _ in range(count):
                name, export_kind, index = part.name(), part.u8(), part.u32()
                if name in exports or export_kind > 3:
                    _refuse()
                exports[name] = (export_kind, index)
        else:
            # I leave other section contents/order to the engine's validator.
            part.pos = len(part.data)
        part.done()
    if (types is None or import_type is None or import_type >= len(types) or
            not memory or exports is None):
        _refuse()
    if types[import_type] != ([0x7f] * 5, [0x7f]):
        _refuse()
    memories = 0
    for name, (kind, index) in exports.items():
        if kind == 2:
            if name != "memory" or index != 0:
                _refuse()
            memories += 1
        if kind == 0 and index >= functions + 1:
            _refuse()
    if memories != 1:
        _refuse()
    return exports


def _spans(size, path, length, destination, capacity, output):
    if length > _PATH_LIMIT or capacity > _TEXT_LIMIT:
        return False
    ranges = ((path, length), (destination, capacity), (output, 4))
    for offset, count in ranges:
        if offset > size or count > size - offset:
            return False
    for i in range(3):
        for j in range(i + 1, 3):
            a, an = ranges[i]
            b, bn = ranges[j]
            if an and bn and a < b + bn and b < a + an:
                return False
    return True


class _ReadTextInstance:
    def __init__(self, module_bytes, paths):
        self._engine = self._store = self._module = self._linker = None
        self._instance = self._memory = None
        self._allowlist = ()
        self._ready = self._terminal = self._export_active = self._callback_active = False
        self._last = dict(status=NPR_OK, opened=False, closeAttempted=False,
                          closeError=False, bytesRead=0)
        data = _bytes_copy(module_bytes, _MODULE_LIMIT)
        self._envelope = _module_envelope(data)
        if not isinstance(paths, (list, tuple)) or len(paths) > 64:
            _refuse()
        copied = []
        for path in paths:
            row = _bytes_copy(path, _PATH_LIMIT)
            if not row or b"\0" in row:
                _refuse()
            copied.append(row)
        self._allowlist = tuple(copied)
        if version("wasmtime") != "43.0.0":
            raise ValueError("I require my pinned Wasmtime43.0.0 binding")
        try:
            config = wt.Config()
            try:
                config.wasm_threads = False
                config.wasm_multi_memory = False
                config.wasm_memory64 = False
                self._engine = wt.Engine(config)
            finally:
                config.close()
            self._module = wt.Module(self._engine, data)
            imports = self._module.imports
            if (len(imports) != 1 or imports[0].module != _NAMESPACE or
                    imports[0].name != _NAME or not isinstance(imports[0].type, wt.FuncType)):
                _refuse()
            function_type = imports[0].type
            if (function_type.params != [wt.ValType.i32()] * 5 or
                    function_type.results != [wt.ValType.i32()]):
                _refuse()
            actual = self._module.exports
            if len(actual) != len(self._envelope):
                _refuse()
            kinds = (wt.FuncType, wt.TableType, wt.MemoryType, wt.GlobalType)
            for export in actual:
                if (export.name not in self._envelope or
                        not isinstance(export.type, kinds[self._envelope[export.name][0]])):
                    _refuse()
            self._store = wt.Store(self._engine)
            self._linker = wt.Linker(self._engine)
            self._linker.define_func(_NAMESPACE, _NAME, function_type,
                                     self._callback, access_caller=True)
            self._instance = self._linker.instantiate(self._store, self._module)
            self._memory = self._instance.exports(self._store)["memory"]
            if not isinstance(self._memory, wt.Memory):
                _refuse()
            memory_type = self._memory.type(self._store)
            if (memory_type.is_64 or memory_type.is_shared or
                    self._memory.data_len(self._store) != 32 * 65536):
                _refuse()
            self._ready = True
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                primary.add_note("I also encountered teardown failure: " + type(cleanup).__name__)
            raise

    def _callback(self, caller, path_offset, path_length, destination_offset, capacity, length_offset):
        if not self._ready or self._terminal or self._callback_active or not self._export_active:
            return NPR_INVALID
        self._callback_active = True
        status, count = NPR_OK, 0
        fd, opened, close_attempted, close_error = None, False, False, False
        try:
            p, n, d, cap, out = (x & 0xffffffff for x in
                                (path_offset, path_length, destination_offset, capacity, length_offset))
            memory = caller.get("memory")
            if not isinstance(memory, wt.Memory):
                status = NPR_INVALID
                return status
            size = memory.data_len(caller)
            if not _spans(size, p, n, d, cap, out):
                status = NPR_INVALID
                return status
            path = bytes(memory.read(caller, p, p + n))
            if not path or b"\0" in path or path not in self._allowlist:
                status = NPR_DENIED
                return status
            data, cell = bytearray(cap + 1), bytearray(4)
            empty = False
            try:
                fd = os.open(path, os.O_RDONLY)
                opened = True
                while count <= cap:
                    got = os.readv(fd, [memoryview(data)[count:cap + 1]])
                    if not isinstance(got, int) or got < 0 or got > cap + 1 - count:
                        raise RuntimeError("I received an invalid host read count")
                    count += got
                    if count > cap:
                        status = NPR_LIMIT
                        break
                    if not got:
                        break
            except OSError:
                empty = True
            except MemoryError:
                status = NPR_MEMORY
            finally:
                if fd is not None:
                    close_attempted = True
                    try:
                        os.close(fd)
                    except OSError:
                        close_error = True
                        empty = True
                    except MemoryError:
                        close_error = True
                        if status == NPR_OK:
                            status = NPR_MEMORY
            if status != NPR_OK:
                return status
            length = 0 if empty else count
            if data.find(b"\0", 0, length) >= 0:
                length = 0
            if memory.data_len(caller) != size or not _spans(size, p, n, d, cap, out):
                status = NPR_INVALID
                return status
            payload = data[:length]
            for i in range(4):
                cell[i] = (length >> (8 * i)) & 0xff
            if length and memory.write(caller, payload, d) != length:
                raise RuntimeError("I did not publish the exact guest payload")
            # Shipped Memory.write allocates its ctypes views before memmove.
            # A publication MemoryError therefore leaves the length cell unchanged.
            if memory.write(caller, cell, out) != 4:
                raise RuntimeError("I did not publish the exact guest length")
            return NPR_OK
        except MemoryError:
            if status == NPR_OK:
                status = NPR_MEMORY
            return status
        except BaseException:
            if status == NPR_OK:
                status = NPR_INVALID
            self._terminal, self._ready = True, False
            raise
        finally:
            self._callback_active = False
            self._last["status"] = status
            self._last["opened"] = opened
            self._last["closeAttempted"] = close_attempted
            self._last["closeError"] = close_error
            self._last["bytesRead"] = count

    def call(self, name, *args):
        if (not self._ready or self._terminal or self._export_active or self._callback_active or
                not isinstance(name, str) or name not in self._envelope or self._envelope[name][0] != 0):
            raise ValueError("I refuse this private guest call")
        self._export_active = True
        try:
            return self._instance.exports(self._store)[name](self._store, *args)
        except BaseException:
            self._terminal, self._ready = True, False
            raise
        finally:
            self._export_active = False

    def close(self):
        if self._export_active or self._callback_active:
            return NPR_INVALID
        self._ready, self._terminal = False, True
        self._memory = self._instance = None
        self._allowlist = ()
        # Linker callback storage releases its bound-method reference before Store.
        failure = None
        for name in ("_linker", "_store", "_module", "_engine"):
            owner = getattr(self, name)
            setattr(self, name, None)
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure
        return NPR_OK

    def report(self):
        return dict(self._last, ready=self._ready, terminal=self._terminal,
                    exportActive=self._export_active, callbackActive=self._callback_active)


def create_read_text_instance(module_bytes, paths):
    """I require explicit close after all calls, including terminal traps.

    I copy bytes before engine creation. Invalid envelopes/configuration raise
    before file effects; calls return guest results unchanged or raise terminally.
    My underscore members are private conventions, not a Python sandbox.
    """
    return _ReadTextInstance(module_bytes, paths)
