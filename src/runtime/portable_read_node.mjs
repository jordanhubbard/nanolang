/* I provide one private synchronous wasm32 read import, not NanoISA authority.
 * My application/intrinsics/hooks are trusted; I do not contain replacement JS. */
import fs from 'node:fs';

export const NPR_OK = 0, NPR_DENIED = 1, NPR_LIMIT = 2, NPR_MEMORY = 3, NPR_INVALID = 4;
const PATH_LIMIT = 4096, TEXT_LIMIT = 1048576;
const MODULE_LIMIT = 16 * 1024 * 1024, TYPE_LIMIT = 4096, FUNCTION_LIMIT = 65536;
const TYPE_ARITY_LIMIT = 64, SECTION_LIMIT = 64;
const NAME = 'read_text', NAMESPACE = 'nanolang_host_v1';
const utf8 = new TextDecoder('utf-8', {fatal: true, ignoreBOM: true});
function refuse() { throw new TypeError('I refuse this private read-text module envelope'); }
function bytesCopy(value, limit) {
    if (!(value instanceof Uint8Array) || value.byteLength > limit || value.buffer instanceof SharedArrayBuffer) refuse();
    return Buffer.from(value);
}
class Cursor {
    constructor(bytes) { this.bytes = bytes; this.pos = 0; }
    take(n) {
        if (!Number.isInteger(n) || n < 0 || n > this.bytes.length - this.pos) refuse();
        const b = this.bytes.subarray(this.pos, this.pos + n); this.pos += n; return b;
    }
    u8() { return this.take(1)[0]; }
    u32() {
        let value = 0;
        for (let i = 0; i < 5; ++i) {
            const b = this.u8(); if (i === 4 && (b & 0xf0)) refuse();
            value += (b & 0x7f) * 2 ** (7 * i);
            if (!(b & 0x80)) return value;
        }
        refuse();
    }
    name() { const n = this.u32(); if (n > PATH_LIMIT) refuse(); return utf8.decode(this.take(n)); }
    done() { if (this.pos !== this.bytes.length) refuse(); }
}
function valueTypes(c) {
    const n = c.u32(); if (n > TYPE_ARITY_LIMIT) refuse();
    const result = [];
    for (let i = 0; i < n; ++i) {
        const tag = c.u8(); if (![0x7f, 0x7e, 0x7d, 0x7c].includes(tag)) refuse(); result.push(tag);
    }
    return result;
}
function moduleEnvelope(bytes) {
    const c = new Cursor(bytes);
    if (!c.take(8).equals(Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]))) refuse();
    let sections = 0, types = null, importType = null, functions = 0, memory = false;
    let exports = null; const seen = new Set();
    while (c.pos < bytes.length) {
        if (++sections > SECTION_LIMIT) refuse();
        const id = c.u8(), part = new Cursor(c.take(c.u32()));
        if (id > 12 || id === 8 || (id && seen.has(id))) refuse();
        if (id) seen.add(id);
        if (id === 0) { part.name(); part.pos = part.bytes.length; }
        else if (id === 1) {
            const n = part.u32(); if (n > TYPE_LIMIT) refuse(); types = [];
            for (let i = 0; i < n; ++i) {
                if (part.u8() !== 0x60) refuse(); types.push([valueTypes(part), valueTypes(part)]);
            }
        } else if (id === 2) {
            if (part.u32() !== 1 || part.name() !== NAMESPACE || part.name() !== NAME || part.u8() !== 0) refuse();
            importType = part.u32();
        } else if (id === 3) {
            functions = part.u32(); if (functions >= FUNCTION_LIMIT) refuse();
            for (let i = 0; i < functions; ++i) { const type = part.u32(); if (!types || type >= types.length) refuse(); }
        } else if (id === 5) {
            if (part.u32() !== 1 || part.u32() !== 1 || part.u32() !== 32 || part.u32() !== 1024) refuse(); memory = true;
        } else if (id === 7) {
            const n = part.u32(); if (n > FUNCTION_LIMIT) refuse(); exports = new Map();
            for (let i = 0; i < n; ++i) {
                const name = part.name(), kind = part.u8(), index = part.u32();
                if (exports.has(name) || kind > 3) refuse(); exports.set(name, {kind, index});
            }
        } else { part.pos = part.bytes.length; } // The engine validates other section contents/order.
        part.done();
    }
    if (!types || importType === null || importType >= types.length || !memory || !exports) refuse();
    const [params, results] = types[importType];
    if (params.length !== 5 || params.some(t => t !== 0x7f) || results.length !== 1 || results[0] !== 0x7f) refuse();
    let memories = 0;
    for (const [name, {kind, index}] of exports) {
        if (kind === 2) { if (name !== 'memory' || index !== 0) refuse(); ++memories; }
        if (kind === 0 && index >= functions + 1) refuse();
    }
    if (memories !== 1) refuse();
    return exports;
}
function spans(size, p, n, d, cap, out) {
    if (n > PATH_LIMIT || cap > TEXT_LIMIT) return false;
    const ranges = [[p, n], [d, cap], [out, 4]];
    for (const [offset, length] of ranges) if (offset > size || length > size - offset) return false;
    for (let i = 0; i < 3; ++i) for (let j = i + 1; j < 3; ++j) {
        const [a, an] = ranges[i], [b, bn] = ranges[j];
        if (an && bn && a < b + bn && b < a + an) return false;
    }
    return true;
}
function systemError(error) {
    return error instanceof Error && typeof error.code === 'string' && Number.isInteger(error.errno);
}

export function createReadTextInstance(moduleBytes, paths) {
    const bytes = bytesCopy(moduleBytes, MODULE_LIMIT), envelope = moduleEnvelope(bytes);
    if (!Array.isArray(paths) || paths.length > 64) refuse();
    let allowlist = [];
    for (const path of paths) {
        const copy = bytesCopy(path, PATH_LIMIT); if (!copy.length || copy.includes(0)) refuse(); allowlist.push(copy);
    }
    let module = new WebAssembly.Module(bytes), instance = null;
    const imports = WebAssembly.Module.imports(module);
    if (imports.length !== 1 || imports[0].module !== NAMESPACE || imports[0].name !== NAME || imports[0].kind !== 'function') refuse();
    const actual = WebAssembly.Module.exports(module);
    if (actual.length !== envelope.size) refuse();
    const kinds = ['function', 'table', 'memory', 'global'];
    for (const {name, kind} of actual) if (!envelope.has(name) || kinds[envelope.get(name).kind] !== kind) refuse();
    let memory = null, ready = false, terminal = false, exportActive = false, callbackActive = false;
    const last = {status: NPR_OK, opened: false, closeAttempted: false, closeError: false, bytesRead: 0};
    function callback(pathOffset, pathLength, destinationOffset, capacity, lengthOffset) {
        if (!ready || terminal || callbackActive || !exportActive) return NPR_INVALID;
        callbackActive = true;
        let status = NPR_OK, fd = null, opened = false, closeAttempted = false, closeError = false, count = 0;
        try {
            const p = pathOffset >>> 0, n = pathLength >>> 0, d = destinationOffset >>> 0;
            const cap = capacity >>> 0, out = lengthOffset >>> 0;
            const buffer = memory.buffer, size = buffer.byteLength;
            if (buffer instanceof SharedArrayBuffer || !spans(size, p, n, d, cap, out)) return (status = NPR_INVALID);
            const path = Buffer.from(new Uint8Array(buffer, p, n));
            if (!n || path.includes(0) || !allowlist.some(row => row.equals(path))) return (status = NPR_DENIED);
            const data = Buffer.alloc(cap + 1), cell = Buffer.alloc(4);
            let empty = false;
            try {
                fd = fs.openSync(path, 'r'); opened = true;
                while (count <= cap) {
                    const got = fs.readSync(fd, data, count, cap + 1 - count, null);
                    if (!Number.isInteger(got) || got < 0 || got > cap + 1 - count) throw new Error('I received an invalid host read count');
                    count += got;
                    if (count > cap) { status = NPR_LIMIT; break; }
                    if (!got) break;
                }
            } catch (error) {
                if (systemError(error)) empty = true;
                else if (error instanceof RangeError) status = NPR_MEMORY;
                else throw error;
            } finally {
                if (fd !== null) {
                    closeAttempted = true;
                    try { fs.closeSync(fd); }
                    catch (error) {
                        if (systemError(error)) { closeError = true; empty = true; }
                        else if (error instanceof RangeError) { closeError = true; if (status === NPR_OK) status = NPR_MEMORY; }
                        else throw error;
                    }
                }
            }
            if (status !== NPR_OK) return status;
            let length = empty ? 0 : count;
            if (data.subarray(0, length).includes(0)) length = 0;
            // Reacquire after I/O. No callback may grow/reenter this memory.
            const current = memory.buffer;
            if (current !== buffer || current.byteLength !== size || !spans(size, p, n, d, cap, out)) return (status = NPR_INVALID);
            const destination = new Uint8Array(current, d, length), lengthView = new Uint8Array(current, out, 4);
            const payload = data.subarray(0, length);
            cell.writeUInt32LE(length, 0);
            destination.set(payload);
            lengthView.set(cell); // All allocating preparation precedes this final publication.
            return NPR_OK;
        } catch (error) {
            // All range-sensitive operations have checked bounded operands.
            // Catchable allocation RangeErrors are MEMORY; fatal engine OOM is not recoverable.
            if (error instanceof RangeError) { if (status === NPR_OK) status = NPR_MEMORY; return status; }
            if (status === NPR_OK) status = NPR_INVALID;
            terminal = true; ready = false; throw error;
        } finally {
            callbackActive = false;
            last.status = status; last.opened = opened; last.closeAttempted = closeAttempted;
            last.closeError = closeError; last.bytesRead = count;
        }
    }
    try {
        instance = new WebAssembly.Instance(module, {[NAMESPACE]: {[NAME]: callback}});
        memory = instance.exports.memory;
        if (!(memory instanceof WebAssembly.Memory) || memory.buffer instanceof SharedArrayBuffer || memory.buffer.byteLength !== 32 * 65536) refuse();
        ready = true;
    } catch (error) { terminal = true; instance = memory = module = allowlist = null; throw error; }
    return Object.freeze({
        call(name, ...args) {
            if (!ready || terminal || exportActive || callbackActive || typeof name !== 'string' ||
                !envelope.has(name) || envelope.get(name).kind !== 0) throw new TypeError('I refuse this private guest call');
            exportActive = true;
            try { return instance.exports[name](...args); }
            catch (error) { terminal = true; ready = false; throw error; }
            finally { exportActive = false; }
        },
        close() {
            if (exportActive || callbackActive) return NPR_INVALID;
            ready = false; terminal = true; instance = memory = module = allowlist = null; return NPR_OK;
        },
        report() { return {...last, ready, terminal, exportActive, callbackActive}; },
    });
}
