#!/usr/bin/env python3
import argparse
import base64
import importlib
import io
import json
import os
import platform
import struct
import sys
import tempfile
import traceback

PROTOCOL_VERSION = 1
HANDLE_KEY = "__pybridge_handle__"
BINARY_KEY = "__pybridge_binary__"
NDARRAY_KEY = "__pybridge_ndarray__"

ERROR_INVALID_REQUEST = -32600
ERROR_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603

LOG_ENABLED = os.getenv("PYBRIDGE_LOG") == "1"


class PyBridgeError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class HandleStore:
    def __init__(self):
        self._next_id = 1
        self._handles = {}

    def add(self, obj):
        handle_id = self._next_id
        self._next_id += 1
        self._handles[handle_id] = obj
        return handle_id

    def get(self, handle_id):
        if handle_id not in self._handles:
            raise PyBridgeError(ERROR_INVALID_PARAMS, f"Unknown handle {handle_id}")
        return self._handles[handle_id]

    def release(self, handle_id):
        return self._handles.pop(handle_id, None) is not None


handles = HandleStore()
privileged = False


def log_event(event, payload=None):
    if not LOG_ENABLED:
        return
    data = {"event": event}
    if payload is not None:
        data["payload"] = payload
    sys.stderr.write(json.dumps(data, separators=(",", ":")) + "\n")
    sys.stderr.flush()


def encode_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode_value(v) for k, v in value.items()}
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            data = value.tobytes()
            return {
                NDARRAY_KEY: base64.b64encode(data).decode("ascii"),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
    except Exception:
        pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {
            BINARY_KEY: base64.b64encode(data).decode("ascii"),
            "encoding": "base64",
        }
    handle_id = handles.add(value)
    return {HANDLE_KEY: handle_id}


def resolve_value(value):
    if isinstance(value, list):
        return [resolve_value(v) for v in value]
    if isinstance(value, dict):
        if HANDLE_KEY in value:
            return handles.get(int(value[HANDLE_KEY]))
        if NDARRAY_KEY in value:
            try:
                import numpy as np
            except Exception:
                raise PyBridgeError(ERROR_INVALID_PARAMS, "NumPy is required to decode ndarray")
            raw = base64.b64decode(value[NDARRAY_KEY])
            dtype = value.get("dtype") or "float64"
            shape = value.get("shape") or []
            arr = np.frombuffer(raw, dtype=dtype)
            if shape:
                arr = arr.reshape(shape)
            return arr
        if BINARY_KEY in value:
            data = base64.b64decode(value[BINARY_KEY])
            return data
        return {k: resolve_value(v) for k, v in value.items()}
    return value


def sysinfo_payload():
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }


def deps_payload():
    try:
        import importlib.metadata as metadata
    except Exception:
        return {"packages": []}

    packages = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.metadata.get("Project-Name")
        version = dist.version
        if name:
            packages.append(f"{name}=={version}")
    packages.sort()
    return {"packages": packages}


def render_matplotlib(spec, inline):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise PyBridgeError(ERROR_INTERNAL, f"Matplotlib import failed: {exc}")

    width = int(spec.get("width", 640))
    height = int(spec.get("height", 480))
    dpi = int(spec.get("dpi", 100))
    title = spec.get("title")
    xlabel = spec.get("xlabel")
    ylabel = spec.get("ylabel")

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    if title:
        ax.set_title(str(title))
    if xlabel:
        ax.set_xlabel(str(xlabel))
    if ylabel:
        ax.set_ylabel(str(ylabel))

    series = spec.get("series")
    if series is None and "x" in spec and "y" in spec:
        series = [{"x": spec.get("x"), "y": spec.get("y"), "label": spec.get("label")}]

    if not isinstance(series, list):
        series = []

    has_label = False
    for item in series:
        if not isinstance(item, dict):
            continue
        xs = item.get("x") or []
        ys = item.get("y") or []
        label = item.get("label")
        if label:
            has_label = True
            ax.plot(xs, ys, label=str(label))
        else:
            ax.plot(xs, ys)

    if has_label:
        ax.legend()

    if inline:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        return {"png": encode_value(buffer.getvalue())}

    fd, path = tempfile.mkstemp(prefix="nanolang_mpl_", suffix=".png")
    os.close(fd)
    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)
    return {"path": path}


def read_message():
    header = sys.stdin.buffer.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise PyBridgeError(ERROR_INVALID_REQUEST, "Incomplete header")
    length = struct.unpack(">I", header)[0]
    if length <= 0:
        raise PyBridgeError(ERROR_INVALID_REQUEST, "Invalid length")
    payload = sys.stdin.buffer.read(length)
    if len(payload) != length:
        raise PyBridgeError(ERROR_INVALID_REQUEST, "Incomplete payload")
    return json.loads(payload.decode("utf-8"))


def write_message(message):
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack(">I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def handle_request(request):
    global privileged

    op = request.get("op")
    params = request.get("params") or {}

    if op == "hello":
        requested_privileged = bool(params.get("privileged"))
        privileged_env = os.getenv("PYBRIDGE_PRIVILEGED") == "1"
        privileged = requested_privileged and privileged_env
        return {
            "server": "nanolang_pybridge",
            "protocol": PROTOCOL_VERSION,
            "privileged": privileged,
        }, False

    if op == "ping":
        return {"ok": True}, False

    if op == "sysinfo":
        return sysinfo_payload(), False

    if op == "deps":
        return deps_payload(), False

    if op == "shutdown":
        return {"ok": True}, True

    if op == "import":
        module_name = params.get("module")
        if not module_name:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing module name")
        module = importlib.import_module(module_name)
        return {"handle": handles.add(module)}, False

    if op == "call":
        handle_id = params.get("handle")
        if handle_id is None:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing handle")
        method = params.get("method") or ""
        args = resolve_value(params.get("args") or [])
        kwargs = resolve_value(params.get("kwargs") or {})
        target = handles.get(int(handle_id))
        if method:
            target = getattr(target, method)
        result = target(*args, **kwargs)
        return {"value": encode_value(result)}, False

    if op == "getattr":
        handle_id = params.get("handle")
        attr = params.get("attr")
        if handle_id is None or not attr:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing handle or attr")
        target = handles.get(int(handle_id))
        value = getattr(target, attr)
        return {"value": encode_value(value)}, False

    if op == "setattr":
        handle_id = params.get("handle")
        attr = params.get("attr")
        value = params.get("value")
        if handle_id is None or not attr:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing handle or attr")
        target = handles.get(int(handle_id))
        setattr(target, attr, resolve_value(value))
        return {"ok": True}, False

    if op == "release":
        handle_id = params.get("handle")
        if handle_id is None:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing handle")
        ok = handles.release(int(handle_id))
        return {"ok": ok}, False

    if op == "eval":
        if not privileged:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Eval requires privileged mode")
        expr = params.get("expr")
        if expr is None:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing expr")
        result = eval(expr, {})
        return {"value": encode_value(result)}, False

    if op == "exec":
        if not privileged:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Exec requires privileged mode")
        code = params.get("code")
        if code is None:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing code")
        exec(code, {})
        return {"ok": True}, False

    if op == "mpl_render_png":
        spec = params.get("spec") or {}
        inline = bool(params.get("inline"))
        result = render_matplotlib(spec, inline)
        return result, False

    raise PyBridgeError(ERROR_NOT_FOUND, f"Unknown operation: {op}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", type=int, default=PROTOCOL_VERSION)
    parser.parse_args()

    while True:
        try:
            request = read_message()
            if request is None:
                break
            log_event("request", request)
            response = {
                "v": PROTOCOL_VERSION,
                "id": request.get("id", 0),
            }
            result, should_exit = handle_request(request)
            response["result"] = result
            write_message(response)
            log_event("response", response)
            if should_exit:
                break
        except PyBridgeError as err:
            response = {
                "v": PROTOCOL_VERSION,
                "id": 0,
                "error": {
                    "code": err.code,
                    "message": err.message,
                    "traceback": "",
                },
            }
            write_message(response)
            log_event("error", response)
        except Exception:
            response = {
                "v": PROTOCOL_VERSION,
                "id": 0,
                "error": {
                    "code": ERROR_INTERNAL,
                    "message": "Internal error",
                    "traceback": traceback.format_exc(),
                },
            }
            write_message(response)
            log_event("error", response)


if __name__ == "__main__":
    main()
