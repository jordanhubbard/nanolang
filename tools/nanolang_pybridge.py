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


# ── Warp fluid simulation ─────────────────────────────────────────────────────
# Implements Jos Stam's Stable Fluids algorithm using NVIDIA Warp (GPU) when
# available, with a pure NumPy CPU fallback.

_warp_fluid_class = None   # lazily initialised on first warp_fluid_init call


def _build_fluid_sim_class():
    """Return the NLFluidSim class, using Warp if available, NumPy otherwise."""
    import numpy as np

    # ── Try Warp (GPU) ────────────────────────────────────────────────────────
    try:
        import warp as wp
        wp.init()

        @wp.kernel
        def _k_advect_scalar(
            q_in:  wp.array2d(dtype=wp.float32),
            q_out: wp.array2d(dtype=wp.float32),
            vel:   wp.array2d(dtype=wp.vec2),
            dt:    float,
            N:     int,
        ):
            i, j = wp.tid()
            v  = vel[i, j]
            x  = float(i) - dt * v[0]
            y  = float(j) - dt * v[1]
            x  = wp.clamp(x, 0.0, float(N - 1))
            y  = wp.clamp(y, 0.0, float(N - 1))
            i0 = int(x); i1 = wp.min(i0 + 1, N - 1)
            j0 = int(y); j1 = wp.min(j0 + 1, N - 1)
            sx = x - float(i0); sy = y - float(j0)
            q_out[i, j] = ((1.0 - sx) * ((1.0 - sy) * q_in[i0, j0] + sy * q_in[i0, j1])
                           + sx        * ((1.0 - sy) * q_in[i1, j0] + sy * q_in[i1, j1]))

        @wp.kernel
        def _k_advect_vel(
            vel_in:  wp.array2d(dtype=wp.vec2),
            vel_out: wp.array2d(dtype=wp.vec2),
            dt:      float,
            N:       int,
        ):
            i, j = wp.tid()
            v  = vel_in[i, j]
            x  = float(i) - dt * v[0]
            y  = float(j) - dt * v[1]
            x  = wp.clamp(x, 0.0, float(N - 1))
            y  = wp.clamp(y, 0.0, float(N - 1))
            i0 = int(x); i1 = wp.min(i0 + 1, N - 1)
            j0 = int(y); j1 = wp.min(j0 + 1, N - 1)
            sx = x - float(i0); sy = y - float(j0)
            vel_out[i, j] = ((1.0 - sx) * ((1.0 - sy) * vel_in[i0, j0] + sy * vel_in[i0, j1])
                             + sx        * ((1.0 - sy) * vel_in[i1, j0] + sy * vel_in[i1, j1]))

        @wp.kernel
        def _k_divergence(
            vel: wp.array2d(dtype=wp.vec2),
            div: wp.array2d(dtype=wp.float32),
            N:   int,
        ):
            i, j = wp.tid()
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                div[i, j] = 0.0
                return
            div[i, j] = -0.5 * (
                vel[i + 1, j][0] - vel[i - 1, j][0] +
                vel[i, j + 1][1] - vel[i, j - 1][1]
            )

        @wp.kernel
        def _k_jacobi(
            p_in:  wp.array2d(dtype=wp.float32),
            div:   wp.array2d(dtype=wp.float32),
            p_out: wp.array2d(dtype=wp.float32),
            N:     int,
        ):
            i, j = wp.tid()
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                p_out[i, j] = 0.0
                return
            p_out[i, j] = (div[i, j]
                           + p_in[i - 1, j] + p_in[i + 1, j]
                           + p_in[i, j - 1] + p_in[i, j + 1]) * 0.25

        @wp.kernel
        def _k_project(
            vel: wp.array2d(dtype=wp.vec2),
            p:   wp.array2d(dtype=wp.float32),
            N:   int,
        ):
            i, j = wp.tid()
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                vel[i, j] = wp.vec2(0.0, 0.0)
                return
            vx = vel[i, j][0] - 0.5 * (p[i + 1, j] - p[i - 1, j])
            vy = vel[i, j][1] - 0.5 * (p[i, j + 1] - p[i, j - 1])
            vel[i, j] = wp.vec2(vx, vy)

        @wp.kernel
        def _k_splat(
            vel:  wp.array2d(dtype=wp.vec2),
            dens: wp.array2d(dtype=wp.float32),
            cx:   int,
            cy:   int,
            fx:   float,
            fy:   float,
            r:    int,
            amt:  float,
        ):
            i, j = wp.tid()
            dx = i - cx; dy = j - cy
            if dx * dx + dy * dy <= r * r:
                vel[i, j]  = vel[i, j]  + wp.vec2(fx, fy)
                dens[i, j] = dens[i, j] + amt

        @wp.kernel
        def _k_scale_scalar(arr: wp.array2d(dtype=wp.float32), s: float):
            i, j = wp.tid()
            arr[i, j] = arr[i, j] * s

        @wp.kernel
        def _k_scale_vel(vel: wp.array2d(dtype=wp.vec2), s: float):
            i, j = wp.tid()
            vel[i, j] = vel[i, j] * s

        class NLFluidSimWarp:
            BACKEND = "warp"  # class-level: "this class uses warp"

            def __init__(self, n, device):
                self.N       = n
                self.device  = device
                self.backend = f"warp/{device}"  # instance-level: actual device chosen
                self.vel    = wp.zeros((n, n), dtype=wp.vec2,    device=device)
                self.vel1   = wp.zeros((n, n), dtype=wp.vec2,    device=device)
                self.dens   = wp.zeros((n, n), dtype=wp.float32, device=device)
                self.dens1  = wp.zeros((n, n), dtype=wp.float32, device=device)
                self.div    = wp.zeros((n, n), dtype=wp.float32, device=device)
                self.p      = wp.zeros((n, n), dtype=wp.float32, device=device)
                self.p1     = wp.zeros((n, n), dtype=wp.float32, device=device)

            def reset(self):
                for arr in (self.vel, self.vel1, self.dens, self.dens1,
                            self.div, self.p, self.p1):
                    arr.zero_()

            def splat(self, cx, cy, fx, fy, radius, amount):
                wp.launch(_k_splat, dim=(self.N, self.N),
                          inputs=[self.vel, self.dens,
                                  int(cx), int(cy),
                                  float(fx), float(fy),
                                  int(radius), float(amount)],
                          device=self.device)

            def step(self, dt=0.15, iters=20):
                N, dev, dim = self.N, self.device, (self.N, self.N)
                wp.launch(_k_advect_vel, dim=dim,
                          inputs=[self.vel, self.vel1, float(dt), N], device=dev)
                self.vel, self.vel1 = self.vel1, self.vel
                wp.launch(_k_divergence, dim=dim,
                          inputs=[self.vel, self.div, N], device=dev)
                for _ in range(iters):
                    wp.launch(_k_jacobi, dim=dim,
                              inputs=[self.p, self.div, self.p1, N], device=dev)
                    self.p, self.p1 = self.p1, self.p
                wp.launch(_k_project, dim=dim,
                          inputs=[self.vel, self.p, N], device=dev)
                wp.launch(_k_scale_vel, dim=dim,
                          inputs=[self.vel, float(0.9995)], device=dev)
                wp.launch(_k_advect_scalar, dim=dim,
                          inputs=[self.dens, self.dens1, self.vel, float(dt), N], device=dev)
                self.dens, self.dens1 = self.dens1, self.dens
                wp.launch(_k_scale_scalar, dim=dim,
                          inputs=[self.dens, float(0.993)], device=dev)

            def get_image_png(self, path):
                from PIL import Image
                d = np.clip(self.dens.numpy().T, 0.0, 1.0).astype(np.float32)
                r = np.clip(d * 3.0,       0.0, 1.0)
                g = np.clip(d * 3.0 - 1.0, 0.0, 1.0)
                b = np.clip(d * 3.0 - 2.0, 0.0, 1.0)
                rgba = np.stack([r, g, b, np.ones_like(d)], axis=-1)
                Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(path)
                return True

        return NLFluidSimWarp

    except Exception:
        pass  # fall through to NumPy CPU implementation

    # ── NumPy CPU fallback ────────────────────────────────────────────────────
    class NLFluidSimNumpy:
        BACKEND = "numpy-cpu"

        def __init__(self, n, device):
            self.N       = n
            self.backend = "numpy-cpu"  # always CPU, device arg is ignored
            self.vel     = np.zeros((n, n, 2), dtype=np.float32)
            self.dens    = np.zeros((n, n),    dtype=np.float32)
            self._div = np.zeros((n, n),    dtype=np.float32)
            self._p   = np.zeros((n, n),    dtype=np.float32)

        def reset(self):
            self.vel[:] = 0.0
            self.dens[:] = 0.0

        def splat(self, cx, cy, fx, fy, radius, amount):
            N = self.N
            xi, yi = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
            mask = (xi - cx) ** 2 + (yi - cy) ** 2 <= radius ** 2
            self.vel[:, :, 0][mask] += fx
            self.vel[:, :, 1][mask] += fy
            self.dens[mask] += amount

        @staticmethod
        def _advect(field, vel, dt, N):
            xs = np.arange(N, dtype=np.float32)
            xi, yi = np.meshgrid(xs, xs, indexing="ij")
            bx = np.clip(xi - dt * vel[:, :, 0], 0, N - 1)
            by = np.clip(yi - dt * vel[:, :, 1], 0, N - 1)
            x0 = bx.astype(np.int32); x1 = np.minimum(x0 + 1, N - 1)
            y0 = by.astype(np.int32); y1 = np.minimum(y0 + 1, N - 1)
            sx = bx - x0.astype(np.float32)
            sy = by - y0.astype(np.float32)
            if field.ndim == 2:
                return ((1 - sx) * ((1 - sy) * field[x0, y0] + sy * field[x0, y1])
                        + sx     * ((1 - sy) * field[x1, y0] + sy * field[x1, y1]))
            out = np.empty_like(field)
            for c in range(field.shape[2]):
                f = field[:, :, c]
                out[:, :, c] = ((1 - sx) * ((1 - sy) * f[x0, y0] + sy * f[x0, y1])
                                + sx     * ((1 - sy) * f[x1, y0] + sy * f[x1, y1]))
            return out

        def step(self, dt=0.15, iters=20):
            N = self.N
            self.vel  = self._advect(self.vel,  self.vel, dt, N)
            self._div[:] = 0.0
            self._div[1:-1, 1:-1] = -0.5 * (
                self.vel[2:,  1:-1, 0] - self.vel[:-2, 1:-1, 0] +
                self.vel[1:-1, 2:,  1] - self.vel[1:-1, :-2, 1]
            )
            self._p[:] = 0.0
            for _ in range(iters):
                p_new = np.zeros_like(self._p)
                p_new[1:-1, 1:-1] = (
                    self._div[1:-1, 1:-1]
                    + self._p[:-2, 1:-1] + self._p[2:, 1:-1]
                    + self._p[1:-1, :-2] + self._p[1:-1, 2:]
                ) * 0.25
                self._p = p_new
            self.vel[1:-1, 1:-1, 0] -= 0.5 * (self._p[2:,  1:-1] - self._p[:-2, 1:-1])
            self.vel[1:-1, 1:-1, 1] -= 0.5 * (self._p[1:-1, 2:]  - self._p[1:-1, :-2])
            self.vel[0, :]  = 0.0; self.vel[-1, :] = 0.0
            self.vel[:, 0]  = 0.0; self.vel[:, -1] = 0.0
            self.vel  *= 0.9995
            self.dens  = self._advect(self.dens, self.vel, dt, N)
            self.dens *= 0.993

        def get_image_png(self, path):
            from PIL import Image
            d = np.clip(self.dens.T, 0.0, 1.0).astype(np.float32)
            r = np.clip(d * 3.0,       0.0, 1.0)
            g = np.clip(d * 3.0 - 1.0, 0.0, 1.0)
            b = np.clip(d * 3.0 - 2.0, 0.0, 1.0)
            rgba = np.stack([r, g, b, np.ones_like(d)], axis=-1)
            Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(path)
            return True

    return NLFluidSimNumpy


def get_fluid_sim_class():
    global _warp_fluid_class
    if _warp_fluid_class is None:
        _warp_fluid_class = _build_fluid_sim_class()
    return _warp_fluid_class


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

    # ── Warp fluid simulation ops ─────────────────────────────────────────────

    if op == "warp_fluid_init":
        n = int(params.get("n") or 128)
        device = str(params.get("device") or "auto")
        FluidSim = get_fluid_sim_class()

        if device == "auto":
            # Probe GPU availability and pick the best backend transparently.
            if FluidSim.BACKEND == "warp":
                # Warp is installed.  Try CUDA first.
                try:
                    import warp as wp
                    wp.get_device("cuda")
                    sim = FluidSim(n, "cuda")
                    return {"handle": handles.add(sim), "backend": sim.backend}, False
                except Exception:
                    pass
                # CUDA unavailable – run warp on CPU instead.
                try:
                    sim = FluidSim(n, "cpu")
                    return {"handle": handles.add(sim), "backend": sim.backend}, False
                except Exception:
                    pass
            # Warp not installed or failed entirely – NumPy CPU.
            try:
                sim = FluidSim(n, "cpu")
                return {"handle": handles.add(sim), "backend": sim.backend}, False
            except Exception as exc:
                raise PyBridgeError(ERROR_INTERNAL,
                    f"No simulation backend available: {exc}")

        elif device == "cuda":
            # Explicit CUDA request: hard-fail with a clear message if unavailable.
            try:
                import warp as wp
            except ImportError as exc:
                raise PyBridgeError(ERROR_INVALID_PARAMS,
                    f"NVIDIA Warp is not installed. "
                    f"Install it with: pip install warp-lang  (error: {exc})")
            if FluidSim.BACKEND != "warp":
                raise PyBridgeError(ERROR_INVALID_PARAMS,
                    "NVIDIA Warp failed to initialise during pybridge startup.")
            try:
                wp.get_device("cuda")
            except Exception as exc:
                raise PyBridgeError(ERROR_INVALID_PARAMS,
                    f"CUDA device is unavailable (NVIDIA GPU required for "
                    f"device='cuda'). Use device='auto' or device='cpu' for "
                    f"the NumPy CPU fallback.  Original error: {exc}")
            try:
                sim = FluidSim(n, "cuda")
            except Exception as exc:
                raise PyBridgeError(ERROR_INTERNAL,
                    f"Failed to create CUDA simulation: {exc}")
            return {"handle": handles.add(sim), "backend": sim.backend}, False

        else:
            # Explicit "cpu" (or any unknown device string) → always succeeds.
            try:
                sim = FluidSim(n, "cpu")
            except Exception as exc:
                raise PyBridgeError(ERROR_INTERNAL,
                    f"Failed to create CPU simulation: {exc}")
            return {"handle": handles.add(sim), "backend": sim.backend}, False

    if op == "warp_fluid_step":
        handle_id = params.get("handle")
        if handle_id is None:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing handle")
        dt    = float(params.get("dt")    or 0.15)
        iters = int(params.get("iters")   or 20)
        sim = handles.get(int(handle_id))
        sim.step(dt, iters)
        return {"ok": True}, False

    if op == "warp_fluid_splat":
        handle_id = params.get("handle")
        if handle_id is None:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing handle")
        cx     = int(params.get("cx")     or 0)
        cy     = int(params.get("cy")     or 0)
        fx     = float(params.get("fx")   or 0.0)
        fy     = float(params.get("fy")   or 0.0)
        radius = int(params.get("radius") or 8)
        amount = float(params.get("amount") or 1.0)
        sim = handles.get(int(handle_id))
        sim.splat(cx, cy, fx, fy, radius, amount)
        return {"ok": True}, False

    if op == "warp_fluid_get_image_png":
        handle_id = params.get("handle")
        path      = params.get("path")
        if handle_id is None or not path:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing handle or path")
        sim = handles.get(int(handle_id))
        sim.get_image_png(path)
        return {"ok": True}, False

    if op == "warp_fluid_reset":
        handle_id = params.get("handle")
        if handle_id is None:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing handle")
        handles.get(int(handle_id)).reset()
        return {"ok": True}, False

    if op == "warp_fluid_destroy":
        handle_id = params.get("handle")
        if handle_id is None:
            raise PyBridgeError(ERROR_INVALID_PARAMS, "Missing handle")
        handles.release(int(handle_id))
        return {"ok": True}, False

    if op == "warp_fluid_backend":
        # Return the best backend that would be chosen by device="auto".
        FluidSim = get_fluid_sim_class()
        if FluidSim.BACKEND == "warp":
            try:
                import warp as wp
                wp.get_device("cuda")
                best = "warp/cuda"
            except Exception:
                best = "warp/cpu"
        else:
            best = "numpy-cpu"
        return {"backend": best}, False

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
