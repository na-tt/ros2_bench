"""Minimal ``ctypes`` bindings to the CUDA runtime (``libcudart``).

TensorRT only needs raw device pointers plus a stream handle, so ~200 lines of
ctypes is enough to avoid depending on ``pycuda`` or ``cuda-python`` (neither is
installed on the target Jetson, the TensorRT Python package is).

Bound entry points:

    cudaSetDevice / cudaGetDevice / cudaGetLastError / cudaGetErrorString
    cudaMalloc / cudaFree / cudaMallocHost / cudaFreeHost
    cudaMemcpy / cudaMemcpyAsync
    cudaStreamCreateWithFlags / cudaStreamDestroy / cudaStreamSynchronize
    cudaDeviceSynchronize

:mod:`yolocar.engine` only uses the asynchronous subset (device pointers plus
one stream); the blocking helpers are kept because they make the module usable
stand-alone and easy to exercise from tests.

Threading: :func:`get_runtime` may be called from several threads and a
:class:`CudaRuntime` holds no per-call state, but CUDA's *current device* is
per-thread state. Buffers and streams belong to whichever device was current
when they were created, so keep a stream, its buffers and the code that
launches work on it on one thread (or call :meth:`CudaRuntime.set_device` on
each thread).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

#: ``cudaStreamNonBlocking``: don't implicitly sync with the legacy stream.
CUDA_STREAM_NON_BLOCKING = 0x01

#: Library names tried in order, after ``ctypes.util.find_library``.
_LIBRARY_CANDIDATES: tuple[str, ...] = (
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "/usr/local/cuda/lib64/libcudart.so",
    "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so",
    "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so",
)


class CudaError(RuntimeError):
    """Raised when a CUDA runtime call returns an error code."""


def _load_library() -> ctypes.CDLL:
    """Find and ``dlopen`` libcudart, raising a helpful error if missing."""
    candidates: list[str] = []
    located = ctypes.util.find_library("cudart")
    if located:
        candidates.append(located)
    candidates.extend(_LIBRARY_CANDIDATES)

    failures: list[str] = []
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:  # pragma: no cover - depends on host
            failures.append(f"{name}: {exc}")
    raise CudaError(
        "could not load the CUDA runtime (libcudart). Tried: "
        + "; ".join(failures)
        + ". Install the CUDA runtime or set LD_LIBRARY_PATH to its lib dir."
    )


class CudaRuntime:
    """Thin wrapper over the handful of libcudart entry points we need."""

    def __init__(self, library: ctypes.CDLL | None = None) -> None:
        self._lib = library if library is not None else _load_library()
        self._bind()

    # -- binding ---------------------------------------------------------
    def _bind(self) -> None:
        lib = self._lib
        void_p = ctypes.c_void_p
        size_t = ctypes.c_size_t
        int_p = ctypes.POINTER(ctypes.c_int)

        try:
            # -- device / errors ------------------------------------------
            lib.cudaSetDevice.argtypes = [ctypes.c_int]
            lib.cudaSetDevice.restype = ctypes.c_int
            lib.cudaGetDevice.argtypes = [int_p]
            lib.cudaGetDevice.restype = ctypes.c_int
            lib.cudaGetLastError.argtypes = []
            lib.cudaGetLastError.restype = ctypes.c_int
            lib.cudaGetErrorString.argtypes = [ctypes.c_int]
            lib.cudaGetErrorString.restype = ctypes.c_char_p

            # -- memory ---------------------------------------------------
            lib.cudaMalloc.argtypes = [ctypes.POINTER(void_p), size_t]
            lib.cudaMalloc.restype = ctypes.c_int
            lib.cudaFree.argtypes = [void_p]
            lib.cudaFree.restype = ctypes.c_int
            lib.cudaMallocHost.argtypes = [ctypes.POINTER(void_p), size_t]
            lib.cudaMallocHost.restype = ctypes.c_int
            lib.cudaFreeHost.argtypes = [void_p]
            lib.cudaFreeHost.restype = ctypes.c_int

            # -- copies ---------------------------------------------------
            lib.cudaMemcpy.argtypes = [void_p, void_p, size_t, ctypes.c_int]
            lib.cudaMemcpy.restype = ctypes.c_int
            lib.cudaMemcpyAsync.argtypes = [void_p, void_p, size_t, ctypes.c_int, void_p]
            lib.cudaMemcpyAsync.restype = ctypes.c_int

            # -- streams --------------------------------------------------
            lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(void_p), ctypes.c_uint]
            lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
            lib.cudaStreamDestroy.argtypes = [void_p]
            lib.cudaStreamDestroy.restype = ctypes.c_int
            lib.cudaStreamSynchronize.argtypes = [void_p]
            lib.cudaStreamSynchronize.restype = ctypes.c_int
            lib.cudaDeviceSynchronize.argtypes = []
            lib.cudaDeviceSynchronize.restype = ctypes.c_int
        except AttributeError as exc:
            name = getattr(lib, "_name", "libcudart")
            raise CudaError(
                f"{name} is missing a required symbol ({exc}); the CUDA runtime "
                "is too old for these bindings"
            ) from exc

    # -- helpers ---------------------------------------------------------
    def error_string(self, code: int) -> str:
        message = self._lib.cudaGetErrorString(code)
        return message.decode("utf-8", "replace") if message else f"error {code}"

    def _check(self, code: int, what: str) -> None:
        if code != 0:
            raise CudaError(f"{what} failed: {self.error_string(code)}")

    # -- device ----------------------------------------------------------
    def set_device(self, device_index: int) -> None:
        self._check(self._lib.cudaSetDevice(int(device_index)), "cudaSetDevice")

    def get_device(self) -> int:
        current = ctypes.c_int()
        self._check(self._lib.cudaGetDevice(ctypes.byref(current)), "cudaGetDevice")
        return int(current.value)

    def device_synchronize(self) -> None:
        """Block until every stream of the current device has finished.

        Not used by :mod:`yolocar.engine` (which synchronizes its own stream);
        useful for teardown code that owns several streams.
        """
        self._check(self._lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def last_error(self) -> int:
        """Read *and clear* the sticky runtime error flag (``cudaGetLastError``).

        Kernel launches never report errors through their own return value, so
        this is the only way to attribute an asynchronous failure -- and it has
        to be drained, otherwise a stale error is reported by the next call.
        """
        return int(self._lib.cudaGetLastError())

    def check_last_error(self, what: str) -> None:
        """Raise if an asynchronous launch failed since the last check."""
        self._check(self.last_error(), what)

    # -- memory ----------------------------------------------------------
    def malloc(self, nbytes: int) -> int:
        """Allocate ``nbytes`` of device memory, returning the raw pointer."""
        if nbytes <= 0:
            raise ValueError(f"cannot allocate {nbytes} bytes on device")
        pointer = ctypes.c_void_p()
        self._check(
            self._lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(nbytes)),
            "cudaMalloc",
        )
        if not pointer.value:
            raise CudaError(f"cudaMalloc returned a null pointer for {nbytes} bytes")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        self._check(self._lib.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def malloc_host(self, nbytes: int) -> int:
        """Allocate page-locked host memory."""
        pointer = ctypes.c_void_p()
        self._check(
            self._lib.cudaMallocHost(ctypes.byref(pointer), ctypes.c_size_t(nbytes)),
            "cudaMallocHost",
        )
        if not pointer.value:
            raise CudaError(f"cudaMallocHost returned a null pointer for {nbytes} bytes")
        return int(pointer.value)

    def free_host(self, pointer: int) -> None:
        self._check(self._lib.cudaFreeHost(ctypes.c_void_p(pointer)), "cudaFreeHost")

    # -- copies ----------------------------------------------------------
    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        """Synchronous copy (blocks the calling thread).

        Not used by :mod:`yolocar.engine`; kept for tests and stand-alone use.
        """
        self._check(
            self._lib.cudaMemcpy(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes),
                kind,
            ),
            "cudaMemcpy",
        )

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
        self._check(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes),
                kind,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync",
        )

    # -- streams ---------------------------------------------------------
    def stream_create(self, non_blocking: bool = True) -> int:
        handle = ctypes.c_void_p()
        flags = CUDA_STREAM_NON_BLOCKING if non_blocking else 0
        self._check(
            self._lib.cudaStreamCreateWithFlags(ctypes.byref(handle), flags),
            "cudaStreamCreateWithFlags",
        )
        # 0 is the legacy default stream: never let a failed/odd creation be
        # mistaken for a valid handle, otherwise work silently moves to it.
        if not handle.value:
            raise CudaError("cudaStreamCreateWithFlags returned a null stream handle")
        return int(handle.value)

    def stream_destroy(self, stream: int) -> None:
        self._check(
            self._lib.cudaStreamDestroy(ctypes.c_void_p(stream)), "cudaStreamDestroy"
        )

    def stream_synchronize(self, stream: int) -> None:
        """Wait for ``stream`` and surface any asynchronous launch failure.

        ``cudaStreamSynchronize`` returns the stream's own error (e.g. an
        illegal address), but a failed kernel *launch* only lands in the sticky
        error flag -- which is why the flag is drained here.
        """
        self.last_error()  # drop stale state so failures are attributed here
        self._check(
            self._lib.cudaStreamSynchronize(ctypes.c_void_p(stream)),
            "cudaStreamSynchronize",
        )
        self.check_last_error("asynchronous kernel execution")


_RUNTIME: CudaRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def get_runtime() -> CudaRuntime:
    """Process-wide CUDA runtime singleton."""
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = CudaRuntime()
    return _RUNTIME


class _OwnedBuffer(np.ndarray):
    """``uint8`` array that carries a strong reference to its allocator.

    Numpy cannot own memory that is reached through an integer address, so the
    owner is tracked explicitly and inherited by every derived view
    (``.view(dtype)``, ``.reshape``, slices, ...). A view that outlives the
    :class:`HostBuffer` it came from therefore keeps that buffer -- and with it
    the underlying allocation -- reachable instead of dangling.
    """

    #: Class attribute (not an instance slot) so ``__array_finalize__`` can
    #: also copy from sources that never had an owner, e.g. a plain ndarray.
    _buffer_owner: Any = None

    def __array_finalize__(self, obj: Any) -> None:
        if obj is not None:
            self._buffer_owner = getattr(obj, "_buffer_owner", None)


def _flat_view(pointer: int, nbytes: int, owner: Any = None) -> np.ndarray:
    """Zero-copy ``uint8`` numpy view over a raw host pointer.

    ``owner`` is kept alive for as long as the returned array (or any view
    derived from it) exists; see :class:`_OwnedBuffer`.
    """
    if not pointer:
        raise CudaError("cannot build a numpy view over a null host pointer")
    if int(nbytes) <= 0:
        raise ValueError(f"cannot build a view of {nbytes} bytes")
    buffer_type = ctypes.c_ubyte * int(nbytes)
    flat = np.frombuffer(buffer_type.from_address(int(pointer)), dtype=np.uint8)
    view = flat.view(_OwnedBuffer)
    view._buffer_owner = owner
    return view


class DeviceBuffer:
    """Owned device allocation."""

    __slots__ = ("_freed", "_runtime", "nbytes", "ptr")

    def __init__(self, runtime: CudaRuntime, nbytes: int) -> None:
        self._runtime = runtime
        self.nbytes = int(nbytes)
        self.ptr = runtime.malloc(self.nbytes)
        self._freed = False

    def free(self) -> None:
        if not self._freed:
            self._runtime.free(self.ptr)
            self._freed = True
            self.ptr = 0

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.free()
        except Exception:  # noqa: BLE001 - interpreter shutdown order
            pass

    def __repr__(self) -> str:
        return f"DeviceBuffer(ptr=0x{self.ptr:x}, nbytes={self.nbytes})"


class HostBuffer:
    """Host allocation exposed as a numpy view.

    Uses page-locked memory (``cudaMallocHost``) when ``pinned`` so that
    ``cudaMemcpyAsync`` actually overlaps with the compute.

    Lifetime rules
    --------------
    * The arrays returned by :attr:`array` / :meth:`view` keep this buffer
      reachable (see :class:`_OwnedBuffer`), so garbage collection cannot free
      the allocation while a view is still around -- this matters for the
      pageable fallback, where the memory is a plain numpy array.
    * :meth:`free` releases the allocation *immediately* (``close()`` has to
      return the memory, otherwise repeated load/close cycles in a benchmark
      leak page-locked memory). The buffer then raises :class:`CudaError` on
      any further use, but views handed out *before* the release are plain
      pointers into freed memory: do not keep them across ``close()``.
    """

    __slots__ = ("_closed", "_flat", "_owner", "_runtime", "nbytes", "pinned", "ptr")

    def __init__(
        self, runtime: CudaRuntime, nbytes: int, pinned: bool = True
    ) -> None:
        self._runtime = runtime
        self.nbytes = int(nbytes)
        self.pinned = bool(pinned)
        self._closed = False
        self._owner: np.ndarray | None = None
        self.ptr = 0
        if self.pinned:
            try:
                self.ptr = runtime.malloc_host(self.nbytes)
            except CudaError as exc:  # pragma: no cover - rare host config
                LOGGER.warning("pinned allocation failed (%s); using pageable memory", exc)
                self.pinned = False
        if not self.pinned:
            self._owner = np.empty(self.nbytes, dtype=np.uint8)
            self.ptr = int(self._owner.ctypes.data)
        self._flat = _flat_view(self.ptr, self.nbytes, owner=self)

    # -- views -----------------------------------------------------------
    @property
    def closed(self) -> bool:
        """True once :meth:`free` released the allocation."""
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise CudaError("host buffer has already been released")

    @property
    def array(self) -> np.ndarray:
        """Flat ``uint8`` view of the buffer."""
        self._require_open()
        return self._flat

    def view(self, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
        """Typed view (no copy) over this buffer."""
        self._require_open()
        values = self._flat.view(np.dtype(dtype))
        return values if shape is None else values.reshape(shape)

    # -- lifetime ---------------------------------------------------------
    def free(self) -> None:
        """Release the allocation; later access through this object raises."""
        if self._closed:
            return
        self._closed = True
        pointer = self.ptr
        # Drop the numpy view *first*: a failing cudaFreeHost must not leave a
        # writable array over memory the driver may already consider free.
        self._flat = np.empty(0, dtype=np.uint8)
        self.ptr = 0
        try:
            if self.pinned and pointer:
                self._runtime.free_host(pointer)
        finally:
            self._owner = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.free()
        except Exception:  # noqa: BLE001 - interpreter shutdown order
            pass

    def __repr__(self) -> str:
        state = ", closed" if self._closed else ""
        return (
            f"HostBuffer(ptr=0x{self.ptr:x}, nbytes={self.nbytes}, "
            f"pinned={self.pinned}{state})"
        )
