"""Minimal ``ctypes`` bindings to the CUDA runtime (``libcudart``).

TensorRT only needs raw device pointers plus a stream handle, so ~200 lines of
ctypes is enough to avoid depending on ``pycuda`` or ``cuda-python`` (neither is
installed on the target Jetson, the TensorRT Python package is).

Only the calls used by :mod:`yolocar.engine` are bound:

    cudaSetDevice / cudaGetDevice
    cudaMalloc / cudaFree
    cudaMallocHost / cudaFreeHost
    cudaMemcpy / cudaMemcpyAsync
    cudaStreamCreateWithFlags / cudaStreamDestroy / cudaStreamSynchronize
    cudaDeviceSynchronize / cudaGetLastError / cudaGetErrorString
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
from typing import Optional

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

    def __init__(self, library: Optional[ctypes.CDLL] = None) -> None:
        self._lib = library if library is not None else _load_library()
        self._bind()

    # -- binding ---------------------------------------------------------
    def _bind(self) -> None:
        lib = self._lib
        void_p = ctypes.c_void_p
        size_t = ctypes.c_size_t
        int_p = ctypes.POINTER(ctypes.c_int)

        lib.cudaSetDevice.argtypes = [ctypes.c_int]
        lib.cudaSetDevice.restype = ctypes.c_int
        lib.cudaGetDevice.argtypes = [int_p]
        lib.cudaGetDevice.restype = ctypes.c_int
        lib.cudaGetLastError.argtypes = []
        lib.cudaGetLastError.restype = ctypes.c_int
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p

        lib.cudaMalloc.argtypes = [ctypes.POINTER(void_p), size_t]
        lib.cudaMalloc.restype = ctypes.c_int
        lib.cudaFree.argtypes = [void_p]
        lib.cudaFree.restype = ctypes.c_int
        lib.cudaMallocHost.argtypes = [ctypes.POINTER(void_p), size_t]
        lib.cudaMallocHost.restype = ctypes.c_int
        lib.cudaFreeHost.argtypes = [void_p]
        lib.cudaFreeHost.restype = ctypes.c_int

        lib.cudaMemcpy.argtypes = [void_p, void_p, size_t, ctypes.c_int]
        lib.cudaMemcpy.restype = ctypes.c_int
        lib.cudaMemcpyAsync.argtypes = [void_p, void_p, size_t, ctypes.c_int, void_p]
        lib.cudaMemcpyAsync.restype = ctypes.c_int

        lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(void_p), ctypes.c_uint]
        lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
        lib.cudaStreamDestroy.argtypes = [void_p]
        lib.cudaStreamDestroy.restype = ctypes.c_int
        lib.cudaStreamSynchronize.argtypes = [void_p]
        lib.cudaStreamSynchronize.restype = ctypes.c_int
        lib.cudaDeviceSynchronize.argtypes = []
        lib.cudaDeviceSynchronize.restype = ctypes.c_int

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
        self._check(self._lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

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
        """Synchronous copy (blocks the calling thread)."""
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
        return int(handle.value or 0)

    def stream_destroy(self, stream: int) -> None:
        self._check(
            self._lib.cudaStreamDestroy(ctypes.c_void_p(stream)), "cudaStreamDestroy"
        )

    def stream_synchronize(self, stream: int) -> None:
        self._check(
            self._lib.cudaStreamSynchronize(ctypes.c_void_p(stream)),
            "cudaStreamSynchronize",
        )


_RUNTIME: Optional[CudaRuntime] = None


def get_runtime() -> CudaRuntime:
    """Process-wide CUDA runtime singleton."""
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = CudaRuntime()
    return _RUNTIME


def _flat_view(pointer: int, nbytes: int) -> np.ndarray:
    """Zero-copy ``uint8`` numpy view over a raw host pointer."""
    buffer_type = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte))
    return np.ctypeslib.as_array(buffer_type, shape=(int(nbytes),))


class DeviceBuffer:
    """Owned device allocation."""

    __slots__ = ("ptr", "nbytes", "_runtime", "_freed")

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
    """

    __slots__ = ("ptr", "nbytes", "pinned", "_runtime", "_owner", "_flat")

    def __init__(
        self, runtime: CudaRuntime, nbytes: int, pinned: bool = True
    ) -> None:
        self._runtime = runtime
        self.nbytes = int(nbytes)
        self.pinned = bool(pinned)
        self._owner: Optional[np.ndarray] = None
        if self.pinned:
            try:
                self.ptr = runtime.malloc_host(self.nbytes)
            except CudaError as exc:  # pragma: no cover - rare host config
                LOGGER.warning("pinned allocation failed (%s); using pageable memory", exc)
                self.pinned = False
        if not self.pinned:
            self._owner = np.empty(self.nbytes, dtype=np.uint8)
            self.ptr = int(self._owner.ctypes.data)
        self._flat = _flat_view(self.ptr, self.nbytes)

    @property
    def array(self) -> np.ndarray:
        """Flat ``uint8`` view of the buffer."""
        return self._flat

    def view(self, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
        """Typed view (no copy) over this buffer."""
        values = self._flat.view(np.dtype(dtype))
        return values if shape is None else values.reshape(shape)

    def free(self) -> None:
        if self.pinned and self.ptr:
            self._runtime.free_host(self.ptr)
        self.ptr = 0
        self._owner = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.free()
        except Exception:  # noqa: BLE001 - interpreter shutdown order
            pass

    def __repr__(self) -> str:
        return (
            f"HostBuffer(ptr=0x{self.ptr:x}, nbytes={self.nbytes}, "
            f"pinned={self.pinned})"
        )
