"""Tests for :mod:`yolocar.cuda` against a fake ``libcudart`` handle.

``CudaRuntime`` takes the ``ctypes.CDLL`` as an injection point, so the bindings
can be exercised on a machine without a GPU. What is checked here is the part
that used to be silent on hardware: sticky-error handling, stream handle
validation, and the lifetime rules of :class:`HostBuffer`.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from yolocar.cuda import (
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CudaError,
    CudaRuntime,
    DeviceBuffer,
    HostBuffer,
)

_SYMBOLS = (
    "cudaSetDevice",
    "cudaGetDevice",
    "cudaGetLastError",
    "cudaGetErrorString",
    "cudaMalloc",
    "cudaFree",
    "cudaMallocHost",
    "cudaFreeHost",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaStreamCreateWithFlags",
    "cudaStreamDestroy",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
)


class _Function:
    """Callable standing in for a ``ctypes`` function pointer."""

    def __init__(self, lib: "FakeLibrary", name: str) -> None:
        self._lib = lib
        self._name = name
        self.argtypes = None
        self.restype = None

    def __call__(self, *args: object) -> int:
        return self._lib.invoke(self._name, args)


class FakeLibrary:
    """Minimal ``libcudart`` double with fault injection."""

    def __init__(self, *, missing: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.return_codes: dict[str, int] = {}
        #: Values returned by ``cudaGetLastError`` in order (then 0).
        self.sticky_errors: list[int] = []
        self.next_pointer = 0x1000
        self.stream_handle = 0xCAFE
        for name in _SYMBOLS:
            if missing and name in missing:
                continue
            setattr(self, name, _Function(self, name))

    def invoke(self, name: str, args: tuple[object, ...]) -> int:
        self.calls.append(name)
        if name == "cudaGetErrorString":
            return b"fake error"  # type: ignore[return-value]
        if name == "cudaGetLastError":
            return self.sticky_errors.pop(0) if self.sticky_errors else 0
        if name in {"cudaMalloc", "cudaMallocHost"}:
            if self.return_codes.get(name, 0) == 0:
                out = ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))
                self.next_pointer += 0x1000
                out.contents.value = self.next_pointer
        elif name == "cudaStreamCreateWithFlags":
            handle = ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))
            handle.contents.value = self.stream_handle
        return self.return_codes.get(name, 0)


class StubAllocator:
    """Allocation backend for :class:`HostBuffer` / :class:`DeviceBuffer`.

    Returns real numpy-backed addresses, so the buffers built on top of them can
    actually be written to (the point of these tests).
    """

    def __init__(self) -> None:
        self.buffers: dict[int, np.ndarray] = {}
        self.freed: list[int] = []

    def _allocate(self, nbytes: int) -> int:
        memory = np.zeros(int(nbytes), dtype=np.uint8)
        self.buffers[int(memory.ctypes.data)] = memory
        return int(memory.ctypes.data)

    def malloc(self, nbytes: int) -> int:
        return self._allocate(nbytes)

    def malloc_host(self, nbytes: int) -> int:
        return self._allocate(nbytes)

    def free(self, pointer: int) -> None:
        self.freed.append(pointer)
        self.buffers.pop(pointer, None)

    # Host and device allocations share the stub's bookkeeping.
    free_host = free


# --------------------------------------------------------------------- runtime
def test_unknown_symbol_reports_a_useful_error() -> None:
    with pytest.raises(CudaError, match="missing a required symbol"):
        CudaRuntime(FakeLibrary(missing={"cudaStreamCreateWithFlags"}))


def test_stream_create_rejects_a_null_handle() -> None:
    lib = FakeLibrary()
    lib.stream_handle = 0
    runtime = CudaRuntime(lib)
    with pytest.raises(CudaError, match="null stream handle"):
        runtime.stream_create()


def test_stream_synchronize_surfaces_sticky_launch_errors() -> None:
    lib = FakeLibrary()
    runtime = CudaRuntime(lib)
    # First read drains stale state, the second reports the launch failure that
    # happened while the stream was running.
    lib.sticky_errors = [0, 700]

    with pytest.raises(CudaError, match="asynchronous kernel execution"):
        runtime.stream_synchronize(0xCAFE)
    runtime.stream_synchronize(0xCAFE)  # errored once, not sticky forever


def test_stream_synchronize_reports_the_stream_error() -> None:
    lib = FakeLibrary()
    lib.return_codes["cudaStreamSynchronize"] = 719
    runtime = CudaRuntime(lib)
    with pytest.raises(CudaError, match="cudaStreamSynchronize failed"):
        runtime.stream_synchronize(0xCAFE)


def test_malloc_rejects_empty_requests() -> None:
    runtime = CudaRuntime(FakeLibrary())
    with pytest.raises(ValueError):
        runtime.malloc(0)


def test_malloc_reports_null_pointer_from_the_driver() -> None:
    lib = FakeLibrary()

    def _null(*_args: object) -> int:
        lib.calls.append("cudaMalloc")
        return 0  # success code, but the out parameter stays null

    lib.cudaMalloc = _null  # type: ignore[assignment]
    runtime = CudaRuntime(lib)
    with pytest.raises(CudaError, match="null pointer"):
        runtime.malloc(64)


# ------------------------------------------------------------------ host buffer
def test_host_buffer_view_keeps_its_allocator_alive() -> None:
    allocator = StubAllocator()
    buffer = HostBuffer(allocator, 64, pinned=True)
    view = buffer.view(np.float32, (16,))

    # The numpy view must keep the buffer (and therefore the allocation)
    # reachable, otherwise the pageable fallback can be collected under it.
    assert view.base is not None
    owner = getattr(view, "_buffer_owner", None)
    assert owner is buffer

    del buffer
    view[:] = 1.0  # still writable: the allocation is still alive
    assert allocator.buffers, "allocation was collected while a view existed"


def test_host_buffer_release_invalidates_views_and_frees_once() -> None:
    allocator = StubAllocator()
    buffer = HostBuffer(allocator, 64, pinned=True)
    pointer = buffer.ptr

    buffer.free()
    assert buffer.closed
    assert buffer.ptr == 0
    assert allocator.freed == [pointer]
    with pytest.raises(CudaError, match="already been released"):
        _ = buffer.array
    with pytest.raises(CudaError, match="already been released"):
        buffer.view(np.float32)

    buffer.free()  # idempotent: no double cudaFreeHost
    assert allocator.freed == [pointer]


def test_host_buffer_falls_back_to_pageable_memory() -> None:
    allocator = StubAllocator()

    def _fail(_nbytes: int) -> int:
        raise CudaError("cudaMallocHost failed: out of memory")

    allocator.malloc_host = _fail  # type: ignore[assignment]
    buffer = HostBuffer(allocator, 32, pinned=True)
    try:
        assert buffer.pinned is False
        assert buffer.array.dtype == np.uint8
        assert buffer.array.size == 32
        buffer.free()
        assert allocator.freed == []  # nothing pinned to release
    finally:
        buffer.free()


def test_device_buffer_is_idempotent_and_reports_writes() -> None:
    allocator = StubAllocator()
    buffer = DeviceBuffer(allocator, 16)  # type: ignore[arg-type]
    pointer = buffer.ptr
    assert "0x" in repr(buffer)
    buffer.free()
    buffer.free()
    assert allocator.freed == [pointer]


# ------------------------------------------------------------------------ views
def test_flat_view_rejects_null_pointer_and_empty_size() -> None:
    from yolocar.cuda import _flat_view

    with pytest.raises(CudaError, match="null host pointer"):
        _flat_view(0, 16)
    with pytest.raises(ValueError):
        _flat_view(0x1000, 0)


def test_memcpy_async_is_bound_with_the_expected_kind() -> None:
    lib = FakeLibrary()
    runtime = CudaRuntime(lib)
    runtime.memcpy_async(0x10, 0x20, 128, CUDA_MEMCPY_HOST_TO_DEVICE, 0xCAFE)
    assert lib.calls[-1] == "cudaMemcpyAsync"
