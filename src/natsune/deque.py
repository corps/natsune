from typing import Literal

import ctypes
import ctypes.util
import sys
from ctypes import pythonapi

# This is a lock free dequeue with work stealing implementation that makes many... reasonable assumptions that
# will break in the future.
# 1.  Assumes CPython -- hoping that new versions of python provide better atomic memory primitives and CAS directly.
# 2.  Assumes x86 so that the virtual memspace is actually 48bits at most
#     This is double whammy because the version tagging is limited to 16bits.  Likely plenty, but in theory...
#     Ideally, again, as python removes the GIL is just makes this easier to do for everyone if true threads become
#     the new norm.
# 3.  id returns a stable virtual address and does not move.

# In practice this likely should become a C module, but I'd like to wait for GIL less python to become more stable first.

Py_IncRef = pythonapi.Py_IncRef
Py_IncRef.argtypes = [ctypes.py_object]
Py_IncRef.restype = None

Py_DecRef = pythonapi.Py_DecRef
Py_DecRef.argtypes = [ctypes.py_object]
Py_DecRef.restype = None


def load_atomic_lib():
    if sys.platform == "darwin":
        # Try to load our wrapper library from the package directory
        try:
            lib_path = ctypes.util.find_library("atomic_wrapper")
            if lib_path:
                return ctypes.CDLL(lib_path)
        except:
            pass

        # Try to load from the package's lib/darwin directory
        try:
            import os

            lib_dir = os.path.join(os.path.dirname(__file__), "lib", "darwin")
            lib_path = os.path.join(lib_dir, "libatomic_wrapper.dylib")
            if os.path.exists(lib_path):
                return ctypes.CDLL(lib_path)
        except:
            pass

        raise RuntimeError("Failed to load atomic library")
    else:
        return ctypes.CDLL(ctypes.util.find_library("atomic"))


libatomic = load_atomic_lib()
ATOMIC_SEQ_RELAXED = 0
ATOMIC_SEQ_ACQUIRE = 2
ATOMIC_SEQ_RELEASE = 3
ATOMIC_SEQ_CST = 5
NOT_WEAK = 0

# Set up the atomic function prototypes
libatomic.__atomic_load_8.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
libatomic.__atomic_load_8.restype = ctypes.c_uint64

libatomic.__atomic_store_8.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint64,
    ctypes.c_int,
]
libatomic.__atomic_store_8.restype = None

libatomic.__atomic_compare_exchange_n_8.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint64,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
libatomic.__atomic_compare_exchange_n_8.restype = ctypes.c_bool


def cas_ptr(ptr: ctypes.c_int64, expected: int, desired: int) -> bool:
    expected_val = ctypes.c_uint64(expected)
    desired_val = ctypes.c_uint64(desired)
    ptr_addr = ctypes.addressof(ptr)
    ptr_ref = ctypes.cast(ptr_addr, ctypes.POINTER(ctypes.c_uint64))
    return libatomic.__atomic_compare_exchange_n_8(
        ptr_ref,
        ctypes.byref(expected_val),
        desired_val,
        False,  # not weak
        ATOMIC_SEQ_CST,
        ATOMIC_SEQ_CST,
    )


def atomic_load(ptr: ctypes.c_int64, mode: int = ATOMIC_SEQ_CST) -> int:
    ptr_addr = ctypes.addressof(ptr)
    ptr_ref = ctypes.cast(ptr_addr, ctypes.POINTER(ctypes.c_uint64))
    return libatomic.__atomic_load_8(ptr_ref, mode)


def atomic_store(ptr: ctypes.c_int64, value: int, mode: int = ATOMIC_SEQ_CST) -> None:
    ptr_addr = ctypes.addressof(ptr)
    ptr_ref = ctypes.cast(ptr_addr, ctypes.POINTER(ctypes.c_uint64))
    libatomic.__atomic_store_8(ptr_ref, ctypes.c_uint64(value), mode)


# Idle counter is a concurrent way to detect when all workers have reached a specific terminal state (idle).
class IdleCounter:
    __slots__ = ("_value", "total")

    def __init__(self, total: int):
        self._value = ctypes.c_int64(0)
        self.total = total

    def signal(self, worker_id: int) -> bool:
        cas_ptr(self._value, worker_id, worker_id + 1)
        # This can only happen when all workers have signaled idle consecutively, since any
        # other state involves a reset.
        return atomic_load(self._value) == self.total

    # If this is called, it implies that not all workers are truly idle.
    def reset(self) -> None:
        atomic_store(self._value, 0)


class LockFreeDeque[A]:
    __slots__ = (
        "left",
        "right",
        "store",
        "capacity",
        "max_reentrant_push",
        "wrap_mask",
    )

    store: list[A | None]
    left: ctypes.c_int64
    right: ctypes.c_int64
    max_reentrant_push: int
    capacity: int
    wrap_mask: int

    def __init__(self, storage_hint: int, max_reentrant_push: int):
        if storage_hint < max_reentrant_push or max_reentrant_push < 1:
            raise ValueError(
                "storage_hint must be greater than max_reentrant_push and max_reentrant_push must be greater than 0"
            )

        storage_hint -= 1
        for i in (1, 2, 4, 8, 16, 32):
            storage_hint |= storage_hint >> i
        self.capacity = storage_hint + 1
        self.wrap_mask = storage_hint

        self.store = [None] * self.capacity
        # Thieve side
        self.left = ctypes.c_int64(0)
        # Owner side
        self.right = ctypes.c_int64(0)
        self.max_reentrant_push = max_reentrant_push

    def push(self, task: A) -> None:
        cur_left = atomic_load(self.left)
        cur_right = atomic_load(self.right)

        # Workers should avoid this case as much as possible by avoiding pop or steal when
        # the capacity to push does not exist.
        assert (
            cur_right - cur_left < self.capacity
        ), "Re-entrant push greater than expected capacity"

        atomic_store(self.right, cur_right + 1)
        self.store[cur_right & self.wrap_mask] = task

    def pop(self) -> A | None | Literal[0]:
        cur_left = atomic_load(self.left)
        cur_right = atomic_load(self.right)

        if cur_right <= cur_left:
            return None

        # Don't pop work that we won't be able to complete with potential additional push
        if cur_right - cur_left - 1 >= self.capacity - self.max_reentrant_push:
            return 0

        if cur_right - cur_left == 1:
            if cas_ptr(self.left, cur_left, cur_left + 1):
                return self.store[cur_left & self.wrap_mask]
            return None

        atomic_store(self.right, cur_right - 1)
        return self.store[(cur_right - 1) & self.wrap_mask]

    def steal(self) -> A | None:
        cur_left = atomic_load(self.left)
        cur_right = atomic_load(self.right)

        if cur_right <= cur_left:
            return None

        if cas_ptr(self.left, cur_left, cur_left + 1):
            return self.store[cur_left & self.wrap_mask]
        return None
