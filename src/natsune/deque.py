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
ATOMIC_SEQ_CST = ctypes.c_int(5)
NOT_WEAK = ctypes.c_int(0)

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
    """Compare and swap 64-bit pointer atomically."""
    expected_val = ctypes.c_uint64(expected)
    desired_val = ctypes.c_uint64(desired)
    # cas_ptr takes a c_int64 (which is a value), we need to get its address
    ptr_addr = ctypes.addressof(ptr)
    ptr_ref = ctypes.cast(ptr_addr, ctypes.POINTER(ctypes.c_uint64))
    return libatomic.__atomic_compare_exchange_n_8(
        ptr_ref,
        ctypes.byref(expected_val),
        desired_val,
        False,  # not weak
        ATOMIC_SEQ_CST.value,
        ATOMIC_SEQ_CST.value,
    )


def atomic_load(ptr: ctypes.c_int64) -> int:
    """Load 64-bit value atomically."""
    ptr_addr = ctypes.addressof(ptr)
    ptr_ref = ctypes.cast(ptr_addr, ctypes.POINTER(ctypes.c_uint64))
    return libatomic.__atomic_load_8(ptr_ref, ATOMIC_SEQ_CST.value)


def atomic_store(ptr: ctypes.c_int64, value: int) -> None:
    """Store 64-bit value atomically."""
    ptr_addr = ctypes.addressof(ptr)
    ptr_ref = ctypes.cast(ptr_addr, ctypes.POINTER(ctypes.c_uint64))
    libatomic.__atomic_store_8(ptr_ref, ctypes.c_uint64(value), ATOMIC_SEQ_CST.value)


# Pack address (48 bits) + version (16 bits) into 64 bits
TAG_MASK = 0xFFFF000000000000  # Upper 16 bits for version
ADDR_MASK = 0x0000FFFFFFFFFFFF  # Lower 48 bits for address


def pack_addr(address: int, version: int) -> int:
    return (version << 48) | (address & ADDR_MASK)


def unpack_addr(packed: int) -> tuple[int, int]:
    return (packed & ADDR_MASK), (packed >> 48)


class Node:
    __slots__ = ("task", "next", "prev")

    def __init__(self, task, next_packed: int = 0, prev_packed: int = 0):
        self.task = task
        self.next = next_packed  # Packed address
        self.prev = prev_packed  # Packed address


class IdleCounter:
    __slots__ = ("_value", "total")

    def __init__(self, total: int):
        self._value = ctypes.c_int64(0)
        self.total = total

    def signal(self, worker_id: int) -> bool:
        return False

    def reset(self) -> None:
        pass


class LockFreeDeque[A]:
    __slots__ = ("head", "tail", "sentinel", "sentinel_addr")

    def __init__(self):
        self.sentinel = Node(task=None)
        self.sentinel_addr = id(self.sentinel) & ADDR_MASK
        sentinel_packed = pack_addr(self.sentinel_addr, 0)
        self.sentinel.next = sentinel_packed
        self.sentinel.prev = sentinel_packed
        self.head = ctypes.c_int64(sentinel_packed)
        self.tail = ctypes.c_int64(sentinel_packed)

    def _deref(self, packed: int) -> tuple[Node, int]:
        if packed == 0:
            raise ValueError("Null pointer dereference")
        addr, version = unpack_addr(packed)
        node_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.py_object))
        node = node_ptr.contents.value
        return node, version

    def push(self, task: A):
        new_node = Node(task=task)
        new_node_addr = id(new_node) & ADDR_MASK
        Py_IncRef(new_node)

        while True:
            head_packed = atomic_load(self.head)
            head_addr, head_ver = unpack_addr(head_packed)

            new_node.next = head_packed
            new_node.prev = pack_addr(self.sentinel_addr, 0)

            if cas_ptr(self.head, head_packed, pack_addr(new_node_addr, head_ver + 1)):
                if head_addr != self.sentinel_addr:
                    head_node, _ = self._deref(head_packed)
                    head_node.prev = pack_addr(new_node_addr, head_ver + 1)
                return

    def pop(self) -> A | None:
        while True:
            head_packed = atomic_load(self.head)
            tail_packed = atomic_load(self.tail)

            head_addr, head_ver = unpack_addr(head_packed)
            tail_addr, tail_ver = unpack_addr(tail_packed)

            if head_addr == tail_addr:
                if head_addr == self.sentinel_addr:
                    return None  # Empty
                head_node, _ = self._deref(head_packed)
                task = head_node.task  # Read BEFORE DecRef

                new_head = pack_addr(self.sentinel_addr, head_ver + 1)
                if cas_ptr(self.head, head_packed, new_head):
                    new_tail = pack_addr(self.sentinel_addr, tail_ver + 1)
                    if cas_ptr(self.tail, tail_packed, new_tail):
                        Py_DecRef(head_node)
                        return task
            else:
                head_node, _ = self._deref(head_packed)
                task = head_node.task  # Read BEFORE DecRef
                next_packed = head_node.next
                next_addr, _ = unpack_addr(next_packed)

                if cas_ptr(self.head, head_packed, pack_addr(next_addr, head_ver + 1)):
                    next_node, _ = self._deref(next_packed)
                    next_node.prev = pack_addr(self.sentinel_addr, head_ver + 1)
                    Py_DecRef(head_node)
                    return task

    def steal(self) -> A | None:
        while True:
            tail_packed = atomic_load(self.tail)
            head_packed = atomic_load(self.head)

            tail_addr, tail_ver = unpack_addr(tail_packed)
            head_addr, head_ver = unpack_addr(head_packed)

            if tail_addr == head_addr:
                if tail_addr == self.sentinel_addr:
                    return None  # Empty
                tail_node, _ = self._deref(tail_packed)
                task = tail_node.task  # Read BEFORE DecRef

                new_tail = pack_addr(self.sentinel_addr, tail_ver + 1)
                if cas_ptr(self.tail, tail_packed, new_tail):
                    new_head = pack_addr(self.sentinel_addr, head_ver + 1)
                    if cas_ptr(self.head, head_packed, new_head):
                        Py_DecRef(tail_node)
                        return task
            else:
                tail_node, _ = self._deref(tail_packed)
                task = tail_node.task  # Read BEFORE DecRef
                prev_packed = tail_node.prev
                prev_addr, _ = unpack_addr(prev_packed)

                if cas_ptr(self.tail, tail_packed, pack_addr(prev_addr, tail_ver + 1)):
                    prev_node, _ = self._deref(prev_packed)
                    prev_node.next = pack_addr(self.sentinel_addr, tail_ver + 1)
                    Py_DecRef(tail_node)
                    return task

    # Approximate length
    def __len__(self) -> int:
        count = 0
        current_packed = atomic_load(self.head)
        while current_packed != self.sentinel_addr:
            current, _ = self._deref(current_packed)
            count += 1
            current_packed = current.next
        return count

    #
    # def __del__(self):
    #     current_packed = atomic_load(self.head)
    #     while True:
    #         current_addr, _ = unpack_addr(current_packed)
    #         if current_addr == self.sentinel_addr:
    #             break
    #         current, _ = self._deref(current_packed)
    #         next_packed = current.next
    #         Py_DecRef(current)
    #         current_packed = next_packed
