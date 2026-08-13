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
    expected_val = ctypes.c_uint64(expected)
    desired_val = ctypes.c_uint64(desired)
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
    ptr_addr = ctypes.addressof(ptr)
    ptr_ref = ctypes.cast(ptr_addr, ctypes.POINTER(ctypes.c_uint64))
    return libatomic.__atomic_load_8(ptr_ref, ATOMIC_SEQ_CST.value)


def atomic_store(ptr: ctypes.c_int64, value: int) -> None:
    ptr_addr = ctypes.addressof(ptr)
    ptr_ref = ctypes.cast(ptr_addr, ctypes.POINTER(ctypes.c_uint64))
    libatomic.__atomic_store_8(ptr_ref, ctypes.c_uint64(value), ATOMIC_SEQ_CST.value)


TAG_MASK = 0xFFFF000000000000
ADDR_MASK = 0x0000FFFFFFFFFFFF


def pack_addr(address: int, version: int) -> int:
    return (version << 48) | (address & ADDR_MASK)


def unpack_addr(packed: int) -> tuple[int, int]:
    return (packed & ADDR_MASK), (packed >> 48)


# We use a python object not a cstruct because we actually /do/ want CPYthon to maintain
# GC ownership of these nodes (implicitly the tasks as well).  This prevents us from
# having to create an ownership model directly inside our deque.
class Node:
    __slots__ = ("task", "next", "prev")

    def __init__(self, task, next_addr: int = 0, prev_addr: int = 0):
        self.task = task
        # These are unpacked because we aren't executing any CAS operations on them,
        # it is safe for them to be raw virtual address pointers.
        self.next = next_addr
        self.prev = prev_addr


# Idle counter is a concurrent way to detect when all workers have reached a specific terminal state (idle).
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
        self.sentinel.next = self.sentinel_addr
        self.sentinel.prev = self.sentinel_addr
        # We don't need to pack here -- it is already implicitly packed by the ADDR_MASK
        # and the version is 0 so....
        self.head = ctypes.c_int64(self.sentinel_addr)
        self.tail = ctypes.c_int64(self.sentinel_addr)

    def _deref(self, unpacked: int) -> Node:
        if unpacked == 0:
            raise ValueError("Null pointer dereference")
        node_ptr = ctypes.cast(ctypes.c_void_p(unpacked), ctypes.py_object)
        node = node_ptr.value
        return node

    def push(self, task: A):
        new_node = Node(task=task)
        new_node_addr = id(new_node) & ADDR_MASK

        while True:
            head_packed = atomic_load(self.head)
            head_addr, head_ver = unpack_addr(head_packed)

            tail_packed = atomic_load(self.tail)
            tail_addr, tail_ver = unpack_addr(tail_packed)

            new_node.next = head_addr
            new_node.prev = self.sentinel_addr

            if cas_ptr(self.head, head_packed, pack_addr(new_node_addr, head_ver + 1)):
                # yes -- we are storing this node into the linked list with implied GC ownership
                Py_IncRef(new_node)

                if head_addr != self.sentinel_addr:
                    head_node = self._deref(head_addr)
                    head_node.prev = pack_addr(new_node_addr, head_ver + 1)

                if tail_addr == self.sentinel_addr:
                    # This is safe -- tail_addr == sentinel_addr is a terminal state for that process, so it cannot
                    # meaningfully concurrently modify this.  We don't want to use CAS because there is no coherent
                    # recovery from being unable to set the second pointer.
                    atomic_store(self.tail, pack_addr(new_node_addr, tail_ver + 1))

                return

    def pop(self) -> A | None:
        while True:
            head_packed = atomic_load(self.head)
            tail_packed = atomic_load(self.tail)

            head_addr, head_ver = unpack_addr(head_packed)
            tail_addr, tail_ver = unpack_addr(tail_packed)

            if head_addr == tail_addr:
                if head_addr == self.sentinel_addr:
                    return None
                head_node = self._deref(head_addr)
                task = head_node.task

                new_head = pack_addr(self.sentinel_addr, head_ver + 1)
                if cas_ptr(self.head, head_packed, new_head):
                    new_tail = pack_addr(self.sentinel_addr, tail_ver + 1)
                    if cas_ptr(self.tail, tail_packed, new_tail):
                        Py_DecRef(head_node)
                        return task
            else:
                head_node = self._deref(head_addr)
                task = head_node.task
                next_packed = head_node.next
                next_addr, _ = unpack_addr(next_packed)

                if cas_ptr(self.head, head_packed, pack_addr(next_addr, head_ver + 1)):
                    next_node = self._deref(next_addr)
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
                    return None
                tail_node = self._deref(tail_addr)
                task = tail_node.task

                new_tail = pack_addr(self.sentinel_addr, tail_ver + 1)
                if cas_ptr(self.tail, tail_packed, new_tail):
                    new_head = pack_addr(self.sentinel_addr, head_ver + 1)
                    if cas_ptr(self.head, head_packed, new_head):
                        Py_DecRef(tail_node)
                        return task
            else:
                tail_node = self._deref(tail_addr)
                task = tail_node.task
                prev_packed = tail_node.prev
                prev_addr, _ = unpack_addr(prev_packed)

                if cas_ptr(self.tail, tail_packed, pack_addr(prev_addr, tail_ver + 1)):
                    prev_node = self._deref(prev_addr)
                    prev_node.next = pack_addr(self.sentinel_addr, tail_ver + 1)
                    Py_DecRef(tail_node)
                    return task

    # Approximate length
    def __len__(self) -> int:
        count = 0
        current_packed = atomic_load(self.head)
        current_addr, _ = unpack_addr(current_packed)
        while current_addr != self.sentinel_addr:
            current = self._deref(current_addr)
            count += 1
            current_packed = current.next
            current_addr, _ = unpack_addr(current_packed)
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
