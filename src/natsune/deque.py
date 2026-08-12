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
        try:
            return ctypes.CDLL(ctypes.util.find_library("atomic"))
        except:
            return ctypes.CDLL(ctypes.util.find_library("c"))
    else:
        return ctypes.CDLL(ctypes.util.find_library("atomic"))


libatomic = load_atomic_lib()
ATOMIC_SEQ_CST = ctypes.c_int(5)
NOT_WEAK = ctypes.c_int(0)


def cas_ptr(ptr: ctypes.c_int64, expected: int, desired: int) -> bool:
    expected_val = ctypes.c_int64(expected)
    desired_val = ctypes.c_int64(desired)
    return libatomic.__atomic_compare_exchange_n_8(
        ctypes.byref(ptr),
        ctypes.byref(expected_val),
        desired_val,
        NOT_WEAK,
        ATOMIC_SEQ_CST,
        ATOMIC_SEQ_CST,
    )


def atomic_load(ptr: ctypes.c_int64) -> int:
    val = ctypes.c_int64()
    libatomic.__atomic_load_8(ctypes.byref(ptr), ctypes.byref(val), ATOMIC_SEQ_CST)
    return val.value


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
        head_packed = atomic_load(self.head)
        head_addr, _ = unpack_addr(head_packed)
        if head_addr == self.sentinel_addr:
            return 0
        count = 0
        current_packed = head_packed
        while True:
            current, _ = self._deref(current_packed)
            count += 1
            current_packed = current.next
            next_addr, _ = unpack_addr(current_packed)
            if next_addr == self.sentinel_addr:
                break
        return count

    def __del__(self):
        current_packed = atomic_load(self.head)
        while True:
            current_addr, _ = unpack_addr(current_packed)
            if current_addr == self.sentinel_addr:
                break
            current, _ = self._deref(current_packed)
            next_packed = current.next
            Py_DecRef(current)
            current_packed = next_packed
