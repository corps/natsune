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


class Node(ctypes.Structure):
    _fields_ = [
        ("task", ctypes.py_object),
        ("next", ctypes.c_int64),
        ("prev", ctypes.c_int64),
    ]


# Assumes x86.
# Pack address (48 bits) + version (16 bits) into 64 bits
TAG_MASK = 0xFFFF000000000000  # Upper 16 bits for version
ADDR_MASK = 0x0000FFFFFFFFFFFF  # Lower 48 bits for address


def pack_addr(address: int, version: int) -> int:
    return (version << 48) | (address & ADDR_MASK)


def unpack_addr(packed: int) -> tuple[int, int]:
    return (packed & ADDR_MASK), (packed >> 48)


class LockFreeDeque[A]:
    """
    Lock-free work-stealing deque using a doubly-linked list.
    **Invariant**:
    - Empty: `head == tail == sentinel`
    - One element: `head == tail == element`
    - Multiple elements: `head` points to first, `tail` points to last
    """

    __slots__ = ("head", "tail", "sentinel", "sentinel_addr")

    head: ctypes.c_int64
    tail: ctypes.c_int64
    sentinel: Node
    sentinel_addr: int

    def __init__(self):
        # tautology indicates empty queue.
        self.sentinel = Node(task=None, next=0, prev=0)
        self.sentinel_addr = ctypes.addressof(self.sentinel)
        self.sentinel.prev = self.sentinel.next = self.sentinel_addr
        self.head = ctypes.c_int64(self.sentinel_addr)
        self.tail = ctypes.c_int64(self.sentinel_addr)

    def push(self, task: A):
        """Push to head (owner operation)."""
        new_node = Node(task=task, next=0, prev=0)
        Py_IncRef(new_node)
        new_node_addr = ctypes.addressof(new_node)

        while True:
            head_addr = atomic_load(self.head)
            head, head_ver = self._deref(head_addr)
            new_node.next = head_addr
            new_node.prev = self.sentinel_addr  # New node becomes the new head

            if cas_ptr(self.head, head_addr, pack_addr(new_node_addr, head_ver + 1)):
                head.prev = new_node_addr  # Link the old head to the new node
                return

    def pop(self) -> A | None:
        """Pop from head (owner operation)."""
        while True:
            head_addr = atomic_load(self.head)
            tail_addr = atomic_load(self.tail)

            if head_addr == tail_addr:
                # Empty or one element
                if head_addr == self.sentinel_addr:
                    return None  # Empty
                # One element: CAS both head and tail to sentinel
                head, head_ver = self._deref(head_addr)
                if cas_ptr(self.head, head_addr, self.sentinel_addr):
                    if cas_ptr(self.tail, tail_addr, self.sentinel_addr):
                        Py_DecRef(head)
                        return head.task
                    else:
                        continue  # Tail changed, retry
            else:
                # Multiple elements: CAS head to next node
                head, head_ver = self._deref(head_addr)
                next_addr = head.next
                if cas_ptr(self.head, head_addr, pack_addr(next_addr, head_ver + 1)):
                    next_node, next_version = self._deref(next_addr)
                    next_node.prev = self.sentinel_addr  # Unlink old head
                    Py_DecRef(head)
                    return head.task

    def steal(self) -> A | None:
        """Steal from tail (thief operation)."""
        while True:
            tail_addr = atomic_load(self.tail)
            head_addr = atomic_load(self.head)

            if tail_addr == head_addr:
                # Empty or one element
                if tail_addr == self.sentinel_addr:
                    return None  # Empty
                # One element: CAS both tail and head to sentinel
                tail, tail_ver = self._deref(tail_addr)
                if cas_ptr(self.tail, tail_addr, self.sentinel_addr):
                    if cas_ptr(self.head, head_addr, self.sentinel_addr):
                        Py_DecRef(tail)
                        return tail.task
                    else:
                        continue  # Head changed, retry
            else:
                # Multiple elements: CAS tail to prev node
                tail, tail_ver = self._deref(tail_addr)
                prev_addr = tail.prev
                if cas_ptr(self.tail, tail_addr, pack_addr(prev_addr, tail_ver + 1)):
                    prev_node, _ = self._deref(prev_addr)
                    prev_node.next = self.sentinel_addr  # Unlink old tail
                    Py_DecRef(tail)
                    return tail.task

    def _deref(self, addr: int) -> tuple[Node, int]:
        if addr == 0:
            raise ValueError("Null pointer dereference")
        addr, version = unpack_addr(addr)
        return ctypes.cast(addr, ctypes.POINTER(Node)).contents, version

    # Approximate only
    def __len__(self) -> int:
        if self.head.value == self.sentinel_addr:
            return 0
        count = 0
        current, _ = self._deref(self.head.value)
        while ctypes.addressof(current) != self.sentinel_addr:
            count += 1
            current, _ = self._deref(current.next)
        return count

    def __del__(self):
        while True:
            head_addr = self.head.value
            tail_addr = self.tail.value
            if head_addr == tail_addr:
                if head_addr == self.sentinel_addr:
                    break
                head, _ = self._deref(head_addr)
                Py_DecRef(head)
                break
            else:
                head, head_ver = self._deref(head_addr)
                next_addr = head.next
                self.head.value = next_addr
                Py_DecRef(head)
