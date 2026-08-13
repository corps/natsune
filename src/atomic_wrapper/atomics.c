// Atomic wrapper for macOS
// Provides GCC libatomic-compatible symbols using clang's __atomic builtins

#include <stdbool.h>
#include <stdint.h>

// Memory order constants (matching GCC libatomic)
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5

// Load 8 bytes atomically
uint64_t __atomic_load_8(uint64_t *ptr, int memorder) {
    return __atomic_load_n(ptr, memorder);
}

// Store 8 bytes atomically
void __atomic_store_8(uint64_t *ptr, uint64_t val, int memorder) {
    __atomic_store_n(ptr, val, memorder);
}

// Compare and exchange 8 bytes atomically
bool __atomic_compare_exchange_n_8(uint64_t *ptr, uint64_t *expected, uint64_t desired, bool weak, int success_memorder, int failure_memorder) {
    return __atomic_compare_exchange_n(ptr, expected, desired, weak, success_memorder, failure_memorder);
}
