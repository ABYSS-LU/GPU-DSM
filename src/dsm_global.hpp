#ifndef DSM_GLOBAL_HPP
#define DSM_GLOBAL_HPP

// A small PGAS-style layer built on NVSHMEM.
//
// Goal:
// - Provide a controllable, single-node friendly programming interface that can
//   later extend to multi-node.
// - Prefer direct mapping (nvshmem_ptr) when available; otherwise fall back to
//   nvshmem get/put/atomics.
//
// Notes:
// - nvshmem_ptr may return nullptr on multi-node or when P2P is not available.
// - The "direct" path is best-effort. For strict ordering/visibility semantics,
//   users should still use NVSHMEM synchronization (quiet/fence/barrier) at the
//   algorithm level.

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

__host__ __device__ inline size_t dsm_ceil_div_size_t(size_t a, size_t b) {
    return (a + b - 1) / b;
}

template <typename T>
struct DSMGlobalPtr {
    T* ptr{nullptr};
    int pe{-1};

    __device__ inline T* try_direct() const {
        return (T*)nvshmem_ptr((void*)ptr, pe);
    }

    __device__ inline T load(int my_pe) const {
        if (pe == my_pe) {
            return *ptr;
        }
        T* direct = try_direct();
        if (direct) {
            return *direct;
        }
        T tmp;
        nvshmem_getmem(&tmp, (const void*)ptr, sizeof(T), pe);
        return tmp;
    }

    __device__ inline void store(const T& value, int my_pe) const {
        if (pe == my_pe) {
            *ptr = value;
            return;
        }
        T* direct = try_direct();
        if (direct) {
            *direct = value;
            // Best-effort visibility for peer/global.
            __threadfence_system();
            return;
        }
        nvshmem_putmem((void*)ptr, (const void*)&value, sizeof(T), pe);
    }
};

// int specialization uses typed NVSHMEM ops and supports atomic add.
template <>
struct DSMGlobalPtr<int> {
    int* ptr{nullptr};
    int pe{-1};

    __device__ inline int* try_direct() const {
        return (int*)nvshmem_ptr((void*)ptr, pe);
    }

    __device__ inline int load(int my_pe) const {
        if (pe == my_pe) {
            return *ptr;
        }
        int* direct = try_direct();
        if (direct) {
            return *direct;
        }
        return nvshmem_int_g(ptr, pe);
    }

    __device__ inline void store(int value, int my_pe) const {
        if (pe == my_pe) {
            *ptr = value;
            return;
        }
        int* direct = try_direct();
        if (direct) {
            *direct = value;
            __threadfence_system();
            return;
        }
        nvshmem_int_p(ptr, value, pe);
    }

    __device__ inline int atomic_add(int value, int my_pe) const {
        if (pe == my_pe) {
            return atomicAdd(ptr, value);
        }
        int* direct = try_direct();
        if (direct) {
            // Best-effort: peer atomics require HW/driver support.
            int old = atomicAdd(direct, value);
            __threadfence_system();
            return old;
        }
        return nvshmem_int_atomic_fetch_add(ptr, value, pe);
    }
};

// Block-cyclic distributed array view.
// base is a symmetric allocation: each PE has local_capacity() elements.
template <typename T>
struct DSMGlobalArray {
    T* base{nullptr};
    size_t global_size{0};
    size_t block_size{1};
    int npes{1};

    __host__ __device__ inline int owner_pe(size_t global_idx) const {
        return (int)((global_idx / block_size) % (size_t)npes);
    }

    __host__ __device__ inline size_t local_index(size_t global_idx) const {
        size_t block = global_idx / block_size;
        size_t in_block = global_idx % block_size;
        size_t cycle = block / (size_t)npes;
        return cycle * block_size + in_block;
    }

    __host__ __device__ inline size_t local_capacity() const {
        size_t global_blocks = dsm_ceil_div_size_t(global_size, block_size);
        size_t blocks_per_pe = dsm_ceil_div_size_t(global_blocks, (size_t)npes);
        return blocks_per_pe * block_size;
    }

    __device__ inline DSMGlobalPtr<T> at(size_t global_idx) const {
        DSMGlobalPtr<T> gp;
        gp.pe = owner_pe(global_idx);
        gp.ptr = base + local_index(global_idx);
        return gp;
    }
};

template <typename T>
inline DSMGlobalArray<T> dsm_alloc_global_array(size_t global_size, size_t block_size) {
    DSMGlobalArray<T> arr{};
    if (block_size == 0) {
        return arr;
    }
    arr.global_size = global_size;
    arr.block_size = block_size;
    arr.npes = nvshmem_n_pes();
    size_t cap = arr.local_capacity();
    arr.base = (T*)nvshmem_malloc(cap * sizeof(T));
    return arr;
}

template <typename T>
inline void dsm_free_global_array(DSMGlobalArray<T>& arr) {
    if (arr.base) {
        nvshmem_free(arr.base);
        arr.base = nullptr;
    }
}

#endif // DSM_GLOBAL_HPP (src/dsm_global.hpp)
