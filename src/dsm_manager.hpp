#ifndef DSM_MANAGER_HPP
#define DSM_MANAGER_HPP

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>

constexpr unsigned long long kDsmAlign = 256ULL;
constexpr unsigned long long kDsmNull = ~0ULL;

struct DSMAllocatorHeader {
    unsigned long long lock;
    unsigned long long pool_size;
    unsigned long long free_head;
    unsigned long long free_bytes;
    unsigned long long used_bytes;
    unsigned long long alloc_count;
    unsigned long long free_count;
};

struct DSMBlockHeader {
    unsigned long long size;
    unsigned long long next;
    unsigned long long prev;
    unsigned long long tag;
};

struct DSMAllocatorStats {
    unsigned long long pool_size;
    unsigned long long free_bytes;
    unsigned long long used_bytes;
    unsigned long long alloc_count;
    unsigned long long free_count;
};

struct DSMDeviceStats {
    unsigned long long pool_size;
    unsigned long long free_bytes;
    unsigned long long used_bytes;
    unsigned long long largest_free;
    unsigned long long free_blocks;
};

__device__ __constant__ DSMAllocatorHeader* g_dsm_alloc_hdr;
__device__ __constant__ unsigned char* g_dsm_pool;

__device__ inline unsigned long long dsm_align_up(unsigned long long v, unsigned long long a) {
    return (v + a - 1ULL) & ~(a - 1ULL);
}

__device__ inline unsigned long long dsm_header_size() {
    return dsm_align_up(sizeof(DSMBlockHeader), kDsmAlign);
}

inline unsigned long long dsm_header_size_host() {
    return ((sizeof(DSMBlockHeader) + kDsmAlign - 1ULL) & ~(kDsmAlign - 1ULL));
}

__device__ inline void dsm_lock() {
    while (atomicCAS(&g_dsm_alloc_hdr->lock, 0ULL, 1ULL) != 0ULL) {
    }
    __threadfence();
}

__device__ inline void dsm_unlock() {
    __threadfence();
    atomicExch(&g_dsm_alloc_hdr->lock, 0ULL);
}

__device__ inline void* dsm_malloc_device(size_t bytes, unsigned long long align = kDsmAlign,
                                          unsigned long long tag = 0ULL) {
    if (!g_dsm_alloc_hdr || !g_dsm_pool) {
        return nullptr;
    }
    if (align < 8ULL) align = 8ULL;
    if (align > kDsmAlign) align = kDsmAlign;
    unsigned long long req = dsm_align_up((unsigned long long)bytes, align);
    unsigned long long hdr_size = dsm_header_size();

    dsm_lock();
    unsigned long long off = g_dsm_alloc_hdr->free_head;
    unsigned long long prev = kDsmNull;
    while (off != kDsmNull) {
        DSMBlockHeader* blk = (DSMBlockHeader*)(g_dsm_pool + off);
        if (blk->size >= req) {
            unsigned long long remaining = blk->size - req;
            if (remaining >= (hdr_size + kDsmAlign)) {
                unsigned long long new_off = off + hdr_size + req;
                DSMBlockHeader* new_blk = (DSMBlockHeader*)(g_dsm_pool + new_off);
                new_blk->size = remaining - hdr_size;
                new_blk->next = blk->next;
                new_blk->prev = blk->prev;
                new_blk->tag = 0ULL;

                if (blk->prev != kDsmNull) {
                    DSMBlockHeader* prev_blk = (DSMBlockHeader*)(g_dsm_pool + blk->prev);
                    prev_blk->next = new_off;
                } else {
                    g_dsm_alloc_hdr->free_head = new_off;
                }
                if (blk->next != kDsmNull) {
                    DSMBlockHeader* next_blk = (DSMBlockHeader*)(g_dsm_pool + blk->next);
                    next_blk->prev = new_off;
                }

                blk->size = req;
            } else {
                if (blk->prev != kDsmNull) {
                    DSMBlockHeader* prev_blk = (DSMBlockHeader*)(g_dsm_pool + blk->prev);
                    prev_blk->next = blk->next;
                } else {
                    g_dsm_alloc_hdr->free_head = blk->next;
                }
                if (blk->next != kDsmNull) {
                    DSMBlockHeader* next_blk = (DSMBlockHeader*)(g_dsm_pool + blk->next);
                    next_blk->prev = blk->prev;
                }
            }

            blk->next = kDsmNull;
            blk->prev = kDsmNull;
            blk->tag = tag;
            g_dsm_alloc_hdr->free_bytes -= blk->size;
            g_dsm_alloc_hdr->used_bytes += blk->size;
            g_dsm_alloc_hdr->alloc_count += 1ULL;
            void* ret = (void*)(g_dsm_pool + off + hdr_size);
            dsm_unlock();
            return ret;
        }

        prev = off;
        off = blk->next;
    }

    dsm_unlock();
    return nullptr;
}

__device__ inline void dsm_free_device(void* ptr) {
    if (!ptr || !g_dsm_alloc_hdr || !g_dsm_pool) return;
    unsigned long long hdr_size = dsm_header_size();
    unsigned long long off = (unsigned long long)((unsigned char*)ptr - g_dsm_pool - hdr_size);
    DSMBlockHeader* blk = (DSMBlockHeader*)(g_dsm_pool + off);
    unsigned long long freed = blk->size;

    dsm_lock();
    g_dsm_alloc_hdr->free_bytes += blk->size;
    g_dsm_alloc_hdr->free_count += 1ULL;

    unsigned long long cur = g_dsm_alloc_hdr->free_head;
    unsigned long long prev = kDsmNull;
    while (cur != kDsmNull && cur < off) {
        prev = cur;
        DSMBlockHeader* cur_blk = (DSMBlockHeader*)(g_dsm_pool + cur);
        cur = cur_blk->next;
    }

    blk->prev = prev;
    blk->next = cur;
    blk->tag = 0ULL;

    if (prev != kDsmNull) {
        DSMBlockHeader* prev_blk = (DSMBlockHeader*)(g_dsm_pool + prev);
        prev_blk->next = off;
    } else {
        g_dsm_alloc_hdr->free_head = off;
    }
    if (cur != kDsmNull) {
        DSMBlockHeader* cur_blk = (DSMBlockHeader*)(g_dsm_pool + cur);
        cur_blk->prev = off;
    }

    // Merge with next
    if (cur != kDsmNull) {
        DSMBlockHeader* next_blk = (DSMBlockHeader*)(g_dsm_pool + cur);
        unsigned long long blk_end = off + hdr_size + blk->size;
        if (blk_end == cur) {
            blk->size += hdr_size + next_blk->size;
            blk->next = next_blk->next;
            if (next_blk->next != kDsmNull) {
                DSMBlockHeader* next_next = (DSMBlockHeader*)(g_dsm_pool + next_blk->next);
                next_next->prev = off;
            }
        }
    }

    // Merge with prev
    if (prev != kDsmNull) {
        DSMBlockHeader* prev_blk = (DSMBlockHeader*)(g_dsm_pool + prev);
        unsigned long long prev_end = prev + hdr_size + prev_blk->size;
        if (prev_end == off) {
            prev_blk->size += hdr_size + blk->size;
            prev_blk->next = blk->next;
            if (blk->next != kDsmNull) {
                DSMBlockHeader* next_blk = (DSMBlockHeader*)(g_dsm_pool + blk->next);
                next_blk->prev = prev;
            }
        }
    }

    g_dsm_alloc_hdr->used_bytes -= freed;

    dsm_unlock();
}

__device__ inline void dsm_mark_device(void* ptr, unsigned long long tag) {
    if (!ptr || !g_dsm_pool) return;
    unsigned long long hdr_size = dsm_header_size();
    unsigned long long off = (unsigned long long)((unsigned char*)ptr - g_dsm_pool - hdr_size);
    DSMBlockHeader* blk = (DSMBlockHeader*)(g_dsm_pool + off);
    blk->tag = tag;
}

__device__ inline void* dsm_malloc_device_batch(size_t count, size_t elem_size, size_t align,
                                                size_t* stride) {
    size_t stride_val = (size_t)dsm_align_up((unsigned long long)elem_size,
                                             (unsigned long long)align);
    if (stride) {
        *stride = stride_val;
    }
    return dsm_malloc_device(count * stride_val, align, 0ULL);
}

__device__ inline void* dsm_ptr_device(void* ptr, int pe) {
    return nvshmem_ptr(ptr, pe);
}

__device__ inline void dsm_putmem_device(void* dest, const void* src, size_t bytes, int pe) {
    nvshmem_putmem(dest, src, bytes, pe);
}

__device__ inline void dsm_getmem_device(void* dest, const void* src, size_t bytes, int pe) {
    nvshmem_getmem(dest, src, bytes, pe);
}

__device__ inline void dsm_put_int_device(int* dest, int value, int pe) {
    nvshmem_int_p(dest, value, pe);
}

__device__ inline int dsm_get_int_device(const int* src, int pe) {
    return nvshmem_int_g(src, pe);
}

__device__ inline int dsm_atomic_add_int(int* dest, int value, int pe) {
    return nvshmem_int_atomic_fetch_add(dest, value, pe);
}

__device__ inline void dsm_stats_device(DSMDeviceStats* out) {
    if (!out || !g_dsm_alloc_hdr || !g_dsm_pool) return;
    unsigned long long free_bytes = 0ULL;
    unsigned long long largest = 0ULL;
    unsigned long long count = 0ULL;
    unsigned long long curr = g_dsm_alloc_hdr->free_head;
    while (curr != kDsmNull) {
        DSMBlockHeader* blk = (DSMBlockHeader*)(g_dsm_pool + curr);
        free_bytes += blk->size;
        if (blk->size > largest) {
            largest = blk->size;
        }
        count += 1ULL;
        curr = blk->next;
    }
    out->pool_size = g_dsm_alloc_hdr->pool_size;
    out->free_bytes = free_bytes;
    out->used_bytes = g_dsm_alloc_hdr->used_bytes;
    out->largest_free = largest;
    out->free_blocks = count;
}

class DSMManager {
public:
    static DSMManager& getInstance() {
        static DSMManager instance;
        return instance;
    }

    void init(MPI_Comm comm, size_t pool_bytes = 1ULL << 26) {
        if (initialized) return;
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        attr.mpi_comm = &comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

        mype = nvshmem_my_pe();
        npes = nvshmem_n_pes();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        cudaSetDevice(mype_node);
        init_allocator(pool_bytes);
        initialized = true;
    }

    void finalize() {
        if (!initialized) return;
        if (alloc_hdr) {
            nvshmem_free(alloc_hdr);
            alloc_hdr = nullptr;
        }
        if (pool) {
            nvshmem_free(pool);
            pool = nullptr;
        }
        nvshmem_finalize();
        initialized = false;
    }

    void* dsm_malloc(size_t bytes) {
        void* ptr = nvshmem_malloc(bytes);
        if (!ptr) {
            throw std::runtime_error("NVSHMEM malloc failed");
        }
        return ptr;
    }

    void dsm_free(void* ptr) {
        nvshmem_free(ptr);
    }

    DSMAllocatorStats get_stats() const {
        DSMAllocatorHeader hdr{};
        cudaMemcpy((void*)&hdr, (void*)alloc_hdr, sizeof(hdr), cudaMemcpyDeviceToHost);
        DSMAllocatorStats stats{};
        stats.pool_size = hdr.pool_size;
        stats.free_bytes = hdr.free_bytes;
        stats.used_bytes = hdr.used_bytes;
        stats.alloc_count = hdr.alloc_count;
        stats.free_count = hdr.free_count;
        return stats;
    }

    int my_pe() const {
        return mype;
    }

    int n_pes() const {
        return npes;
    }

    int my_pe_node() const {
        return mype_node;
    }

private:
    DSMManager() = default;

    void init_allocator(size_t pool_bytes) {
        alloc_hdr = (DSMAllocatorHeader*)nvshmem_malloc(sizeof(DSMAllocatorHeader));
        if (!alloc_hdr) {
            throw std::runtime_error("DSM allocator header allocation failed");
        }
        pool = (unsigned char*)nvshmem_malloc(pool_bytes);
        if (!pool) {
            throw std::runtime_error("DSM pool allocation failed");
        }
        unsigned long long hdr_size = dsm_header_size_host();
        if (pool_bytes <= hdr_size + kDsmAlign) {
            throw std::runtime_error("DSM pool size too small");
        }
        DSMAllocatorHeader init_hdr{};
        init_hdr.lock = 0ULL;
        init_hdr.pool_size = pool_bytes;
        init_hdr.free_head = 0ULL;
        init_hdr.free_bytes = pool_bytes - hdr_size;
        init_hdr.used_bytes = 0ULL;
        init_hdr.alloc_count = 0ULL;
        init_hdr.free_count = 0ULL;

        DSMBlockHeader init_blk{};
        init_blk.size = pool_bytes - hdr_size;
        init_blk.next = kDsmNull;
        init_blk.prev = kDsmNull;
        init_blk.tag = 0ULL;

        cudaMemcpy(pool, &init_blk, sizeof(init_blk), cudaMemcpyHostToDevice);
        cudaMemcpy(alloc_hdr, &init_hdr, sizeof(init_hdr), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(g_dsm_alloc_hdr, &alloc_hdr, sizeof(alloc_hdr));
        cudaMemcpyToSymbol(g_dsm_pool, &pool, sizeof(pool));
    }

    bool initialized = false;
    int mype = 0;
    int npes = 1;
    int mype_node = 0;
    DSMAllocatorHeader* alloc_hdr = nullptr;
    unsigned char* pool = nullptr;
};

#endif // DSM_MANAGER_HPP (src/dsm_manager.hpp)
