#include "dsm_manager.hpp" // examples/global_array_demo.cu
#include "dsm_global.hpp"

#include <iostream>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

__global__ void init_local_segment_kernel(int* base, size_t n, int mype) {
    size_t idx = (size_t)threadIdx.x + (size_t)blockIdx.x * (size_t)blockDim.x;
    if (idx < n) {
        base[idx] = mype * 100000 + (int)idx;
    }
}

__global__ void write_global_indices_kernel(DSMGlobalArray<int> arr, size_t idx0, int v0,
                                            size_t idx1, int v1, int my_pe) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        arr.at(idx0).store(v0, my_pe);
        arr.at(idx1).store(v1, my_pe);
    }
}

__global__ void read_global_indices_kernel(DSMGlobalArray<int> arr, size_t idx0, size_t idx1,
                                           int* out, int my_pe) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = arr.at(idx0).load(my_pe);
        out[1] = arr.at(idx1).load(my_pe);
    }
}

int main(int argc, char **argv) {
    (void)argv;
    MPI_Init(&argc, &argv);

    DSMManager::getInstance().init(MPI_COMM_WORLD);
    int mype = DSMManager::getInstance().my_pe();
    int npes = DSMManager::getInstance().n_pes();

    std::cout << "[global] PE " << mype << " of " << npes << " initialized" << std::endl;

    try {
        if (npes < 2) {
            std::cout << "需要至少 2 个 PE" << std::endl;
            DSMManager::getInstance().finalize();
            MPI_Finalize();
            return 1;
        }

        cudaStream_t stream;
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

        // GlobalArray demo (single-node fast path; multi-node will fallback)
        size_t global_n = 1024;
        size_t block_size = 128;
        DSMGlobalArray<int> garr = dsm_alloc_global_array<int>(global_n, block_size);
        if (!garr.base) {
            throw std::runtime_error("dsm_alloc_global_array failed");
        }

        // Initialize local segment (each PE only writes its local storage).
        size_t local_cap = garr.local_capacity();
        int init_threads = 256;
        int init_blocks = (int)((local_cap + (size_t)init_threads - 1) / (size_t)init_threads);
        init_local_segment_kernel<<<init_blocks, init_threads, 0, stream>>>(garr.base, local_cap, mype);
        cuda_check(cudaGetLastError(), "init_local_segment_kernel launch");
        cuda_check(cudaStreamSynchronize(stream), "init_local_segment_kernel sync");

        nvshmem_barrier_all();

        // Pick two global indices:
        // - idx_remote: owned by PE1 (if npes>=2)
        // - idx_local:  owned by PE0
        size_t idx_local = 0;
        size_t idx_remote = block_size; // owner_pe = 1 when npes>=2

        if (mype == 0) {
            write_global_indices_kernel<<<1, 1, 0, stream>>>(garr, idx_remote, 777, idx_local, 555, mype);
            cuda_check(cudaGetLastError(), "write_global_indices_kernel launch");
            // Ensure NVSHMEM put path completes (direct path uses threadfence_system).
            nvshmemx_quiet_on_stream(stream);
            cuda_check(cudaStreamSynchronize(stream), "write_global_indices_kernel sync");
            std::cout << "[global] PE 0: 写入 idx_remote=" << idx_remote << " idx_local=" << idx_local
                      << std::endl;
        }

        nvshmemx_barrier_all_on_stream(stream);
        cuda_check(cudaStreamSynchronize(stream), "barrier sync");
        nvshmem_barrier_all();

        int* d_out = nullptr;
        cuda_check(cudaMalloc(&d_out, 2 * sizeof(int)), "cudaMalloc d_out");
        if (mype == 1) {
            read_global_indices_kernel<<<1, 1, 0, stream>>>(garr, idx_remote, idx_local, d_out, mype);
            cuda_check(cudaGetLastError(), "read_global_indices_kernel launch");
            cuda_check(cudaStreamSynchronize(stream), "read_global_indices_kernel sync");
            int h_out[2] = {0, 0};
            cuda_check(cudaMemcpy(h_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost),
                       "copy out");
            std::cout << "[global] PE 1: load idx_remote=" << idx_remote << " => " << h_out[0]
                      << ", idx_local=" << idx_local << " => " << h_out[1] << std::endl;
        }
        cuda_check(cudaFree(d_out), "cudaFree d_out");
        dsm_free_global_array(garr);

        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        DSMManager::getInstance().finalize();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        DSMManager::getInstance().finalize();
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
