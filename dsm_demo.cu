#include "dsm_manager.hpp"
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

__global__ void init_kernel(int *ptr, int base, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        ptr[idx] = base + idx;
    }
}

__global__ void put_kernel(int *src, int n, int target_pe) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        nvshmem_int_p(&src[idx], src[idx], target_pe);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    DSMManager::getInstance().init(MPI_COMM_WORLD);
    int mype = DSMManager::getInstance().my_pe();
    int npes = DSMManager::getInstance().n_pes();

    std::cout << "PE " << mype << " of " << npes << " initialized" << std::endl;

    try {
        if (npes < 2) {
            std::cout << "需要至少 2 个 PE" << std::endl;
            DSMManager::getInstance().finalize();
            MPI_Finalize();
            return 1;
        }

        int *logical_array = (int*)DSMManager::getInstance().dsm_malloc(10 * sizeof(int));
        if (!logical_array) {
            std::cout << "DSM malloc failed" << std::endl;
            DSMManager::getInstance().finalize();
            MPI_Finalize();
            return 1;
        }

        int *physical_array = logical_array;
        cudaPointerAttributes attr;
        cuda_check(cudaPointerGetAttributes(&attr, physical_array), "cudaPointerGetAttributes");
        if (mype == 0) {
            std::cout << "Pointer memory type: "
                      << (attr.type == cudaMemoryTypeDevice ? "DEVICE" : "HOST/OTHER")
                      << std::endl;
        }

        int n = 10;
        int threads = 128;
        int blocks = (n + threads - 1) / threads;
        cudaStream_t stream;
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

        // GPU 上初始化数据
        init_kernel<<<blocks, threads, 0, stream>>>(physical_array, mype * 10, n);
        cuda_check(cudaGetLastError(), "init_kernel launch");
        cuda_check(cudaStreamSynchronize(stream), "init_kernel sync");

        nvshmem_barrier_all();

        if (mype == 1) {
            int host_buf_init[10] = {0};
            cuda_check(cudaMemcpy(host_buf_init, physical_array, n * sizeof(int), cudaMemcpyDeviceToHost), "copy init");
            std::cout << "PE 1: 本地初始化数据: ";
            for (int i = 0; i < n; i++) {
                std::cout << host_buf_init[i] << " ";
            }
            std::cout << std::endl;
        }

        nvshmem_barrier_all();

        if (mype == 0) {
            int target_pe = 1;
            put_kernel<<<blocks, threads, 0, stream>>>(physical_array, n, target_pe);
            nvshmemx_quiet_on_stream(stream);
            cuda_check(cudaGetLastError(), "put_kernel launch");
            cuda_check(cudaStreamSynchronize(stream), "put_kernel sync");
            std::cout << "PE 0: put 数据到 PE 1" << std::endl;
        }

        // All PEs must participate in the device-side barrier.
        nvshmemx_barrier_all_on_stream(stream);
        cuda_check(cudaStreamSynchronize(stream), "barrier sync");
        nvshmem_barrier_all();

        if (mype == 1) {
            int host_buf_put[10] = {0};
            cuda_check(cudaMemcpy(host_buf_put, physical_array, n * sizeof(int), cudaMemcpyDeviceToHost), "copy put");
            std::cout << "PE 1: 远端写入后数据: ";
            for (int i = 0; i < n; i++) {
                std::cout << host_buf_put[i] << " ";
            }
            std::cout << std::endl;
        }

        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        DSMManager::getInstance().dsm_free(logical_array);
        DSMManager::getInstance().finalize();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        DSMManager::getInstance().finalize();
        MPI_Finalize();
        return 1;
    }

    DSMManager::getInstance().finalize();
    MPI_Finalize();
    return 0;
}
