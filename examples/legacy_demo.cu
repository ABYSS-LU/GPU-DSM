#include "dsm_manager.hpp" // examples/legacy_demo.cu

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

__device__ int* g_dsm_ptr;

__global__ void alloc_init_kernel(int base, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_dsm_ptr = (int*)dsm_malloc_device(n * sizeof(int));
    }
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && g_dsm_ptr) {
        g_dsm_ptr[idx] = base + idx;
    }
}

__global__ void put_kernel(int n, int target_pe) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && g_dsm_ptr) {
        dsm_put_int_device(&g_dsm_ptr[idx], g_dsm_ptr[idx], target_pe);
    }
}

int main(int argc, char **argv) {
    (void)argv;
    MPI_Init(&argc, &argv);

    DSMManager::getInstance().init(MPI_COMM_WORLD);
    int mype = DSMManager::getInstance().my_pe();
    int npes = DSMManager::getInstance().n_pes();

    std::cout << "[legacy] PE " << mype << " of " << npes << " initialized" << std::endl;

    try {
        if (npes < 2) {
            std::cout << "需要至少 2 个 PE" << std::endl;
            DSMManager::getInstance().finalize();
            MPI_Finalize();
            return 1;
        }

        int n = 10;
        int threads = 128;
        int blocks = (n + threads - 1) / threads;
        cudaStream_t stream;
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

        alloc_init_kernel<<<blocks, threads, 0, stream>>>(mype * 10, n);
        cuda_check(cudaGetLastError(), "alloc_init_kernel launch");
        cuda_check(cudaStreamSynchronize(stream), "alloc_init_kernel sync");

        nvshmem_barrier_all();

        if (mype == 1) {
            int host_buf_init[10] = {0};
            int* ptr = nullptr;
            cuda_check(cudaMemcpyFromSymbol(&ptr, g_dsm_ptr, sizeof(ptr)), "get g_dsm_ptr");
            cuda_check(cudaMemcpy(host_buf_init, ptr, n * sizeof(int), cudaMemcpyDeviceToHost), "copy init");
            std::cout << "[legacy] PE 1: 本地初始化数据: ";
            for (int i = 0; i < n; i++) {
                std::cout << host_buf_init[i] << " ";
            }
            std::cout << std::endl;
        }

        nvshmem_barrier_all();

        if (mype == 0) {
            int target_pe = 1;
            put_kernel<<<blocks, threads, 0, stream>>>(n, target_pe);
            nvshmemx_quiet_on_stream(stream);
            cuda_check(cudaGetLastError(), "put_kernel launch");
            cuda_check(cudaStreamSynchronize(stream), "put_kernel sync");
            std::cout << "[legacy] PE 0: put 数据到 PE 1" << std::endl;
        }

        nvshmemx_barrier_all_on_stream(stream);
        cuda_check(cudaStreamSynchronize(stream), "barrier sync");
        nvshmem_barrier_all();

        if (mype == 1) {
            int host_buf_put[10] = {0};
            int* ptr = nullptr;
            cuda_check(cudaMemcpyFromSymbol(&ptr, g_dsm_ptr, sizeof(ptr)), "get g_dsm_ptr");
            cuda_check(cudaMemcpy(host_buf_put, ptr, n * sizeof(int), cudaMemcpyDeviceToHost), "copy put");
            std::cout << "[legacy] PE 1: 远端写入后数据: ";
            for (int i = 0; i < n; i++) {
                std::cout << host_buf_put[i] << " ";
            }
            std::cout << std::endl;
        }

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
