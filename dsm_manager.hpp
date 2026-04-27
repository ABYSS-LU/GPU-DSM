#ifndef DSM_MANAGER_HPP
#define DSM_MANAGER_HPP

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <stdexcept>

class DSMManager {
public:
    static DSMManager& getInstance() {
        static DSMManager instance;
        return instance;
    }

    void init(MPI_Comm comm) {
        if (initialized) return;
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        attr.mpi_comm = &comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

        mype = nvshmem_my_pe();
        npes = nvshmem_n_pes();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        cudaSetDevice(mype_node);
        initialized = true;
    }

    void finalize() {
        if (!initialized) return;
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

    bool initialized = false;
    int mype = 0;
    int npes = 1;
    int mype_node = 0;
};

#endif // DSM_MANAGER_HPP
