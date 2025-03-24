/*%****************************************************************************80
%  Code:
%   ncclBcast.cu
%
%  Purpose:
%   Implements sample BROADCAST code using the package NCCL (ncclBcast).
%   Using 'Multiples Devices per Thread'.
%   The code multiple the vector position per 2 on GPUs.
%
%  Modified:
%   Aug 17 2020 10:57
%
%  Author:
%    Murilo Boratto <murilo.boratto 'at' fieb.org.br>
%
%  How to Compile:
%   nvcc ncclBcast.cu -o ncclBcast -lnccl
%
%  Execute:
%   ./ncclBcast
%
%****************************************************************************80*/

#include <nccl.h>

#include <cstdio>
#include <cstdlib>

__global__ void kernel(int *a)
{
    int index = threadIdx.x;

    a[index] *= 2;
    printf("%d\t", a[index]);

} /*kernel*/

void print_vector(int *in, int n)
{
    for (int i = 0; i < n; i++) printf("%d\t", in[i]);

    printf("\n");

} /*print_vector*/

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (myrank == 0) cerr << "Nprocs: " << nprocs << endl;

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;
    int num = 10;
    // get NCCL unique ID at rank 0 and broadcast it to all others
    if (myrank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // picking a GPU based on localRank, allocate device buffers
    CUDA_CHECK(cudaSetDevice(myrank));
    CUDA_CHECK(cudaMalloc(&sendbuff, num * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recvbuff, num * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&s));

    // initializing NCCL
    NCCL_CHECK(ncclCommInitRank(&comm, nprocs, id, myrank));

    // communicating using NCCL
    NCCL_CHECK(ncclAllReduce((const void *)sendbuff, (void *)recvbuff, num, ncclFloat, ncclSum, comm, s));

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDA_CHECK(cudaStreamSynchronize(s));

    // free device buffers
    CUDA_CHECK(cudaFree(sendbuff));
    CUDA_CHECK(cudaFree(recvbuff));

    return 0;

} /*main*/
