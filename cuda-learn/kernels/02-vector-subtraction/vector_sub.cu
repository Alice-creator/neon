#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_subtract(const float* alpha, const float* beta, float* result, int max_length){
    //                                                                  ↑ không có const — result là output, cần ghi vào
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //                               ↑ blockDimx → blockDim.x

    if(index < max_length){
        result[index] = alpha[index] - beta[index];
    }
}

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main() {
    int max_length = 1 << 24;
    size_t bytes = max_length * sizeof(float);

    float* host_alpha  = (float*)malloc(bytes);
    float* host_beta   = (float*)malloc(bytes);
    float* host_result = (float*)malloc(bytes);

    // Fill input data
    for (int i = 0; i < max_length; i++) {
        host_alpha[i] = (float)i;
        host_beta[i]  = (float)(i / 2);
    }

    float *device_alpha, *device_beta, *device_result;
    CUDA_CHECK(cudaMalloc(&device_alpha, bytes));
    CUDA_CHECK(cudaMalloc(&device_beta, bytes));
    CUDA_CHECK(cudaMalloc(&device_result, bytes));

    // Copy CPU to GPU
    CUDA_CHECK(cudaMemcpy(device_alpha, host_alpha, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_beta, host_beta, bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks = (max_length + threads_per_block - 1) / threads_per_block;

    vector_subtract<<<blocks, threads_per_block>>>(device_alpha, device_beta, device_result, max_length);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy GPU → CPU
    CUDA_CHECK(cudaMemcpy(host_result, device_result, bytes, cudaMemcpyDeviceToHost));

    // Verify
    int errors = 0;
    for (int i = 0; i < max_length; i++) {
        if (host_result[i] != host_alpha[i] - host_beta[i]) errors++;
    }
    printf("n      = %d\n", max_length);
    printf("errors = %d\n", errors);
    printf("result[0]   = %.0f (expected %.0f)\n", host_result[0],   host_alpha[0]   - host_beta[0]);
    printf("result[100] = %.0f (expected %.0f)\n", host_result[100], host_alpha[100] - host_beta[100]);

    // Free
    cudaFree(device_alpha); cudaFree(device_beta); cudaFree(device_result);
    free(host_alpha); free(host_beta); free(host_result);

    return 0;
}