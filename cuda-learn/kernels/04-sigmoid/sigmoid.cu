#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void sigmoid(const float* params, float* result, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n) return;
    result[index] = 1.0f / (1.0f + expf(-params[index]));
    //                                            ↑ index không phải i
}

int main() {
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    float* host_params = (float*)malloc(bytes);
    float* host_result = (float*)malloc(bytes);

    float *device_params, *device_result;
    CUDA_CHECK(cudaMalloc(&device_params, bytes));
    CUDA_CHECK(cudaMalloc(&device_result, bytes));

    for (int i = 0; i < n; i++){
        host_params[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
    }

    CUDA_CHECK(cudaMemcpy(device_params, host_params, bytes, cudaMemcpyHostToDevice));
    
    int threads_per_block = 1024;
    int blocks = n / threads_per_block + 1;

    sigmoid<<<blocks, threads_per_block>>>(device_params, device_result, n);
    // ↑ <<< >>> không phải << >>                              ↑ thêm n
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host_result, device_result, bytes, cudaMemcpyDeviceToHost));

    cudaFree(device_params); cudaFree(device_result);
    free(host_params); free(host_result);
    return 0;
}