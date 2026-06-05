#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void relu(const float* in, float* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fmaxf(0.0f, in[i]);
}

// ── helper: chạy kernel, trả về kết quả trên CPU ─────────────────────────────
std::vector<float> run_relu(std::vector<float> input) {
    int n = input.size();
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    relu<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return output;
}

// ── tests ─────────────────────────────────────────────────────────────────────
TEST(ReLU, negative_becomes_zero) {
    auto out = run_relu({-5.0f, -1.0f, -0.001f, -100.0f});
    for (float v : out)
        EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(ReLU, positive_unchanged) {
    auto out = run_relu({1.0f, 2.5f, 0.001f, 100.0f});
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.5f);
    EXPECT_FLOAT_EQ(out[2], 0.001f);
    EXPECT_FLOAT_EQ(out[3], 100.0f);
}

TEST(ReLU, zero_stays_zero) {
    auto out = run_relu({0.0f});
    EXPECT_FLOAT_EQ(out[0], 0.0f);
}

TEST(ReLU, mixed_values) {
    auto out = run_relu({-3.0f, 0.0f, 2.0f, -1.0f, 5.0f});
    std::vector<float> expected = {0.0f, 0.0f, 2.0f, 0.0f, 5.0f};
    for (int i = 0; i < 5; i++)
        EXPECT_FLOAT_EQ(out[i], expected[i]);
}

TEST(ReLU, large_n) {
    int n = 1 << 20;
    std::vector<float> input(n);
    for (int i = 0; i < n; i++)
        input[i] = (i % 2 == 0) ? 1.0f : -1.0f;

    auto out = run_relu(input);

    for (int i = 0; i < n; i++) {
        float expected = (i % 2 == 0) ? 1.0f : 0.0f;
        EXPECT_FLOAT_EQ(out[i], expected);
    }
}
