#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>

__global__ void sigmoid(const float* in, float* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / (1.0f + expf(-in[i]));
}

std::vector<float> run_sigmoid(std::vector<float> input) {
    int n = input.size();
    size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    sigmoid<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return output;
}

// sigmoid(0) = 0.5 — điểm đối xứng
TEST(Sigmoid, zero_is_half) {
    auto out = run_sigmoid({0.0f});
    EXPECT_NEAR(out[0], 0.5f, 1e-6f);
}

// output luôn trong (0, 1) — dùng giá trị vừa phải, float32 mất precision ở ±100
TEST(Sigmoid, output_between_zero_and_one) {
    auto out = run_sigmoid({-10.0f, -1.0f, 0.0f, 1.0f, 10.0f});
    for (float v : out) {
        EXPECT_GT(v, 0.0f);
        EXPECT_LT(v, 1.0f);
    }
}

// số dương lớn → gần 1
TEST(Sigmoid, large_positive_approaches_one) {
    auto out = run_sigmoid({100.0f});
    EXPECT_NEAR(out[0], 1.0f, 1e-4f);
}

// số âm lớn → gần 0
TEST(Sigmoid, large_negative_approaches_zero) {
    auto out = run_sigmoid({-100.0f});
    EXPECT_NEAR(out[0], 0.0f, 1e-4f);
}

// đối xứng: sigmoid(-x) = 1 - sigmoid(x)
TEST(Sigmoid, symmetry) {
    std::vector<float> xs = {0.5f, 1.0f, 2.0f, 5.0f};
    auto pos = run_sigmoid(xs);
    std::vector<float> neg_xs = {-0.5f, -1.0f, -2.0f, -5.0f};
    auto neg = run_sigmoid(neg_xs);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(neg[i], 1.0f - pos[i], 1e-6f);
}

// giá trị cụ thể — sigmoid(1) ≈ 0.7311
TEST(Sigmoid, known_values) {
    auto out = run_sigmoid({1.0f, -1.0f, 2.0f});
    EXPECT_NEAR(out[0], 0.7310586f, 1e-5f);
    EXPECT_NEAR(out[1], 0.2689414f, 1e-5f);
    EXPECT_NEAR(out[2], 0.8807970f, 1e-5f);
}

// large n
TEST(Sigmoid, large_n) {
    int n = 1 << 20;
    std::vector<float> input(n, 0.0f);  // tất cả = 0 → output = 0.5
    auto out = run_sigmoid(input);
    for (float v : out)
        EXPECT_NEAR(v, 0.5f, 1e-6f);
}
