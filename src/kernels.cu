#include "kernels.cuh"
#include "doctest/doctest.h"
#include <random>
#include "common.h"

template <IsKernel K>
__global__ void _test_kernels(
    float3* x, float* ws, float3* dws, uint N, const K W)
{
    auto i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= N)
        return;
    ws[i] = W(x[i]);
    dws[i] = W.nabla(x[i]);
}

// ADD NEW KERNELS TO THE TEMPLATE :

TEST_CASE_TEMPLATE("Kernel function properties", K, B3, C2, W6, COS)
{
    // settings
    const float DEV_HOST_EQ_TOL { 0.005 };
    const float KERNEL_GRAD_TOL { 0.01 };
    const uint SAMPLE_COUNT { 25 };

    // other quantities
    const uint N = SAMPLE_COUNT * SAMPLE_COUNT * SAMPLE_COUNT;
    const float h_bar { 1. * SAMPLE_COUNT };
    const K W(h_bar);

    // add interesting values / edge cases to the array, fill the rest
    // pseudo-randomly
    const float EPS = std::numeric_limits<float>::epsilon();
    std::vector<float3> x_rand_h(N);
    x_rand_h[0] = v3(0.f);
    x_rand_h[1] = v3(h_bar);
    x_rand_h[2] = v3(h_bar - EPS);
    x_rand_h[3] = v3(h_bar + EPS);
    x_rand_h[4] = v3(EPS);
    x_rand_h[5] = v3(h_bar + 1e-20);
    x_rand_h[6] = v3(h_bar - 1e-20);
    x_rand_h[7] = v3(1e-20);
    x_rand_h[8] = v3(std::numeric_limits<float>::max());
    std::srand(16142069);
    for (int i { 9 }; i < N; ++i) {
        x_rand_h[i] = v3(h_bar * 1.5 * static_cast<float>(std::rand())
                / static_cast<float>(RAND_MAX),
            h_bar * 1.5 * static_cast<float>(std::rand())
                / static_cast<float>(RAND_MAX),
            h_bar * 1.5 * static_cast<float>(std::rand())
                / static_cast<float>(RAND_MAX));
    }

    // also construct positions on a regular grid
    std::vector<float3> x_grid_h(N);
    const float spacing { h_bar * 2. / static_cast<float>(SAMPLE_COUNT) };
    uint j { 0 };
    for (uint xi = 0; xi < SAMPLE_COUNT; ++xi) {
        for (uint yi = 0; yi < SAMPLE_COUNT; ++yi) {
            for (uint zi = 0; zi < SAMPLE_COUNT; ++zi) {
                x_grid_h[j] = v3(-h_bar + spacing / 2)
                    + v3(xi * spacing, yi * spacing, zi * spacing);
                ++j;
            }
        }
    }
    // allocate device memory
    float3* x_rand_d { nullptr };
    float* ws_d { nullptr };
    float3* dws_d { nullptr };

    float3* x_grid_d { nullptr };
    float* reg_ws_d { nullptr };
    float3* reg_dws_d { nullptr };
    CUDA_CHECK(cudaMalloc((void**)&x_rand_d, sizeof(float3) * N));
    CUDA_CHECK(cudaMalloc((void**)&ws_d, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc((void**)&dws_d, sizeof(float3) * N));

    CUDA_CHECK(cudaMalloc((void**)&x_grid_d, sizeof(float3) * N));
    CUDA_CHECK(cudaMalloc((void**)&reg_ws_d, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc((void**)&reg_dws_d, sizeof(float3) * N));

    // copy pseudo-random positions to the device
    CUDA_CHECK(cudaMemcpy(
        x_rand_d, x_rand_h.data(), sizeof(float3) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        x_grid_d, x_grid_h.data(), sizeof(float3) * N, cudaMemcpyHostToDevice));

    // launch the test kernel
    _test_kernels<K><<<BLOCKS(N), BLOCK_SIZE>>>(x_rand_d, ws_d, dws_d, N, W);
    _test_kernels<K>
        <<<BLOCKS(N), BLOCK_SIZE>>>(x_grid_d, reg_ws_d, reg_dws_d, N, W);

    // place results on the heap to avoid stack limits for large tests
    std::vector<float> ws_h(N);
    std::vector<float3> dws_h(N);
    std::vector<float> reg_ws_h(N);
    std::vector<float3> reg_dws_h(N);
    CUDA_CHECK(cudaMemcpy(
        ws_h.data(), ws_d, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        dws_h.data(), dws_d, sizeof(float3) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        reg_ws_h.data(), reg_ws_d, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(reg_dws_h.data(), reg_dws_d, sizeof(float3) * N,
        cudaMemcpyDeviceToHost));

    // loop over all results and make assertions
    SUBCASE("Pseudo-random samples: positivity, compact support and similar "
            "behaviour on device and host")
    {
        for (int i { 0 }; i < N; ++i) {
            // capture values for more helpful error messages in case a test
            // fails
            CAPTURE(i);
            CAPTURE(x_rand_h[i].x);
            CAPTURE(x_rand_h[i].y);
            CAPTURE(x_rand_h[i].z);

            // check positivity of kernel shape function
            CHECK(ws_h[i] >= doctest::Approx(0.f));
            // check compact support of both kernel shape and its derivative
            if (norm(x_rand_h[i]) >= h_bar) {
                CHECK(ws_h[i] == doctest::Approx(0.0f));
                CHECK(norm(dws_h[i]) == doctest::Approx(0.0f));
            }
            // check host-device compatibility
            CHECK(ws_h[i]
                == doctest::Approx(W(x_rand_h[i])).epsilon(DEV_HOST_EQ_TOL));
            CHECK(dws_h[i].x
                == doctest::Approx(W.nabla(x_rand_h[i]).x)
                       .epsilon(DEV_HOST_EQ_TOL));
            CHECK(dws_h[i].y
                == doctest::Approx(W.nabla(x_rand_h[i]).y)
                       .epsilon(DEV_HOST_EQ_TOL));
            CHECK(dws_h[i].z
                == doctest::Approx(W.nabla(x_rand_h[i]).z)
                       .epsilon(DEV_HOST_EQ_TOL));
        }
        // check that w(0) is not zero, i.e. the call did not go wrong
        CHECK(ws_h[0] > doctest::Approx(0.f));
        // check that dw([0,0,0]) is the zero vector
        CHECK(dws_h[0].x == doctest::Approx(0.f));
        CHECK(dws_h[0].y == doctest::Approx(0.f));
        CHECK(dws_h[0].z == doctest::Approx(0.f));
    }

    SUBCASE("Kernel gradient antisymmetry")
    {
        float kernel_sum { 0. };
        float3 kernel_grad_sum { v3(0.) };
        for (int i { 0 }; i < N; ++i) {
            kernel_sum += reg_ws_h[i];
            kernel_grad_sum += reg_dws_h[i];

            // assert kernel gradient symmetry
            uint j { (N - 1) - i };
            CAPTURE(i);
            CHECK(-reg_dws_h[i].x == doctest::Approx(reg_dws_h[j].x));
            CHECK(-reg_dws_h[i].y == doctest::Approx(reg_dws_h[j].y));
            CHECK(-reg_dws_h[i].z == doctest::Approx(reg_dws_h[j].z));
        }

        SUBCASE("Kernel sum is one over volume")
        {
            // the kernel sum should be one over the volume
            CHECK(kernel_sum
                == doctest::Approx(1. / (spacing * spacing * spacing))
                       .epsilon(DEV_HOST_EQ_TOL));
        }
        SUBCASE("Kernel gradient sum is zero vector")
        {
            // the kernel gradient sum should be zero
            CHECK(kernel_grad_sum.x
                == doctest::Approx(0.).epsilon(KERNEL_GRAD_TOL));
            CHECK(kernel_grad_sum.y
                == doctest::Approx(0.).epsilon(KERNEL_GRAD_TOL));
            CHECK(kernel_grad_sum.z
                == doctest::Approx(0.).epsilon(KERNEL_GRAD_TOL));
        }
    }

    // free the device memory, host vectors are handled by their destructor
    CUDA_CHECK(cudaFree(x_rand_d));
    CUDA_CHECK(cudaFree(ws_d));
    CUDA_CHECK(cudaFree(dws_d));
    CUDA_CHECK(cudaFree(x_grid_d));
    CUDA_CHECK(cudaFree(reg_ws_d));
    CUDA_CHECK(cudaFree(reg_dws_d));
}