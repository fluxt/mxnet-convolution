
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <iostream>
#define TILE_SIZE 8

using namespace std;
namespace mxnet
{
namespace op
{

__constant__ float mask[2400];

__global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    int H_out = H-K+1;
    int W_out = W-K+1;
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    //load input tile
    int W_grid = (W_out-1)/TILE_SIZE +1;
    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid)*TILE_SIZE  + threadIdx.y;
    int w = (blockIdx.y % W_grid)*TILE_SIZE + threadIdx.x;

    if (h < H_out && w < W_out)
    {
        float acc = 0.0f;
        for (int c = 0; c < C; c++)  // sum over all input channels
        for (int p = 0; p < K; p++)  // loop over KxK  filter
        for (int q = 0; q < K; q++){
            printf("x(%d, %d, %d, %d) = %.20f; k(%d, %d, %d, %d) = %.4f\n", b, c, h+p, w+q, x4d(b, c, h + p, w + q), m, c, p, q, k4d(m,c,p,q));
            acc += x4d(b, c, h + p, w + q) * k4d(m,c,p,q);
        }
        y4d(b,m, h, w) = acc;
    }    

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    const int B = x.shape_[0]; // Bench Size
    const int M = y.shape_[1]; // Number of Output Feature Maps
    const int C = x.shape_[1]; // Number of Input Feature Maps
    const int H = x.shape_[2]; // Input Height(Rows)
    const int W = x.shape_[3]; // Input Width(Columns)
    const int K = w.shape_[3]; // Filter Size

    const int H_out = H-K+1;
    const int W_out = W-K+1;

    cudaMemcpyToSymbol(mask, w.dptr_ ,sizeof(float)*(M*C*K*K));

    int grid_h = (H_out -1)/ TILE_SIZE +1;
    int grid_w = (W_out -1)/ TILE_SIZE +1;
    int total_grid = grid_h * grid_w;

    dim3 gridDim(M, total_grid, B);
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);

    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
