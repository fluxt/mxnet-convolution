#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#define UNROLL_BLOCK_SIZE 1024
#define MM_TILE_SIZE 16

using namespace std;
namespace mxnet
{
namespace op
{

__constant__ float mask[2400];

__global__ void unroll_kernel(const float *x, float *x_unroll, const int B,  const int C, const int H, const int W, const int K)
{
    int H_out = H-K+1;
    int W_out = W-K+1;
    int W_unroll = H_out * W_out;
    int t = blockIdx.x * UNROLL_BLOCK_SIZE+threadIdx.x;
    int b = blockIdx.y;
    int c, s, h_out, w_out, h_base, h_unroll;
    int size_for_each_input_section = H_out * W_out * K * K;
    int size_for_each_batch = C * size_for_each_input_section;
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_unroll3d(b, h, w) x_unroll[b * size_for_each_batch + h * (W_unroll) + w]

    if(t < C*W_unroll){
        c = t/W_unroll; //which section
        s = t%W_unroll; //which col
        h_out = s/W_out; //corresponding to which output height
        w_out = s%W_out; //corresponding to which output width
        h_base = c*K*K; 
        for(int p =0; p < K; p++){
            for(int q=0; q < K; q ++){
                h_unroll = h_base+p*K+q;
                x_unroll3d(b, h_unroll, s)= x4d(b, c, h_out+p, w_out+q);
            }
        }
    }
    __syncthreads();
#undef x4d
#undef x_unroll3d
}

__global__ void matrixMultiply(float *y, float *x_unroll, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out,
                               int numARows, int numAColumns,
                               int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {

#define y3d(i3, i2,     i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) +                  i0]
#define x_unroll3d(b, h, w) x_unroll[(b) * H_out * W_out * C * K * K + (h) * (H_out * W_out) + (w)     ]
#define w2d(i3,         i0) mask[(i3) * (C * K * K) +                               (i0)]

  __shared__ float x_s[MM_TILE_SIZE][MM_TILE_SIZE];
  __shared__ float w_s[MM_TILE_SIZE][MM_TILE_SIZE];

  const int tx = threadIdx.x; const int ty = threadIdx.y;

  const int b   = blockIdx.z;
  const int row = blockIdx.y*blockDim.y+threadIdx.y;
  const int col = blockIdx.x*blockDim.x+threadIdx.x;

  float sum = 0.0;

  for (int i = 0; i < (numAColumns-1)/MM_TILE_SIZE+1; i++) {
    if (row < numARows && (i*MM_TILE_SIZE+tx) < numAColumns) {
      w_s[ty][tx] = w2d(row, i*MM_TILE_SIZE+tx);
    } else {
      w_s[ty][tx] = 0.0;
    }
    if (i*MM_TILE_SIZE+ty < numBRows && col < numBColumns) {
      x_s[ty][tx] = x_unroll3d(b,i*MM_TILE_SIZE+ty,col);
    } else {
      x_s[ty][tx] = 0.0;
    }
    __syncthreads();
    for (int j = 0; j < MM_TILE_SIZE; j++) {
      sum += w_s[ty][j] * x_s[j][tx];
    }
    __syncthreads();
  }

  if (row < numCRows && col < numCColumns) {
    y3d(b,row,col) = sum;
  }

#undef y3d
#undef x_unroll3d
#undef k2d
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

    float * x_unroll_device;

    cudaMemcpyToSymbol(mask, w.dptr_ ,sizeof(float)*(M*C*K*K));

    int size_of_unrolled_x = H_out*W_out*C*K*K*B;
    cudaMalloc ((void **) &x_unroll_device, size_of_unrolled_x * sizeof(float));

    int grid_num = H_out*W_out*C;
    int grid_dim_x = ((grid_num-1)/UNROLL_BLOCK_SIZE) +1;
    dim3 gridDim(grid_dim_x, B,1);
    dim3 blockDim(UNROLL_BLOCK_SIZE,1,1);
    unroll_kernel<<<gridDim, blockDim>>>(x.dptr_, x_unroll_device, B,C,H,W,K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    gridDim = dim3((H_out*W_out-1)/MM_TILE_SIZE+1, (M-1)/MM_TILE_SIZE+1, B);
    blockDim = dim3(MM_TILE_SIZE, MM_TILE_SIZE, 1);
    matrixMultiply<<<gridDim, blockDim>>>(y.dptr_, x_unroll_device, B,M,C,H,W,K,H_out,W_out, M, C*K*K, C*K*K, W_out*H_out, M, W_out*H_out);
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
