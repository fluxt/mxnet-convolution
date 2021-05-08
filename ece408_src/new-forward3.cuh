#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#define BLOCK_SIZE 1024

using namespace std;
namespace mxnet
{
namespace op
{

__constant__ float mask[2400];

//kernel threads maps to input tile size W/H=64 ->(64 * 16); W/H = 30 ->(30 * 16)
//every thread load one data into shared memory
//threads contribute to output W/H=64 -> (60 * 12); W/H=30-> (26*12)
//W/H = 64 -> gridDim(5*1,  B, 1)
//W/H = 30 -> gridDim(3*6,  B, 1)
__global__ void unroll_kernel(const float *x, float *x_unroll, const int output_sec, const int B,  const int C, const int H, const int W, const int K)
{
    __shared__ float tile[1024]; //64 width * 16 height
    int H_out = H-K+1;
    int W_out = W-K+1;
    int W_unroll = H_out * W_out;

    int tx = threadIdx.x; //[0-63] first 60 thread used for unrolling
    int ty = threadIdx.y; //[0-15] first 12 used for unrolling
    int bx = blockIdx.x; //[0-4] for H/W=64, = 5 * C [0-17] for H/W = 30
    int input_c_idx = bx / output_sec;
    int output_sec_idx = bx % output_sec;
    int b = blockIdx.y; //[0-10k]

    int base_output_h = output_sec_idx * 12; //within a input img
    
    int size_for_each_input_c = H_out * W_out * K * K;
    int size_for_each_batch = C * size_for_each_input_c;

    int h_base, w_idx, h_unroll;

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_unroll3d(b, h, w) x_unroll[b * size_for_each_batch + h * (W_unroll) + w]
#define tile2d(h,w) tile[h * W+ w]

    int input_h = base_output_h + ty;

    if(input_h < H){
        tile2d(ty, tx) = x4d(b, input_c_idx, input_h, tx);
    }
    __syncthreads();

    //real_output_w == tx;
    int real_output_h = base_output_h + ty;
    if(tx < (W_out) && ty < 12 && real_output_h < H_out){
        h_base = input_c_idx * K*K;
        w_idx = real_output_h  * W_out + tx;
        if(b==1 && blockIdx.x == 17 && H==30){
            //printf("real_output_h = %d, tx = %d, ty=%d, w_idx = %d, h_base = %d, input_c_idx=%d\n", real_output_h, tx, ty, w_idx, h_base, input_c_idx);
        }
        for(int p =0; p < K; p++){
            for(int q=0; q < K; q ++){
                h_unroll = h_base+p*K+q;
                // if(b==0 && blockIdx.x == 0 && H == 64 && p == 4 && q==4 ){
                    //printf("tx = %d, ty = %d, x_unroll(%d, %d, %d) = tile2d(%d, %d)\n", tx, ty, b, h_unroll, w_idx, ty+p, tx+q);
                //}
                x_unroll3d(b, h_unroll, w_idx) = tile2d(ty+p, tx+q);
                __syncthreads();
            }
        }
    }
    __syncthreads();
#undef x4d
#undef x_unroll3d
#undef tile2d
}

__global__ void matrixMultiply(float *y, float *x_unroll, const int B, const int M, const int C, const int H, const int W, const int K) {
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define y3d(i3, i2,     i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) +                  i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_unroll3d(b, h, w) x_unroll[b * H_out * W_out * C * K * K + h * (H_out * W_out) + w]
#define k4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k2d(i3,         i0) mask[(i3) * (C * K * K) +                               i0]

  const int H_out = H-K+1;
  const int W_out = W-K+1;

  int b = blockIdx.z;
  int m = blockIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x; //output pixel sequence idx
  if (col < H_out * W_out) {
    float sum = 0.0;
    for (int i=0; i<C*K*K; i++) {
      sum += k2d(m, i) * x_unroll3d(b, i, col);
    }

    y3d(b, m, col) = sum;
  }

#undef y4d
#undef y3d
#undef x4d
#undef k4d
#undef k2d
#undef x_unroll3d
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

    float * x_unroll_device;
    int size_of_unrolled_x = H_out*W_out*C*K*K*B;
    cudaMalloc ((void **) &x_unroll_device, size_of_unrolled_x * sizeof(float));

    int output_block_dim_y = 16-4; //12
    int output_sec = (H_out-1)/output_block_dim_y +1; //5
    dim3 gridDim(output_sec*C, B,1); //(5, 10k,1)
    dim3 blockDim(W,16,1); //(64, 16, 1)
    unroll_kernel<<<gridDim, blockDim>>>(x.dptr_, x_unroll_device,output_sec, B,C,H,W,K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    gridDim = dim3((H_out*W_out-1)/1024+1, M, B);
    blockDim = dim3(1024, 1, 1);
    matrixMultiply<<<gridDim, blockDim>>>(y.dptr_, x_unroll_device,B,M,C,H,W,K);
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
