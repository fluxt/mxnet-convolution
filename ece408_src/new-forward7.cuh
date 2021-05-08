#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#define BLOCK_SIZE 256
#define BLOCK_SIZE_2 54

using namespace std;
namespace mxnet
{
namespace op
{

__constant__ float mask[2400];


__global__ void first_kernel(float * y, const float *x){
        int t = blockIdx.x * BLOCK_SIZE+threadIdx.x;
        int b = blockIdx.y;
        int h_out, w_out, h_unroll;

        __shared__ float tile[25 * BLOCK_SIZE];
        __shared__ float result [6 * BLOCK_SIZE];
        
#define x4d(i3, i2, i1, i0) x[(i3) * 4096 + (i2) * 4096 + (i1) * 64 + i0]
#define tile2d(h, w) tile[h * BLOCK_SIZE + w]
#define result2d(m, w) result[m * BLOCK_SIZE + w]
#define mask2d(h,w) mask[h * 25 + w]
#define y3d(b, m, w) y[b * 21600 + m * 3600 +  w]

        if(t < 3600){
        h_out = t/60; //corresponding to which output height
        w_out = t%60; //corresponding to which output width
        for(int p =0; p < 5; p++){
            for(int q=0; q < 5; q ++){
                h_unroll = p*5+q;
                tile2d(h_unroll, threadIdx.x) = x4d(b, 0, h_out+p, w_out+q);
            }
        }
        __syncthreads();

        //set shared memory to 0 
        for(int i = 0; i < 6; i ++){
            result2d(i, threadIdx.x) = 0.0f;
        }
        __syncthreads();

        for (int i=0; i <25; i++) {
            float temp = tile2d(i, threadIdx.x);
            for(int j = 0; j < 6; j++){
                result2d(j, threadIdx.x) += mask2d(j,i)*temp;
            }
        }
        __syncthreads();

        for(int j = 0; j < 6; j++){
            y3d(b, j, t) = result2d(j, threadIdx.x);
        }
        }
#undef x4d
#undef tile2d
#undef result2d
#undef mask2d
#undef y3d
    }

__global__ void second_kernel(float *y, const float *x){
    //int H_out = H-K+1; 26
    //int W_out = W-K+1; 26
    int t = blockIdx.x * BLOCK_SIZE_2+threadIdx.x;
    int b = blockIdx.y;
    int h_out, w_out, h_base, h_unroll;
    //int size_for_each_input_section = H_out * W_out * K * K; 16900
    //int size_for_each_batch = C * size_for_each_input_section; 101400


    __shared__ float tile [150 * BLOCK_SIZE_2];
    __shared__ float result [16 * BLOCK_SIZE_2];
    

#define x4d(i3, i2, i1, i0) x[(i3) * (5400) + (i2) * (900) + (i1) * (30) + i0]
#define tile2d(h, w) tile[h * BLOCK_SIZE_2 + w]
#define result2d(m, w) result[m * BLOCK_SIZE_2 + w]
#define y3d(b, m, w) y[b * 10816 + m * (676) + w]
#define mask2d(h, w) mask[h * (150) + w]

    if(t < 676){
        h_out = t/26; //corresponding to which output height
        w_out = t%26; //corresponding to which output width

        //loading data to shared memory
        for(int i = 0; i < 6; i ++){
            h_base = i*25;
            for(int p =0; p < 5; p++){
                for(int q=0; q < 5; q ++){
                    h_unroll = h_base+p*5+q;
                    tile2d(h_unroll, threadIdx.x)= x4d(b, i, h_out+p, w_out+q);
                }
            }
        }

        __syncthreads();

        //set shared memory to 0
        for(int i = 0; i < 16; i ++){
            result2d(i, threadIdx.x) = 0.0f;
        }
        __syncthreads();

        for (int i=0; i<150; i++) {
            float temp = tile2d(i, threadIdx.x);
            for(int j = 0; j <16; j++){
                result2d(j, threadIdx.x) += mask2d(j, i)*temp;
            }
        }
        __syncthreads();

        for(int i = 0; i < 16; i++){
            y3d(b, i, t) = result2d(i,threadIdx.x);
        }
    }
#undef x4d
#undef tile2d
#undef result2d
#undef y3d
#undef mask2d

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

    int grid_num = H_out*W_out;
    if(H == 64){

        int grid_dim_x = ((grid_num-1)/BLOCK_SIZE) +1;
        dim3 gridDim(grid_dim_x, B,1);
        dim3 blockDim(BLOCK_SIZE,1,1);
        //cout<<"first kernel launch"<<endl;
        first_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_);
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }else{
        int grid_dim_x = ((grid_num-1)/BLOCK_SIZE_2) +1;
        dim3 gridDim(grid_dim_x, B,1);
        dim3 blockDim(BLOCK_SIZE_2,1,1);
        //cout<<"second kernel launch"<<endl;
        second_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_);
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
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
