
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>
#include <iostream>

using namespace std;

namespace mxnet
{
namespace op
{


template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    const int B = x.shape_[0]; // Bench Size
    const int M = y.shape_[1]; // Number of Output Feature Maps
    const int C = x.shape_[1]; // Number of Input Feature Maps
    const int H = x.shape_[2]; // Input Height(Rows)
    const int W = x.shape_[3]; // Input Width(Columns)
    const int K = k.shape_[3]; // Filter Size
    //cout<<"Batch sixe B= "<<B<<"; # of Output feaure map M= "<<M<<"; # of Input feature map C= "<< C<<" Input height H= "<<H<<"; Input width W= "<<W<<"; filter size K= "<<K<<endl;

// B   Bench Size
// M   Number of Output Feature Maps
// C   Number of Input Feature Maps
// H   Input Height(Rows)
// W   Input Width(Columns)
// K   Filter Size
// H_o Output Height(Rows)
// W_o Input Height(Columns)

// B   = 10k, 10k
// M   = 6  , 16
// C   = 1  , 6
// H   = 64 , 30
// W   = 64 , 30
// K   = 5  , 5
// H_o = 60 , 26
// W_o = 60 , 26

//  X
//  B   C   H   W

//  W
//  M   C   K   K

//  Y
//  B   M   H_o W_o

    int H_out = H-K+1; // Output Height(Rows)
    int W_out = W-K+1; // Output Width(Rows)

    for (int b=0; b<B; b++)
    for (int m=0; m<M; m++)
    for (int h=0; h<H_out; h++)
    for (int w=0; w<W_out; w++)
    {
        y[b][m][h][w] = 0.0;       
        for (int c=0; c<C; c++)
        for (int p=0; p<K; p++)
        for (int q=0; q<K; q++) {
            y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
        }
    }
}
}
}

#endif
