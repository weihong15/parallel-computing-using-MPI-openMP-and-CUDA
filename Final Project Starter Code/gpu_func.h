#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"

int myGEMM(nn_real* A, nn_real* B, nn_real* C, nn_real* alpha, nn_real* beta, int M, int N,
           int K);

// TODO
// Add additional function declarations

__global__
void GPU_GEMM_brandNew(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real alpha, nn_real beta,
           int M, int N, int K);

int brandNewGeMM(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K);



// alpha*AB +beta * C
void copy2GPUwrap(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K);

__global__
void GPU_GEMM_noob(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real alpha, nn_real beta,
           int M, int N, int K);

__global__
void gpu_sigmoid_noob(nn_real* v,int M, int N);

void my_sigmoid(nn_real* v, nn_real* u, int M, int N);
__global__
void gpu_softmax_noob(nn_real* v,int M,int N);
void my_softmax(nn_real* u, nn_real* v, int M, int N);


#endif
