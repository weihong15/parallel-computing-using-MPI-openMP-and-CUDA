#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"



__global__
void GPU_GEMM_brandNew(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real alpha, nn_real beta,
           int M, int N, int K){

			   int temp = ( blockDim.x * threadIdx.y) + threadIdx.x;
			   int row = blockDim.x * blockDim.y *blockIdx.y + temp;
			   int col = 16* blockIdx.x;

			//Within each iteration,  a 4×16 sub-matrix ofBis loaded into the thread block’sshared memory,
			__shared__ nn_real B_[64];

			//Each thread then reads in a 1×4 sub-matrixofAinto a local arraya[4],
			nn_real a_[4] = {};

			//Each thread computes a row of size 1×16 in the outputD.
			nn_real d_[16] = {};

			// each 16X4 block need to compute a 64X16 matrix
			// first store shared memory
			for (int i = 0;i <(K+3)/4;i++){
				
				//storage for shared memory
				//each thread store 1, 16X4, need 64 shared
				//threadIdx.y + 4*threadIdx.x --> 0-63
				// global wise, i am confused???
				// each ith is within some 16 subblock,
				int global_row_B = i * 4+threadIdx.y;
				int global_col_B = col +threadIdx.x;
				int global_ind_B = N*global_row_B + global_col_B;
				//column major
				global_ind_B = global_row_B+global_col_B*K;

				if (global_row_B<K && global_col_B < N) B_[threadIdx.y+4*threadIdx.x] = B[global_ind_B];


				//Each thread then reads in a 1×4 sub-matrixofAinto a local arraya[4],
				if (row<M){
					for (int j=0; j<4;j++) a_[j] = A[row+M*(4*i+j)];
				}

				//sync threads, to ensure shared memory is stored properly
				__syncthreads();

				//each a[4] will multiply by b shared to get 16 elements.
				for(int j = 0; j<16; j++){
					//k is confusing, change ind to ij
					for(int ij = 0; ij < 4; ij++){
						if(row<M && (4*i+ij)<K && (col+j)<N)
							d_[j]+=a_[ij]*B_[4*j+ij];
					}
				}
				__syncthreads();
			}
			//now we got true value of AB for 16 elements, we need to write back to C
			for(int i = 0;i<16;i++){
				if (row<M &&(col+i)<N)
					C[row+(col+i)*M] = alpha*d_[i]+beta*C[row+(col+i)*M];
			}
		   }

//attempt to follow part 1 notes 16*4 blocks
//16*4 blocks will compute 64X16
int brandNewGeMM(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K) {
			dim3 block(16,4);
			int block_X_PerGrid = (N+15)/16;
			int block_Y_PerGrid = (M +63)/64;
			dim3 grid(block_X_PerGrid,block_Y_PerGrid);
			GPU_GEMM_brandNew<<<grid,block>>>(A,B,C, *alpha, *beta,M,N,K);
    		return 0;
		   }








/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
__global__
void GPU_GEMM_amateur(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real alpha, nn_real beta,
           int M, int N, int K){
        // alpha*AB +beta * C
        // 
        const int BLOCK_SIZE=32;
        nn_real total = 0.0;
        int iter = (K+BLOCK_SIZE-1)/BLOCK_SIZE;
//each thread will handle the row and column of the C matrix calculation
        //true index
        int globalX = blockIdx.x* blockDim.x + threadIdx.x;
        int globalY = blockIdx.y* blockDim.y+ threadIdx.y;
// we load shared memory, this square matrix is utilized but all people in thread

	//int subBlocks = (K+blockDim.y-1)/blockDim.y;
	__shared__ nn_real A_[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ nn_real B_[BLOCK_SIZE][BLOCK_SIZE];
	
	int colOne,rowOne,colTwo,rowTwo,indOne,indTwo;

	for (int i =0; i< iter; i++){
		//2 points one for A, one for B
		colOne = i*BLOCK_SIZE+threadIdx.x;
		rowOne = globalY;
		colTwo = globalX;
		rowTwo = i*BLOCK_SIZE + threadIdx.y;

		indOne = colOne*M+rowOne;
		indTwo = colTwo*K+rowTwo;

		if (colOne<K)A_[threadIdx.y][threadIdx.x] = A[indOne];
		else A_[threadIdx.y][threadIdx.x] = 0.0;

		if (rowTwo<K)B_[threadIdx.y][threadIdx.x] = B[indTwo];
		else B_[threadIdx.y][threadIdx.x] = 0.0;

                __syncthreads();

		if(globalX<N && globalY<M){
			for(int j=0; j<blockDim.y;j++) total+= A_[threadIdx.y][j]*B_[j][threadIdx.x];
		}

        }

        //
        //if(threadIdx.x<M && threadIdx.y<N) C[globalY*M+globalX] = alpha*total +beta*C[globalY*M+globalX];
	if(globalX<N && globalY<M) C[globalX*M+globalY] = alpha*total+beta*C[globalX*M+globalY];
}
/*
        if (row<M && col < N){
        //dot product of A[row] and B[col],
                int ind = M* col + row;;
                nn_real cumm = 0.0;
                for(int i = 0;i<K;i++) cumm += alpha*A[M*i+row]*B[K*col+i];
                        //cumm += alpha*A[row*N+i] *B[col+i*N];
                cumm+= beta*C[ind];
                C[ind] = cumm;

        } */

__global__
void GPU_GEMM_amateur2(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real alpha, nn_real beta,
           int M, int N, int K){
	// alpha*AB +beta * C
	// 
	const int BLOCK_SIZE=32;
	double total = 0.0;
	int iter = (K+BLOCK_SIZE-1)/BLOCK_SIZE;
//each thread will handle the row and column of the C matrix calculation
	//true index
	int globalX = blockIdx.x* blockDim.x + threadIdx.x;
	int globalY = blockIdx.y* blockDim.y+ threadIdx.y;
// we load shared memory, this square matrix is utilized but all people in thread
	for (int i =0; i< iter; i++){
		__shared__ double A_[BLOCK_SIZE][BLOCK_SIZE+1];
		__shared__ double B_[BLOCK_SIZE][BLOCK_SIZE+1];

		int colA = BLOCK_SIZE*i+threadIdx.y;
		if (globalX< M && colA< K) A_[threadIdx.x][threadIdx.y] = A[M*colA+globalX];
		int rowB = threadIdx.x+BLOCK_SIZE*i;
		if (globalY<N && rowB<K) B_[threadIdx.x][threadIdx.y] = B[N*rowB+globalY];

		__syncthreads();

		//num_elements is normally 32, unless it towards the end.
		int num_ele = BLOCK_SIZE;
		if ((K-i*BLOCK_SIZE)<BLOCK_SIZE) num_ele= K-i*BLOCK_SIZE;
		//accumualte the total matrix multiplication
		for (int j=0;j<num_ele;j++) total+= A_[threadIdx.x][j]*B_[j][threadIdx.y];

	__syncthreads();
	}

	//
	if(threadIdx.x<M && threadIdx.y<N) C[globalY*M+globalX] = alpha*total +beta*C[globalY*M+globalX];

/*
	if (row<M && col < N){
	//dot product of A[row] and B[col],
		int ind = M* col + row;;
		nn_real cumm = 0.0;
		for(int i = 0;i<K;i++) cumm += alpha*A[M*i+row]*B[K*col+i];
			//cumm += alpha*A[row*N+i] *B[col+i*N];
		cumm+= beta*C[ind];
		C[ind] = cumm;

	} */
}

void copy2GPUwrap(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K) {
	nn_real* d_A;nn_real* d_B; nn_real* d_C;
	//A is size M,K, //B is size K N, C is size M,N
	cudaMalloc(&d_A, sizeof(nn_real)*M*K);
	cudaMalloc(&d_B, sizeof(nn_real)*K*N);
	cudaMalloc(&d_C, sizeof(nn_real)*M*N);

	//writing back to C, so we don't need d_out
	
	//copying
	cudaMemcpy(d_A,A, sizeof(nn_real)*M*K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B, sizeof(nn_real)*N*K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C,C, sizeof(nn_real)*M*N, cudaMemcpyHostToDevice);

	check_launch("copy to gpu GEMM");

	//this will be a wrapper
	//GPU code here.
	//myGEMM(d_A,d_B,d_C, alpha, beta,M,N,K);
	
	brandNewGeMM(d_A,d_B,d_C, alpha, beta,M,N,K);
	//overwrite current C with d_C
	cudaMemcpy(C,d_C,sizeof(nn_real)*M*N,cudaMemcpyDeviceToHost);

	check_launch("copy from gpu GEMM");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
           }

__global__
void GPU_GEMM_noob(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real alpha, nn_real beta,
           int M, int N, int K){
	// alpha*AB +beta * C
	int row = blockIdx.x*blockDim.x+threadIdx.x;
	int col = blockIdx.y*blockDim.y+threadIdx.y;
	if (row<M && col < N){
	//dot product of A[row] and B[col],
		int ind = M* col + row;;
		nn_real cumm = 0.0;
		for(int i = 0;i<K;i++) cumm += alpha*A[M*i+row]*B[K*col+i];
			//cumm += alpha*A[row*N+i] *B[col+i*N];
		cumm+= beta*C[ind];
		C[ind] = cumm;

	}
}


/*
  Caller function GEMM
*/
int BLOCK_SIZE = 32;
int NUM_THREADS = 256;

int myGEMM(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K) {
//short circuit to update this
	brandNewGeMM(A,B,C, alpha, beta,M,N,K);
	return 0;

    // size of C is MN, so we need at least MN threads working
	dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);
	int block_X_PerGrid = (M+block.x-1)/block.x;
	int block_Y_PerGrid = (N +block.y-1)/block.y;
	dim3 grid(block_X_PerGrid,block_Y_PerGrid);
	
	GPU_GEMM_noob<<<grid,block>>>(A,B,C, *alpha, *beta,M,N,K);
    return 0;
}


/* Helper functions for neural networks */
// TODO

__global__
void gpu_sigmoid_noob(nn_real* v,int M, int N){
        int row = threadIdx.x+blockIdx.x*blockDim.x;
        int col = threadIdx.y+blockIdx.y*blockDim.y;
        if (row<M && col<N) v[row+col*M] = 1/(1+exp(-v[row+col*M]));
}

void my_sigmoid(nn_real* v, nn_real* u, int M, int N){
	//will do sigmoid on vector v
	//store the output from GPU in u
	nn_real* d_u = nullptr;
	cudaMalloc(&d_u,M*N*sizeof(nn_real));
	cudaMemcpy(d_u,v, M*N*sizeof(nn_real), cudaMemcpyHostToDevice);
	check_launch("copy to gpu");
	dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);
        int block_X_PerGrid = (M+block.x-1)/block.x;
        int block_Y_PerGrid = (N +block.y-1)/block.y;
        dim3 grid(block_X_PerGrid,block_Y_PerGrid);
		
	//kernel
	gpu_sigmoid_noob<<<grid,block>>>(d_u,M,N);
	cudaMemcpy(u,d_u,M*N*sizeof(nn_real),cudaMemcpyDeviceToHost);
	check_launch("copy from gpu whyyyyy???");
	cudaFree(d_u);	
}

__global__
void gpu_softmax_noob(nn_real* v,int M,int N){
	//softmax every col
	int col = threadIdx.x + blockIdx.x*blockDim.x;
	while (col<N){
	nn_real cumm = 0.00;
	for (int i = 0; i<M;i++){
		//exponential the whole column
		//store the sum, and divide the whole column
		//int ind = M*col+ i;
		//v[ind] = exp(v[ind]);
		//cumm += v[ind];

		int ind = col*M+i;
		v[ind] = exp(v[ind]);
		cumm += v[ind];

	}
	for (int i = 0;i<M;i++) v[col*M+i]/=cumm;

//	printf("finish with column %d",col);
	col += blockDim.x*gridDim.x;
	}
}


void my_softmax(nn_real* u, nn_real* v, int M, int N){
//soft max each column of matrix u, size M,N
	nn_real* d_v = nullptr;
        cudaMalloc(&d_v,M*N*sizeof(nn_real));
        cudaMemcpy(d_v,u, M*N*sizeof(nn_real), cudaMemcpyHostToDevice);
	check_launch("copy to gpu softmax");
	int block_per_grid = (N+BLOCK_SIZE-1)/BLOCK_SIZE;

        //kernel
        gpu_softmax_noob<<<block_per_grid,BLOCK_SIZE>>>(d_v,M,N);
	check_launch("mid point");
	//printf("two values M = %d, N= %d",M,N);
        cudaMemcpy(v,d_v,M*N*sizeof(nn_real),cudaMemcpyDeviceToHost);
        check_launch("copy from gpu softmax");
	cudaFree(d_v);
}
