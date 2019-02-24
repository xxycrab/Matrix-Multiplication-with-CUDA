// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include <stdio.h>

using namespace std;
#define TW  32 // C block size 64 * 64
#define TS 32 // B: 32 * 64  A: 64 * 32
#define NUMOPT 4 // 4 output per thread
#define OPS 2 // output per side

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;

    __shared__ _DOUBLE_ AS[TW][TS];
    __shared__ _DOUBLE_ BS[TS][TW];

    _DOUBLE_ CR[NUMOPT] = {0.0};

    int I = by * TW + ty;
    int J = bx * TW + tx;
    int KNUM = (N % TS) ? 1 + N / TS : N / TS;
    

    for(int k = 0; k < KNUM; ++k) {
        int K0 = k * TS;
        
        #pragma unroll
        for(int j = 0; j < TS; j += blockDim.y) {
            int Ax = K0 + tx + j;
            int By = K0 + ty + j;

            #pragma unroll
            for(int i = 0; i < TW; i += blockDim.y) {
                int Ay = I + i;
                int Bx = J + i;
    
                AS[ty + i][tx + j] = (Ay < N && Ax < N) ? __ldg(&A[Ay * N + Ax]) : 0.0;
                BS[ty + j][tx + i] = (By < N && Bx < N) ? __ldg(&B[By * N + Bx]) : 0.0;
            }
        }

        __syncthreads();

        for(int kk = 0; kk < TS; ++kk) {

            #pragma unroll
            for(int i = 0; i < OPS; ++i) {

                #pragma unroll
                for(int j = 0; j < OPS; ++j) {
                    CR[i * OPS + j] += AS[ty + i * blockDim.y][kk] * BS[kk][tx + j * blockDim.y]; 
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i = 0; i < OPS; ++i) {
        int ii = I + i * blockDim.y;

        #pragma unroll
        for(int j = 0; j < OPS; ++j) {
            int jj = J + j * blockDim.y;
            if(ii < N && jj < N) C[ii * N + jj] = CR[i * OPS + j];
        }
    }
}
