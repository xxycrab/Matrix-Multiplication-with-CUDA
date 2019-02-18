// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include <stdio.h>

using namespace std;
#define TW BLOCKDIM_X // block size
#define NUMOPT 8 // 8 output per thread

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    int ystep = TW / NUMOPT;

    __shared__ _DOUBLE_ AS[TW][TW];
    __shared__ _DOUBLE_ BS[TW][TW];

    _DOUBLE_ CR[NUMOPT] = {0.0};

    int I = by * TW + ty;
    int J = bx * TW + tx;
    

    for(int k = 0; k < gridDim.x; ++k) {
        int K0 = k * TW;
        #pragma unroll
        for(int i = 0; i < TW; i += ystep) {
            int Ay = I + i;
            int Ax = K0 + tx;
            int By = K0 + ty + i;

            AS[ty + i][tx] = (Ay < N && Ax < N) ? __ldg(&A[Ay * N + Ax]) : 0.0;
            BS[ty + i][tx] = (By < N &&  J < N) ? __ldg(&B[By * N +  J]) : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for(int kk = 0; kk < TW; ++kk) {

            #pragma unroll
            for(int i = 0; i < NUMOPT; ++i) {
                CR[i] += AS[ty + i * ystep][kk] * BS[kk][tx];
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < NUMOPT; ++i) {
        int ii = I + i * ystep;
        if(ii < N && J < N) C[ii * N + J] = CR[i];
    }
}
