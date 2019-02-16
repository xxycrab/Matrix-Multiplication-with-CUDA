// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include <stdio.h>

using namespace std;
#define TW BLOCKDIM_X//32

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    int limit = N / TW;

    __shared__ _DOUBLE_ AS[TW][TW];
    __shared__ _DOUBLE_ BS[TW][TW];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        AS[ty + TW / 8 * i][tx] = 0;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        BS[ty + TW / 8 * i][tx] = 0;
    }
    _DOUBLE_ Cij = 0;
    _DOUBLE_ Cij_4 = 0;
    _DOUBLE_ Cij_8 = 0;
    _DOUBLE_ Cij_12 = 0;
    _DOUBLE_ Cij_16 = 0;
    _DOUBLE_ Cij_20 = 0;
    _DOUBLE_ Cij_24 = 0;
    _DOUBLE_ Cij_28 = 0;

    int I = by * TW + ty;
    int J = bx * TW + tx;

// This section of the code handles the corner cases.
    if(N%TW !=0)
    {
        register int a1 = ty - I;
        register int a2 = limit*TW+tx;
#pragma unroll
        for(int Ai =I; (Ai < N) && (Ai < I + TW); Ai +=TW/8)
        {
            if(limit*TW+tx < N)
                AS[a1+Ai][tx] = __ldg(&A[(Ai)*N+a2]);
        }
        register int b_start = limit*TW+ty;
        register int b_1 = limit*TW;
#pragma unroll
        for(int Bi = b_start;(Bi<N) && (Bi < b_start+TW) ; Bi+=TW/8)
        {
            if(J < N)
                BS[Bi - b_1][tx] = __ldg(&B[Bi*N+J]);

        }
        __syncthreads();

        // This section handles the computation of CS
#pragma unroll
        for(int k = 0; k < TW; k++)
        {
            Cij += AS[ty][k] * BS[k][tx];
            Cij_4 += AS[ty + (TW / 8)][k] * BS[k][tx];
            Cij_8 += AS[ty + (TW / 4)][k] * BS[k][tx];
            Cij_12 += AS[ty + (3 * TW / 8)][k] * BS[k][tx];
            Cij_16 += AS[ty + (TW / 2)][k] * BS[k][tx];
            Cij_20 += AS[ty + (5 * TW / 8)][k] * BS[k][tx];
            Cij_24 += AS[ty + (3 * TW / 4)][k] * BS[k][tx];
            Cij_28 += AS[ty + (7 * TW / 8)][k] * BS[k][tx];
        }

    }
    __syncthreads();
    
    // perfect situation

    for (int kk = 0; kk < limit; kk++) {
        AS[ty][tx] = __ldg(&A[I * N + kk * TW + tx]);
        BS[ty][tx] = __ldg(&B[(kk * TW + ty) * N + J]);
        AS[ty + (TW / 8)][tx] = __ldg(&A[(I + (TW / 8)) * N + kk * TW + tx]);
        BS[ty + (TW / 8)][tx] = __ldg(&B[(kk * TW + ty + (TW / 8)) * N + J]);
        AS[ty + (TW / 4)][tx] = __ldg(&A[(I + (TW / 4)) * N + kk * TW + tx]);
        BS[ty + (TW / 4)][tx] = __ldg(&B[(kk * TW + ty + (TW / 4)) * N + J]);
        AS[ty + (3 * TW / 8)][tx] = __ldg(&A[(I + (3 * TW / 8)) * N + kk * TW + tx]);
        BS[ty + (3 * TW / 8)][tx] = __ldg(&B[(kk * TW + ty + (3 * TW / 8)) * N + J]);
        AS[ty + (TW / 2)][tx] = __ldg(&A[(I + (TW / 2)) * N + kk * TW + tx]);
        BS[ty + (TW / 2)][tx] = __ldg(&B[(kk * TW + ty + (TW / 2)) * N + J]);
        AS[ty + (5 * TW / 8)][tx] = __ldg(&A[(I + (5 * TW / 8)) * N + kk * TW + tx]);
        BS[ty + (5 * TW / 8)][tx] = __ldg(&B[(kk * TW + ty + (5 * TW / 8)) * N + J]);
        AS[ty + (3 * TW / 4)][tx] = __ldg(&A[(I + (3 * TW / 4)) * N + kk * TW + tx]);
        BS[ty + (3 * TW / 4)][tx] = __ldg(&B[(kk * TW + ty + (3 * TW / 4)) * N + J]);
        AS[ty + (7 * TW / 8)][tx] = __ldg(&A[(I + (7 * TW / 8)) * N + kk * TW + tx]);
        BS[ty + (7 * TW / 8)][tx] = __ldg(&B[(kk * TW + ty + (7 * TW / 8)) * N + J]);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TW; k++) {
            Cij += AS[ty][k] * BS[k][tx];
            Cij_4 += AS[ty + (TW / 8)][k] * BS[k][tx];
            Cij_8 += AS[ty + (TW / 4)][k] * BS[k][tx];
            Cij_12 += AS[ty + (3 * TW / 8)][k] * BS[k][tx];
            Cij_16 += AS[ty + (TW / 2)][k] * BS[k][tx];
            Cij_20 += AS[ty + (5 * TW / 8)][k] * BS[k][tx];
            Cij_24 += AS[ty + (3 * TW / 4)][k] * BS[k][tx];
            Cij_28 += AS[ty + (7 * TW / 8)][k] * BS[k][tx];
        }
        __syncthreads();
    }


    if ((I < N) && (J < N)) {
        C[I * N + J] = Cij;
        if (I + TW / 8 < N)
            C[(I + TW / 8) * N + J] = Cij_4;
        if (I + TW / 4 < N)
            C[(I + TW / 4) * N + J] = Cij_8;
        if (I + TW * 3 / 8 < N)
            C[(I + TW * 3 / 8) * N + J] = Cij_12;
        if (I + TW / 2 < N)
            C[(I + TW / 2) * N + J] = Cij_16;
        if (I + TW * 5 / 8 < N)
            C[(I + TW * 5 / 8) * N + J] = Cij_20;
        if (I + TW * 3 / 4 < N)
            C[(I + TW * 3 / 4) * N + J] = Cij_24;
        if (I + TW * 7 / 8 < N)
            C[(I + TW * 7 / 8) * N + J] = Cij_28;
        if (I + TW * 7 / 8 < N)
            C[(I + TW * 7 / 8) * N + J] = Cij_28;
    }
}
