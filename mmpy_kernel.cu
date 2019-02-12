// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include <stdio.h>

using namespace std;
#define TW BLOCKDIM_X//32

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ double As[TW][TW], Bs[TW][TW];
    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    double Cij = 0;
    double Cij_4 = 0;
    double Cij_8 = 0;
    double Cij_12 = 0;
    double Cij_16 = 0;
    double Cij_20 = 0;
    double Cij_24 = 0;
    double Cij_28 = 0;
    if (N % TW || BLOCKDIM_X != BLOCKDIM_Y * 8) {
        int I = min(N - 1, by * TW + ty);
        int J = min(N - 1, bx * TW + tx);

        if ((I < N) && (J < N)) {
#pragma unroll
            for (int kk = 0; kk < (N / TW + int(bool(N % TW))); kk++) {
                As[ty][tx] = __ldg(&A[I * N + kk * TW + tx]);
                Bs[ty][tx] = __ldg(&B[(kk * TW + ty) * N + J]);

                __syncthreads();
#pragma unroll
                for (int k = 0; k < min(TW, N - kk * TW); k++) {
                    Cij += As[ty][k] * Bs[k][tx];
                }
                __syncthreads();
            }
            C[I * N + J] = Cij;

        }
    } else {
        int I = by * TW + ty;
        int J = bx * TW + tx;
        if ((I < N) && (J < N)) {
#pragma unroll
            for (int kk = 0; kk < N / TW; kk++) {
                As[ty][tx] = __ldg(&A[I * N + kk * TW + tx]);
                Bs[ty][tx] = __ldg(&B[(kk * TW + ty) * N + J]);
                As[ty + (TW / 8)][tx] = __ldg(&A[(I + (TW / 8)) * N + kk * TW + tx]);
                Bs[ty + (TW / 8)][tx] = __ldg(&B[(kk * TW + ty + (TW / 8)) * N + J]);
                As[ty + (TW / 4)][tx] = __ldg(&A[(I + (TW / 4)) * N + kk * TW + tx]);
                Bs[ty + (TW / 4)][tx] = __ldg(&B[(kk * TW + ty + (TW / 4)) * N + J]);
                As[ty + (3 * TW / 8)][tx] = __ldg(&A[(I + (3 * TW / 8)) * N + kk * TW + tx]);
                Bs[ty + (3 * TW / 8)][tx] = __ldg(&B[(kk * TW + ty + (3 * TW / 8)) * N + J]);
                As[ty + (TW / 2)][tx] = __ldg(&A[(I + (TW / 2)) * N + kk * TW + tx]);
                Bs[ty + (TW / 2)][tx] = __ldg(&B[(kk * TW + ty + (TW / 2)) * N + J]);
                As[ty + (5 * TW / 8)][tx] = __ldg(&A[(I + (5 * TW / 8)) * N + kk * TW + tx]);
                Bs[ty + (5 * TW / 8)][tx] = __ldg(&B[(kk * TW + ty + (5 * TW / 8)) * N + J]);
                As[ty + (3 * TW / 4)][tx] = __ldg(&A[(I + (3 * TW / 4)) * N + kk * TW + tx]);
                Bs[ty + (3 * TW / 4)][tx] = __ldg(&B[(kk * TW + ty + (3 * TW / 4)) * N + J]);
                As[ty + (7 * TW / 8)][tx] = __ldg(&A[(I + (7 * TW / 8)) * N + kk * TW + tx]);
                Bs[ty + (7 * TW / 8)][tx] = __ldg(&B[(kk * TW + ty + (7 * TW / 8)) * N + J]);
                __syncthreads();
#pragma unroll
                for (int k = 0; k < TW; k++) {
                    Cij += As[ty][k] * Bs[k][tx];
                    Cij_4 += As[ty + (TW / 8)][k] * Bs[k][tx];
                    Cij_8 += As[ty + (TW / 4)][k] * Bs[k][tx];
                    Cij_12 += As[ty + (3 * TW / 8)][k] * Bs[k][tx];
                    Cij_16 += As[ty + (TW / 2)][k] * Bs[k][tx];
                    Cij_20 += As[ty + (5 * TW / 8)][k] * Bs[k][tx];
                    Cij_24 += As[ty + (3 * TW / 4)][k] * Bs[k][tx];
                    Cij_28 += As[ty + (7 * TW / 8)][k] * Bs[k][tx];
                }
                __syncthreads();
            }
            C[I * N + J] = Cij;
            C[(I + (TW / 8)) * N + J] = Cij_4;
            C[(I + (TW / 4)) * N + J] = Cij_8;
            C[(I + (3 * TW / 8)) * N + J] = Cij_12;
            C[(I + (TW / 2)) * N + J] = Cij_16;
            C[(I + (5 * TW / 8)) * N + J] = Cij_20;
            C[(I + (3 * TW / 4)) * N + J] = Cij_24;
            C[(I + (7 * TW / 8)) * N + J] = Cij_28;
        }
    }
}
