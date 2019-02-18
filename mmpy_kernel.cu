// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

# define SIZEH 16
# define SIZEV 256
#define min(a,b) (((a)<(b))?(a):(b))
// block size C 256 * 16
// block size A 256 * 16
// block size B 16 * 16

//Thread block dim: 256(y) * 1(x)

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    __shared__ _DOUBLE_ BB[SIZEH][SIZEH];
    
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int I = by * SIZEV + ty;
    int J = bx * SIZEH;
    int Bwidth = min(SIZEH, N - J);
    _DOUBLE_ CI[SIZEH] = {0.0};

    for(int k = 0; k < gridDim.x; ++k) {
        int K0 = k * SIZEH;
        int BBy = ty / SIZEH;
        int BBx = ty % SIZEH;
        int By = K0 + BBy;
        int Bx = J + BBx;
        int Bheight = min(SIZEH, N - K0);
        BB[BBy][BBx] = (By < N && Bx < N) ? B[By * N + Bx]: 0.0;
        __syncthreads();

        #pragma unroll
        for(int kk = 0; kk < Bheight; ++kk) {

            #pragma unroll
            for(int i = 0; i < Bwidth; ++i) {
                CI[i] += (I < N) ? A[I * N + K0 + kk] * BB[kk][i] : 0.0;
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for(int i = 0; i < Bwidth; ++i) {
        if(I < N) C[I * N + J + i] = CI[i];
    }
}
