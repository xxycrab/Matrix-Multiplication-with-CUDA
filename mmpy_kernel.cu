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
    
    //int I =  blockIdx.y*blockDim.y + threadIdx.y;
    //int J =  blockIdx.x*blockDim.x + threadIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int I = by * SIZEV + ty;
    int J = bx * SIZEH;
    int Bwidth = min(SIZEH, N - J);
    _DOUBLE_ CI[SIZEH] = {0};

    for(int k = 0; k < gridDim.x; ++k) {
        int Bheight = min(SIZEH, N - k * SIZEH);
        //BB[ty/SIZEH][ty%SIZEH] = (ty < Bheight * Bwidth) ? B[(k * SIZEH + (ty / Bwidth)) * N + J + (ty % Bwidth)]: 0;
        if(ty < Bheight * Bwidth) BB[ty/Bwidth][ty%Bwidth] = B[(k * SIZEH + (ty / Bwidth)) * N + J + (ty % Bwidth)];
        __syncthreads();

        int Aheight = min(SIZEV, N - by * SIZEV);
        if(ty < Aheight) {
            #pragma unroll
            for(int j = 0; j < Bheight; ++j){

                #pragma unroll
                int a = A[I * N + k * SIZEH + j];
                for(int i = 0; i < Bwidth; ++i) {
                    CI[i] += a * BB[j][i];
                }
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for(int i = 0; i < Bwidth; ++i) {
        C[I * N + J + i] = CI[i];
    }

    // if((I < N) && (J < N)){
    //     _DOUBLE_ _c = 0;
    //     for (unsigned int k = 0; k < N; k++) {
    //         _DOUBLE_ a = A[I * N + k];
    //         _DOUBLE_ b = B[k * N + J];
    //         _c += a * b;
    //     }
    //     C[I * N + J] = _c;
    // }
}
