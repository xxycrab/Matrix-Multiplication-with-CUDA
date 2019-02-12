
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"

using namespace std;
#define BLOCK_SIZE 32
#define ROW_BLOCK_SIZE BLOCK_SIZE
#define COL_BLOCK_SIZE BLOCK_SIZE
#define NUM_SIMULTANEOUS_C_ROW_ELEMENTS 2
#define NUM_SIMULTANEOUS_C_COL_ELEMENTS 2

__global__ void matMul(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    int num_elements = square_dim * square_dim;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    //since we are operating on adjacent squares
    int by = (blockIdx.y * NUM_SIMULTANEOUS_C_COL_ELEMENTS);
    int bx = (blockIdx.x * NUM_SIMULTANEOUS_C_ROW_ELEMENTS);

    __shared__ _DOUBLE_ A_0y[ROW_BLOCK_SIZE][COL_BLOCK_SIZE];
    __shared__ _DOUBLE_ A_1y[ROW_BLOCK_SIZE][COL_BLOCK_SIZE];
    __shared__ _DOUBLE_ B_x0[COL_BLOCK_SIZE][ROW_BLOCK_SIZE];
    __shared__ _DOUBLE_ B_x1[COL_BLOCK_SIZE][ROW_BLOCK_SIZE];

    // the four element to be updated
    _DOUBLE_ c_00 = 0;
    _DOUBLE_ c_01 = 0;
    _DOUBLE_ c_10 = 0;
    _DOUBLE_ c_11 = 0;

    int r0 = (by) * ROW_BLOCK_SIZE + ty;
    int r1 = (by + 1) * ROW_BLOCK_SIZE + ty;
    int c0 = (bx) * COL_BLOCK_SIZE + tx;
    int c1 = (bx + 1) * COL_BLOCK_SIZE + tx;

    //parameters for loop
    int B_step = ROW_BLOCK_SIZE * square_dim;
    int B_x0_index = c0 + ty * square_dim;
    int B_x1_index = c1 + ty * square_dim;
    int A_0y_index = r0 * square_dim + tx;
    int A_1y_index = r1 * square_dim + tx;

#pragma unroll
    for (unsigned int stride = 0;
         stride < gridDim.x * NUM_SIMULTANEOUS_C_COL_ELEMENTS;
         ++stride
                 , A_0y_index += COL_BLOCK_SIZE
                 , A_1y_index += COL_BLOCK_SIZE
                 , B_x0_index += B_step
                 , B_x1_index += B_step) {

/*load sub-blocks into shared memory: each thread does one load to each array*/
        //check if both rows of A are within block
        if (A_1y_index < num_elements) {
            A_0y[ty][tx] = A[A_0y_index];
            A_1y[ty][tx] = A[A_1y_index];
        } else {// if second row out of border, padding with 0
            A_1y[ty][tx] = 0;
            //check if A0y is within block
            if (A_0y_index < num_elements) {
                A_0y[ty][tx] = A[A_0y_index];
            } else {// if not, padding with 0
                A_0y[ty][tx] = 0;
            }
        }

        // same as for sub block A
        if (B_x1_index < num_elements) {
            B_x0[ty][tx] = B[B_x0_index];
            B_x1[ty][tx] = B[B_x1_index];
        } else {
            B_x1[ty][tx] = 0;
            if (B_x0_index < num_elements) {
                B_x0[ty][tx] = B[B_x0_index];
            } else {
                B_x0[ty][tx] = 0;
            }
        }
        __syncthreads();


/* Compute and update c_00 - c_11
 * Each thread within the block dim loops over rows to add results
 * Note: due to thread divergence, produced marginally better results than having all threads compute*/

        if (r1 < square_dim) { // if within row bound
            if (c1 < square_dim) {//all fit
#pragma unroll
                for (unsigned int k = 0; k < COL_BLOCK_SIZE; ++k) {
                    c_00 += A_0y[ty][k] * B_x0[k][tx];//
                    c_01 += A_0y[ty][k] * B_x1[k][tx];//
                    c_10 += A_1y[ty][k] * B_x0[k][tx];//
                    c_11 += A_1y[ty][k] * B_x1[k][tx];//
                }
            } else if (c0 < square_dim) {//if within col bound
#pragma unroll
                for (unsigned int k = 0; k < COL_BLOCK_SIZE; ++k) {
                    c_00 += A_0y[ty][k] * B_x0[k][tx];//
                    c_10 += A_1y[ty][k] * B_x0[k][tx];//
                }
            }
        } else if ((r0 < square_dim)) { // if within matrix bounds
            if (c1 < square_dim) {//both cols fit
#pragma unroll
                for (unsigned int k = 0; k < COL_BLOCK_SIZE; ++k) {
                    c_00 += A_0y[ty][k] * B_x0[k][tx];//
                    c_01 += A_0y[ty][k] * B_x1[k][tx];//
                }
            } else if (c0 < square_dim) {//only 1 col fits
#pragma unroll
                for (unsigned int k = 0; k < COL_BLOCK_SIZE; ++k) {
                    c_00 += A_0y[ty][k] * B_x0[k][tx];//
                }
            }
        }
        __syncthreads();
    }
// Update C matrix
    if (r1 < square_dim) { // if within row bound
        if (c1 < square_dim) {//all fit
            C[r0 * square_dim + c0] = c_00;
            C[r0 * square_dim + c1] = c_01;
            C[r1 * square_dim + c0] = c_10;
            C[r1 * square_dim + c1] = c_11;
        } else if (c0 < square_dim) {//if within col bound
            C[r0 * square_dim + c0] = c_00;
            C[r1 * square_dim + c0] = c_10;
        }
    } else if ((r0 < square_dim)) { // if within matrix bounds
        if (c1 < square_dim) {//both cols fit
            C[r0 * square_dim + c0] = c_00;
            C[r0 * square_dim + c1] = c_01;
        } else if (c0 < square_dim) {//only 1 col fits
            C[r0 * square_dim + c0] = c_00;
        }
    }
}

