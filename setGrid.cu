#define TW BLOCKDIM_X

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   // set your block dimensions and grid dimensions here
   blockDim.y = (n%TW || BLOCKDIM_X != BLOCKDIM_Y*8) ? blockDim.x : blockDim.y;
   gridDim.x = n / blockDim.x;
   gridDim.y = n / blockDim.y;
   if(n % blockDim.x != 0)
   	gridDim.x++;
   if(n % blockDim.y != 0)
    	gridDim.y++;
}
