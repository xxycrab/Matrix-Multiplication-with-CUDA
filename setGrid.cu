void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   // set your block dimensions and grid dimensions here
   gridDim.x = ceil(((double)n / blockDim.x)/2);
   gridDim.y = ceil(((double)n / blockDim.y)/2);
}
