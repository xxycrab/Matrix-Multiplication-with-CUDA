void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
    // set your block dimensions and grid dimensions here
    // remember to edit these two parameters each time you change the block size
    gridDim.x = n / (blockDim.x * 2);
    gridDim.y = n / (blockDim.y * 2);
    if(n % (blockDim.x*2) != 0)
        gridDim.x++;
    if(n % (blockDim.y*2) != 0)
        gridDim.y++;
}
