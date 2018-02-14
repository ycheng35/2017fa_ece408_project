#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define STREAM_TOT 32
#define MAX_THREADS 256
#define TILE_WIDTH_MULT 32

#include <mxnet/base.h>



namespace mxnet
{
namespace op
{

template<typename gpu, typename DType>
__global__ void matrixMultiplyShared(DType *A, DType *B, DType *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{

  __shared__ float subTileM[TILE_WIDTH_MULT][TILE_WIDTH_MULT];
  __shared__ float subTileN[TILE_WIDTH_MULT][TILE_WIDTH_MULT];
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x;  int ty = threadIdx.y;
  
  int Row = by*TILE_WIDTH_MULT + ty;
  int Col = bx*TILE_WIDTH_MULT + tx;
  float pvalue = 0;
  for (int m=0; m<(numAColumns-1)/TILE_WIDTH_MULT+1; ++m){
    if (Row < numARows && m*TILE_WIDTH_MULT+tx < numAColumns){
      subTileM[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH_MULT+tx];
    }
    else {
      subTileM[ty][tx] = 0;
    }
    if (Col < numBColumns && m*TILE_WIDTH_MULT+ty < numBRows){
      subTileN[ty][tx] = B[(m*TILE_WIDTH_MULT+ty)*numBColumns + Col];
    }
    else {
      subTileN[ty][tx] = 0;
    }
    __syncthreads();
    if (Row < numARows && Col < numBColumns){
      for (int k=0; k<TILE_WIDTH_MULT; ++k){
        pvalue += subTileM[ty][k] * subTileN[k][tx];
      }
    }
    __syncthreads();  
  }
  if (Row < numARows && Col < numBColumns){
    C[Row*numCColumns+Col] = pvalue;
  }
  
}

template<typename gpu, typename DType>
__global__ void unroll_Kernel(int C, int H, int W, int K, DType* X, DType* X_unroll) {
  int c, s, h_out, w_out, p, q, w_unroll, w_base, h_unroll;
  int t = blockIdx.x * MAX_THREADS + threadIdx.x;
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int W_unroll = H_out * W_out;
  
  #define x3d(i3,i2,i1) X[(i3) * (H * W) + (i2)*(W) + (i1)]
  if (t < C*W_unroll) {
    c = t/W_unroll;
    s = t%W_unroll;
    h_out = s/W_out;
    w_out = s%W_out;
    h_unroll = h_out * W_out + w_out;
    w_base = c*K*K;
    for(p = 0; p < K; p++) {
      for(q = 0; q < K; q++) {
        w_unroll = w_base + p * K + q;
        //X_unroll[h_unroll, w_unroll] = X[c, h_out + p, w_out + q]
        X_unroll[(w_unroll)*W_unroll + h_unroll] = x3d(c,(h_out + p),(w_out + q)); 
      }
    }
  }
  #undef x3d
}


// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    const int B = x.shape_[0];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int M = y.shape_[1];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    //variables
    int HW = H_out*W_out;
    int CHW = C*H_out*W_out;
    int numARows = M;    
    int numAColumns = C*K*K; 
    int numBRows = C*K*K;
    int numBColumns = H_out*W_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    DType* xout;
    int curStream;
    cudaStream_t s;

    //use multiple stream to parallel the execution
    cudaStream_t stream_arr[STREAM_TOT];
    for (int i = 0; i < STREAM_TOT; ++i) {
      cudaStreamCreate(&stream_arr[i]); 
    }

    cudaMalloc((void**) &xout, sizeof(DType) * HW * numAColumns * STREAM_TOT);


    for (int n = 0; n < B; n++) {
        curStream = n % STREAM_TOT;
        s = stream_arr[curStream];

        int blockDim = ceil(float(CHW) / float(MAX_THREADS));
        unroll_Kernel <gpu, DType> <<<blockDim, MAX_THREADS, 0, s>>> (C, H, W, K, x.dptr_ + n*C*H*W, xout+curStream*HW*numAColumns);


        //matrix multiply
        dim3 DimGrid((numCColumns-1)/TILE_WIDTH_MULT+1, (numCRows-1)/TILE_WIDTH_MULT+1, 1);
        dim3 DimBlock(TILE_WIDTH_MULT, TILE_WIDTH_MULT, 1);

        matrixMultiplyShared <gpu, DType> <<<DimGrid, DimBlock, 2*TILE_WIDTH_MULT*TILE_WIDTH_MULT, s>>> (w.dptr_, xout+curStream*HW*numAColumns, y.dptr_+n*M*H_out*W_out,numARows, numAColumns,                                      numBRows, numBColumns, numCRows, numCColumns);
    }

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}



}
}

#endif


