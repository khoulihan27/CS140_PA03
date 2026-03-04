# Statistics
CUDA matrix-matrix multiply (50x50 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 0.113 ms   GFLOPS:  2.219
2. cuBLAS dgemv in a loop for GEMM     Latency: 0.156 ms   GFLOPS:  1.606
3. Naive GEMM (parallelized 3 loops)   Latency: 0.009 ms   GFLOPS:  27.127

Mid-point verification looks OK: DGEMM=12.7049, DGEMV=12.7049, Naive=12.7049

CUDA matrix-matrix multiply (200x200 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 12.547 ms   GFLOPS:  1.275
2. cuBLAS dgemv in a loop for GEMM     Latency: 3.375 ms   GFLOPS:  4.741
3. Naive GEMM (parallelized 3 loops)   Latency: 0.030 ms   GFLOPS:  538.793

Mid-point verification looks OK: DGEMM=48.5952, DGEMV=48.5952, Naive=48.5952

CUDA matrix-matrix multiply (800x800 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 0.230 ms   GFLOPS:  4456.824
2. cuBLAS dgemv in a loop for GEMM     Latency: 13.269 ms   GFLOPS:  77.172
3. Naive GEMM (parallelized 3 loops)   Latency: 1.051 ms   GFLOPS:  974.659

Mid-point verification looks OK: DGEMM=203.2539, DGEMV=203.2539, Naive=203.2539

CUDA matrix-matrix multiply (1600x1600 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 1.330 ms   GFLOPS:  6161.696
2. cuBLAS dgemv in a loop for GEMM     Latency: 46.572 ms   GFLOPS:  175.901
3. Naive GEMM (parallelized 3 loops)   Latency: 8.386 ms   GFLOPS:  976.920

Mid-point verification looks OK: DGEMM=385.0199, DGEMV=385.0199, Naive=385.0199

CUDA matrix-matrix multiply (3200x3200 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 9.352 ms   GFLOPS:  7008.010
2. cuBLAS dgemv in a loop for GEMM     Latency: 314.202 ms   GFLOPS:  208.579
3. Naive GEMM (parallelized 3 loops)   Latency: 58.999 ms   GFLOPS:  1110.803

Mid-point verification looks OK: DGEMM=800.1449, DGEMV=800.1449, Naive=800.1449

# Answers
I called the following function to cublasDgemm

cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size_N, size_N, size_N, &alpha, d_A, size_N, d_B, size_N, &beta, d_C, size_N),


The number of CUDA threads used in Method 3 via the Naive GEMM is gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z. BLOCK_SIZE = 20 and N is the matrix size so threads = ((N + 20 - 1)/20)^2 * 20^2.
For 50x50, we have 3,600
200x200, we have 40,000
800x800, 640,000
1600x1600, 2,560,000
3200x3200, 10,240,000

The highest GFLOPS is consistently cuBLAS dgemm as expected then the naive GEMM and lastly cuBLAS dgemv. here the highest GFLOPS when N=1600 was 6161.696 GFLOPS.
The PA2 MKL GEMM code with N=1600 was 42.67 GFLOPS. Thus speedup is 6161.696 / 42.67 = 144.4 speedup. This is a tremendous speedup and shows why using the cuda library makes such a big difference.
