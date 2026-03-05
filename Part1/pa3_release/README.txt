Last name of Student 1: Houlihan
First name of Student 1: Kai
Email of Student 1: khoulihan@ucsb.edu
GradeScope account name of Student 1: khoulihan
Last name of Student 2: Vannier
First name of Student 2: Olivier
Email of Student 2: ovannier@ucsb.edu
GradeScope account name of Student 2: ovannier


----------------------------------------------------------------------------
Report for Question 1 

List your code change for this question 

>>>>>>>>>>>>>>>>>>>
int idx = blockIdx.x * blockDim.x + threadIdx.x; /*Assign a linearized thread ID, which
										 will be used to determine what I own*/

	for (int i = 0; i < rows_per_thread; i++)
	{
		int row_index = idx * rows_per_thread + i;
		double sum = d[row_index];
		for (int j = 0; j < n; j++)
		{
			sum += A[row_index * n + j] * x[j];
		}
		y[row_index] = sum;

		diff[row_index] = fabs(sum - x[row_index]);
	}
<<<<<<<<<<<<<<<<<<<

Parallel time for n=4K, t=1K,  4x128  threads
4.101887s

Parallel time for n=4K, t=1K,  8x128  threads
2.050574s

Parallel time for n=4K, t=1K,  16x128 threads
1.035280s

Parallel time for n=4K, t=1K,  32x128 threads
0.551504s

Do you see a trend of  speedup improvement  with more threads? We expect a good speedup and explain the reason.

Yes, whenever the block count is doubled, there is about a 2x speedup from one size to the other. Since this method utilizes Jacobi,
all data is in-sync so increasing the amount of threads allows each iteration to run faster while having no negative effect on data sharing
between threads unlike asynchronous Gauss-Seidel which operates on both new and old values. 

----------------------------------------------------------------------------


Report for Question 2 
List your code change for this question

>>>>>>>>>>>>>>>>>>>
int idx = blockIdx.x * blockDim.x + threadIdx.x; /*Assign a linearized thread ID, so I can
										   be responsible for some rows*/

	for (int i = 0; i < rows_per_thread; i++)
	{
		int row_index = idx * rows_per_thread + i;
		y[row_index] = x[row_index]; // Start with current value of x
	}
	for (int k = 0; k < num_async_iter; k++)
	{
		/*Perform asynchronous Gauss-Seidel method for y=d+Ay*/
		/*Your solution*/
		for (int i = 0; i < rows_per_thread; i++)
		{
			int row_index = idx * rows_per_thread + i;
			float sum = d[row_index];
			for (int j = 0; j < n; j++)
			{
				sum += A[row_index * n + j] * y[j];
			}
			y[row_index] = sum;
		}
	}
<<<<<<<<<<<<<<<<<<<

Let the default number of asynchronous iterations be 5 in a batch as specified in it_mult_vec.h.
List reported parallel time and the number of actual iterations executed  for n=4K, t=1K, 8x128  threads with asynchronous Gauss Seidel
0.038567s
# of iterations executed: 15.

List reported parallel time and the number of actual iterations executed  for n=4K, t=1K,  32x128 threads with asynchronous Gauss Seidel
0.439733s
# of iterations executed: 1025.

Sequential Time/Iterations (n = 4k, t = 1k):
0.133155s
# of iterations executed: 6.


Is the number of iterations  executed by  above parallel asynchronous Gauss Seidel-Seidel method  bigger or smaller  than that
of the sequential Gauss Seidel-Seidel code under the same converging error threshold (1e-3)?  
Explain the reason based on the running trace of above two thread configurations that more threads may not yield more time reduction in this case. 

The number of iterations for parallel asynchronous Gauss-Seidel is typically larger than sequential Gauss-Seidel. Increasing the number of 
available threads doesn't always result in a greater time reduction due to the nature of asynchronous Gauss-Seidel. Since it operates 
asynchronously, each iteration across each thread/block reads and operates on both old and new values, resulting in more iterations 
in total. Each iteration is faster in parallel but the benefits from that grow less as the number of total iterations increases.
 While it makes parallelization far easier, it may result in smaller time reduction gains as opposed to if the method was synchronous.


Make sure you attach the  output trace  of your code below in running the tests of the unmodified it_mult_vec_test.cu on Expanse GPU for Q1 and Q2

>>>>>>>>>>>>>>>>>>>

>>>>>>>>>>>>>>>>>>>>>>>>>
Start running itmv tests.
>>>>>>>>>>>>>>>>>>>>>>>>>

Test 1:n=4, t=1, 1x2 threads:
With totally 1*2 threads, matrix size being 4, t being 1
Time cost in seconds: 0.107268
Final error (|y-x|): 1.750000.
# of iterations executed: 1.
Final y[0]=1.750000. y[n-1]=1.750000

Test 2:n=4, t=2, 1x2 threads:
With totally 1*2 threads, matrix size being 4, t being 2
Time cost in seconds: 0.000257
Final error (|y-x|): 1.312500.
# of iterations executed: 2.
Final y[0]=0.437500. y[n-1]=0.437500

Test 3:n=8, t=1, 1x2 threads:
With totally 1*2 threads, matrix size being 8, t being 1
Time cost in seconds: 0.000231
Final error (|y-x|): 1.875000.
# of iterations executed: 1.
Final y[0]=1.875000. y[n-1]=1.875000

Test 4:n=8, t=2, 1x2 threads:
With totally 1*2 threads, matrix size being 8, t being 2
Time cost in seconds: 0.000246
Final error (|y-x|): 1.640625.
# of iterations executed: 2.
Final y[0]=0.234375. y[n-1]=0.234375

Test 8a:n=4, t=1, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 4, t being 1
Time cost in seconds: 0.000237
Final error (|y-x|): 1.000193.
# of iterations executed: 5.
Final y[0]=1.000089. y[n-1]=1.000193

Test 8b:n=4, t=2, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 4, t being 2
Time cost in seconds: 0.000230
Final error (|y-x|): 1.000193.
# of iterations executed: 5.
Final y[0]=1.000089. y[n-1]=1.000193

Test 8c:n=8, t=1, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 8, t being 1
Time cost in seconds: 0.000228
Final error (|y-x|): 1.001155.
# of iterations executed: 5.
Final y[0]=1.001155. y[n-1]=0.999790

Test 8d:n=8, t=2, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 8, t being 2
Time cost in seconds: 0.000232
Final error (|y-x|): 1.001155.
# of iterations executed: 5.
Final y[0]=1.001155. y[n-1]=0.999790

Test 9: n=4K t=1K 32x128 threads:
With totally 32*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.551504
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225

Test 9a: n=4K t=1K 16x128 threads:
With totally 16*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 1.035280
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225

Test 9b: n=4K t=1K 8x128 threads:
With totally 8*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 2.050574
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225

Test 9c: n=4K t=1K 4x128 threads:
With totally 4*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 4.101887
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225

Test 11: n=4K t=1K 32x128 threads/Async:
With totally 32*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.439733
Final error (|y-x|): 0.001949.
# of iterations executed: 1025.
Final y[0]=1.000973. y[n-1]=1.000934

Test 11a: n=4K t=1K 8x128 threads/Async:
With totally 8*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.038567
Final error (|y-x|): 0.000000.
# of iterations executed: 15.
Early exit due to convergence, even asked for 1024 iterations.
Asynchronous code actually runs 15 iterations.
Final y[0]=1.000000. y[n-1]=1.000000

Summary: Failed 0 out of 14 tests

<<<<<<<<<<<<<<<<<<<

----------------------------------------------------------------------------

Report for Question 3

List your solution to call  cublasDgemm() in Method 1.
 
 handle_cublas_error( (cublasStatus_t) cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size_N, size_N, size_N, &alpha, d_A, size_N, d_B, size_N, &beta, d_C, size_N),
                 "cublasDgemm (Main)", __FILE__, __LINE__);

List the latency and GFLOPs of the above 3 version of implementation and the number of Cuda threads used in executing Method 3 
when matrix dimension N varies as 50, 200, 800,  and 1600.  

The number of CUDA threads used in Method 3 via the Naive GEMM is gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z. BLOCK_SIZE = 20 and N is the matrix size so threads = ((N + 20 - 1)/20)^2 * 20^2.
For 50x50, we have 3,600
200x200, we have 40,000
800x800, 640,000
1600x1600, 2,560,000
3200x3200, 10,240,000

List the highest gigaflops you have observed with V100 from this question and the highest gigaflops  you have observed from PA2 MKL GEMM code  when N=1600.  
Compute the ratio between these two numbers as the speedup of V100 over a CPU host. 

The highest GFLOPS is consistently cuBLAS dgemm as expected then the naive GEMM and lastly cuBLAS dgemv. here the highest GFLOPS when N=1600 was 6161.696 GFLOPS.
The PA2 MKL GEMM code with N=1600 was 42.67 GFLOPS. Thus speedup is 6161.696 / 42.67 = 144.4 speedup. This is a tremendous speedup and shows why using the cuda library makes such a big difference.

