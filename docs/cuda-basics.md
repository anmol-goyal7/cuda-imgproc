# CUDA Basics — Study Notes

Personal notes on CUDA fundamentals. These are the core concepts needed to understand and write the kernels in this project.

---

## What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA's programming model for general-purpose GPU computing. It extends C++ with syntax for writing functions that run on the GPU (kernels) and managing GPU memory.

Key idea: write a function for **one** element, then launch it on **thousands** of threads that each process a different element in parallel.

---

## Thread Hierarchy

CUDA organizes parallel execution into a hierarchy:

```
Grid (entire kernel launch)
├── Block (0,0)
│   ├── Thread (0,0)
│   ├── Thread (1,0)
│   ├── Thread (2,0)
│   └── ...
├── Block (1,0)
│   ├── Thread (0,0)
│   ├── Thread (1,0)
│   └── ...
├── Block (0,1)
│   └── ...
└── ...
```

- **Thread**: Smallest unit of execution. Each thread has a unique ID within its block.
- **Warp**: 32 threads that execute in lockstep on the same GPU core (SM). This is a hardware detail — you don't configure it, but you need to be aware of it for performance.
- **Block**: A group of threads (up to 1024) that can cooperate via shared memory and synchronize with `__syncthreads()`. Threads in different blocks cannot directly communicate.
- **Grid**: The full set of blocks for one kernel launch. Can be 1D, 2D, or 3D.

### Built-in Variables

```cuda
threadIdx.x, threadIdx.y, threadIdx.z    // Thread index within a block
blockIdx.x, blockIdx.y, blockIdx.z       // Block index within the grid
blockDim.x, blockDim.y, blockDim.z       // Number of threads per block
gridDim.x, gridDim.y, gridDim.z          // Number of blocks in the grid
```

### Computing a Global Thread Index

```cuda
// 1D grid of 1D blocks
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid of 2D blocks (for images)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

---

## Memory Hierarchy

From fastest to slowest:

| Memory | Scope | Size | Latency | Notes |
|---|---|---|---|---|
| **Registers** | Per-thread | ~256 KB per SM | 0 cycles | Fastest. Local variables live here |
| **Shared Memory** | Per-block | 48–164 KB per SM | ~5 cycles | Programmer-managed cache. Key to optimization |
| **L2 Cache** | All SMs | 4–40 MB | ~200 cycles | Automatic, hardware-managed |
| **Global Memory** | All threads | 8–80 GB (VRAM) | ~400 cycles | Slow but large. Input/output data lives here |

### Key Insight

Global memory is ~100x slower than shared memory. If multiple threads in a block need the same data, load it into shared memory once and reuse it. This is the core optimization for convolution.

---

## Kernel Launch Syntax

```cuda
// Declaration
__global__ void my_kernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = data[i] * 2.0f;
    }
}

// Launch
int N = 1000000;
int threads_per_block = 256;
int blocks = (N + threads_per_block - 1) / threads_per_block;
my_kernel<<<blocks, threads_per_block>>>(d_data, N);

// 2D launch (for images)
dim3 block(16, 16);          // 256 threads per block
dim3 grid((width + 15) / 16, (height + 15) / 16);
image_kernel<<<grid, block>>>(d_img, width, height);
```

The `<<<blocks, threads>>>` syntax is called the **execution configuration**. It determines how many threads are launched and how they're organized.

---

## Common Patterns

### Boundary Checking

Always check that the thread index is within bounds. The grid size is rounded up to a multiple of the block size, so some threads in the last block(s) will have out-of-bounds indices.

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) {
    // safe to access data[i]
}
```

### Memory Coalescing

Adjacent threads should access adjacent memory locations. This allows the hardware to combine multiple memory accesses into a single transaction.

```
Good (coalesced):   Thread 0 → data[0], Thread 1 → data[1], Thread 2 → data[2]
Bad (strided):      Thread 0 → data[0], Thread 1 → data[100], Thread 2 → data[200]
```

For images stored in row-major order, iterate over x (columns) in the inner loop and assign consecutive threads to consecutive columns.

### Shared Memory Usage

```cuda
__global__ void kernel(...) {
    __shared__ float cache[BLOCK_SIZE];

    // Load from global memory into shared memory
    cache[threadIdx.x] = global_data[...];

    // Wait for ALL threads in the block to finish loading
    __syncthreads();

    // Now safe to read any element in cache[]
    float neighbor = cache[threadIdx.x + 1];
}
```

`__syncthreads()` is a barrier — no thread in the block proceeds past it until every thread has reached it.

### Error Checking

```cuda
// Check for launch errors
kernel<<<grid, block>>>(args);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Launch failed: %s\n", cudaGetErrorString(err));
}

// Check for execution errors
cudaDeviceSynchronize();
err = cudaGetLastError();
```

---

## Memory Transfers

```cuda
// Allocate on GPU
float* d_data;
cudaMalloc(&d_data, N * sizeof(float));

// Copy host → device
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

// Copy device → host
cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_data);
```

Transfer bandwidth is limited by the PCIe bus (~12–32 GB/s). For small data, transfer time can dominate kernel execution time. Minimize transfers by keeping data on the GPU across multiple kernel calls.
