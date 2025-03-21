# CUDA Parallel Particle Simulation

## Overview

On the GPU exists the full set of the particles in the `particle_t* parts` array. In order to segment `parts` into boxes, I will utilize two arrays. The first array is  `int* particle_idx`, and it stores the indices of the particles needed to access them from `parts`. `particle_idx` stores the particle indices in order of the boxes that they are assigned to. To know the boundaries of each box, we have another `int* boxes` array, and each value stored at `boxes[i]` is the starting index in`particle_idx` of the particles that are stored within box[i]. For example, `boxes[0] = 0`, meaning the first particles of box 0 start at index 0 of `particle_idx`. Then, perhaps `boxes[1] = 2`, meaning the particles starting at idx 2 in `particle_idx` are stored in box 1.

## CPU assignToBoxes

For testing purposes, first pass implementations will all be on the CPU side. Will utilize cudaMemcpy to copy the calculated box assignments to mirrored ararys on the GPU. For assignToBoxes, the passed in `parts` array is a pointer to memory on the GPU. It thus cannot be accessed unless from within a `__global__` or `__device__` function, which are run on the GPU (called from host or device, respectively). To do proper CPU testing, we need to copy the `parts` array from the GPU and store it in `cpu_parts` to be able to properly access the particles on the CPU side.  

In the future, we will rewrite these functions using CUDA kernels so that everything is handled on the GPU side.

## GPU boxed force calculations

Currently, each thread is assigned to a particle. It then checks each other particle in the parts array (on the GPU) to calculate forces, leading to O(N^2) runtime. I have created box assignment logic using the prefixSums and particle_ids arrays. For each particle, I can find the box this particle is assigned to. Then, using these two arrays, I can perform force calculations only for other particles within the same box, and other particles within neighboring boxes.

In each simulation step, I call the `compute_forces_gpu` CUDA kernel, which assigns a single thread per particle. The threads are assigned to the order of particles defined by the `particle_ids` array, which means in order of boxes. This ensures that each thread accesses at least the `particle_ids` contiguously. Need to test if just accessing from the `particles` array directly is better. 

For each particle, I call the `apply_force_from_neighbor_gpu` CUDA kernel for each of the neighboring boxes. Currently the calling of this neighbor forces kernel is loop unrolled, but need to profile and check if this provides actual performance improvements. The `apply_force_from_neighbor_gpu` iterates through all particles in the neighboring box, and calls the given `apply_force_gpu` kernel to apply forces from the neighbor particle to thisParticle.

## GPU assignToBoxes

The idea is to parallelize counting the number of particles per box, computation of the prefix sum, and assignment of `parts` indices to `particle_ids` through the GPU. Each thread will have access to the shared `gpu_boxCounts`, `gpu_prefixSums`, and `gpu_particle_ids` arrays. Atomic operations for adding in `gpu_boxCounts` will be necessary to prevent race conditions. Specifically, I can use `atomicAdd(int* address, int val)` to add val to the integer array at address `int* address`. Need to explore the best method for computing a prefixSum on the GPU: probably `thrust::exclusive_scan`. The `thrust` library allows for me to perform reductions and scans (such as generating a prefixSum) on the GPU without explicitly defining a CUDA kernel. The `thrust` library already implemented a parallel prefixSum calculation, and I just leveraged the existing functions to do so. Important to note is that `exclusive_scan` omits the very last prefixSum at index totalBoxes, which is just used to calculate the ending index for the last box. That value was manually added into the `gpu_prefixSums` array.

## Debugging Information For GPU countParticlesPerBox

For 1000 particles, 71 boxes by 71 boxes. boxSize1D: 0.01  

cur parts idx: 63. boxIndex: 3638. Coords: (0.177000, 0.519787)
Row = 51. Col = 17.
BoxIndex = row * 71 + col = 3638.

Box calculation is correct on the GPU side.

Sum of gpu_boxCounts: 999

For some reason, not all the particles are being assigned to a box initially.

[X] FIXED: atomicAdd ensures all particles are counted and added to boxCounts
[X] FIXED: correctness check. Incorrectly changed memory size when copying gpu_particles to cpu particles.

## Initial Kernel Implementation Times

- First pass for CUDA kernels for boxCounts, prefixSums, particle_ids assignment, and force calculations. With -o outputting.
  - | # Particles  | Time (s) |
    |---|---|
    | 1000  | 0.296442 |
    | 10000  | 2.15889 |
    | 100000  | 20.4864 |
- Without -o outputting
  - | # Particles  | Time (s) |
    |---|---|
    | 1,000  | 0.0802011 |
    | 10,000  | 0.0832291 |
    | 100,000  | 0.116139 |
    | 1,000,000  | 1.49235 |
    | 10,000,000  | 15.6889 |

## Useful Commands

salloc -A mp309 -N 1 -C gpu -q interactive -t 00:05:00

./gpu -s 1 -o $SCRATCH/1000.out
./gpu -s 1 -n 10000 -o $SCRATCH/10k.out
./gpu -s 1 -n 100000 -o $SCRATCH/100k.out

~/hw2-correctness/correctness-check.py $SCRATCH/1000.out ~/hw2-correctness/verf.out
~/hw2-correctness/correctness-check.py $SCRATCH/10k.out ~/hw2-correctness/10k.out

~/hw2-rendering/render.py 1000.out 1000_gpu.gif 0.01
