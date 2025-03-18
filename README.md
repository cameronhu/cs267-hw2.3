# CUDA Parallel Particle Simulation

## Overview

On the GPU exists the full set of the particles in the `particle_t* parts` array. In order to segment `parts` into boxes, I will utilize two arrays. The first array is  `int* particle_idx`, and it stores the indices of the particles needed to access them from `parts`. `particle_idx` stores the particle indices in order of the boxes that they are assigned to. To know the boundaries of each box, we have another `int* boxes` array, and each value stored at `boxes[i]` is the starting index in`particle_idx` of the particles that are stored within box[i]. For example, `boxes[0] = 0`, meaning the first particles of box 0 start at index 0 of `particle_idx`. Then, perhaps `boxes[1] = 2`, meaning the particles starting at idx 2 in `particle_idx` are stored in box 1.

## CPU assignToBoxes

For testing purposes, first pass implementations will all be on the CPU side. Will utilize cudaMemcpy to copy the calculated box assignments to mirrored ararys on the GPU. For assignToBoxes, the passed in `parts` array is a pointer to memory on the GPU. It thus cannot be accessed unless from within a `__global__` or `__device__` function, which are run on the GPU (called from host or device, respectively). To do proper CPU testing, we need to copy the `parts` array from the GPU and store it in `cpu_parts` to be able to properly access the particles on the CPU side.  

In the future, we will rewrite these functions using CUDA kernels so that everything is handled on the GPU side.

## Useful Commands

salloc -A mp309 -N 1 -C gpu -q interactive -t 00:05:00

~/hw2-correctness/correctness-check.py 1000.out ~/hw2-correctness/verf.out
