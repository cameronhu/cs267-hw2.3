#include "common.h"
#include <cuda.h>
#include <stdio.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <numeric>

#define NUM_THREADS 256
#define INDEX(row, col) ((row) * numBoxes1D + (col))

// ---------------------------------------------------------------------
// Global variables (or static). 
// ---------------------------------------------------------------------
int    blks;
double boxSize1D = cutoff;
int    numBoxes1D;
int    totalBoxes;

// We'll store both counts & prefix sums in a single array [0..totalBoxes] 
//   (the last entry = total # of particles).
size_t boxPrefixMemSize;
size_t particleIDMemSize;

// Single GPU array for "counts + prefix sums"
int* gpu_boxPrefix       = nullptr;

// Single GPU array to hold the sorted list of particle IDs
int* gpu_particle_ids    = nullptr;


// ---------------------------------------------------------------------
// Device/Host utility to find the box index
// ---------------------------------------------------------------------
__device__ __host__
int findBox(const particle_t& p, int numBoxes1D, double boxSize1D)
{
    int col = (int)floor(p.x / boxSize1D);
    int row = (int)floor(p.y / boxSize1D);
    return INDEX(row, col);
}

// ---------------------------------------------------------------------
// Device function to apply force from neighbor
// ---------------------------------------------------------------------
__device__
void apply_force_gpu(particle_t& particle, const particle_t& neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;

    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1.0 - cutoff / r) / (r2) / mass;

    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// ---------------------------------------------------------------------
// For each neighboring box, read its subrange [startIdx, endIdx)
//    in the gpu_particle_ids array, and apply forces
// ---------------------------------------------------------------------
__device__
void apply_force_from_neighbor_gpu(int row, int col,
                                   particle_t& thisParticle,
                                   particle_t* particles,
                                   int*        gpu_particle_ids,
                                   int*        gpu_boxPrefix,
                                   int         numBoxes1D,
                                   double      boxSize1D)
{
    // Check bounds
    if (row < 0 || row >= numBoxes1D) return;
    if (col < 0 || col >= numBoxes1D) return;

    int boxIndex = INDEX(row, col);

    int startIdx = gpu_boxPrefix[boxIndex];
    int endIdx   = gpu_boxPrefix[boxIndex + 1];

    for (int i = startIdx; i < endIdx; ++i) {
        int neighborID = gpu_particle_ids[i];
        apply_force_gpu(thisParticle, particles[neighborID]);
    }
}

// ---------------------------------------------------------------------
// Kernel: compute_forces_gpu
//    We assume that gpu_particle_ids[] has been populated in "box order."
//    So thread tid is in [0..num_parts).  We interpret that as the 
//    "subrange index" in a box-ordered array. 
//    Then we find which actual particle that corresponds to, apply 
//    neighbor computations, etc.
// ---------------------------------------------------------------------
__global__
void compute_forces_gpu(particle_t* particles,
                        int         num_parts,
                        int*        gpu_particle_ids,
                        int*        gpu_boxPrefix,
                        int         numBoxes1D,
                        double      boxSize1D)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // 'tid' is the index in the "box-ordered" particle list
    int pid = gpu_particle_ids[tid];
    particle_t& thisParticle = particles[pid];
    thisParticle.ax = 0.0;
    thisParticle.ay = 0.0;

    // find the box for this particle
    int row = (int)floor(thisParticle.y / boxSize1D);
    int col = (int)floor(thisParticle.x / boxSize1D);

    // Apply neighbor forces from the 8 surrounding boxes + self
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            apply_force_from_neighbor_gpu(row + dr, col + dc,
                                          thisParticle,
                                          particles,
                                          gpu_particle_ids,
                                          gpu_boxPrefix,
                                          numBoxes1D,
                                          boxSize1D);
        }
    }
}

// ---------------------------------------------------------------------
// Kernel: move_gpu (Velocity Verlet + bounce from walls)
// ---------------------------------------------------------------------
__global__
void move_gpu(particle_t* particles,
              int         num_parts,
              double      size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particle_t* p = &particles[tid];

    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    // bounce from walls
    while (p->x < 0 || p->x > size) {
        p->x  = (p->x < 0) ? -(p->x) : 2.0 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y  = (p->y < 0) ? -(p->y) : 2.0 * size - p->y;
        p->vy = -(p->vy);
    }
}

// ---------------------------------------------------------------------
// Kernel: count how many particles land in each box
//         store that in gpu_boxPrefix[boxIndex]. 
// ---------------------------------------------------------------------
__global__
void countParticlesPerBox(particle_t* gpu_parts,
                          int         num_parts,
                          int*        gpu_boxPrefix,
                          int         numBoxes1D,
                          double      boxSize1D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    int boxIndex = findBox(gpu_parts[tid], numBoxes1D, boxSize1D);
    atomicAdd(&gpu_boxPrefix[boxIndex], 1);
}

// ---------------------------------------------------------------------
// computePrefixSum: do an exclusive scan over the first "totalBoxes"
//                   entries in gpu_boxPrefix
//  After scanning, 
//      gpu_boxPrefix[i]   = sum of counts for boxes [0..i-1],
//      gpu_boxPrefix[i+1] = sum of counts for boxes [0..i],
//  and so forth. We'll also fill gpu_boxPrefix[totalBoxes] with the total
//  number of particles, just for convenience. 
// ---------------------------------------------------------------------
void computePrefixSum(int* gpu_boxPrefix, int totalBoxes, int num_parts)
{
    thrust::device_ptr<int> devPtr(gpu_boxPrefix);

    // exclusive scan of the first "totalBoxes" elements
    thrust::exclusive_scan(devPtr, devPtr + totalBoxes, devPtr);

    // we also want to store the total # of particles in [totalBoxes]
    // i.e. prefix[totalBoxes] = prefix[totalBoxes-1] + the old count of last box
    // But we know the sum must be "num_parts," so let's just store num_parts:
    cudaMemcpy(gpu_boxPrefix + totalBoxes,
               &num_parts,
               sizeof(int),
               cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------
// Kernel: place each particle ID in gpu_particle_ids, using the 
//         (already scanned) boxPrefix array. 
//   "pos = prefix[boxIndex] + atomicAdd(...)" approach overwrites the 
//   prefix array entries for each box, but that's OK if we don't need 
//   them afterward.
// ---------------------------------------------------------------------
__global__
void populateParticleID(particle_t* gpu_parts,
                        int         num_parts,
                        int*        gpu_boxPrefix,
                        int*        gpu_particle_ids,
                        int         numBoxes1D,
                        double      boxSize1D)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // find which box
    int boxIndex = findBox(gpu_parts[tid], numBoxes1D, boxSize1D);

    // The position for this particle ID in the final array
    //   =  old prefix sum + "how many we've inserted so far"
    int pos = atomicAdd(&gpu_boxPrefix[boxIndex], 1);

    // store the ID
    gpu_particle_ids[pos] = tid;
}

// ---------------------------------------------------------------------
// HELPER (like your old assignToBoxes) but broken into two smaller steps
// ---------------------------------------------------------------------
static void countAndScan(particle_t* gpu_parts, int num_parts)
{
    // Clear the array [0..(totalBoxes-1)]
    cudaMemset(gpu_boxPrefix, 0, totalBoxes * sizeof(int));

    // Kernel: count how many in each box
    countParticlesPerBox<<<blks, NUM_THREADS>>>(
        gpu_parts, num_parts,
        gpu_boxPrefix,
        numBoxes1D,
        boxSize1D
    );
    cudaDeviceSynchronize();

    // Thrust: exclusive scan -> prefix sums
    computePrefixSum(gpu_boxPrefix, totalBoxes, num_parts);
    cudaDeviceSynchronize();
}

static void fillParticleIDs(particle_t* gpu_parts, int num_parts)
{
    // Now that we have prefix sums, place each particle ID
    // into the array in box order. This overwrites prefix sums 
    // inside gpu_boxPrefix, so we re-scan if we need them again.
    populateParticleID<<<blks, NUM_THREADS>>>(
        gpu_parts, num_parts,
        gpu_boxPrefix,
        gpu_particle_ids,
        numBoxes1D,
        boxSize1D
    );
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------
// init_simulation: called once before the time loop
// ---------------------------------------------------------------------
void init_simulation(particle_t* gpu_parts, int num_parts, double size)
{
    setbuf(stdout, NULL);

    blks        = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    numBoxes1D  = (int) ceil(size / boxSize1D);
    totalBoxes  = numBoxes1D * numBoxes1D;

    // We'll make a single array of length "totalBoxes+1"
    // so index [totalBoxes] can store the total # of particles
    boxPrefixMemSize   = (totalBoxes + 1) * sizeof(int);
    particleIDMemSize  = num_parts * sizeof(int);

    // Allocate GPU memory
    cudaMalloc((void**)&gpu_boxPrefix,   boxPrefixMemSize);
    cudaMalloc((void**)&gpu_particle_ids, particleIDMemSize);
}

// ---------------------------------------------------------------------
// simulate_one_step: called each iteration
// ---------------------------------------------------------------------
void simulate_one_step(particle_t* gpu_parts, int num_parts, double size)
{
    // 1) Count how many parts in each bin -> prefix sums 
    countAndScan(gpu_parts, num_parts);

    // 2) Place (sort) the particles in "box order" 
    fillParticleIDs(gpu_parts, num_parts);

    // 3) If we still want the prefix sums for neighbor search:
    //    we must RE-scan. Because fillParticleIDs overwrote them.
    countAndScan(gpu_parts, num_parts);

    // 4) Now we can do neighbor-based forces, reading subranges 
    //    from gpu_boxPrefix + gpu_particle_ids
    compute_forces_gpu<<<blks, NUM_THREADS>>>(
        gpu_parts, num_parts,
        gpu_particle_ids,
        gpu_boxPrefix,
        numBoxes1D,
        boxSize1D
    );

    // 5) Move the particles
    move_gpu<<<blks, NUM_THREADS>>>(gpu_parts, num_parts, size);

    cudaDeviceSynchronize();
}