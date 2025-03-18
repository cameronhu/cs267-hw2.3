#include "common.h"
#include <cuda.h>
#include <stdio.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <numeric>

#define NUM_THREADS 256
#define INDEX(row, col) ((row) * numBoxes1D + (col))

// Put any static global variables here that you will use throughout the simulation.
int blks;
double boxSize1D = cutoff;
int numBoxes1D;
int totalBoxes;
size_t boxesMemSize;
size_t prefixMemSize;
size_t particle_idMemSize;

// ============ Array pointers for boxes and particle_idx ============

// CPU arrays
int* boxCounts;
int* prefixSums;
int* particle_ids;

// GPU arrays
int* gpu_boxCounts;
int* gpu_prefixSums;
int* gpu_particle_ids;

// =================
// Helper Functions
// =================

// Calculate the box row of the particle
__device__ __host__ int findRow(const particle_t& p, double boxSize1D) {
    return floor(p.y / boxSize1D);
}

// Calculate the box column of the particle
__device__ __host__ int findCol(const particle_t& p, double boxSize1D) {
    return floor(p.x / boxSize1D);
}

/**
* Helper function to calculate the box index of a given particle
*/
__device__ __host__ int findBox(const particle_t& p, int numBoxes1D, double boxSize1D) {
    int col = floor(p.x / boxSize1D);
    int row = floor(p.y / boxSize1D);
    return INDEX(row, col);
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

/**
 * Given a particle, apply force from all other particles in the given box (row, col)
 * @param row Row of the neighbor box
 * @param col Column of the neighbor box
 * @param thisParticle Particle to apply force to
 */
__device__ void apply_force_from_neighbor_gpu(int row, int col, particle_t& thisParticle, particle_t* particles, int* particle_ids, int* prefixSums, int numBoxes1D, int boxSize1D) {
    // Check if the neighbor is within bounds
    if (row >= 0 && row < numBoxes1D && col >= 0 && col < numBoxes1D) {
        int boxIndex = INDEX(row, col);
        int startIdx = prefixSums[boxIndex];
        int endIdx = prefixSums[boxIndex + 1];

        // Check if there are particles in the box
        // A box with no particles will have same prefixSum as the next box
        // Apply forces for all particles in this neighboring box
        for (int i = startIdx; i < endIdx; ++i) {
            int parts_idx = particle_ids[i];
            particle_t& neighbor = particles[parts_idx];
            apply_force_gpu(thisParticle, neighbor);
        }
    }
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* particle_ids, int* prefixSums, int numBoxes1D, double boxSize1D) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    //
    // TODO: check indexing through particle_ids array
    // Access through particle_ids array for coalesced memory access
    // int parts_idx = particle_ids[tid];
    // particle_t& thisParticle = particles[parts_idx];
    //

    particle_t& thisParticle = particles[tid];
    thisParticle.ax = thisParticle.ay = 0;
    int row = findRow(thisParticle, boxSize1D);
    int col = findCol(thisParticle, boxSize1D);

    // TODO: profile loop unrolling
    apply_force_from_neighbor_gpu(row - 1, col - 1, thisParticle, particles, particle_ids, prefixSums, numBoxes1D, boxSize1D); // Up Left
    apply_force_from_neighbor_gpu(row - 1, col, thisParticle, particles, particle_ids, prefixSums, numBoxes1D, boxSize1D);     // Up
    apply_force_from_neighbor_gpu(row - 1, col + 1, thisParticle, particles, particle_ids, prefixSums, numBoxes1D, boxSize1D); // Up Right
    apply_force_from_neighbor_gpu(row, col - 1, thisParticle, particles, particle_ids, prefixSums, numBoxes1D, boxSize1D);     // Left
    apply_force_from_neighbor_gpu(row, col + 1, thisParticle, particles, particle_ids, prefixSums, numBoxes1D, boxSize1D);     // Right
    apply_force_from_neighbor_gpu(row + 1, col - 1, thisParticle, particles, particle_ids, prefixSums, numBoxes1D, boxSize1D); // Down Left
    apply_force_from_neighbor_gpu(row + 1, col, thisParticle, particles, particle_ids, prefixSums, numBoxes1D, boxSize1D);     // Down
    apply_force_from_neighbor_gpu(row + 1, col + 1, thisParticle, particles, particle_ids, prefixSums, numBoxes1D, boxSize1D); // Down Right
    apply_force_from_neighbor_gpu(row, col, thisParticle, particles, particle_ids, prefixSums, numBoxes1D, boxSize1D);         // Self
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

// Iterates through parts and increments boxCounts
__global__ void countParticlesPerBox(particle_t* gpu_parts, int num_parts, int* gpu_boxCounts, int numBoxes1D, double boxSize1D) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int boxIndex = findBox(gpu_parts[tid], numBoxes1D, boxSize1D);
    // printf("cur parts idx: %i. boxIndex: %i. Coords: (%f, %f)\n", tid, boxIndex, gpu_parts[tid].x, gpu_parts[tid].y);
    gpu_boxCounts[boxIndex]++;
}

// Iterates through boxCounts and computes a prefixSum
void computePrefixSum() {
    int prefixSum = 0;
    for (int boxIndex = 0; boxIndex <= totalBoxes; ++boxIndex) {
        prefixSums[boxIndex] = prefixSum;
        prefixSum += boxCounts[boxIndex];
        // printf("%i\n", boxCounts[boxIndex]);
    }
}
// Organizes parts by box, in particle_id array
// Uses prefixSum and a reset boxCounts to compute where in particle_id the particle should be inserted 
void populateParticleID(particle_t* parts, int num_parts) {
    memset(boxCounts, 0, boxesMemSize);
    for (int i = 0; i < num_parts; ++i) {
        int boxIndex = findBox(parts[i], numBoxes1D, boxSize1D);
        int pos = prefixSums[boxIndex] + boxCounts[boxIndex];
        particle_ids[pos] = i;
        boxCounts[boxIndex]++;
    }
}

void printAssignmentStats(particle_t* parts) {
    int numEmpty = 0;
    int numFilled = 0;
    int partCount = 0;
    for (int i = 0; i < totalBoxes; ++i) {
        // prefixSums[i] = (boxCounts[i] > 0) ? prefixSums[i] : -1;
        if (boxCounts[i] == 0) {
            printf("Check for empty box %i. prefixSums[%i]: %i\n", i, i, prefixSums[i]);
            numEmpty += 1;
        }
        else {
            numFilled += 1;
            partCount += boxCounts[i];
            printf("Particle id: %i with coords: (%f, %f) in box %i. findBox_output: %i.\n",
            particle_ids[prefixSums[i]], parts[particle_ids[prefixSums[i]]].x, parts[particle_ids[prefixSums[i]]].y, i, findBox(parts[particle_ids[prefixSums[i]]], numBoxes1D, boxSize1D));
        }
    }
    printf("prefixSums[0]: %i. prefixSums[totalBoxes]: %i\n", prefixSums[0], prefixSums[totalBoxes]);
    printf("Num empty boxes: %i. Num boxes w/ particles: %i. Num particles: %i. Average particles per box: %f\n", 
        numEmpty, numFilled, partCount, (double)(partCount / numFilled));
}

// Initializes the particle_id and prefixSums arrays, on GPU
void assignToBoxes(particle_t* parts, int num_parts, int* gpu_boxCounts) {
    // setbuf(stdout, NULL);

    // Copy from parts (gpu_parts) to cpu_parts
    particle_t* cpu_parts = new particle_t[num_parts];
    cudaMemcpy(cpu_parts, parts, particle_idMemSize, cudaMemcpyDeviceToHost);    

    // First pass: count particles in each box. Reset box counts from past iteration
    cudaMemset(gpu_boxCounts, 0, boxesMemSize);
    countParticlesPerBox<<<blks, NUM_THREADS>>>(parts, num_parts, gpu_boxCounts, numBoxes1D, boxSize1D);


    //
    // TEST countParticlesPerBox
    // Use thrust to calculate the sum of all values in gpu_boxCounts
    thrust::device_ptr<int> dev_ptr(gpu_boxCounts);
    int totalParticles = thrust::reduce(dev_ptr, dev_ptr + totalBoxes, 0, thrust::plus<int>());
    printf("Sum of gpu_boxCounts: %d\n", totalParticles);
    

    // Wait for all threads to finish. Then copy gpu_boxCounts to CPU, use for computePrefixSum
    cudaDeviceSynchronize();
    cudaMemcpy(boxCounts, gpu_boxCounts, boxesMemSize, cudaMemcpyDeviceToHost);

    // TEST cpu boxCounts sum
    int totalParticlesCPU = std::accumulate(boxCounts, boxCounts + totalBoxes, 0);
    printf("Sum of cpu boxCounts: %d\n", totalParticlesCPU);

    // Compute starting index for each box in particle_idx from boxCounts
    computePrefixSum();

    populateParticleID(cpu_parts, num_parts);

    // printAssignmentStats(cpu_parts);
}

// Copies data from CPU particle_id and prefixSums to mirrored arrs on GPU
void copyArraysToGPU() {
    cudaMemcpy(gpu_particle_ids, particle_ids, particle_idMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_prefixSums, prefixSums, prefixMemSize, cudaMemcpyHostToDevice);
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    setbuf(stdout, NULL);

    // Assign global variables
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    numBoxes1D = (int) ceil(size / boxSize1D);
    totalBoxes = numBoxes1D * numBoxes1D;
    boxesMemSize = totalBoxes * sizeof(int);
    prefixMemSize = (totalBoxes + 1) * sizeof(int);
    particle_idMemSize = num_parts * sizeof(int);

    // Allocate memory for CPU-side arrays
    boxCounts = new int[totalBoxes]();
    prefixSums = new int[totalBoxes + 1];
    particle_ids = new int[num_parts];

    // Allocate memory for GPU-side arrays and copy from CPU-side arrays
    cudaMalloc((void**)&gpu_boxCounts, boxesMemSize);
    cudaMemset(gpu_boxCounts, 0, boxesMemSize);
    cudaMalloc((void**)&gpu_prefixSums, prefixMemSize);
    cudaMalloc((void**)&gpu_particle_ids, num_parts * sizeof(int));

    // printf("Numboxes1d: %i. totalBoxes: %i\n", numBoxes1D, totalBoxes);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Assign all particles to boxes
    assignToBoxes(parts, num_parts, gpu_boxCounts);

    // Copy CPU arrays that were updated by assignToBoxes to GPU
    copyArraysToGPU();

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, gpu_particle_ids, gpu_prefixSums, numBoxes1D, boxSize1D);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}