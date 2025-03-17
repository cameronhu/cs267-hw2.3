#include "common.h"
#include <cuda.h>
#include <stdio.h>

#define NUM_THREADS 256
#define INDEX(row, col) ((row) * numBoxes1D + (col))

// Put any static global variables here that you will use throughout the simulation.
int blks;
double boxSize1D = cutoff;
int numBoxes1D;
int totalBoxes;
size_t boxesMemSize;
size_t prefixMemSize;

// ============ Array pointers for boxes and particle_idx ============

// CPU arrays
int* boxCounts;
int* prefixSums;
int* particle_ids;
int* boxes;

// GPU arrays
int* gpu_boxCounts;
int* gpu_prefixSums;
int* gpu_particle_ids;
int* gpu_boxes;

// =================
// Helper Functions
// =================

/**
* Helper function to calculate the box index of a given particle
*/
int findBox(const particle_t& p) {
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

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
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

void assignToBoxes(particle_t* parts, int num_parts) {
    setbuf(stdout, NULL);
    printf("Inside assignToBoxes\n");

    // Copy from parts (gpu_parts) to cpu_parts
    particle_t* cpu_parts = new particle_t[num_parts];
    cudaMemcpy(cpu_parts, parts, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
    size_t actual_size = 0;
    for (int i = 0; i < num_parts; ++i) {
        actual_size += sizeof(cpu_parts[i]);
    }
    printf("Actual size of cpu_parts: %lu\n", actual_size);
    printf("Num particles in cpu_parts: %lu\n", actual_size / sizeof(particle_t));
    
    // First pass: count particles in each box. Reset box counts from past iteration
    memset(boxCounts, 0, boxesMemSize);
    for (int i = 0; i < num_parts; ++i) {
        // printf("cur parts idx: %i\n", i);
        int boxIndex = findBox(cpu_parts[i]);
        // printf("boxIndex: %i\n", boxIndex);
        boxCounts[boxIndex]++;
    }

    // printf("Fin counting particles per box\n");

    // Compute starting index for each box in particle_idx
    int prefixSum = 0;
    for (int boxIndex = 0; boxIndex <= totalBoxes; ++boxIndex) {
        prefixSums[boxIndex] = prefixSum;
        prefixSum += boxCounts[boxIndex];
        // printf("%i\n", boxCounts[boxIndex]);
    }
    // printf("Last value of prefix sums should be num_parts. prefixSums[-1]: %i. num_parts: %i\n", prefixSums[totalBoxes], num_parts);

    // printf("Fin calc starting particle_idx index for each box's first part\n");

    // Reset box counts for use in the second pass
    memset(boxCounts, 0, boxesMemSize);

    // Second pass: assign particles to particle_idx and update boxes
    for (int i = 0; i < num_parts; ++i) {
        int boxIndex = findBox(cpu_parts[i]);
        int pos = prefixSums[boxIndex] + boxCounts[boxIndex];
        particle_ids[pos] = i;
        boxCounts[boxIndex]++;
    }
    // printf("Fin second pass to assign particles `parts` index to particle_ids in proper box order.\n");

    // Update boxes array: -1 if box has no particles
    // for (int i = 0; i < totalBoxes; ++i) {
    //     boxes[i] = (boxCounts[i] > 0) ? prefixSums[i] : -1;
    // }
    int numEmpty = 0;
    int numFilled = 0;
    int partCount = 0;
    for (int i = 0; i < totalBoxes; ++i) {
        prefixSums[i] = (boxCounts[i] > 0) ? prefixSums[i] : -1;
        if (boxCounts[i] == 0) {
            // printf("Check for empty box %i. prefixSums[%i]: %i\n", i, i, prefixSums[i]);
            numEmpty += 1;
        }
        else {
            numFilled += 1;
            partCount += boxCounts[i];
        }
    }
    // printf("Updating `boxes` array with starting indices if box has particles.\n");
    printf("prefixSums[0]: %i. prefixSums[totalBoxes]: %i\n", prefixSums[0], prefixSums[totalBoxes]);
    printf("Num empty boxes: %i. Num boxes w/ particles: %i. Num particles: %i. Average particles per box: %f\n", 
        numEmpty, numFilled, partCount, numFilled/partCount);

    // ================ Copy all CPU arrays to mirrored GPU arrays ================

}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    setbuf(stdout, NULL);

    // Assign global variables
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    numBoxes1D = ceil(size / boxSize1D);
    totalBoxes = numBoxes1D * numBoxes1D;
    boxesMemSize = totalBoxes * sizeof(int);
    prefixMemSize = (totalBoxes + 1) * sizeof(int);

    // Allocate memory for CPU-side arrays
    boxCounts = new int[totalBoxes]();
    prefixSums = new int[totalBoxes + 1];
    particle_ids = new int[num_parts];
    boxes = new int[totalBoxes];

    // Allocate memory for GPU-side arrays and copy from CPU-side arrays
    cudaMalloc((void**)&gpu_boxCounts, boxesMemSize);
    cudaMemset(gpu_boxCounts, 0, boxesMemSize);
    // cudaMemcpy(gpu_boxCounts, boxCounts, boxesMemSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_prefixSums, prefixMemSize);

    cudaMalloc((void**)&gpu_particle_ids, num_parts * sizeof(int));

    cudaMalloc((void**)&gpu_boxes, boxesMemSize);
    printf("Numboxes1d: %i. totalBoxes: %i\n", numBoxes1D, totalBoxes);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Assign all particles to boxes
    assignToBoxes(parts, num_parts);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
