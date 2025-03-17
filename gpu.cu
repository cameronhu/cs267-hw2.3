#include "common.h"
#include <cuda.h>
#include <stdio.h>

#define NUM_THREADS 256
#define INDEX(row, col) ((row) * numBoxes1D + (col))

// Put any static global variables here that you will use throughout the simulation.
int blks;
double boxSize1D = cutoff;
int numBoxes1D;
int numBoxesTotal;


// ============ Array pointers for boxes and particle_idx ============

// CPU arrays
int* boxes;
int* particle_idx;

// GPU arrays
int* gpu_boxes;
int* gpu_particle_idx;

// =================
// Helper Functions
// =================

/**
* Helper function to calculate the box index of a given particle
*/
int findBox(const particle_t& p) {
    int box_x = floor(p.x / boxSize1D);
    int box_y = floor(p.y / boxSize1D);
    return INDEX(box_x, box_y);
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

    // Initialize box counts
    int* box_counts = new int[numBoxesTotal]();
    
    // First pass: count particles in each box
    for (int i = 0; i < num_parts; ++i) {
        int curr_box_idx = findBox(parts[i]);
        box_counts[curr_box_idx]++;
    }

    printf("Fin counting particles per box\n");

    // Compute starting index for each box in particle_idx
    int* box_start_idx = new int[numBoxesTotal];
    int idx = 0;
    for (int i = 0; i < numBoxesTotal; ++i) {
        box_start_idx[i] = idx;
        idx += box_counts[i];
    }

    printf("Fin calc starting particle_idx index for each box's first part\n");

    // Reset box counts for use in the second pass
    memset(box_counts, 0, numBoxesTotal * sizeof(int));

    // Second pass: assign particles to particle_idx and update boxes
    for (int i = 0; i < num_parts; ++i) {
        int curr_box_idx = findBox(parts[i]);
        int pos = box_start_idx[curr_box_idx] + box_counts[curr_box_idx];
        particle_idx[pos] = i;
        box_counts[curr_box_idx]++;
    }
    printf("Fin second pass to assign particles `parts` index to particle_id in proper box order.\n");


    // Update boxes array: -1 if box has no particles
    for (int i = 0; i < numBoxesTotal; ++i) {
        boxes[i] = (box_counts[i] > 0) ? box_start_idx[i] : -1;
    }
    printf("Updating `boxes` array with starting indices if box has particles.\n");

    // Clean up
    delete[] box_counts;
    delete[] box_start_idx;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    // Assign global variables
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    numBoxes1D = ceil(size / boxSize1D);
    numBoxesTotal = numBoxes1D * numBoxes1D;

    // Allocate memory for CPU-side arrays
    boxes = new int[numBoxesTotal];
    particle_idx = new int[num_parts];

    // Allocate memory for GPU-side arrays and copy from CPU-side arrays
    cudaMalloc((void**)&gpu_boxes, numBoxesTotal * sizeof(int));
    cudaMemcpy(gpu_boxes, boxes, numBoxesTotal * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_particle_idx, num_parts * sizeof(int));
    cudaMemcpy(gpu_particle_idx, particle_idx, num_parts * sizeof(int), cudaMemcpyHostToDevice);
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
