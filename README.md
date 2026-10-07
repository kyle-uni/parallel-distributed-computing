# Parallel and Distributed Computing — Assignment Part 2

This repository contains the source code and supporting files for Assignment Part 2, which investigates OpenMP synchronization, scheduling, and performance in an N-body simulation.

## Source Files

### Part 2A — OpenMP Synchronization

- `part2a_critical.c` — OpenMP implementation using a `critical` section to protect concurrent updates to the shared force array.
- `part2a_locks.c` — OpenMP implementation using one lock per particle to protect concurrent force updates.

### Part 2B — OpenMP Scheduling and Performance

- `omp_nbody_basic.c` — Basic OpenMP N-body implementation.
- `omp_nbody_red_default.c` — Reduced-force implementation using default OpenMP scheduling.
- `omp_nbody_red.c` — Reduced-force implementation using cyclic scheduling for the force-calculation loop.
- `omp_nbody_red_allcyclic.c` — Reduced-force implementation using cyclic scheduling for all OpenMP work-sharing loops.

---

## Compilation

All programs require GCC with OpenMP support.

### Part 2A Critical-Section Version

```bash
gcc -g -Wall -fopenmp -o part2a_critical part2a_critical.c -lm
```

### Part 2A Lock-Based Version

```bash
gcc -g -Wall -fopenmp -o part2a_locks part2a_locks.c -lm
```

### Part 2B Basic Version

```bash
gcc -g -Wall -fopenmp -o omp_nbody_basic omp_nbody_basic.c -lm
```

### Part 2B Reduced — Default Scheduling

```bash
gcc -g -Wall -fopenmp -o omp_nbody_red_default omp_nbody_red_default.c -lm
```

### Part 2B Reduced — Forces Cyclic

```bash
gcc -g -Wall -fopenmp -o omp_nbody_red omp_nbody_red.c -lm
```

### Part 2B Reduced — All Cyclic

```bash
gcc -g -Wall -fopenmp -o omp_nbody_red_allcyclic omp_nbody_allcyclic.c -lm
```

For performance testing, detailed particle output can be disabled using `-DNO_OUTPUT`, and optimisation can be enabled with `-O2`.

Example:

```bash
gcc -O2 -Wall -fopenmp -DNO_OUTPUT -o omp_nbody_basic omp_nbody_basic.c -lm
```

---

## Execution

All programs use the following command-line format:

```bash
./<program> <number_of_threads> <number_of_particles> <number_of_timesteps> <timestep_size> <output_frequency> <g|i>
```

Arguments:

- `<number_of_threads>` — number of OpenMP threads.
- `<number_of_particles>` — total number of particles in the simulation.
- `<number_of_timesteps>` — number of simulation timesteps.
- `<timestep_size>` — size of each timestep.
- `<output_frequency>` — frequency at which particle state is printed.
- `g` — generate the initial conditions automatically.
- `i` — read the initial conditions from standard input.

### Example Correctness Run

```bash
./part2a_critical 4 8 2 0.01 1 g
```

```bash
./part2a_locks 4 8 2 0.01 1 g
```

This runs the simulation using:

- 4 OpenMP threads
- 8 particles
- 2 timesteps
- timestep size `0.01`
- output every timestep
- generated initial conditions

### Example Part 2B Performance Run

```bash
./part2b_red_forces_cyclic 16 10000 20 0.01 1 g
```

The Part 2B performance experiments used:

- 10,000 particles
- 20 timesteps
- timestep size `0.01`
- generated initial conditions
- 1, 4, 8, 16, and 32 OpenMP threads
- detailed output disabled with `-DNO_OUTPUT`
- five runs per configuration, with mean execution time reported

The same simulation parameters and compilation options were used for all four Part 2B implementations.

---

## Performance Test Executables

Example commands for the Part 2B tests are:

### Basic

```bash
./omp_nbody_basic 1 10000 20 0.01 1 g
./omp_nbody_basic 4 10000 20 0.01 1 g
./omp_nbody_basic 8 10000 20 0.01 1 g
./omp_nbody_basic 16 10000 20 0.01 1 g
./omp_nbody_basic 32 10000 20 0.01 1 g
```

The same thread counts were used for:

```bash
./omp_nbody_red_default
./omp_nbody_red
./omp_nbody_red_allcyclic
```

with the same remaining simulation arguments.

---

## Testing Environment

Performance testing was conducted on an Apple MacBook Air with an Apple M1 8-core CPU using Visual Studio Code and a Linux Docker container.

The programs were compiled using GCC with OpenMP support.

---

## Private Repository

Private Git repository:

`https://github.com/kyle-uni/parallel-distributed-computing.git`