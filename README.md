# Parallel and Distributed Computing — Assignment Part 1

This repository contains the source code and supporting files for Assignment Part 1 of the N-body MPI assignment.

## Files

- `part1a.c` — Part 1A implementation using explicit ring point-to-point communication instead of the timestep `MPI_Allgather`.
- `part1b.c` — Part 1B reduced-memory implementation. Each MPI process permanently stores only its locally owned masses, positions, velocities, and forces, while mass and position blocks circulate through the ring during force calculation.
- `assignment_part1.pdf` — Submission report covering the Part 1A and Part 1B designs, validation, and design comparison.

## Compilation

Compile Part 1A with:

```bash
mpicc -g -Wall -o part1a part1a.c -lm
```

Compile Part 1B with:

```bash
mpicc -g -Wall -o part1b part1b.c -lm
```

For performance testing with detailed particle output disabled, compile with `NO_OUTPUT`:

```bash
mpicc -O2 -DNO_OUTPUT -o part1a part1a.c -lm
```

```bash
mpicc -O2 -DNO_OUTPUT -o part1b part1b.c -lm
```

## Execution

General execution format:

```bash
mpiexec -n <number_of_processes> ./<program> <number_of_particles> <number_of_timesteps> <timestep_size> <output_frequency> <g|i>
```

Arguments:

- `<number_of_processes>` — number of MPI processes.
- `<number_of_particles>` — total number of particles.
- `<number_of_timesteps>` — number of simulation timesteps.
- `<timestep_size>` — timestep size used by the Euler update.
- `<output_frequency>` — how often particle state is printed.
- `g` — generate the initial conditions.
- `i` — read the initial conditions from standard input.

The number of particles should be evenly divisible by the number of MPI processes.

## Example Runs

Part 1A:

```bash
mpiexec -n 4 ./part1a 8 2 0.01 1 g
```

Part 1B:

```bash
mpiexec -n 4 ./part1b 8 2 0.01 1 g
```

Example performance run:

```bash
mpiexec -n 4 ./part1b 2000 50 0.01 1 g
```

When the program has been compiled with `-DNO_OUTPUT`, the particle state is not printed and only the elapsed execution time is reported.

## Correctness Testing

Correctness was checked by comparing the output of the implementations against the supplied `mpi_nbody_basic.c` solver using identical generated initial conditions.

The following configurations were used:

- 1 MPI process, 4 particles, 2 timesteps
- 2 MPI processes, 4 particles, 2 timesteps
- 4 MPI processes, 8 particles, 2 timesteps

The reported positions and velocities matched the supplied solver for the tested configurations.

## Performance Testing

Performance testing used:

- 2000 particles
- 50 timesteps
- timestep size of `0.01`
- 1, 2, and 4 MPI processes

Each configuration was executed three times and the mean runtime was used for comparison.

## Repository

Private repository:

`https://github.com/kyle-uni/parallel-distributed-computing.git`
