# k-NN_search
Implementation of distributed k-NN search

All the files needed by the compiler are aviable in the "Source" folder.
In the MAKEFILE there 3 different policy to compile three different executables.
Note that the code was tested only on unitn HPC server and ment to be reproduced there for accademic reason.

make sharedMake: gcc -O3 -mavx2 -mfma -fopenmpi shared.c -o shared.o
  produce an executable where all the algorithms are tested in a shared memory sistem.

make aosMake: mpicc -O3 -mavx2 -mfma -fopenmpi MPI_AoS.c -o MPI_AoS.o
  produce an executable where sequential, thread and OMP algorithms are test in a distributed memory sistem.

make saoMake: mpicc -O3 -mavx2 -mfma -fopenmpi MPI_AoS.c -o MPI_SoA.o
  produce an executable where AVX and AVX+thread algorithms are test in a distributed memory sistem.

Change directory to PBS folder to run all the aviable .pbs files used to generate the reported results.
