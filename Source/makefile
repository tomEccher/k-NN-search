sharedMake:
	gcc -O3 -mavx2 -mfma -std=c11 -fopenmp shared.c -o shared.o

aosMake:  
	mpicc -O3 -mavx2 -mfma -std=c11 -fopenmp MPI_AoS.c -o MPI_AoS.o

saoMake: 
	mpicc -O3 -mavx2 -mfma -std=c11 -fopenmp MPI_SoA.c -o MPI_SoA.o


