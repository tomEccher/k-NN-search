#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Elementi.h"
#include "max_heap.h"
#include "seq_knn.h"
#include "min_heap.h"
#include "par_knn.h"
#include "thread_knn.h"
#include "omp_knn.h"
#include "mpi_knn.h"

void guida_uso();

    long int n=2500000;
    int k=7;


int main(int argc, char **argv)
{
    srand(time(NULL));
    int a;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &a);
    if(a<MPI_THREAD_FUNNELED) {fprintf(stderr, "Thread non supportati\n");}
    int rank, size, last;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_thread=omp_get_max_threads();

	
    if(2==argc){
	    k=strtol(argv[1], NULL, 10);
	    n=n*size;
    }
    if(3==argc){
	    k=strtol(argv[1], NULL, 10);
	    n=strtol(argv[2], NULL, 10);
    }
    if((1==argc)||(argc>3)){
	    if(rank==0) guida_uso();
	    MPI_Finalize();
    }

    last=size-1;

    
    if(rank==0)  printf("\n\nEsecuzione su %lld elementi con %d thread distribuiti su %d processi MPI\n", n, num_thread, size);


    int MPI_chunk_size=n/size;
    int MPI_last_chunk_size=n-MPI_chunk_size*last;

    Campione* vicini;
    Elementi query;
    query.x=(float)(rand()%(MAX-MIN)+MIN)/100.00;
    query.y=(float)(rand()%(MAX-MIN)+MIN)/100.00;
    query.z=(float)(rand()%(MAX-MIN)+MIN)/100.00;

    create_MPI_Campione_type();
    Database d;

    if(rank==0){
		double t0=0, t1=0;

                d=creata_Data(n);
                init_database(d);

	        stampa_query(query);
		printf("\n===== ESECUZIONI MEMORIA CONDIVISA ======\n");

	        printf("\n---Esecuzione avx---\n");
		t0=now_sec();
	        vicini=avx_CalcolaVicini(d, 0, n, query, k);
		qsort(vicini, k, sizeof(Campione), compara_camp);
		t1=now_sec();
	        stampa_ViciniD( vicini, d, k);
		printf("Tempo di esecuzione con AVX prefetch %8.4fs\n", t1-t0);


                printf("\n---Esecuzione Thread + avx---\n");
                t0=now_sec();
                vicini=threadAVX_CalcolaVicini(d, query, k, num_thread);
                qsort(vicini, k, sizeof(Campione), compara_camp);
                t1=now_sec();
                stampa_ViciniD( vicini, d, k);
                printf("Tempo di esecuzione con thread AVX %8.4fs\n", t1-t0);


		printf("\n===== ESECUZIONI MEMORIA DISTRIBUITA =====\n");

	}

    MPI_Barrier(MPI_COMM_WORLD);

    Database local_d=creata_Data(MPI_chunk_size);
	
    double t=0, ta=0;

    for(int i=0; i<5; ++i){
    if(0==rank){
	    t=now_sec();
	    MPI_Scatter(d.x, MPI_chunk_size, MPI_FLOAT, local_d.x, MPI_chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }else{
            MPI_Scatter(NULL, MPI_chunk_size, MPI_FLOAT, local_d.x, MPI_chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

//    MPI_Barrier(MPI_COMM_WORLD);

    if(0==rank){
	    MPI_Scatter(d.y, MPI_chunk_size, MPI_FLOAT, local_d.y, MPI_chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
   }else{
	    MPI_Scatter(NULL, MPI_chunk_size, MPI_FLOAT, local_d.y, MPI_chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
   }

//    MPI_Barrier(MPI_COMM_WORLD);


    if(0==rank){
            MPI_Scatter(d.z, MPI_chunk_size, MPI_FLOAT, local_d.z, MPI_chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
   }else{
            MPI_Scatter(NULL, MPI_chunk_size, MPI_FLOAT, local_d.z, MPI_chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
   }

//    MPI_Barrier(MPI_COMM_WORLD);

    if(0==rank){
	    MPI_Scatter(d.idx, MPI_chunk_size, MPI_UNSIGNED_LONG, local_d.idx, MPI_chunk_size, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	    t=now_sec()-t;
	    ta+=t;
    } else {
	    MPI_Scatter(NULL, MPI_chunk_size, MPI_UNSIGNED_LONG, local_d.idx, MPI_chunk_size, MPI_UNSIGNED_LONG,0, MPI_COMM_WORLD);
    }
    }

	if(0==rank){ printf("Tempo scatter SoA  %8.4f\n==== Esecuzione AVX ====\n", ta/5);}
     MPI_Barrier(MPI_COMM_WORLD);

	Campione* local_k;
	double avg=0, exec, best=10000, tgather;
	Max_Heap result=createHeap(k);


	for(int i=0; i<5; ++i){
		

		t=now_sec();
	    local_k=avx_CalcolaVicini(local_d, 0, MPI_chunk_size, query, k);
	    qsort(local_k, k, sizeof(Campione), compara_camp);

	     MPI_Barrier(MPI_COMM_WORLD);


	     if(0==rank){
		     Campione *all_k=(Campione*)malloc(sizeof(Campione)*k*size);
		     tgather=now_sec();
		     MPI_Gather(local_k, k, MPI_CAMPIONE, all_k, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
		     tgather=now_sec()-tgather;
		     result=createHeap(k);
	
		     merge_P_lists(all_k, size, result.c);
		     exec=now_sec()-t-tgather;
		     avg+=exec;
		     best=(best>exec) ? exec:best;
		     result.dim=k;
		
		     
	     }else{
		     MPI_Gather(local_k, k, MPI_CAMPIONE, NULL, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
	     }
	}

	if(0==rank){
//		stampa_ViciniD(result.c, d, k);
	       	printf("Tempo di esecuzione AVX migliore %8.4f, medio %8.4f\n", best, avg/5);
		avg=0; best=1000;
		printf("====== Esecuzione thread + AVX\n");
	}

	for(int i=0; i<5; ++i){


                t=now_sec();
            local_k=threadAVX_CalcolaVicini(local_d, query, k, num_thread);
            qsort(local_k, k, sizeof(Campione), compara_camp);

             MPI_Barrier(MPI_COMM_WORLD);

             if(0==rank){
                     Campione *all_k=(Campione*)malloc(sizeof(Campione)*k*size);
		     tgather=now_sec();
                     MPI_Gather(local_k, k, MPI_CAMPIONE, all_k, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
		     tgather=now_sec()-tgather;
                     result=createHeap(k);
		     double tmerge=now_sec();
                     merge_P_lists(all_k, size, result.c);
		     tmerge=now_sec()-tmerge;
                     exec=now_sec()-t-tgather;
                     avg+=exec;
                     best=(best>exec) ? exec:best;
                     result.dim=k;


		     


                     //stampa_ViciniD(result, d);
             }else{
                     MPI_Gather(local_k, k, MPI_CAMPIONE, NULL, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
             }
        }

        if(0==rank){
//		stampa_ViciniD(result.c, d, k);

                printf("Tempo di esecuzione thread+AVX migliore %8.4f, medio %8.4f\n", best, avg/5);
        }



    MPI_Finalize();

}

void guida_uso(){
	printf("Guida all'uso\n");
	printf("Paramentri: ./MPI_SoA,o [k] numero di elementi per deafult 2.5*10^6 per processo MPI\n");
	printf("Paramentri: ./MPI_SoA,o [k] [N elementi]\n");
	printf("Il numero di thread è sempre ottenuto da omp_get_num_thread()\n");
}
