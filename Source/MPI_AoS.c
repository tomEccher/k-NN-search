#include <unistd.h>
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


    size_t n=2500000;
    int k=7;
    int num_thread;

    void guida_uso();


int main(int argc, char **argv)
{
    
    srand(time(NULL));

    int prov;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);

    if(prov<MPI_THREAD_FUNNELED){
	    fprintf(stderr, "Impossibile fornire il livello richiesto\n");
    }


    int rank, size, last;
    num_thread=omp_get_max_threads();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


	
    if(2==argc){
	    k=strtol(argv[1], NULL, 10);
	    n*=size;
    }
    if(3==argc){
	    k=strtol(argv[1], NULL, 10);
	    n=strtol(argv[2], NULL, 10);
    }
    if((1==argc)||(argc>3)){
	    if(rank==0) guida_uso();
	    MPI_Finalize();
	    return 0;
    }

    last=size-1;
    
    if(rank==0)  printf("\nEsecuzione su %lld elementi con %d thread distribuiti su %d processi MPI\n", n, num_thread, size);
    if(0==rank) fprintf(stderr, "Dimensione \"Elementi\" %d\n Dimensione dataset %lld\n", sizeof(Elementi), sizeof(Elementi)*n);

    int MPI_chunk_size=n/size;
    int MPI_last_chunk_size=n-MPI_chunk_size*last;

    Campione* vicini;
    Elementi* el;
    Elementi query;
    Elementi *rec_buff;
    query.x=(float)(rand()%(MAX-MIN)+MIN)/100.0;
    query.y=(float)(rand()%(MAX-MIN)+MIN)/100.0;
    query.z=(float)(rand()%(MAX-MIN)+MIN)/100.0;

    create_MPI_Campione_type();
    create_MPI_Elementi_type();

    if(rank==0){

	        el=(Elementi *) malloc(sizeof(Elementi)*n);
		double t0=0, t1=0;

	        init(el, n);
	        stampa_query(query);

		printf("=== ESECUZIONE LINEARE SU RANK === %d\n", rank);
		for(int i=0; i<2; ++i){
			t0=now_sec();
			vicini=seq_CalcolaVicini(el, n, query, k);
			qsort(vicini, k, sizeof(Campione), compara_camp);
			t1+=now_sec()-t0;
		}
		printf("Tempo di esecuzione medio %8.4f s\n\n", t1/2);
		t0=0; t1=0;	
		printf("=== ESECUZIONE OMP SU RANK === %d\n", rank);
                for(int i=0; i<2; ++i){
                        t0=now_sec();
                        vicini=omp_CalcolaVicini(el, n, query, k);
                        qsort(vicini, k, sizeof(Campione), compara_camp);
                        t1+=now_sec()-t0;
                }
                printf("Tempo di esecuzione medio %8.4f s\n\n", t1/2);


    }

    MPI_Barrier(MPI_COMM_WORLD);

    rec_buff=(Elementi *)malloc(sizeof(Elementi)*MPI_chunk_size);

    double t=0, ta=0;
    for(int i=0; i<5; ++i){

    if(rank==0){
	    t=now_sec();
	    MPI_Scatter(el, MPI_chunk_size, MPI_ELEMENTI, rec_buff, MPI_chunk_size, MPI_ELEMENTI, 0, MPI_COMM_WORLD);
	    t=now_sec()-t;
	    ta+=t;
    }else{
	    MPI_Scatter(NULL, 0, MPI_ELEMENTI, rec_buff, MPI_chunk_size, MPI_ELEMENTI, 0, MPI_COMM_WORLD);
    }
    }

    if(rank==0) printf("Tempo di scatter %8.4f\n", ta/5);

    double exec=0, avg=0, best=10000, tgather;

    if(0==rank){
            printf("=== ESECUZIONE LINEARE ===\n");
    }


    for(int i=0; i<5; ++i){

	   t=now_sec();
	    vicini=seq_CalcolaVicini(rec_buff, MPI_chunk_size, query, k);
	    qsort(vicini, k, sizeof(Campione), compara_camp);	    
	    

	    MPI_Barrier(MPI_COMM_WORLD);


	    if(rank==0){
		    Max_Heap all_k=createHeap(k*size);
		    Max_Heap result=createHeap(k);
		    tgather=now_sec();
		    MPI_Gather(vicini, k, MPI_CAMPIONE, all_k.c, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
		    tgather=now_sec()-tgather;
		    merge_P_lists(all_k.c, size, result.c);
		  //  printf("Tempo di gather %8.4f\n", tgather);
		    exec=now_sec()-t-tgather;
		    avg+=exec;

		    best = (best>exec) ? exec : best;

		    result.dim=k;
		    vicini=result.c;
	    }else{
		    MPI_Gather(vicini, k, MPI_CAMPIONE, NULL, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
	    }
    }

    if(0==rank){
//	   printf("Vicini da processo distribuito:\n");
//                  stampa_Vicini(el, vicini);

	    printf("ESECUZIONE LINEARE: BEST %8.4fs, MEDIO %8.4fS\n", best, avg/5);
	    best=10000; avg=0;
	    printf("=== ESECUZIONE THREAD ===\n");
	    
    }


    for(int i=0; i<5; ++i){
//	    fprintf(stderr, "Esecuzione numero %d del processo %d\n", i, rank);
	    t=now_sec();
    	vicini=thread_CalcolaVicini(rec_buff, MPI_chunk_size, query, k, num_thread);
            qsort(vicini, k, sizeof(Campione), compara_camp);


            MPI_Barrier(MPI_COMM_WORLD);


            if(rank==0){
                    Max_Heap all_k=createHeap(k*size);
                    Max_Heap result=createHeap(k);

		    tgather=now_sec();
                    MPI_Gather(vicini, k, MPI_CAMPIONE, all_k.c, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
		    tgather=now_sec()-tgather;
                    merge_P_lists(all_k.c, size, result.c);
//		    printf("Tempo di gather %8.4f\n", tgather);
                    exec=now_sec()-t-tgather;
                    avg+=exec;

                    best = (best>exec) ? exec : best;

                    result.dim=k;
		    vicini=result.c;
            }else{
                    MPI_Gather(vicini, k, MPI_CAMPIONE, NULL, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
            }
    }
    

    if(0==rank){
//           printf("Vicini da processo distribuito:\n");
//                  stampa_Vicini(el, vicini);

            printf("ESECUZIONE THREAD: BEST %8.4fs, MEDIO %8.4fS\n", best, avg/5);
            best=10000; avg=0;
	    printf("=== ESECUZIONE OMP ===\n");
    }


    for(int i=0; i<5; ++i){
            t=now_sec();
        vicini=omp_CalcolaVicini(rec_buff, MPI_chunk_size, query, k);
            qsort(vicini, k, sizeof(Campione), compara_camp);


            MPI_Barrier(MPI_COMM_WORLD);


            if(rank==0){
                    Max_Heap all_k=createHeap(k*size);
                    Max_Heap result=createHeap(k);

		    tgather=now_sec();
                    MPI_Gather(vicini, k, MPI_CAMPIONE, all_k.c, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
		    tgather=now_sec()-tgather;
                    merge_P_lists(all_k.c, size, result.c);
//		    printf("Tempo di gather %8.4f\n", tgather);
                    exec=now_sec()-t-tgather;
                    avg+=exec;

                    best = (best>exec) ? exec : best;

                    result.dim=k;
                    vicini=result.c;
            }else{
                    MPI_Gather(vicini, k, MPI_CAMPIONE, NULL, k, MPI_CAMPIONE, 0, MPI_COMM_WORLD);
            }
    }

    if(0==rank){
//           printf("Vicini da processo distribuito:\n");
//                  stampa_Vicini(el, vicini);

            printf("ESECUZIONE OMP: BEST %8.4fs, MEDIO %8.4fS\n", best, avg/5);
            printf("\n\n");
    }



    MPI_Finalize();

}

void guida_uso(){
	printf("Guida all'uso\n");
	printf("Parametri: MPI_AoS.o [K] di default N elementi 2.5*10^8 per processo MPI\n");
	printf("Parametri: MPI_AoS.o [K] [N elementi]\n");
	printf("Il numero di thread è sempre ottenuto da omp_get_num_threads()\n");
}
