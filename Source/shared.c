#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "Elementi.h"
#include "max_heap.h"
#include "seq_knn.h"
#include "par_knn.h"
#include "min_heap.h"
#include "thread_knn.h"
#include "omp_knn.h"


void guida_uso();

    size_t n=250000000;
    int k=7;


int main(int argc, char **argv)
{
	
	int n_th;

	srand(time(NULL));
	if(3==argc){
		k=strtol(argv[1], NULL, 10);
		n_th=strtol(argv[2], NULL, 10);
		n*=n_th;
		omp_set_num_threads(n_th);
	}
	if(4==argc){
		k=strtol(argv[1], NULL, 10);
    	n_th=strtol(argv[2], NULL, 10);
		n=strtol(argv[3], NULL, 10)*n_th;
		omp_set_num_threads(n_th);
	}
	if(argc<3){
		guida_uso();
		return 0;
	}


    Campione* vicini;
    Elementi* el;
    Elementi query;
    Database d=creata_Data(n);

    query.x=(float)(rand()%(MAX-MIN)+MIN)/100.00;
    query.y=(float)(rand()%(MAX-MIN)+MIN)/100.00;
    query.z=(float)(rand()%(MAX-MIN)+MIN)/100.00;

    el=(Elementi *) malloc(sizeof(Elementi)*n);
    
    init(el, n);
    if(el_to_data(el, d)!=0){
	    fprintf(stderr, "Errore inizializzazione data");
    }


	printf("\n\nESECUZIONE CON %d THREADS SU %lld ELEMENTI\n", n_th, n);


    stampa_query(query);

	double t0, t1, best=10000, avg=0;

    printf("\n---Esecuzione lineare---\n");

	for(int i=0; i<5; ++i){
		t0=now_sec();
	        vicini=seq_CalcolaVicini(el, n, query, k);

        	qsort(vicini, k, sizeof(Campione), compara_camp);
		t1=now_sec();
		double exec=t1-t0;
		avg+=exec;
		best = (best>exec) ? exec: best;
	}
	printf("Tempo di esecuzione migliore %8.8fs, medio %8.8fs\n\n", best, avg/5);

	stampa_Vicini(el, vicini, k);

	avg=0; best=10000;
    printf("\n---Esecuzione AVX---\n");

    for(int i=0; i<5; ++i){
    	t0=now_sec();
        vicini=avx_CalcolaVicini(d, 0, n, query, k);

        t1=now_sec();
        double exec=t1-t0;
        avg+=exec;
        best = (best>exec) ? exec: best;
    }
	
    printf("Tempo di esecuzione migliore %8.8fs, medio %8.8fs\n\n", best, avg/5);
    stampa_ViciniD(vicini, d, k);

    printf("\n---Esecuzione thread---\n");
	avg=0; best=10000;
	for(int i=0; i<5; ++i){
		t0=now_sec();

        vicini=thread_CalcolaVicini(el, n, query, k, omp_get_max_threads());
		t1=now_sec();
		double exec=t1-t0;
		avg+=exec;
		best = (best>exec) ? exec: best;
	        
	}
	printf("Tempo di esecuzione migliore %8.8fs, medio %8.8fs\n\n", best, avg/5);
	stampa_Vicini(el, vicini, k);

	printf("\n---Esecuzione OMP---\n");
	avg=0; best=1000;
	for(int i=0; i<5; ++i){

		t0=now_sec();
	    	vicini= omp_CalcolaVicini(el, n, query, k);
		t1=now_sec();
		double exec=t1-t0;
		avg+=exec;
		best = (best>exec) ? exec:best;
	}
	printf("Tempo di esecuzione migliore %8.8fs, medio %8.8fs\n\n", best, avg/5);
	stampa_Vicini(el, vicini, k);

	printf("\n---Esecuzione THREAD AVX---\n");
    avg=0; best=1000;
    for(int i=0; i<5; ++i){
		t0=now_sec();
        vicini= threadAVX_CalcolaVicini(d, query, k, omp_get_max_threads());
        t1=now_sec();
		double exec=t1-t0;
        avg+=exec;
        best = (best>exec) ? exec:best;
    }
        printf("Tempo di esecuzione migliore %8.8fs, medio %8.8fs\n\n", best, avg/5);
	stampa_ViciniD(vicini, d, k);

	free(el); free(d.x); free(d.y); free(d.z); free(d.idx);
	return 0;
}

void guida_uso(){
	printf("Giuda all'uso\n");
	printf("Parametrei: shared.o [k] [N thread] di deafult 2.5*10^8 elementi per thread\n");
	printf("Parametrei: shared.o [k] [N thread] [N elementi]\n");
}


