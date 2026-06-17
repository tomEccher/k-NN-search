#ifndef THREAD_KNN_H_INCLUDED
#define THREAD_KNN_H_INCLUDED 
#include <pthread.h>
#include "par_knn.h"

int k;
Elementi q;
Max_Heap *out;
Elementi *pts;
size_t chunk_size;
size_t last_chunk_size;
int n_thread;
Database d;
int MPI_rank;

//function for qsort ordering
static int compara_camp(const void *a, const void* b){
    Campione c1=*(Campione*)a;
    Campione c2=*(Campione*)b;

    if(c1.d2 != c2.d2) return c1.d2>c2.d2;
    return c1.idx>c2.idx;

}

void* thread_work(void* rank){

    int r=*(int *)rank;

    Campione tmp;

    out[r]=createHeap(k);

    size_t start=r*chunk_size;
    size_t stop=0;

    if(r==n_thread-1){
        stop=start+last_chunk_size;
    }else{
        stop=start+chunk_size;
    }

	float threshold;
	size_t i=start;
//fill the heap with k values
	for(; i<start+k; ++i){
	        volatile float dx, dy, dz;
	        dx=pts[i].x-q.x;
        	dy=pts[i].y-q.y;
	        dz=pts[i].z-q.z;

        	tmp.d2=dx*dx+dy*dy+dz*dz;
			tmp.idx=pts[i].idx;
            heap_push(&out[r], tmp);
	}
//set thre threshold 
	threshold=out[r].c[0].d2;
    for(; i<stop; ++i){
        volatile float dx, dy, dz;
        dx=pts[i].x-q.x;
        dy=pts[i].y-q.y;
        dz=pts[i].z-q.z;

        float d2=dx*dx+dy*dy+dz*dz;

	if(d2<=threshold){
		tmp.d2=d2;
	        tmp.idx=pts[i].idx;
        	heap_push(&out[r], tmp);
		threshold=heap_maxd2(out[r]);
	}
    }

    qsort(out[r].c, k, sizeof(Campione), compara_camp);

    return NULL;

}

void* thread_AVX(void *rank){
	int r=*(int*)rank;

	size_t start=r*chunk_size;
    size_t stop=0;

	if(r==n_thread-1){
		stop=d.n;
	}else{
		stop=start+chunk_size;
	}

	out[r]=createHeap(k);

	 Campione tmp;
    __m256 QX = _mm256_set1_ps(q.x);
    __m256 QY = _mm256_set1_ps(q.y);
    __m256 QZ = _mm256_set1_ps(q.z);

    float *d2buff;
    posix_memalign((void**)&d2buff, 64, sizeof(float)*8);
    float threshold;


    for(size_t i=start; i<start+k; ++i){
            volatile float dx=d.x[i]-q.x;
            volatile float dy=d.y[i]-q.y;
            volatile float dz=d.z[i]-q.z;

            tmp.d2=dx*dx+dy*dy+dz*dz;
            tmp.idx=d.idx[i];
            heap_push(&out[r], tmp);
    }

    threshold=out[r].c[0].d2;
    size_t i=start;
    for(; i+7<stop; i+=8){
	    if(i+16<stop){
        __builtin_prefetch(&d.x[i+16], 0, 1);
        __builtin_prefetch(&d.y[i+16], 0, 1);
        __builtin_prefetch(&d.z[i+16], 0, 1);
	    } 

        __m256 X = _mm256_loadu_ps(&d.x[i]);
        __m256 Y = _mm256_loadu_ps(&d.y[i]);
        __m256 Z = _mm256_loadu_ps(&d.z[i]);

        __m256 DX = _mm256_sub_ps(X, QX);
        __m256 DY = _mm256_sub_ps(Y, QY);
        __m256 DZ = _mm256_sub_ps(Z, QZ);

	    __m256 D2 = _mm256_fmadd_ps(DZ, DZ, _mm256_fmadd_ps(DY, DY, _mm256_mul_ps(DX, DX)));

        _mm256_store_ps(d2buff, D2);

        for(int l=0; l<8; l++){
                if(d2buff[l]<=threshold){

                    Campione cp;
                    cp.d2=d2buff[l];
                    cp.idx=d.idx[i+l];
                    heap_push(&out[r], cp);
                    threshold=heap_maxd2(out[r]);
                }
        }
    }


    for (; i < stop; ++i) {
        volatile float dx = d.x[i] - q.x;
        volatile float dy = d.y[i] - q.y;
        volatile float dz = d.z[i] - q.z;

        volatile float d2=dx*dx+dy*dy+dz*dz;

        if(d2<=threshold){

                Campione cp;
                cp.idx=d.idx[i];
                cp.d2=d2;
                heap_push(&out[r], cp);
                threshold=heap_maxd2(out[r]);
        }
    }

	qsort(out[r].c, k, sizeof(Campione), compara_camp);
	return NULL;
}

void merge_P_lists(Campione *all, int P, Campione *out){

    Min_Heap m=create_mHeap(P);

    for(int i=0; i<P; i++){
        m.n[i].val=all[i*k];
        m.n[i].r=i;
        m.n[i].pos=0;

        m.dim++;
    }


    for(int i=(P-1)/2; i>=0; i--)
        node_sift_down(&m, i);

    for(int i=0; i<k; i++){
        Node best= node_pop(&m);
        out[i]=best.val;

        int pos=best.pos+1;
        int r=best.r;

        if(pos<k){
            Node nxt;
            nxt.pos=pos;
            nxt.r=r;
            nxt.val=all[r*k+pos];
            node_push(&m, nxt);
        }
    }
}

Campione *threadAVX_CalcolaVicini(Database database, Elementi query, int k, int n_th){
	
	k=k;
	q=query;
	d=database;
	n_thread=n_th;

	pthread_t *tID=(pthread_t*)malloc(sizeof(pthread_t)*n_thread);
	out=(Max_Heap*)malloc(sizeof(Max_Heap)*n_thread);
	chunk_size=d.n/n_thread;
	last_chunk_size=d.n-chunk_size*(n_thread-1);

	int *par=(int *)malloc(sizeof(int)*n_thread);
	for(int i=0; i<n_thread; ++i){
		par[i]=i;
		pthread_create(&tID[i], NULL, thread_AVX, (void*)&par[i]);

	}

	for(int i=0; i<n_thread; i++){
		pthread_join(tID[i], NULL);
	}

	Campione *all_k=(Campione*)malloc(sizeof(Campione)*k*n_thread);

#pragma omp for collapse(2)
	for(int j=0; j<n_thread; ++j){
		for(int i=0; i<k; i++){
			all_k[k*j+i]=out[j].c[i];
		}
	}

	Campione *result=(Campione*)malloc(sizeof(Campione)*k);

	free(out);

	merge_P_lists(all_k, n_thread, result);
	free(all_k);
	return result;

}


Campione* thread_CalcolaVicini(Elementi *el, size_t n, Elementi query, int k, int n_th){


    k=k;
    q=query;
    pts=el;
    n_thread=n_th;

    pthread_t *tID=(pthread_t*)malloc(sizeof(pthread_t)*n_thread);
    out=(Max_Heap*)malloc(sizeof(Max_Heap)*n_thread);

    chunk_size=n/n_thread;
    last_chunk_size=n-chunk_size*(n_thread-1);

	int *par=(int *)malloc(sizeof(int)*n_thread);
    for(int i=0; i< n_thread; i++){
	    par[i]=i;
        pthread_create(&tID[i], NULL, thread_work, (void *)&par[i]);
    }


    for(int i=0; i<n_thread; i++){
        pthread_join(tID[i], NULL);
    }

    Campione *all_k=(Campione *)malloc(sizeof(Campione)*k*n_thread);

#pragma omp for collapse(2) 
   for(int j=0; j<n_thread; ++j){
        for(int i=0; i<k; ++i){
            all_k[k*j+i]=out[j].c[i];
        }
    }

    Campione *result=(Campione *)malloc(sizeof(Campione)*k);
    free(out);
    merge_P_lists(all_k, n_thread, result);
    free(all_k);

    return result;
}

#endif // THREAD_KNN_H_INCLUDED
