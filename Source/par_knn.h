#ifndef PAR_KNN_H_INCLUDED
#define PAR_KNN_H_INCLUDED
#include <omp.h>
#include <immintrin.h>
#include <time.h>
#include <omp.h>

typedef struct{
    float * x;
    float * y;
    float * z;
    size_t * idx;
    size_t n;

}Database;


static inline double now_sec(void) {
    return omp_get_wtime();
}


Database creata_Data(size_t n){

    Database data;
    data.n=n;
    size_t b=sizeof(float)*n;

	if( posix_memalign((void**)&data.x, 64, b) ||
    posix_memalign((void**)&data.y, 64, b) ||
    posix_memalign((void**)&data.z, 64, b) ||
    posix_memalign((void**)&data.idx, 64, sizeof(size_t)*n)){
		fprintf(stderr, "posix fail\n");
	}

    return data;
}


void init_database(Database d){

    for(size_t i=0; i<d.n; ++i){
        d.idx[i]=i;
        d.x[i]=(rand()%(MAX-MIN)+MIN)/100.00;
        d.y[i]=(rand()%(MAX-MIN)+MIN)/100.00;
        d.z[i]=(rand()%(MAX-MIN)+MIN)/100.00;

    }

}

int el_to_data(Elementi *e, Database d){

	for(size_t i=0; i<d.n; ++i){
		d.x[i]=e[i].x;
		d.y[i]=e[i].y;
		d.z[i]=e[i].z;
		d.idx[i]=e[i].idx;
	}

	return 0;
}



void stampaD(Database d){
	for(size_t i=0; i<d.n; i++){
		printf("Elemento %ld x = %.2f\ty = %.2f\tz = %.2f",i, d.x[i], d.y[i], d.z[i]);
	}
}


void stampa_ViciniD(Campione *c, Database d, int k){
	for(int j=0; j<k; j++){
		int i=c[j].idx;
		printf("Vicnino #%d: indice = %ld\tx = %.2f\ty = %.2f\tz = %.2f\t con distanza^2=%.3f\n",
				j+1, i, d.x[i], d.y[i], d.z[i], c[j].d2);
	}
}




Campione* avx_CalcolaVicini(Database d, size_t start, size_t stop, Elementi query, int k){

    Max_Heap m=createHeap(k);
    Campione tmp;
    __m256 QX = _mm256_set1_ps(query.x);
    __m256 QY = _mm256_set1_ps(query.y);
    __m256 QZ = _mm256_set1_ps(query.z);

    float *d2buff;
    posix_memalign((void**)&d2buff, 64, sizeof(float)*8);
    float threshold;
	
//	printf("Start %d, stop %d\n", start, stop);

    for(size_t i=start; i<start+k; ++i){
	    volatile float dx=d.x[i]-query.x;
	    volatile float dy=d.y[i]-query.y;
	    volatile float dz=d.z[i]-query.z;

	    tmp.d2=dx*dx+dy*dy+dz*dz;
	    tmp.idx=d.idx[i];
	    heap_push(&m, tmp);
    }

	threshold=m.c[0].d2;
    size_t i=start;
    for(; i+7<stop; i+=8){
        __builtin_prefetch(&d.x[i+16], 0, 1);
        __builtin_prefetch(&d.y[i+16], 0, 1);
        __builtin_prefetch(&d.z[i+16], 0, 1);


        __m256 X = _mm256_load_ps(&d.x[i]);
        __m256 Y = _mm256_load_ps(&d.y[i]);
        __m256 Z = _mm256_load_ps(&d.z[i]);



        __m256 DX = _mm256_sub_ps(X, QX);
        __m256 DY = _mm256_sub_ps(Y, QY);
        __m256 DZ = _mm256_sub_ps(Z, QZ);

        __m256 D2 = _mm256_fmadd_ps(DZ, DZ, _mm256_fmadd_ps(DY, DY, _mm256_mul_ps(DX, DX)));

        _mm256_store_ps(d2buff, D2);


        for(int l=0; l<8; l++){
		if(d2buff[l]<threshold){

	            Campione cp;
        	    cp.d2=d2buff[l];
	            cp.idx=d.idx[i+l];
        	    heap_push(&m, cp);
		    threshold=heap_maxd2(m);
		}
        }
    }



    // Coda scalare
    for (; i < stop; ++i) {
        volatile float dx = d.x[i] - query.x;
        volatile float dy = d.y[i] - query.y;
        volatile float dz = d.z[i] - query.z;

	volatile float d2=dx*dx+dy*dy+dz*dz;

	if(d2<threshold){

	        Campione cp;
		cp.idx=d.idx[i];
		cp.d2=d2;
	        heap_push(&m, cp);
		threshold=heap_maxd2(m);
	}
    }

    return m.c;


   // printf("%d", m->c[0].d2);

}

#endif // PAR_KNN_H_INCLUDED
