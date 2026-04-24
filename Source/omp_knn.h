#ifndef OMP_KNN_H_INCLUDED
#define OMP_KNN_H_INCLUDED
#include <omp.h>
#include <immintrin.h>
#include <time.h>

Campione* omp_CalcolaVicini(Elementi *e, size_t n, Elementi query, int k){
	int num_threads=omp_get_max_threads();

	Max_Heap *out=(Max_Heap*)malloc(sizeof(Max_Heap)*num_threads);

//	printf("Numero di thread OMP %d\n", num_threads);

	for(int i=0; i<num_threads; ++i){
		out[i]=createHeap(k);
	}

	size_t chunk=n/num_threads;

	#pragma omp parallel
	{
		int id=omp_get_thread_num();
//		printf("Id lavoro %d\n", id);
		Max_Heap *local=&out[id];
		Campione tmp;
		float dMAX=0;
		
		
		for(int i=0; i<k; ++i){
			float dx=e[i].x-query.x;
			float dy=e[i].y-query.y;
			float dz=e[i].z-query.z;

			tmp.d2=dx*dx+dy*dy+dz*dz;
			tmp.idx=e[i].idx;

			heap_push(local, tmp);
		}

		dMAX=local->c[0].d2;
		
		#pragma omp for schedule(static, chunk)
		for(size_t i=k; i<n; ++i){

			__builtin_prefetch(&e[i+16], 0, 1);

			float dx=e[i].x-query.x;
                        float dy=e[i].y-query.y;
                        float dz=e[i].z-query.z;
			
			float d2=dx*dx+dy*dy+dz*dz;

			if(d2<=dMAX){
	                        tmp.d2=d2;
        	                tmp.idx=e[i].idx;
	                        heap_push(local, tmp);
				dMAX=local->c[0].d2;
			}
		}

	}

	for(int i=0; i< num_threads; i++){
		 qsort(out[i].c, k, sizeof(Campione), compara_camp);
	}

	Campione *all=(Campione *)malloc(sizeof(Campione)*k*num_threads);

#pragma omp for collapse(2)
	for(int j=0; j<num_threads; ++j){
		for(int i=0; i<k; ++i){
			all[k*j+i]=out[j].c[i];
		}
	}

	Campione *result=(Campione*)malloc(sizeof(Campione)*k);

	merge_P_lists(all, num_threads, result);
	free(all);
	return result;

}



Campione* ompAVX_CalcolaVicini(Database d, Elementi query, int k){
	int num_threads=omp_get_max_threads();
	size_t N=d.n;

	Max_Heap *out=(Max_Heap *)malloc(sizeof(Max_Heap)*num_threads);
	for(int i=0; i<num_threads; ++i){
		out[i]=createHeap(k);
	}

//	printf("Numero di omp thread + avx %d\n", num_threads);
	size_t chunk=N/num_threads;
	int last=num_threads-1;

	__m256 QX = _mm256_set1_ps(query.x);
	__m256 QY = _mm256_set1_ps(query.y);
	__m256 QZ = _mm256_set1_ps(query.z);

#pragma omp parallel
	{
		int id=omp_get_thread_num();

		Max_Heap *local=&out[id];

		Campione tmp;
		float *d2buff; posix_memalign((void**)&d2buff, 64, sizeof(float)*8);
		float dMAX;

		
		for(int i=0; i<k; ++i){
			volatile float dx=d.x[i]-query.x;
			volatile float dy=d.y[i]-query.y;
			volatile float dz=d.z[i]-query.z;

			tmp.d2=dx*dx+dy*dy+dz*dz;
			tmp.idx=d.idx[i];
			heap_push(local, tmp);
		}


		size_t i=0;
		dMAX=out[id].c[0].d2;
		for(; i+7<N; i+=8){
			__builtin_prefetch(&d.x[i+16], 0, 1);
			__builtin_prefetch(&d.y[i+16], 0, 1);
			__builtin_prefetch(&d.z[i+16], 0, 1);

			__m256 X = _mm256_load_ps(&d.x[i]);
			__m256 Y = _mm256_load_ps(&d.y[i]);
			__m256 Z = _mm256_load_ps(&d.z[i]);

			__m256 DX = _mm256_sub_ps(X, QX);
			__m256 DY = _mm256_sub_ps(Y, QY);
			__m256 DZ = _mm256_sub_ps(Z, QZ);

			__m256 D2 = _mm256_fmadd_ps(DX, DX, _mm256_fmadd_ps(DY, DY, _mm256_mul_ps(DZ, DZ)));

			_mm256_store_ps(d2buff, D2);

			#pragma omp for schedule (static)
			for(int j=0; j<8; ++j){
				if(d2buff[j]<dMAX){
					tmp.d2=d2buff[j];
					tmp.idx=d.idx[i+j];
					heap_push(local, tmp);
					dMAX=local->c[0].d2;
				}
			}
		}

		for(; i<N; ++i){
			volatile float dx=d.x[i]-query.x;
			volatile float dy=d.y[i]-query.y;
			volatile float dz=d.z[i]-query.z;

			volatile float d2=dx*dx+dy*dy+dz*dz;

			if(d2<dMAX){
				tmp.idx=d.idx[i];
				tmp.d2=d2;
				heap_push(local, tmp);
				dMAX=local->c[0].d2;
			}
		}

		qsort(out[id].c, k, sizeof(Campione), compara_camp);
	}


	Campione *all=(Campione*)malloc(sizeof(Campione)*k*num_threads);

	#pragma omp for collapse(2)
	for(int j=0; j<num_threads; ++j){
		for(int i=0; i<k; ++i){
			all[j*k+i]=out[j].c[i];
		}
	}

	Campione *result=(Campione *)malloc(sizeof(Campione)*k);

	merge_P_lists(all, num_threads, result);
	free(all);

	return result;
}	


#endif // OMP_KNN_H_INCLUDED
