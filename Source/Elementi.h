#ifndef ELEMENTI_H_INCLUDED
#define ELEMENTI_H_INCLUDED
#include "max_heap.h"

const int MAX=50000;
const int MIN=1;

typedef struct{

    float x;
    float y;
    float z;

    size_t idx;

}Elementi;


void init(Elementi *e, size_t n){

#pragma omp for schedule(static)
    for(size_t i=0; i<n; i++){
        e[i].idx=i;
        e[i].x=(float)(rand()%(MAX-MIN)+MIN)/100.00;
        e[i].y=(float)(rand()%(MAX-MIN)+MIN)/100.00;
        e[i].z=(float)(rand()%(MAX-MIN)+MIN)/100.00;
    }

}

void stampa_Elementi(Elementi *e, size_t n){

    for(size_t i=0; i<n; i++){
        printf("Elemento %ld -> x = %.2f \ty = %.2f \tz = %.2f\n", e[i].idx, e[i].x, e[i].y, e[i].z);

    }
    printf("\n");
}


void stampa_query(Elementi e){
    printf("Elemento di query -> x = %.2f \ty = %.2f \tz = %.2f\n", e.x, e.y, e.z);
}


void stampa_Vicini(Elementi *e, Campione* c, int k){



        for(int i=0; i<k; i++){
            int idx=c[i].idx;
            printf("Vicino #%d: indice = %ld\tx = %.2f \ty = %.2f \tz = %.2f\tcon distanza^2=%.3f\n",
                i+1, idx, e[idx].x, e[idx].y, e[idx].z, c[i].d2);
        }


}



#endif // ELEMENTI_H_INCLUDED
