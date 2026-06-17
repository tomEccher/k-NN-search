#ifndef SEQ_KNN_H_INCLUDED
#define SEQ_KNN_H_INCLUDED


Campione* seq_CalcolaVicini(Elementi *e, size_t n, Elementi query, int k){

    Max_Heap m=createHeap(k);
    Campione tmp;
    float treshold=999999;
    size_t i=0;

    for(; i<k; ++i){

        volatile float dx=e[i].x-query.x;
        volatile float dy=e[i].y-query.y;
        volatile float dz=e[i].z-query.z;

        tmp.d2=dx*dx+dy*dy+dz*dz;
        tmp.idx=e[i].idx;

	heap_push(&m, tmp);
    }


    for(; i<n; i++){

        volatile float dx=e[i].x-query.x;
        volatile float dy=e[i].y-query.y;
        volatile float dz=e[i].z-query.z;

        float d2=dx*dx+dy*dy+dz*dz;

	if(d2<=treshold){
		tmp.d2=d2;
		tmp.idx=e[i].idx;
        	heap_push(&m, tmp);
		treshold=heap_maxd2(m);
	}
    }

   // printf("%d", m->c[0].d2);

    return m.c;
}

#endif // SEQ_KNN_H_INCLUDED
