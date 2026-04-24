#ifndef MAX_HEAP_H_INCLUDED
#define MAX_HEAP_H_INCLUDED


typedef struct{
    float d2;
    size_t idx;

}Campione;


typedef struct{
    Campione* c;
    int dim;
    int cap;

}Max_Heap;


Max_Heap createHeap(int k){

    Max_Heap m;
    m.dim=0;
    m.cap= (k>0) ?  k: 0;
    m.c=(k>0)?(Campione*)malloc(sizeof(Campione)*m.cap) : NULL;

    return m;
}

void freeHeap(Max_Heap *m){
    free(m->c);
    m->c=NULL;
    m->cap=m->dim=0;
}

float heap_maxd2(Max_Heap m){
	return m.c[0].d2;
}

int heap_size(Max_Heap *m){
    return m->dim;
}

static inline int heap_full(Max_Heap *m){
    return m->dim==m->cap;
}


/* swap */
static inline void heap_swap(Campione *a, Campione *b) {
    Campione t = *a; *a = *b; *b = t;
}

static inline int compara_mag(Campione c1, Campione c2){
    if(c1.d2!=c2.d2) return c1.d2 >c2.d2;
	return c1.idx > c2.idx;
}

/* risale finché la proprietà di heap è rispettata */
static inline void heap_sift_up(Max_Heap *m, int i) {
	Campione tmp=m->c[i];
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (!compara_mag(m->c[i], m->c[p])) break;
        m->c[i] = m->c[p];
        i = p;
    }
    m->c[i]=tmp;
}

/* scende per ripristinare la proprietà di heap */
static inline void heap_sift_down(Max_Heap *m, int i) {
    int n = m->dim;
    while (1) {
        int l = (i << 1) + 1;
        int r = l + 1;
        int g = i;
        if (l < n && compara_mag(m->c[l], m->c[g])) g = l;
        if (r < n && compara_mag(m->c[r], m->c[g])) g = r;
        if (g == i) break;
        heap_swap(&m->c[i], &m->c[g]);
        i = g;
    }
}

Campione heap_pop(Max_Heap *m){
    Campione head = m->c[0];
    m->c[0]=m->c[--m->dim];
    heap_sift_down(m, 0);
    return head;
}
void stampa_Campion(Campione *c, int n){

    for(int i=0; i<n; i++){
        printf("%d, %.2f\n", c[i].idx, c[i].d2);
    }
}

static inline void heap_push(Max_Heap *m, Campione cp){
    if(m->dim<m->cap){

        m->c[m->dim]=cp;
        heap_sift_up(m, m->dim);
        m->dim++;
    }else{
            m->c[0] = cp;
            heap_sift_down(m, 0);
    }


}









#endif // MAX_HEAP_H_INCLUDED
