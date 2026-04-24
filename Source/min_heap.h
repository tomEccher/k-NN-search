#ifndef MIN_HEAP_H_INCLUDED
#define MIN_HEAP_H_INCLUDED

typedef struct{
    Campione val;
    int r;
    int pos;
}Node;




typedef struct{
    Node* n;
    int dim;
    int cap;

}Min_Heap;


Min_Heap create_mHeap(int k){

    Min_Heap m;
    m.dim=0;
    m.cap= (k>0) ?  k: 0;
    m.n=(k>0)?(Node*)malloc(sizeof(Node)*m.cap) : NULL;

    return m;
}

void free_mHeap(Min_Heap *m){
    free(m->n);
    m->n=NULL;
    m->cap=m->dim=0;
}



/* swap */
static inline void node_swap(Node *a, Node *b) {
    Node t = *a; *a = *b; *b = t;
}

static inline int compara_min(Node n1, Node n2){
   if(n1.val.d2 != n2.val.d2) return n1.val.d2 < n2.val.d2;
	   return n1.val.idx < n2.val.idx;
}

/* risale finché la proprietà di heap è rispettata */
static void node_sift_up(Min_Heap *m, int i) {
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (!compara_min(m->n[i], m->n[p])) break;
        node_swap(&m->n[i], &m->n[p]);
        i = p;
    }
}

/* scende per ripristinare la proprietà di heap */
static void node_sift_down(Min_Heap *m, int i) {
    int n = m->dim;
    while (1) {
        int l = (i << 1) + 1;
        int r = l + 1;
        int g = i;
        if (l < n && compara_min(m->n[l], m->n[g])) g = l;
        if (r < n && compara_min(m->n[r], m->n[g])) g = r;
        if (g == i) break;
        node_swap(&m->n[i], &m->n[g]);
        i = g;
    }
}

Node node_pop(Min_Heap *m){
    Node head = m->n[0];
    m->n[0]=m->n[--m->dim];
    node_sift_down(m, 0);
    return head;
}


void node_push(Min_Heap *m, Node nd){
    if(m->dim < m->cap){

        m->n[m->dim]=nd;
        node_sift_up(m, m->dim);
        m->dim++;
    }
}











#endif // MIN_HEAP_H_INCLUDED
