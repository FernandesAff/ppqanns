#include "k_min.h"

void my_k_min(dis_t q, int ktmp, float *dis, int *ids, int inner_threads){
	int *qidx = (int*)malloc(sizeof(int)*q.idx.n), j=1;
	float *qdis = (float*)malloc(sizeof(float)*q.dis.n);

	if (q.dis.n == 0 || ktmp == 0)return;
	
	memcpy(&qdis[0],&q.dis.mat[0], sizeof(float)*q.dis.n);
	
	memcpy(&qidx[0],&q.idx.mat[0], sizeof(int)*q.idx.n);
	
	constroiHeap(q.dis.n, qdis, qidx, inner_threads);
	
	dis[0] = qdis[0];
    	
	qdis[0] = qdis[q.dis.n-1];
    	
	ids[0] = qidx[0];
    	
	qidx[0] = qidx[q.dis.n-1];
	
	#pragma omp parallel for schedule(dynamic) num_threads(inner_threads)
	for (int i = q.dis.n-1; i > q.dis.n-ktmp; i--){
		
		trocarRaiz(q.dis.n, qdis, qidx);
		
		dis[j] = qdis[0];
       		
		qdis[0] = qdis[i-1];
        	
		ids[j] = qidx[0];
        	
		qidx[0] = qidx[i-1];
		
        	j++;
	}
	free(qidx);
	free(qdis);

}

static void constroiHeap (int n, float *qdis, int *qidx, int inner_threads){
	int k, aux;
	float aux2;

	#pragma omp parallel for schedule(dynamic) num_threads(inner_threads)
	for (k = 1; k < n; k++) {
		int f = k+1;
		while (f > 1 && qdis[(f/2)-1] > qdis[f-1]) {
        	aux2 = qdis[(f/2)-1];
        	qdis[(f/2)-1] = qdis[f-1];
        	qdis[f-1] = aux2;
        	aux = qidx[(f/2)-1];
        	qidx[(f/2)-1] = qidx[f-1];
        	qidx[f-1] = aux;
			f /= 2;
		}
	}
}

static void trocarRaiz (int n, float *qdis, int *qidx){
	int p = 1, f = 2, aux2 = qidx[0];
	float aux = qdis[0];
	while (f < n) {
		if (f < n-1 && qdis[f-1] > qdis[f])  f++;
		if (aux <= qdis[f]) break;
		qdis[p-1] = qdis[f-1];
		qidx[p-1] = qidx[f-1];
		p = f;
		f = 2*p;
	}
	qdis[p-1] = aux;
	qidx[p-1] = aux2;
}
