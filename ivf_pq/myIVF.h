#ifndef H_MYIVF
#define H_MYIVF

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../pq-utils/pq_assign.h"
#include "../pq-utils/pq_new.h"
#include "../pq-utils/pq_search.h"

extern "C" {
	#include "../yael/vector.h"
	#include "../yael/kmeans.h"
}

typedef struct {
	mat dis;
	matI idx;
}dis_t;

typedef struct ivfpq{
	pqtipo pq;
	int coarsek;
	float *coa_centroids;
	int coa_centroidsn;
	int coa_centroidsd;
}ivfpq_t;

typedef struct ivf{
	int* ids;
	int idstam;
	matI codes;
}ivf_t;

typedef struct query_id{
	int tam;
	int id;
}query_id_t;

void set_last (int comm_sz, int *last_assign, int *last_search, int *last_aggregator);

#endif
