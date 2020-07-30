#ifndef H_SEARCH
#define H_SEARCH

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "pq_test_load_vectors.h"
#include "pq_new.h"
#include "pq_assign.h"
extern "C" {
#include "../yael/kmeans.h"
#include "../yael/nn.h"
#include "../yael/vector.h"
#include "../yael/matrix.h"
#include "../yael/sorting.h"
}


void pq_search(pqtipo pq, matI codebook, mat vquery, int k, float *dis, int *ids);
float* sumidxtab(mat distab, matI codebook);
void k_min (mat disquerybase, int k, float *dis, int *ids);

#endif
