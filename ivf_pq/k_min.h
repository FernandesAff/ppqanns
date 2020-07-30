

#ifndef K_MIN
#define K_MIN

// A C++ program to find k'th smallest element using min heap
#include<iostream>
#include<climits>
#include <list>
#include "../pq-utils/pq_new.h"
#include "../pq-utils/pq_test_load_vectors.h"
#include "../pq-utils/pq_test_compute_stats.h"
#include "myIVF.h"

void my_k_min(dis_t q, int ktmp, float *dis, int *ids, int inner_threads);
static void constroiHeap (int n, float *qdis, int *qidx, int inner_threads);
static void trocarRaiz (int n, float *qdis, int *qidx);

#endif
