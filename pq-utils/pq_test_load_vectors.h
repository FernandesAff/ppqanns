#ifndef H_LOAD_VECTORS
#define H_LOAD_VECTORS

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <string.h>
extern "C" {
#include "../yael/matrix.h"
#include "../yael/vector.h"
#include "../yael/nn.h"
}

using namespace std;

#define BASE_DIR "/home/andreff/bases/sift/"

typedef struct{
	float *mat;
	int n,d;
}mat;

typedef struct{
	int *mat;
	int n,d;
}matI;

typedef struct{
	mat train,base,query;
	matI ids_gnd;
}data;

typedef struct{
	char 	*train, *groundtruth;
}namefile;

mat pq_test_load_train(char* dataset, int tam);
matI pq_test_load_gdn(char* dataset, int tam);
mat pq_test_load_query(char* dataset, int threads);
mat pq_test_load_base(char* dataset, int offset, int my_rank);
void load_random (float *v, int n, int d);
int ivecs_read (const char *fname, int d, int n, int *a);
int my_bvecs_read (int offset, const char *fname, int d, int n, float *a);

#endif
