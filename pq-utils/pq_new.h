#ifndef H_NEW
#define H_NEW

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "pq_test_load_vectors.h"
extern "C" {
#include "../yael/kmeans.h"
#include "../yael/vector.h"
#include "../yael/matrix.h"
}

/* flags para o kmeans */
#define KMEANS_QUIET                    0x10000
#define KMEANS_INIT_BERKELEY            0x20000
#define KMEANS_NORMALIZE_CENTS          0x40000
#define KMEANS_INIT_RANDOM              0x80000
#define KMEANS_INIT_USER               	0x100000
#define KMEANS_L1                       0x200000
#define KMEANS_CHI2                     0x400000

using namespace std;

typedef struct pqtipo{
	int nsq;
	int ks;
	int ds;
	float *centroids;
	int centroidsn;
	int centroidsd;
} pqtipo;

pqtipo pq_new(int, mat, int, int);

void check_new();

/* Concatena dois vetores, caso um não exista ele é criado exatamente igual ao vin
 * vinout : vetor a ser aumentado com os valores de vin
 * vinout_n : tamanho do vetor
 * vin : vetor a ser copiado
 *  ...
 */

void copySubVectors(float *vout, mat vin, int ds, int n1row, int n1col, int n2col);

void ivec_concat_transp(matI vinout, int* vin, int nsq);

#endif
