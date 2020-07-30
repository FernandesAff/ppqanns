#ifndef H_IVFNEW
#define H_IVFNEW

	#include <stdio.h>
	#include <stdlib.h>
	#include <mpi.h>
	#include <omp.h>
	extern "C"{
		#include "../yael/vector.h"
		#include "../yael/nn.h"
		#include "../yael/kmeans.h"
	}
	#include "../pq-utils/pq_test_load_vectors.h"
	#include "../pq-utils/pq_new.h"
	#include "../pq-utils/pq_assign.h"
	#include "myIVF.h"

	void parallel_training (char *dataset, int coarsek, int nsq, long long tam, int comm_sz, int threads);
	void subtract(mat v, float* v2, int* idx, int c_d);
	ivfpq_t ivfpq_new(int coarsek, int nsq, mat vtrain, int threads);
	void write_cent(char *file, char *file2, char *file3, ivfpq_t ivfpq);
	void read_cent(char *file, char *file2, char *file3, ivfpq_t *ivfpq);

#endif
