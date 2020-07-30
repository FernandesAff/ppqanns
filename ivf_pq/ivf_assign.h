#ifndef H_IVFASSIGN
#define H_IVFASSIGN

	#include <stdio.h>
	#include <stdlib.h>
	#include <mpi.h>
	extern "C"{
		#include "../yael/nn.h"
	}
	#include "../pq-utils/pq_test_load_vectors.h"
	#include "../pq-utils/pq_new.h"
	#include "myIVF.h"

	void parallel_assign (char *dataset, int w, int comm_sz, MPI_Comm search_comm, int threads);
	void bsxfunMINUS(float *mout, mat vin, float* vin2, int nq, int* qcoaidx, int ncoaidx);

#endif
