#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <unistd.h>
#include "pq-utils/pq_new.h"
#include "pq-utils/pq_test_load_vectors.h"

#include "ivf_pq/ivf_assign.h"
#include "ivf_pq/ivf_training.h"
#include "ivf_pq/ivf_search.h"
#include "ivf_pq/ivf_aggregator.h"

int main(int argc, char **argv){

	if(argc < 6){
		cout << "Usage: mpirun -n <n_processes> ./ivfpq_test <dataset> <threads_search> <tam> <coarsek> <nsq> <w> <threads_training> <tipo de paralelizacao> <outer> <inner>" << endl;
		return -1;
	}

	/*
		nsq - numero de subdimensões em que os vetores são divididos
		coarsek - numero de coarse centroides usados para indexacao da base
		tam - tamanho da base
		threads - numero de threads de busca
		w - numero de entradas a serem verificadas na lista invertida na etapa de busca
		k - quantidade de vetores a serem retornados na busca
		tamt - tamanho da base apos divisão entre os processos
		threads_training - numero de threads usadas na criacao dos centroides
		dataset - base a ser utilizada na busca
	*/
	int nsq, coarsek, threads, w, k, threads_training, comm_sz, provided, inner_threads, outer_threads;
	char* dataset, *parallel_type;
	long long tam, tamt;

	dataset = argv[1];
	threads  = atoi(argv[2]);
	tam  = atoll(argv[3]);
	coarsek = atoi(argv[4]);
	nsq = atoi(argv[5]);
	w = atoi(argv[6]);
	threads_training = atoi(argv[7]);
	parallel_type = argv[8];
	outer_threads = atoi(argv[9]);
	inner_threads = atoi(argv[10]);
	comm_sz = 1;
	k = 100;

	//Modo treinamento - com essa flag o algoritmo roda apenas a criação dos centroides a partir da base de treinamento
	#ifdef TRAIN

		struct timeval start, end;
		gettimeofday(&start, NULL);
		parallel_training (dataset, coarsek, nsq, tam, comm_sz, threads_training);
		gettimeofday(&end, NULL);
		double time = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));
		cout << time << endl;

	#else
		/*
			my_rank - guarda o rank do processo atual
			last_assign, last_search, last_aggregator - guardam os ranks dos ultimos processos de cada estagio
		*/
		int my_rank, last_assign, last_search, last_aggregator, n;
		MPI_Group world_group, search_group;
		MPI_Comm search_comm;

		MPI_Init_thread(&argc, &argv,MPI_THREAD_SERIALIZED,&provided);
		MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
		MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		last_assign=1;
		last_search=comm_sz-2;
		last_aggregator=comm_sz-1;
		tamt = tam/(last_search-last_assign);

		n = comm_sz-2;
		int ranks[n];
		for(int i=0; i<n; i++){
			ranks[i]=i+1;
		}
		MPI_Group_incl(world_group, n, ranks, &search_group);
		MPI_Comm_create_group(MPI_COMM_WORLD, search_group, 0, &search_comm);

		int rc;
                char hostname[50];
                rc = gethostname(hostname,sizeof(hostname));
                if(rc == 0){
                	printf("rank = %d, hostname = %s\n",my_rank,hostname);
                }

		if (my_rank<last_assign){
			parallel_training (dataset, coarsek, nsq, tam, comm_sz, threads_training);
		}
		else if(my_rank<=last_assign){
			parallel_assign (dataset, w, comm_sz,search_comm, threads);
		}
		else if(my_rank<=last_search){
			parallel_search (nsq, k, comm_sz, threads, tamt, search_comm, dataset, w, parallel_type, outer_threads, inner_threads);
		}
		else{
			parallel_aggregator(k, w, my_rank, comm_sz, tam, threads, dataset);
		}
		
		MPI_Finalize();
	#endif

	return 0;
}
