
#include "ivf_search.h"

int iter;
static int last_assign, last_search, last_aggregator;
static sem_t sem;

void parallel_search (int nsq, int k, int comm_sz, int threads, long long tam, MPI_Comm search_comm, char *dataset, int w, char *parallel_type, int outer_threads_1, int inner_threads_1){
	int *coaidx, my_rank, finish_aux, count, **ids, outer_threads, inner_threads;
	float **dis;
	double time;
	char arqNome[100];
	FILE *fp;
	mat residual;
	ivfpq_t ivfpq;
	ivf_t *ivf;
	query_id_t *fila;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	sprintf(arqNome, "%dtestes%d.txt", comm_sz, my_rank);
	fp = fopen(arqNome, "a");
	fprintf(fp, "Thread ivfpq_search min k_min critical cross_distances sumidx \n");
	fclose(fp);

	set_last (comm_sz, &last_assign, &last_search, &last_aggregator);

	//Recebe os centroides

	MPI_Recv(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivfpq.pq.centroidsn*ivfpq.pq.centroidsd);
	MPI_Recv(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivfpq.coa_centroidsd*ivfpq.coa_centroidsn);
	MPI_Recv(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsn*ivfpq.coa_centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//guarda a lista invertida em um arquivo
	#ifdef WRITE_IVF
		write_ivf(ivfpq, threads, tam, my_rank, nsq, dataset);

		ivf = read_ivf(ivfpq, tam, my_rank);
	#else
		//le a lista de um arquivo
		#ifdef READ_IVF

			ivf = read_ivf(ivfpq, tam, my_rank);

		#else
			ivf = create_ivf(ivfpq, threads, tam, my_rank, nsq, dataset);
		#endif
	#endif

	finish_aux=0;
	count =0;

	MPI_Barrier(search_comm);
	sem_init(&sem, 0, 1);

	omp_set_nested(1);
	#pragma omp parallel sections num_threads(2)
	{
		#pragma omp section
		{
			printf("do nothing");
		}
		#pragma omp section
		{
			while(1){
				int my_omp_rank;
				double f1=0, f2=0, f3=0, f4=0, g1=0, g2=0, g3=0;

				MPI_Bcast(&residual.n, 1, MPI_INT, 0, search_comm);
				MPI_Bcast(&residual.d, 1, MPI_INT, 0, search_comm);

				residual.mat = (float*)malloc(sizeof(float)*residual.n*residual.d);

				MPI_Bcast(&residual.mat[0], residual.d*residual.n, MPI_FLOAT, 0, search_comm);

				coaidx = (int*)malloc(sizeof(int)*residual.n);

				MPI_Bcast(&coaidx[0], residual.n, MPI_INT, 0, search_comm);
				MPI_Bcast(&finish_aux, 1, MPI_INT, 0, search_comm);

				fila = (query_id_t*)malloc(sizeof(query_id_t)*(residual.n/w));

				for(int it=0; it<residual.n/w; it++){
					fila[it].tam = 0;
					fila[it].id = count;
				}

				iter = 0;

				dis = (float**)malloc(sizeof(float *)*(residual.n/w));
				ids = (int**)malloc(sizeof(int *)*(residual.n/w));

				if(!strcmp(parallel_type, "manual")){
					outer_threads = outer_threads_1;
					inner_threads = inner_threads_1;
					printf("%d", inner_threads);
				}else{
					int iteration=0;
					outer_threads = 0;
					while(outer_threads<threads){
						outer_threads = pow(2,iteration);
						if(outer_threads>=(residual.n/w)){
							break;
						}
						iteration++;
					}
					inner_threads = threads/outer_threads;
				}

				#pragma omp sections num_threads(2)
				{
					#pragma omp section
					{
						send_aggregator(residual.n, w, fila, ids, dis, finish_aux, count);

						count += iter;

						free(dis);
						free(ids);
						free(coaidx);
						free(residual.mat);
						free(fila);
					}
					#pragma omp section
					{
						#pragma omp parallel for num_threads(outer_threads)
						for(int i=0; i<residual.n/w; i++){

							int ktmp;
							dis_t qt;
							query_id_t element;

							qt.idx.mat = (int*) malloc(sizeof(int));
							qt.dis.mat = (float*) malloc(sizeof(float));
							qt.idx.n=0;
							qt.idx.d=1;
							qt.dis.n=0;
							qt.dis.d=1;

							struct timeval a1, a2, a3, a4, a5, b1, b2;

							gettimeofday(&a1, NULL);

							for(int j=0; j<w; j++){
								dis_t q;
								q = ivfpq_search(ivf, &residual.mat[0]+(i*w+j)*residual.d, ivfpq.pq, coaidx[i*w+j], &g1, &g2, inner_threads);

								qt.idx.mat = (int*) realloc(qt.idx.mat, sizeof(int)*(qt.idx.n+q.idx.n));
								qt.dis.mat = (float*) realloc(qt.dis.mat, sizeof(float)*(qt.dis.n+q.dis.n));

								memcpy(&qt.idx.mat[qt.idx.n], &q.idx.mat[0], sizeof(int)*q.idx.n);
								memcpy(&qt.dis.mat[qt.dis.n], &q.dis.mat[0], sizeof(float)*q.dis.n);

								qt.idx.n+=q.idx.n;
								qt.dis.n+=q.dis.n;
								free(q.dis.mat);
								free(q.idx.mat);
							}

							gettimeofday(&a2, NULL);
							ktmp = min(qt.idx.n, k);

							ids[i] = (int*) malloc(sizeof(int)*ktmp);
							dis[i] = (float*) malloc(sizeof(float)*ktmp);
							gettimeofday(&a3, NULL);

							my_k_min(qt, ktmp, &dis[i][0], &ids[i][0], inner_threads);

							gettimeofday(&a4, NULL);

							element.id = i;
							element.tam = ktmp;
							sem_wait(&sem);
							fila[iter].id+=i;
							fila[iter].tam+=ktmp;
							iter++;
							sem_post(&sem);
							gettimeofday(&a5, NULL);

							f1 += ((a2.tv_sec * 1000000 + a2.tv_usec)-(a1.tv_sec * 1000000 + a1.tv_usec));
							f2 += ((a3.tv_sec * 1000000 + a3.tv_usec)-(a2.tv_sec * 1000000 + a2.tv_usec));
							f3 += ((a4.tv_sec * 1000000 + a4.tv_usec)-(a3.tv_sec * 1000000 + a3.tv_usec));
							f4 += ((a5.tv_sec * 1000000 + a5.tv_usec)-(a4.tv_sec * 1000000 + a4.tv_usec));

							free(qt.dis.mat);
							free(qt.idx.mat);
						}
						FILE *fp;

						fp = fopen(arqNome, "a");

						my_omp_rank = omp_get_thread_num();

						fprintf(fp, "%d %g %g %g %g %g %g \n",my_omp_rank, f1/1000,f2/1000,f3/1000,f4/1000,g1/1000,g2/1000);

						fclose(fp);
					}
				}
				if (finish_aux==1)break;
			}
		}
	}
	cout << "." << endl;
	sem_destroy(&sem);
	free(ivf);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
}

dis_t ivfpq_search(ivf_t *ivf, float *residual, pqtipo pq, int centroid_idx, double *g1, double *g2, int inner_threads){
	dis_t q;
	int ds, ks, nsq;
	struct timeval b1, b2, b3;

	ds = pq.ds;
	ks = pq.ks;
	nsq = pq.nsq;

	mat distab;
	distab.mat = (float*)malloc(sizeof(float)*ks*nsq);
	distab.n = nsq;
	distab.d = ks;

	float *distab_temp=(float*)malloc(sizeof(float)*ks);

	float* AUXSUMIDX;

	q.dis.n = ivf[centroid_idx].codes.n;
	q.dis.d = 1;
	q.dis.mat = (float*)malloc(sizeof(float)*q.dis.n);

	q.idx.n = ivf[centroid_idx].codes.n;
	q.idx.d = 1;
	q.idx.mat = (int*)malloc(sizeof(int)*q.idx.n);

	gettimeofday(&b1, NULL);

	for (int query = 0; query < nsq; query++) {

		compute_cross_distances_thread(ds, 1, distab.d, &residual[query*ds], &pq.centroids[query*ks*ds], distab_temp, 1);

		memcpy(distab.mat+query*ks, distab_temp, sizeof(float)*ks);
	}

	gettimeofday(&b2, NULL);
	
	AUXSUMIDX = sumidxtab2(distab, ivf[centroid_idx].codes, 0, inner_threads);

	gettimeofday(&b3, NULL);

	memcpy(q.idx.mat, ivf[centroid_idx].ids,  sizeof(int)*ivf[centroid_idx].idstam);
	memcpy(q.dis.mat, AUXSUMIDX, sizeof(float)*ivf[centroid_idx].codes.n);

	free (AUXSUMIDX);
	free (distab_temp);
	free (distab.mat);

	*g1 += ((b2.tv_sec * 1000000 + b2.tv_usec)-(b1.tv_sec * 1000000 + b1.tv_usec));
	*g2 += ((b3.tv_sec * 1000000 + b3.tv_usec)-(b2.tv_sec * 1000000 + b2.tv_usec));

	return q;
}

void send_aggregator(int residualn, int w, query_id_t *fila, int **ids, float **dis, int finish_aux, int count){

	int i=0, num=0, ttam=0, *ids2, it=0, my_rank, finish=0;
	float *dis2;
	query_id_t *element;

	ids2 = (int*)malloc(sizeof(int));
	dis2 = (float*)malloc(sizeof(float));
	element = (query_id_t*)malloc(sizeof(query_id_t)*10);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);



	while(i<residualn/w){

		while(it==iter);

		element[num] = fila[it];

		ids2 = (int*)realloc(ids2,sizeof(int)*(ttam+element[num].tam));
		dis2 = (float*)realloc(dis2,sizeof(float)*(ttam+element[num].tam));
		memcpy(&ids2[0] + ttam, ids[element[num].id-count], sizeof(int)*element[num].tam);
		memcpy(&dis2[0] + ttam, dis[element[num].id-count], sizeof(float)*element[num].tam);
		free(ids[element[num].id-count]);
		free(dis[element[num].id-count]);
		ttam+=element[num].tam;
		num++;
		it++;

		if(num==10 || num==residualn/w-i){
			if(num==residualn/w-i && finish_aux==1)finish=1;

			MPI_Send(&my_rank, 1, MPI_INT, last_aggregator, 1, MPI_COMM_WORLD);
			MPI_Send(&num, 1, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD);
			MPI_Send(&element[0], sizeof(query_id_t)*num, MPI_BYTE, last_aggregator, 0, MPI_COMM_WORLD);
			MPI_Send(&ids2[0], ttam, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD);
			MPI_Send(&dis2[0], ttam, MPI_FLOAT, last_aggregator, 0, MPI_COMM_WORLD);
			MPI_Send(&finish, 1, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD);

			i+=num;
			num=0;
			ttam=0;
		}
	}
	free(element);
	free(ids2);
	free(dis2);
}

ivf_t* create_ivf(ivfpq_t ivfpq, int threads, long long tam, int my_rank, int nsq, char* dataset){
	ivf_t *ivf;
	struct timeval start, end;
	double time;
	int lim;

	printf("\nIndexing\n");

	gettimeofday(&start, NULL);

	ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);
	for(int i=0; i<ivfpq.coarsek; i++){
		ivf[i].ids = (int*)malloc(sizeof(int));
		ivf[i].idstam = 0;
		ivf[i].codes.mat = (int*)malloc(sizeof(int));
		ivf[i].codes.n = 0;
		ivf[i].codes.d = nsq;
	}
	lim = tam/1000000;
	if(tam%1000000!=0){
		lim = (tam/1000000) + 1;
	}

	//Cria a lista invertida correspondente ao trecho da base assinalado a esse processo
	#pragma omp parallel for num_threads(threads) schedule(dynamic)
		for(int i=0; i<lim; i++){

			ivf_t *ivf2;
			int aux;
				mat vbase;
			ivf2 = (ivf_t *)malloc(sizeof(ivf_t)*ivfpq.coarsek);

			vbase = pq_test_load_base(dataset, i, my_rank-last_assign);

			ivfpq_assign(ivfpq, vbase, ivf2);

			for(int j=0; j<ivfpq.coarsek; j++){
				for(int l=0; l<ivf2[j].idstam; l++){
					ivf2[j].ids[l]+=1000000*i+tam*(my_rank-last_assign-1);
				}

				aux = ivf[j].idstam;
				#pragma omp critical
				{
					ivf[j].idstam += ivf2[j].idstam;
					ivf[j].ids = (int*)realloc(ivf[j].ids,sizeof(int)*ivf[j].idstam);
					memcpy (ivf[j].ids+aux, ivf2[j].ids, sizeof(int)*ivf2[j].idstam);
					ivf[j].codes.n += ivf2[j].codes.n;
					ivf[j].codes.mat = (int*)realloc(ivf[j].codes.mat,sizeof(int)*ivf[j].codes.n*ivf[j].codes.d);
					memcpy (ivf[j].codes.mat+aux*ivf[i].codes.d, ivf2[j].codes.mat, sizeof(int)*ivf2[j].codes.n*ivf2[j].codes.d);
				}
				free(ivf2[j].ids);
				free(ivf2[j].codes.mat);
			}
			free(vbase.mat);
			free(ivf2);
		}

	gettimeofday(&end, NULL);
	time = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec))/1000;

	printf ("\nTempo de criacao da lista invertida: %g\n",time);

	return ivf;
}

void write_ivf(ivfpq_t ivfpq, int threads, long long tam, int my_rank, int nsq, char* dataset){
	FILE *fp;
	char name_arq[100];
	struct timeval start, end;
	double time;
	int lim, i;

	printf("\nIndexing\n");

	gettimeofday(&start, NULL);

	lim = tam/1000000;
	if(tam%1000000!=0){
		lim = (tam/1000000) + 1;
	}

	//Cria a lista invertida correspondente ao trecho da base assinalado a esse processo
	#pragma omp parallel for num_threads(threads) schedule(dynamic)
		for(i=0; i<lim; i++){

			ivf_t *ivf;
			int aux;
			mat vbase;
			ivf = (ivf_t *)malloc(sizeof(ivf_t)*ivfpq.coarsek);

			vbase = pq_test_load_base(dataset, i, my_rank-last_assign);

			ivfpq_assign(ivfpq, vbase, ivf);

			free(vbase.mat);

			for(int j=0; j<ivfpq.coarsek; j++){
				for(int l=0; l<ivf[j].idstam; l++){
					ivf[j].ids[l]+=1000000*i+tam*(my_rank-last_assign-1);
				}

				aux = ivf[j].idstam;
				#pragma omp critical
				{
					sprintf(name_arq, "/pylon5/ac3uump/freire/ivf/ivf_%d_%d_%d.bin", ivfpq.coarsek, tam, j);
					fp = fopen(name_arq,"ab");
					fwrite(&ivfpq.coarsek, sizeof(int), 1, fp);
					fwrite(&ivf[j].idstam, sizeof(int), 1, fp);
					fwrite(&ivf[j].ids[0], sizeof(int), ivf[j].idstam, fp);
					fwrite(&ivf[j].codes.n, sizeof(int), 1, fp);
					fwrite(&ivf[j].codes.d, sizeof(int), 1, fp);
					fwrite(&ivf[j].codes.mat[0], sizeof(int), ivf[j].codes.n*ivf[j].codes.d, fp);
					fclose(fp);
				}
				free(ivf[j].ids);
				free(ivf[j].codes.mat);
			}
			free(ivf);
		}

	gettimeofday(&end, NULL);
	time = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec))/1000;

	printf ("\nTempo de criacao da lista invertida: %g\n",time);
}

ivf_t* read_ivf(ivfpq_t ivfpq, long long tam, int my_rank){
	FILE *fp;

	ivf_t* ivf;
	char name_arq[150];
	int coarsek;

	ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);

	for(int i=0; i<ivfpq.coarsek; i++){
		int idstam, codesn, codesd;

		ivf[i].ids = (int*)malloc(sizeof(int));
		ivf[i].idstam = 0;
		ivf[i].codes.mat = (int*)malloc(sizeof(int));
		ivf[i].codes.n = 0;
		ivf[i].codes.d = ivfpq.pq.nsq;

		sprintf(name_arq, "/pylon5/ac3uump/freire/ivf/ivf_%d_%lld_%d.bin", ivfpq.coarsek, tam, i);
		fp = fopen(name_arq,"rb");

		for(int j=0; j<tam/1000000; j++){
			fread(&coarsek, sizeof(int), 1, fp);
			fread(&idstam, sizeof(int), 1, fp);
			ivf[i].idstam += idstam;
			ivf[i].ids = (int*)realloc(ivf[i].ids,sizeof(int)*ivf[i].idstam);
			fread(&ivf[i].ids[ivf[i].idstam-idstam], sizeof(int), idstam, fp);
			fread(&codesn, sizeof(int), 1, fp);
			ivf[i].codes.n += codesn;
			fread(&codesd, sizeof(int), 1, fp);
			ivf[i].codes.d = codesd;
			ivf[i].codes.mat = (int*)realloc(ivf[i].codes.mat,sizeof(int)*ivf[i].codes.n*ivf[i].codes.d);
			fread(&ivf[i].codes.mat[((ivf[i].codes.n)*(ivf[i].codes.d))-codesn*codesd], sizeof(int), codesn*codesd, fp);
		}
		fclose(fp);
	}

	return ivf;
}

int min(int a, int b){
	if(a>b){
		return b;
	}
	else
		return a;
}

float * sumidxtab2(mat D, matI ids, int offset, int inner_threads){
	//aloca o vetor a ser retornado

	float *dis = (float*)malloc(sizeof(float)*ids.n);

	//soma as distancias para cada vetor
	#pragma omp parallel for schedule(dynamic) num_threads(inner_threads)
	for (int i = 0; i < ids.n ; i++) {
		
		float dis_tmp = 0;
		for(int j=0; j<D.n; j++){
			dis_tmp += D.mat[ids.mat[i*ids.d+j]+ offset + j*D.d];
		}
		dis[i]=dis_tmp;
	}

	return dis;
}

int* imat_new_transp (const int *a, int ncol, int nrow){
	int i,j;
	int *vt=(int*)malloc(sizeof(int)*ncol*nrow);

	for(i=0;i<ncol;i++)
		for(j=0;j<nrow;j++)
			vt[i*nrow+j]=a[j*ncol+i];

	return vt;
}

void copySubVectorsI(int* qcoaidx, int* coaidx, int query, int nq, int w){
	for (int i = 0; i < w; i++) {
		qcoaidx[i] = coaidx[query*w + i];
	}
}

void ivfpq_assign(ivfpq_t ivfpq, mat vbase, ivf_t *ivf){
	int k = 1;


	int *assign = (int*)malloc(sizeof(int)*vbase.n);
	float *dis = (float*)malloc(sizeof(float)*vbase.n);

	//acha os indices para o coarse quantizer
	knn_full(2, vbase.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, k,
					 ivfpq.coa_centroids, vbase.mat, NULL, assign, dis);

	//residuos
	subtract(vbase, ivfpq.coa_centroids, assign, ivfpq.coa_centroidsd);

	matI codebook = pq_assign(ivfpq.pq, vbase);
	int *codeaux = (int*) malloc(sizeof(int)*codebook.d*codebook.n);
	memcpy(codeaux, codebook.mat, sizeof(int)*codebook.n*codebook.d);
	int * hist = (int*) calloc(ivfpq.coarsek,sizeof(int));

	histogram(assign, vbase.n ,ivfpq.coarsek, hist);

	// Gera codebook com ids ordenados
	int* ids = (int*)malloc(sizeof(int)*vbase.n);
	ivec_sort_index(assign, vbase.n, ids);

	for(int i=0; i<codebook.n; i++){
		memcpy(codebook.mat+i*codebook.d, codeaux+codebook.d*ids[i], sizeof(int)*codebook.d);
	}
	int pos = 0;

	for (int i = 0; i < ivfpq.coarsek; i++) {
		int nextpos;

		ivf[i].ids = (int*)malloc(sizeof(int)*hist[i]);

		ivf[i].idstam = hist[i];

		ivf[i].codes.mat = (int*)malloc(sizeof(int)*hist[i]*ivfpq.pq.nsq);

		ivf[i].codes.n = hist[i];

		ivf[i].codes.d = ivfpq.pq.nsq;

		nextpos = pos+hist[i];

		memcpy(ivf[i].ids, &ids[pos], sizeof(int)*hist[i]);

		for (int p = pos; p < nextpos; p++) {
			copySubVectorsI(ivf[i].codes.mat + (p-pos)*ivf[i].codes.d, codebook.mat, p, 0,  ivf[i].codes.d);
		}

		pos += hist[i];
	}
	free(codebook.mat);
	free(ids);
	free(hist);
	free(assign);
	free(dis);
}

void histogram(const int* vec, int n, int range, int *hist){

	for(int j=0; j<n; j++){
		hist[vec[j]]++;
	}
}
