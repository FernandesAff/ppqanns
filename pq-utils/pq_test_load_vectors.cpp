#include "pq_test_load_vectors.h"

mat pq_test_load_train(char* dataset, int tam){
	mat vtrain;
	char *ftrain;

	ftrain = (char*) malloc(sizeof(char)*100);

	if(strcmp(dataset, "siftsmall")==0){
		strcpy (ftrain, BASE_DIR);
		strcat (ftrain,"siftsmall_learn.fvecs");
		vtrain.n=25000;
		vtrain.d=128;
		vtrain.mat= fmat_new (vtrain.d, vtrain.n);
	}
	else if(strcmp(dataset, "sift")==0){
		strcpy (ftrain, BASE_DIR);
		strcat (ftrain,"sift_learn.fvecs");
		vtrain.n=100000;
		vtrain.d=128;
		vtrain.mat= fmat_new (vtrain.d, vtrain.n);
	}
	else if(strcmp(dataset, "siftbig")==0 ){
		strcpy (ftrain, BASE_DIR);
		strcat (ftrain,"siftbig_learn.bvecs");
		vtrain.n=tam/10;
		vtrain.d=128;
		vtrain.mat= fmat_new (vtrain.d, vtrain.n);
	}
	else if(strcmp(dataset, "gist")==0){
		strcpy (ftrain, BASE_DIR);
		strcat (ftrain,"gist_learn.fvecs");
		vtrain.n=500000;
		vtrain.d=960;
		vtrain.mat= fmat_new (vtrain.d, vtrain.n);
	}

	if(strcmp(dataset, "siftbig")!=0){
		fvecs_read (ftrain, vtrain.d, vtrain.n, vtrain.mat);
	}
	else{
		b2fvecs_read (ftrain, vtrain.d, vtrain.n, vtrain.mat);
	}
	free(ftrain);
	return vtrain;
}

matI pq_test_load_gdn(char* dataset, int tam){
	matI vids_gnd;
	int *ids_gnd;
	char *fgroundtruth;

	fgroundtruth= (char*) malloc(sizeof(char)*100);

	if(strcmp(dataset, "siftsmall")==0){
		strcpy (fgroundtruth, BASE_DIR);
		strcat (fgroundtruth,"siftsmall_gnd.ivecs");
		vids_gnd.n=100;
		vids_gnd.d=100;
		ids_gnd= ivec_new (vids_gnd.n*vids_gnd.d);
	}
	else if(strcmp(dataset, "sift")==0){
		strcpy (fgroundtruth, BASE_DIR);
		strcat (fgroundtruth,"sift_gnd.ivecs");
		vids_gnd.n=10000;
		vids_gnd.d=100;
		ids_gnd= ivec_new (vids_gnd.n*vids_gnd.d);
	}
	else if(strcmp(dataset, "siftbig")==0 ){
		if(tam==1000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_1M.ivecs");
		}
		else if(tam==2000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_2M.ivecs");
		}
		else if(tam==5000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_5M.ivecs");
		}
		else if(tam==10000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_10M.ivecs");
		}
		else if(tam==20000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_20M.ivecs");
		}
		else if(tam==50000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_50M.ivecs");
		}
		else if(tam==100000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_100M.ivecs");
		}
		else if(tam==200000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_200M.ivecs");
		}
		else if(tam==500000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_500M.ivecs");
		}
		else if(tam==1000000000){
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_1000M.ivecs");
		}
		else{
			strcpy (fgroundtruth, BASE_DIR);
			strcat (fgroundtruth,"gnd/idx_1M.ivecs");
		}
		vids_gnd.n=10000;
		vids_gnd.d=1000;
		ids_gnd= ivec_new (vids_gnd.n*vids_gnd.d);
	}
	else if(strcmp(dataset, "gist")==0){
		strcpy (fgroundtruth, BASE_DIR);
		strcat (fgroundtruth,"gist_gnd.ivecs");
		vids_gnd.n=1000;
		vids_gnd.d=100;
		ids_gnd= ivec_new (vids_gnd.n*vids_gnd.d);
	}

	ivecs_read (fgroundtruth, vids_gnd.d , vids_gnd.n, ids_gnd);

	vids_gnd.mat= ivec_new (vids_gnd.n);
	for(int i=0; i<vids_gnd.n;i++){
		vids_gnd.mat[i]=ids_gnd[i*vids_gnd.d];
	}
	vids_gnd.d=1;
	free(ids_gnd);
	free(fgroundtruth);
	return vids_gnd;
}

mat pq_test_load_query(char* dataset, int threads){
	mat vquery;
	char srank[4];
	char *fquery;
	fquery = (char*) malloc(sizeof(char)*100);

	if(strcmp(dataset, "siftsmall")==0){
		strcpy (fquery, BASE_DIR);
		strcat (fquery,"siftsmall_query.fvecs");
		vquery.n=1;
		vquery.d=128;
		vquery.mat= fmat_new (vquery.d, vquery.n);
	}
	else if(strcmp(dataset, "sift")==0){
		strcpy (fquery, BASE_DIR);
		strcat (fquery,"sift_query.fvecs");
		vquery.n=10000;
		vquery.d=128;
		vquery.mat= fmat_new (vquery.d, vquery.n);
	}
	else if(strcmp(dataset, "siftbig")==0){
		strcpy (fquery, BASE_DIR);
		strcat (fquery,"siftbig_1_query.bvecs");
		vquery.n=1;
		vquery.d=128;
		vquery.mat= fmat_new (vquery.d, vquery.n);
	}
	else if(strcmp(dataset, "gist")==0){
		strcpy (fquery, BASE_DIR);
		strcat (fquery,"gist_query.fvecs");
		vquery.n=1000;
		vquery.d=960;
		vquery.mat= fmat_new (vquery.d, vquery.n);
	}
	if(strcmp(dataset, "siftbig")!=0){
		fvecs_read (fquery, vquery.d, vquery.n, vquery.mat);
	}
	else{
		b2fvecs_read (fquery, vquery.d, vquery.n, vquery.mat);
	}
	free(fquery);
	return vquery;
}

mat pq_test_load_base(char* dataset, int offset, int my_rank){
	mat vbase;
	char srank[4];
	sprintf (srank, "%d",my_rank);
	char *fbase;
	fbase= (char*) malloc(sizeof(char)*100);

	if(strcmp(dataset, "siftsmall")==0){
		strcpy (fbase, BASE_DIR);
		strcat (fbase,"siftsmall_base.fvecs");
		vbase.n=10000;
		vbase.d=128;
		vbase.mat= fmat_new (vbase.d, vbase.n);
	}
	else if(strcmp(dataset, "sift")==0){
		strcpy (fbase, BASE_DIR);
		strcat (fbase,"sift_base.fvecs");
		vbase.n=1000000;
		vbase.d=128;
		vbase.mat= fmat_new (vbase.d, vbase.n);
	}
	else if(strcmp(dataset, "siftbig")==0 ){
		strcpy (fbase, BASE_DIR);
		strcat (fbase,"siftbig_base.bvecs");
		vbase.n=1000000;
		vbase.d=128;
		vbase.mat= (float*) malloc(sizeof(float)*vbase.d*vbase.n);
	}
	else if(strcmp(dataset, "gist")==0){
		strcpy (fbase, BASE_DIR);
		strcat (fbase,"gist_base.fvecs");
		vbase.n=100000;
		vbase.d=960;
		vbase.mat= fmat_new (vbase.d, vbase.n);
	}
	strcat(fbase, srank);
	if(strcmp(dataset, "siftbig")!=0){
		fvecs_read (fbase, vbase.d, vbase.n, vbase.mat);
	}
	else{
		my_bvecs_read (offset, fbase, vbase.d, vbase.n, vbase.mat);
	}
	free(fbase);
	return vbase;
}

void load_random (float *v, int n, int d){
	int i;

	for(i=0;i<n*d;i++){
		v[i]= (float) rand()/RAND_MAX;
	}
}

int ivecs_read (const char *fname, int d, int n, int *a){
	FILE *f = fopen (fname, "r");

	if (!f) {
		fprintf (stderr, "ivecs_read: could not open %s\n", fname);
		perror ("");
		return -1;
	}

	long i;
	for (i = 0; i < n; i++) {
		int new_d;

		if (fread (&new_d, sizeof (int), 1, f) != 1) {
			if (feof (f))break;
			else {
				perror ("ivecs_read error 1");
				fclose(f);
				return -1;
			}
		}

		if (new_d != d) {
			fprintf (stderr, "ivecs_read error 2: unexpected vector dimension\n");
			fclose(f);
			return -1;
		}

		if (fread (a + d * (long) i, sizeof (int), d, f) != d) {
			fprintf (stderr, "ivecs_read error 3\n");
			fclose(f);
			return -1;
		}
	}
	fclose (f);

	return i;
}

int my_bvecs_read (int offset, const char *fname, int d, int n, float *a){
	FILE *f = fopen (fname, "r");

	if (!f) {
		fprintf (stderr, "my_bvecs_read: could not open %s\n", fname);
		perror ("");
		return -1;
	}

	unsigned long long b;
	b=(unsigned long long) offset*n*(d+4);

	fseek (f, b, SEEK_SET);
	long i;

	for (i = 0; i < n; i++) {
		int new_d;

		if (fread (&new_d, sizeof (int), 1, f) != 1) {
			if (feof (f))break;
			else {
				perror ("my_bvecs_read error 1");
				fclose(f);
				return -1;
			}
		}

		if (new_d != d) {
			printf("d%dnd%d",d,new_d);
			fprintf (stderr, "my_bvecs_read error 2: unexpected vector dimension\n");
			fclose(f);
			return -1;
		}

		unsigned char * vb = (unsigned char *) malloc (sizeof (*vb) * d);

		if (fread (vb, sizeof (*vb), d, f) != d) {
			fprintf (stderr, "my_bvecs_read error 3\n");
			fclose(f);
			return -1;
		}

		for (int j = 0 ; j < d ; j++){
    		a[i*d+j] = vb[j];
		}
  		free (vb);
	}
	fclose (f);

	return i;
}
