.SUFFIXES: .cpp .c .o

include makefile.inc

CC=gcc -g
CXX=g++  -g
MF =
IVF =
MPICXX = mpicxx $(MF) $(IVF) -fopenmp  -g
PQ_UTILS_DIR = ./pq-utils
OBJDIR=./obj
IVF_DIR = ./ivf_pq
OMP = -fopenmp
PTHREAD = -lpthread

all: pq_test_load_vectors.o pq_new.o pq_assign.o pq_search.o pq_test.o pq_test $(OBJDIR)/myIVF.o $(OBJDIR)/ivf_training.o $(OBJDIR)/ivf_assign.o $(OBJDIR)/ivf_search.o $(OBJDIR)/k_min.o ivfpq_test.o ivfpq_test


ifeq "$(USEARPACK)" "yes"
  EXTRAYAELLDFLAG=$(ARPACKLDFLAGS)
  EXTRAMATRIXFLAG=-DHAVE_ARPACK
endif

ifeq "$(USEOPENMP)" "no"
  EXTRAMATRIXFLAG+=-fopenmp
  EXTRAYAELLDFLAG+=-fopenmp
endif

# Various

# .c.o:
# 	$(cc) $(CFLAGS) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)
#
# .cpp.o:
# 	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_assign.o:
$(OBJDIR)/pq_assign.o: $(PQ_UTILS_DIR)/pq_assign.cpp $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(PQ_UTILS_DIR)/pq_new.h yael//nn.h yael//vector.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_new.o:
$(OBJDIR)/pq_new.o: $(PQ_UTILS_DIR)/pq_new.cpp $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael//kmeans.h yael//vector.h yael//matrix.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test_load_vectors.o:
$(OBJDIR)/pq_test_load_vectors.o: $(PQ_UTILS_DIR)/pq_test_load_vectors.cpp $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael//matrix.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test_compute_stats.o:
$(OBJDIR)/pq_test_compute_stats.o: $(PQ_UTILS_DIR)/pq_test_compute_stats.cpp $(PQ_UTILS_DIR)/pq_test_compute_stats.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_search.o:
$(OBJDIR)/pq_search.o: $(PQ_UTILS_DIR)/pq_search.cpp $(PQ_UTILS_DIR)/pq_search.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(PQ_UTILS_DIR)/pq_new.h yael//matrix.h yael//kmeans.h yael//vector.h yael//nn.h
				$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test.o:

$(OBJDIR)/pq_test.o: pq_test.cpp yael/vector.h yael/kmeans.h $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h
	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test: $(OBJDIR)/pq_test.o $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h  $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h
	$(CXX) $(cflags) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS)


# IVF_TEST

$(OBJDIR)/myIVF.o:  $(IVF_DIR)/myIVF.cpp $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/kmeans.h yael/vector.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/ivf_training.o: $(IVF_DIR)/ivf_training.cpp $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h  $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/kmeans.h yael/vector.h yael/nn.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/ivf_assign.o: $(IVF_DIR)/ivf_assign.cpp $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/myIVF.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h  yael/nn.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/ivf_search.o: $(IVF_DIR)/ivf_search.cpp $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/kmeans.h yael/vector.h yael/nn.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/ivf_aggregator.o: $(IVF_DIR)/ivf_aggregator.cpp $(IVF_DIR)/ivf_search.h $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h yael/kmeans.h yael/vector.h yael/nn.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/k_min.o: $(IVF_DIR)/k_min.cpp $(IVF_DIR)/k_min.h $(IVF_DIR)/ivf_aggregator.cpp $(IVF_DIR)/ivf_search.h $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h yael/kmeans.h yael/vector.h yael/nn.h
	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

ivfpq_test.o: ivf_test.cpp $(OBJDIR)/ivf_aggregator.o $(OBJDIR)/ivf_assign.o $(OBJDIR)/myIVF.o $(OBJDIR)/ivf_training.o $(OBJDIR)/ivf_search.o $(IVF_DIR)/ivf_search.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(IVF_DIR)/ivf_aggregator.h $(IVF_DIR)/ivf_assign.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

ivfpq_test: ivfpq_test.o $(OBJDIR)/ivf_aggregator.o $(OBJDIR)/ivf_assign.o $(OBJDIR)/myIVF.o $(OBJDIR)/ivf_training.o $(OBJDIR)/ivf_search.o $(OBJDIR)/k_min.o $(IVF_DIR)/k_min.h $(IVF_DIR)/ivf_search.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/ivf_aggregator.h  $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h
		$(MPICXX) $(cflags) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS) $(PTHREAD)

clean:
	rm -f $(OBJDIR)/*.o pq_test *.o ivfpq_test
