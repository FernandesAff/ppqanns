.SUFFIXES: .c .o 

include ../makefile.inc

all: libyael.a libyael.$(SHAREDEXT) 

CC=gcc

ifeq "$(USEARPACK)" "yes"
  EXTRAYAELLDFLAG=$(ARPACKLDFLAGS)
  EXTRAMATRIXFLAG=-DHAVE_ARPACK
  EXTRAEIGSFLAG=-DHAVE_ARPACK
endif

ifeq "$(USEOPENMP)" "yes"
  OPENMPFLAG = -fopenmp 
  EXTRAMATRIXFLAG+=-fopenmp
  EXTRAYAELLDFLAG+=-fopenmp
endif


#############################
# Various  

LIBOBJ=matrix.o vector.o nn.o kmeans.o eigs.o sorting.o binheap.o machinedeps.o 

libyael.a: $(LIBOBJ)
	ar r libyael.a $^

libyael.$(SHAREDEXT): $(LIBOBJ)
	$(CC) $(LDFLAGS) $(YAELSHAREDFLAGS) -o libyael.$(SHAREDEXT) $^ $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@ $(FLAGS) $(EXTRACFLAGS)

nn.o: EXTRACFLAGS=$(LAPACKCFLAGS) $(THREADCFLAGS) $(OPENMPFLAG)

matrix.o: EXTRACFLAGS=$(LAPACKCFLAGS) $(THREADCFLAGS) $(EXTRAMATRIXFLAG) $(OPENMPFLAG)

kmeans.o: EXTRACFLAGS=$(THREADCFLAGS)

eigs.o: EXTRACFLAGS=$(LAPACKCFLAGS) $(LAPACKCFLAGS) $(EXTRAEIGSFLAG) $(OPENMPFLAG)

machinedeps.o:  EXTRACFLAGS=$(OPENMPFLAG)

#############################
# Dependencies  

# for i in *.c ; do cpp -I.. -MM $i; done

binheap.o: binheap.c binheap.h sorting.h
eigs.o: eigs.c eigs.h vector.h sorting.h machinedeps.h 
kmeans.o: kmeans.c vector.h kmeans.h nn.h sorting.h
machinedeps.o: machinedeps.c machinedeps.h
matrix.o: matrix.c matrix.h vector.h sorting.h machinedeps.h eigs.h
nn.o: nn.c machinedeps.h vector.h nn.h binheap.h sorting.h
sorting.o: sorting.c sorting.h machinedeps.h binheap.h vector.h
vector.o: vector.c vector.h

clean:
	rm -f libyael.a libyael.$(SHAREDEXT)* *.o 
