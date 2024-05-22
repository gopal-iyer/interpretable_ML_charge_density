override CC=mpicc


DEBUG_MODE    = 0

LDLIBS   = -lrt

LIBBASE = ./main

CFLAGS = -std=gnu99 -g -O3 -fopenmp -Iinclude/

CPPFLAGS += -m64 -I${MKLROOT}/include -DUSE_MKL
LDFLAGS   = -L${MKLROOT}/lib/intel64
LDLIBS   += -Wl,-rpath=${MKLROOT}/lib/intel64,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl


OBJSC = main.o initialization.o readfiles.o fitting.o common.o

ifeq ($(DEBUG_MODE), 1)
CPPFLAGS += -Wall -g -DDEBUG
endif

all: main

main: $(OBJSC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o  $(LIBBASE) $^ $(LDLIBS)

.PHONY: clean

clean:
	rm -f  $(OBJSC) $(LIBBASE)