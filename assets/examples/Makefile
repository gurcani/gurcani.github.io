CFLAGS=-O3 -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
        -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native

libmult.so: mult.o
	gcc $(CFLAGS) -shared -o libmult.so mult.o

mult.o: mult.c
	gcc $(CFLAGS) -fpic -c mult.c
