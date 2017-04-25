CC = g++
INCLUDE = -I../
OPT = -O2 -Wall #-std=c99 
LIBS  = -lm -fopenmp

# fp64
#LIBS += -lfftw3

# fp 32
LIBS += -lfftw3f

all: fftw

fftw: fftw.cpp
	$(CC) $(INCLUDE) $(OPT) $^ -o $@ $(LIBS)  

.PHONY: clean
clean:
	rm fftw
