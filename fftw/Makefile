CC = g++
INCLUDE = -I../
OPT = -O2 -Wall
LIBS  = -lm -fopenmp

# fp64
#LIBS += -lfftw3

# fp 32
LIBS += -lfftw3f


OBJS = fftw_1d fftw_2d fftw_3d

all: $(OBJS) 

fftw_1d: fftw_1d.cpp
	$(CC) $(INCLUDE) $(OPT) $^ -o $@ $(LIBS)  

fftw_2d: fftw_2d.cpp
	$(CC) $(INCLUDE) $(OPT) $^ -o $@ $(LIBS)  

fftw_3d: fftw_3d.cpp
	$(CC) $(INCLUDE) $(OPT) $^ -o $@ $(LIBS)  

.PHONY: clean
clean:
	rm -rf $(OBJS) 
