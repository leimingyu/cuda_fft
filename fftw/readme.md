# FFTW (v3.3.4)

### Doc
* [html](http://www.fftw.org/fftw3_doc/index.html#SEC_Contents)
* [pdf](http://www.fftw.org/fftw3.pdf)

### Installation
```
sudo apt-get install libfftw3-dev
```

### Program Usage
```
leiming@homedesktop:~/git/cuda_fft/fftw$ ./fftw -l 10240 -r 1 -t 10
[LOG] Signal Length: 10240
[LOG] FFT run: 1
[LOG] Trials: 10
[LOG] Run FFTW ...
0.000063 sec
0.000023 sec
0.000023 sec
0.000023 sec
0.000023 sec
0.000023 sec
0.000023 sec
0.000023 sec
0.000023 sec
0.000023 sec
[LOG] Finished!
[LOG] Average: 0.000027 sec
```

### Notes
* multi-threaded FFTW ([doc](http://www.fftw.org/fftw3_doc/Usage-of-Multi_002dthreaded-FFTW.html))
* dht can be more efficient than dft for real data
* Saving (optimized) plans : FFTW implements a method for saving plans to disk and restoring them. 
([Page 18](http://www.fftw.org/fftw3.pdf))
* FFTW_MEASURE tells FFTW to find ** an optimized plan ** by actually computing several FFTs and measuring their execution time. Depending on your machine, this can take some time (often a few seconds). FFTW_MEASURE is the default planning option. ([P26](http://www.fftw.org/fftw3.pdf))

### Examples
* http://people.sc.fsu.edu/~jburkardt/c_src/fftw3/fftw3_prb.c
* https://www.hoffman2.idre.ucla.edu/fftw/
* http://toto-share.com/2012/07/cc-fftw-tutorial/
* http://fftwpp.sourceforge.net/
* http://people.sc.fsu.edu/~jburkardt/c_src/fftw3/fftw3.html
