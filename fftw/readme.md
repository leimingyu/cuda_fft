# FFTW

### Doc
[official manual](http://www.fftw.org/fftw3.pdf)

### Program Usage
```
leiming@homedesktop:~/git/cuda_fft/fftw$ ./fftw 
[LOG] Signal Length: 4096
[LOG] FFT run: 10000
[LOG] Trials: 10

leiming@homedesktop:~/git/cuda_fft/fftw$ ./fftw -l 1024 -r 1024 -t 20
[LOG] Signal Length: 1024
[LOG] FFT run: 1024
[LOG] Trials: 20
```

### Notes
* multi-threaded FFTW ([doc](http://www.fftw.org/fftw3_doc/Usage-of-Multi_002dthreaded-FFTW.html))
* dht can be more efficient than dft for real data
* Saving (optimized) plans : FFTW implements a method for saving plans to disk and restoring them. 
([Page 18](http://www.fftw.org/fftw3.pdf))


### Examples
* http://people.sc.fsu.edu/~jburkardt/c_src/fftw3/fftw3_prb.c
* https://www.hoffman2.idre.ucla.edu/fftw/
