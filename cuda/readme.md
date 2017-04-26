# cuFFT

### Doc
* http://developer.download.nvidia.com/compute/cuda/1.0/CUFFT_Library_1.0.pdf

### tutorials
* https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf

### Interfaces
* cufftType
```
    typedef enum cufftType_t { 
    CUFFT_R2C = 0x2a, // Real to complex (interleaved) 
    CUFFT_C2R = 0x2c, // Complex (interleaved) to real 
    CUFFT_C2C = 0x29, // Complex to complex (interleaved) 
    CUFFT_D2Z = 0x6a, // Double to double-complex (interleaved) 
    CUFFT_Z2D = 0x6c, // Double-complex (interleaved) to double 
    CUFFT_Z2Z = 0x69 // Double-complex to double-complex (interleaved) 
    } cufftType;
```

* Create a simple plan for a 1D/2D/3D transform respectively. 
[link](http://docs.nvidia.com/cuda/cufft/index.html#function-cufftplan1d)
```
cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch);
cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);
cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type);
```

* Run cuda fft
```
cufftExecC2C() / cufftExecZ2Z() - complex-to-complex transforms for single/double precision.
cufftExecR2C() / cufftExecD2Z() - real-to-complex forward transform for single/double precision.
cufftExecC2R() / cufftExecZ2D() - complex-to-real inverse transform for single/double precision.
```
