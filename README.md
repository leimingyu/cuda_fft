# cuda_fft

### Parameters
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

### 1D 
```
cufftPlan1d
```


### benchmarking
* http://www.cv.nrao.edu/~pdemores/gpu/
