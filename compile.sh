# compile files
gcc convolution_seq.c util.c -o convolution_seq
gcc -fopenmp convolution_omp.c util.c -o convolution_omp