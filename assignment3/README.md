Compile a `.cu` file, run
```sh
$ nvcc –arch=sm_32 <filename> –o <output> –lcuda -lcudart
```

For example, to run `mm-cuda.cu` to create runable file `mm-cuda`, we run
```sh
$ nvcc –arch=sm_32 mm-cuda.cu –o mm-cuda –lcuda -lcudart
```

Compile the `mm-seq.c`, run
```sh
$ gcc -O3 mm-seq.c -o mm-seq -lrt
```