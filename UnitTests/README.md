# Unit Tests for CombBLAS Components


# Unit Tests

## Test OpenMP availability
```bash
yhong4@gh095:/work/nvme/bdyd/yhong4/workspace/kk/tfCombBLAS-minor/build> OMP_NUM_THREADS=16 ./UnitTests/test_openmp
Hello from thread 0 out of 16 threads.
Hello from thread 1 out of 16 threads.
Hello from thread 2 out of 16 threads.
Hello from thread 3 out of 16 threads.
Hello from thread 4 out of 16 threads.
Hello from thread 5 out of 16 threads.
Hello from thread 6 out of 16 threads.
Hello from thread 7 out of 16 threads.
Hello from thread 8 out of 16 threads.
Hello from thread 9 out of 16 threads.
Hello from thread 10 out of 16 threads.
Hello from thread 11 out of 16 threads.
Hello from thread 12 out of 16 threads.
Hello from thread 13 out of 16 threads.
Hello from thread 14 out of 16 threads.
Hello from thread 15 out of 16 threads.
```

## Test CUDA-Aware MPI availability
```bash
yhong4@gh095:/work/nvme/bdyd/yhong4/workspace/kk/tfCombBLAS-minor/build> srun -n 2 ./UnitTests/test_cudampi
Rank 0: MPI_Send with GPU memory succeeded. Your MPI library is CUDA-awared.
Rank 1: MPI_Recv succeeded. Received data: 0 1 2 3 4 5 6 7 8 9
```


## CUDA Memory Manager Unit Tests
```bash
yhong4@gh095:/work/nvme/bdyd/yhong4/workspace/kk/tfCombBLAS-minor/build> ./UnitTests/test_cmm
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from CudaMemoryManagerTest
[ RUN      ] CudaMemoryManagerTest.BasicMallocFree
[       OK ] CudaMemoryManagerTest.BasicMallocFree (3540 ms)
[ RUN      ] CudaMemoryManagerTest.MultiStreamAllocation
[       OK ] CudaMemoryManagerTest.MultiStreamAllocation (1 ms)
[ RUN      ] CudaMemoryManagerTest.InvalidFreeThrows
[       OK ] CudaMemoryManagerTest.InvalidFreeThrows (1 ms)
[ RUN      ] CudaMemoryManagerTest.RepeatedAllocFree
[       OK ] CudaMemoryManagerTest.RepeatedAllocFree (1 ms)
[ RUN      ] CudaMemoryManagerTest.ZeroSizeAllocation
[       OK ] CudaMemoryManagerTest.ZeroSizeAllocation (1 ms)
[----------] 5 tests from CudaMemoryManagerTest (3548 ms total)

[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (3548 ms total)
[  PASSED  ] 5 tests.
```

# Benchmarks 

## CUDA Memory Manager Benchmark 
Benchmark of CUDA memory pool on a single GH200.
```bash
yhong4@gh095:/work/nvme/bdyd/yhong4/workspace/kk/tfCombBLAS-minor/build> ./UnitTests/bm-cmm
2025-02-09T22:53:02-06:00
Running ./UnitTests/bm-cmm
Run on (288 X 3708 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB (x288)
  L1 Instruction 64 KiB (x288)
  L2 Unified 1024 KiB (x288)
  L3 Unified 116736 KiB (x4)
Load Average: 0.32, 0.17, 0.25
-----------------------------------------------------------------------------------------
Benchmark                               Time             CPU   Iterations UserCounters...
-----------------------------------------------------------------------------------------
BM_CudaMallocFree/1024         3601394269 ns   3573833024 ns            1 bytes_per_second=286.527/s
BM_CudaMallocFree/10240             91374 ns        91370 ns         7525 bytes_per_second=106.88M/s
BM_CudaMallocFree/102400            91284 ns        91286 ns         7664 bytes_per_second=1069.78M/s
BM_CudaMallocFree/1048576           90912 ns        90915 ns         7699 bytes_per_second=10.7415G/s
BM_CudaMallocFree/10485760         107374 ns       107370 ns         6567 bytes_per_second=90.9533G/s
BM_CudaMallocFree/104857600        305035 ns       305028 ns         2281 bytes_per_second=320.155G/s
BM_CudaMemoryManager/1024             386 ns          386 ns      1818558 bytes_per_second=2.47265G/s
BM_CudaMemoryManager/10240            386 ns          386 ns      1809467 bytes_per_second=24.6908G/s
BM_CudaMemoryManager/102400           386 ns          386 ns      1812572 bytes_per_second=247.082G/s
BM_CudaMemoryManager/1048576          386 ns          386 ns      1810040 bytes_per_second=2.46806T/s
BM_CudaMemoryManager/10485760         385 ns          385 ns      1812565 bytes_per_second=24.7628T/s
BM_CudaMemoryManager/104857600        385 ns          385 ns      1815557 bytes_per_second=247.765T/s
```
