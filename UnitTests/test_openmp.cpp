#include <iostream>
#include <omp.h>

int main() {
    int nthreads = 0;

    // Create a parallel region.
    #pragma omp parallel
    {
        // Save the number of threads from one thread.
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
        }

        // Each thread obtains its unique ID.
        int tid = omp_get_thread_num();

        // Ensure all threads know the number of threads.
        #pragma omp barrier

        // Loop from 0 to the total number of threads.
        for (int i = 0; i < nthreads; i++) {
            // Only the thread with ID equal to 'i' prints its message.
            if (tid == i) {
                std::cout << "Hello from thread " << tid 
                          << " out of " << nthreads << " threads." 
                          << std::endl;
            }
            // Synchronize all threads before moving to the next iteration.
            #pragma omp barrier
        }
    }

    return 0;
}