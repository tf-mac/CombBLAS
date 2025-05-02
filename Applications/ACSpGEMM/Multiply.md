ACSpGEMM Implementation

# Paper

## Four major goals 

Our adaptive chunk-based GPU SpGEMM approach (ACSpGEMM) focuses on four major goals 
1. performing computations in local on-chip memory 
2. coherent memory access 
3. independence of row lengths 
4. ensuring deterministic results 

## Four-stage approach to finish the four major goals

![ACSpGEMM Figure 2](./figures/acspgemmfig2.png){width="500px"}

To achieve these goals, AC-SpGEMM follows a four-stage approach, as outlined in Figure 2. 
In the first stage, ACSpGEMM prepares data for global load balancing. 
In the second stage, we perform chunk-based ESC, producing deterministically bit-stable results. With the help of our local work distribution, this stage performs multiple iterations of ESC, fully utilizing local memory resources while keeping temporary results in scratchpad memory. 
Merging of rows shared across chunks happens in the third stage. 
Finally, in the fourth stage, we allocate the output matrix $\mathbf{C}$ and fill it with data from the chunks.

Before discussing the details of each stage, we motivate some design choices. Analysing the row length of matrices from the SuiteSparse matrix collection (todo: add citation), it can be observed that the majority of matrices in common problem domains have average row lengths of less than 200 elements, cf. Figure 1. Considering register sizes of current GPUs and reasonably small thread block sizes, up to 4000 temporary elements can be held by each block. If there is reasonable overlap between rows, the resulting output can be stored in scratchpad memory for another iteration of ESC. Given 200 entries per row, ideally another 3800 temporary elements can be loaded and compacted. In the best case, these local load and compaction steps continue until yielding completed rows of C, without ever going through slow global memory.

# Code
## CUDA Configurations 
The block size is 256. 
```cpp
int main(){
    return hello;
}
```






