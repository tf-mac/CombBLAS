#ifndef NSPARSE_DEF_H
#define NSPARSE_DEF_H

/* Hardware Specific Parameters */
#define warp_BIT 5
#define NSPARSE_WARP_SIZE 32
#define MAX_LOCAL_THREAD_NUM 1024
#define MAX_THREAD_BLOCK (MAX_LOCAL_THREAD_NUM / NSPARSE_WARP_SIZE)

/* Number of SpMV Execution for Evaluation or Test */
#define TRI_NUM 101
#define TEST_NUM 2
#define SpGEMM_TRI_NUM 11

/* Define 2 related */
#define sfFLT_MAX 1000000000
#define SHORT_MAX 32768
#define SHORT_MAX_BIT 15
#define USHORT_MAX 65536
#define USHORT_MAX_BIT 16

#define SCL_BORDER 16
#define SCL_BIT ((1 << SCL_BORDER) - 1)

#define MAX_BLOCK_SIZE 20

/* Check the answer */
#define sfDEBUG

#endif
