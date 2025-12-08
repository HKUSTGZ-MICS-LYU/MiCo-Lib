// Test configuration for quantized average pooling
// This file defines test parameters for Q8_AvgPool2D

#ifndef Q8_AVGPOOL_TEST_H
#define Q8_AVGPOOL_TEST_H

// Test case 1: Small image, 2x2 kernel, stride 2, no padding
#define TEST1_N 1
#define TEST1_INC 2
#define TEST1_INH 4
#define TEST1_INW 4
#define TEST1_K 2
#define TEST1_S 2
#define TEST1_P 0

// Test case 2: Larger image, 3x3 kernel, stride 1, padding 1
#define TEST2_N 1
#define TEST2_INC 4
#define TEST2_INH 8
#define TEST2_INW 8
#define TEST2_K 3
#define TEST2_S 1
#define TEST2_P 1

// Test case 3: 2x2 kernel, stride 1, padding 0
#define TEST3_N 1
#define TEST3_INC 8
#define TEST3_INH 6
#define TEST3_INW 6
#define TEST3_K 2
#define TEST3_S 1
#define TEST3_P 0

// Test case 4: 3x3 kernel, stride 2, padding 0
#define TEST4_N 1
#define TEST4_INC 3
#define TEST4_INH 7
#define TEST4_INW 7
#define TEST4_K 3
#define TEST4_S 2
#define TEST4_P 0

#endif // Q8_AVGPOOL_TEST_H
