# Survey on Different Block Sizes of Im2Col+Conv2D

## Setup

+ Precision: INT8
+ Input: 1 x 16 x 32 x 32
+ Kernel: 16 x 5 x 5
+ Hardware: VexiiMiCo-High (RV32IMFC+MiCoExt)
+ Data Cache Size: 2 Ways, 64 Sets (2\*64\*64=1 KB)

| Block Size | Total Cycles | 
| --- | --- |
| 1   | 29288483 |
| 2   | 29158519 |
| 4   | 30061630 |
| 8   | 35518159 |

+ Data Cache Size: 4 Ways, 64 Sets (4\*64\*64=2 KB)

| Block Size | Total Cycles | 
| --- | --- |
| 1   | 29330071 |
| 2   | 29158375 |
| 4   | 30061404 |
| 8   | 35517969 |