#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_INPUT2%
%DTYPE_DEFINES_OUTPUT%

// Minimal test version: just write zeros to verify dispatch works
#include "MatMulCommon.shaderh"

#define WG_SIZE 256
#define COLS_PER_WG 4
#define THREADS_PER_COL 64

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID,
          uint3 GId  : SV_GroupID) {
    uint tid = GTid.x;
    uint m = GId.y;
    uint colIdx   = tid / THREADS_PER_COL;
    uint tidInCol = tid % THREADS_PER_COL;
    uint n = GId.x * COLS_PER_WG + colIdx;

    if (tidInCol == 0 && n < pc.N && m < pc.M) {
        writeOutput(m, n, (%SCALAR_DTYPE_OUTPUT%)(0));
    }
}
