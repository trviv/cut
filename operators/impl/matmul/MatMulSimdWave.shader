#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT%

// SIMD group matmul: WR=%WR%, WC=%WC%, TM=%TM%, TN=%TN%
// 32 threads = 1 SIMD wave, no shared memory, data exchange via WaveReadLaneAt

#define WR %WR%
#define WC %WC%
#define TM %TM%
#define TN %TN%

#include "MatMulCommon.shaderh"

[numthreads(WR * WC, 1, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint laneId = GTid.x;
    uint wr = laneId / WC;  // wave row index 0..WR-1
    uint wc = laneId % WC;  // wave col index 0..WC-1

    uint blockRowStart = Gid.y * (WR * TM);
    uint blockColStart = Gid.x * (WC * TN);

    // Each thread computes a TM x TN sub-tile of output
    %SCALAR_DTYPE_INPUT% acc[TM][TN];
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            acc[m][n] = (%SCALAR_DTYPE_INPUT%)(0);

    uint numKTiles = (pc.K + WC - 1) / WC;

    for (uint kt = 0; kt < numKTiles; kt++) {
        uint kBase = kt * WC;

        // Each thread loads TM values from A at its unique K offset (kBase + wc)
        %SCALAR_DTYPE_INPUT% a_reg[TM];
        [unroll] for (uint m = 0; m < TM; m++) {
            uint aRow = blockRowStart + wr * TM + m;
            uint aCol = kBase + wc;
            a_reg[m] = loadA(aRow, aCol);
        }

        // Each thread loads TN values from B at K offset (kBase + wr)
        // Only threads with wr < WC have valid K indices
        %SCALAR_DTYPE_INPUT% b_reg[TN];
        [unroll] for (uint n = 0; n < TN; n++) {
            uint bRow = kBase + wr;
            uint bCol = blockColStart + wc * TN + n;
            b_reg[n] = (wr < WC) ? loadB(bRow, bCol) : (%SCALAR_DTYPE_INPUT%)(0);
        }

        // Inner loop: broadcast A and B across the wave using WaveReadLaneAt
        [unroll] for (uint kk = 0; kk < WC; kk++) {
            // Broadcast A from the thread that has wc == kk (same wave row)
            %SCALAR_DTYPE_INPUT% a_k[TM];
            [unroll] for (uint m = 0; m < TM; m++) {
                a_k[m] = WaveReadLaneAt(a_reg[m], wr * WC + kk);
            }

            // Broadcast B from the thread that has wr == kk (same wave col)
            %SCALAR_DTYPE_INPUT% b_k[TN];
            [unroll] for (uint n = 0; n < TN; n++) {
                b_k[n] = WaveReadLaneAt(b_reg[n], kk * WC + wc);
            }

            // Outer product accumulation
            [unroll] for (uint m = 0; m < TM; m++)
                [unroll] for (uint n = 0; n < TN; n++)
                    acc[m][n] += a_k[m] * b_k[n];
        }
    }

    // Write output
    [unroll] for (uint m = 0; m < TM; m++) {
        [unroll] for (uint n = 0; n < TN; n++) {
            uint outRow = blockRowStart + wr * TM + m;
            uint outCol = blockColStart + wc * TN + n;
            if (outRow < pc.M && outCol < pc.N) {
                dataC[outRow * pc.strideB + outCol] = acc[m][n];
            }
        }
    }
}
