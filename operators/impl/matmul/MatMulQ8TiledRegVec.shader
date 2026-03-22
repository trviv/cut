#include "ComputeOpsShared.h"

%DTYPE_DEFINES_INPUT1%
%DTYPE_DEFINES_SCALES%
%DTYPE_DEFINES_OUTPUT%

// Q8_0 dequant matmul with vectorized cooperative loading.
// TILE_SIZE=%TILE_SIZE%, TM=%TM%, TN=%TN%
//
// Fixes vs MatMulQ8TiledReg:
//   1. Vec4 cooperative loading for tileA (4x fewer A load instructions)
//   2. Packed uint32 cooperative loading for tileB (4x fewer B loads,
//      each uint32 unpacks to 4 dequantized floats inline)
//   3. Vec4 scale loading (one load for 4 column scales vs 1 per element)
//   4. B-register pre-loading in compute inner loop
//
// Thread mapping for cooperative loads (TILE_SIZE=16, TM=4, TN=4):
//   256 threads → tileA: 64×16 = 1024 elements via 256 vec4 loads (1/thread)
//   256 threads → tileB: 16×64 = 1024 elements via 256 packed uint32 loads (1/thread)

#define TILE_SIZE %TILE_SIZE%
#define TM %TM%
#define TN %TN%

#include "MatMulQ8Common.shaderh"

// +1 padding on inner dimension to avoid shared memory bank conflicts
groupshared %SCALAR_DTYPE_INPUT1% tileA[TILE_SIZE * TM][TILE_SIZE + 1];
groupshared float tileB[TILE_SIZE][TILE_SIZE * TN + 1];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID) {
    uint localRow = GTid.y;
    uint localCol = GTid.x;
    uint tid = localRow * TILE_SIZE + localCol;

    uint2 tileId = swizzleTileId(Gid);
    uint blockRowStart = tileId.y * TILE_SIZE * TM;
    uint blockColStart = tileId.x * TILE_SIZE * TN;

    %SCALAR_DTYPE_OUTPUT% acc[TM][TN];
    [unroll] for (uint m = 0; m < TM; m++)
        [unroll] for (uint n = 0; n < TN; n++)
            acc[m][n] = (%SCALAR_DTYPE_OUTPUT%)(0);

    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;
    uint fullKTiles = pc.K / TILE_SIZE;
    bool hasPartialK = (pc.K % TILE_SIZE) != 0;
    bool fullM = (blockRowStart + TILE_SIZE * TM <= pc.M);
    bool fullN = (blockColStart + TILE_SIZE * TN <= pc.N);

    // Vec4 cooperative load mapping for tileA [TILE_SIZE*TM × TILE_SIZE]
    // 256 threads, each loads one vec4 (4 floats along K dimension)
    uint tileA_row = tid / (TILE_SIZE / 4);
    uint tileA_col = (tid % (TILE_SIZE / 4)) * 4;

    // Packed cooperative load mapping for tileB [TILE_SIZE × TILE_SIZE*TN]
    // 256 threads, each loads one uint32 (4 int8 values along N dimension)
    uint tileB_row = tid / (TILE_SIZE * TN / 4);
    uint tileB_col = (tid % (TILE_SIZE * TN / 4)) * 4;

    if (fullM && fullN) {
        // Fast path: interior workgroup — all loads are in bounds
        for (uint t = 0; t < fullKTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            // Vec4 cooperative load for tileA (standard float data)
            uint gA_idx = (blockRowStart + tileA_row) * pc.strideA + tileKStart + tileA_col;
            %VEC_DTYPE_INPUT1% vA = dataA[gA_idx >> 2];
            tileA[tileA_row][tileA_col + 0] = vA[0];
            tileA[tileA_row][tileA_col + 1] = vA[1];
            tileA[tileA_row][tileA_col + 2] = vA[2];
            tileA[tileA_row][tileA_col + 3] = vA[3];

            // Packed cooperative load for tileB (int8 → dequant → float)
            // One uint32 = 4 adjacent int8 values; one vec4 scale load for 4 columns
            {
                uint gB_k = tileKStart + tileB_row;
                uint gB_n = blockColStart + tileB_col;
                uint byteIdx = gB_k * pc.strideBN + gB_n;
                uint packed = packedB[byteIdx >> 2];

                // Unpack 4 int8 values with sign extension
                int sp = int(packed);
                float b0 = float((sp << 24) >> 24);
                float b1 = float((sp << 16) >> 24);
                float b2 = float((sp <<  8) >> 24);
                float b3 = float(sp >> 24);

                // Load 4 scales (one per column, same K-block)
                uint scaleBase = (gB_k >> 5) * pc.scaleStride + gB_n;
#ifdef DTYPE_SCALES_IS_INT8
                int packedScales = scalesB[scaleBase >> 4][(scaleBase >> 2) & 3];
                int ss = packedScales;
                float s0 = float((ss << 24) >> 24);
                float s1 = float((ss << 16) >> 24);
                float s2 = float((ss <<  8) >> 24);
                float s3 = float(ss >> 24);
#else
                %VEC_DTYPE_SCALES% sv = scalesB[scaleBase >> 2];
                float s0 = float(sv[0]);
                float s1 = float(sv[1]);
                float s2 = float(sv[2]);
                float s3 = float(sv[3]);
#endif
                // Dequantize and store into shared memory
                tileB[tileB_row][tileB_col + 0] = b0 * s0;
                tileB[tileB_row][tileB_col + 1] = b1 * s1;
                tileB[tileB_row][tileB_col + 2] = b2 * s2;
                tileB[tileB_row][tileB_col + 3] = b3 * s3;
            }

            GroupMemoryBarrierWithGroupSync();

            // Compute: K-unroll by 4 with B-register pre-load
            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                %SCALAR_DTYPE_OUTPUT% b_reg[4][TN];
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k    ][bc];
                    b_reg[1][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 1][bc];
                    b_reg[2][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 2][bc];
                    b_reg[3][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 3][bc];
                }
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Partial K tile: fall back to scalar loads (rare, only last tile)
        if (hasPartialK) {
            uint tileKStart = fullKTiles * TILE_SIZE;

            [unroll] for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    loadA(blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            [unroll] for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    loadB(tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }

            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                %SCALAR_DTYPE_OUTPUT% b_reg[4][TN];
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k    ][bc];
                    b_reg[1][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 1][bc];
                    b_reg[2][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 2][bc];
                    b_reg[3][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 3][bc];
                }
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Unchecked output write
        [unroll] for (uint m = 0; m < TM; m++) {
            [unroll] for (uint n = 0; n < TN; n++) {
                uint outRow = blockRowStart + localRow + m * TILE_SIZE;
                uint outCol = blockColStart + localCol + n * TILE_SIZE;
                writeOutput(outRow, outCol, acc[m][n]);
            }
        }
    } else {
        // Edge workgroup: use checked scalar loads
        for (uint t = 0; t < numTiles; t++) {
            uint tileKStart = t * TILE_SIZE;

            [unroll] for (uint m = 0; m < TM; m++) {
                tileA[localRow + m * TILE_SIZE][localCol] =
                    loadA(blockRowStart + localRow + m * TILE_SIZE, tileKStart + localCol);
            }
            [unroll] for (uint n = 0; n < TN; n++) {
                tileB[localRow][localCol + n * TILE_SIZE] =
                    loadB(tileKStart + localRow, blockColStart + localCol + n * TILE_SIZE);
            }

            GroupMemoryBarrierWithGroupSync();

            [unroll] for (uint k = 0; k < TILE_SIZE; k += 4) {
                %SCALAR_DTYPE_OUTPUT% b_reg[4][TN];
                [unroll] for (uint n = 0; n < TN; n++) {
                    uint bc = localCol + n * TILE_SIZE;
                    b_reg[0][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k    ][bc];
                    b_reg[1][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 1][bc];
                    b_reg[2][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 2][bc];
                    b_reg[3][n] = (%SCALAR_DTYPE_OUTPUT%)tileB[k + 3][bc];
                }
                [unroll] for (uint m = 0; m < TM; m++) {
                    %SCALAR_DTYPE_OUTPUT% a0 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k];
                    %SCALAR_DTYPE_OUTPUT% a1 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 1];
                    %SCALAR_DTYPE_OUTPUT% a2 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 2];
                    %SCALAR_DTYPE_OUTPUT% a3 = (%SCALAR_DTYPE_OUTPUT%)tileA[localRow + m * TILE_SIZE][k + 3];
                    [unroll] for (uint n = 0; n < TN; n++) {
                        acc[m][n] = mad(a0, b_reg[0][n], acc[m][n]);
                        acc[m][n] = mad(a1, b_reg[1][n], acc[m][n]);
                        acc[m][n] = mad(a2, b_reg[2][n], acc[m][n]);
                        acc[m][n] = mad(a3, b_reg[3][n], acc[m][n]);
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Checked output write
        [unroll] for (uint m = 0; m < TM; m++) {
            [unroll] for (uint n = 0; n < TN; n++) {
                uint outRow = blockRowStart + localRow + m * TILE_SIZE;
                uint outCol = blockColStart + localCol + n * TILE_SIZE;
                if (outRow < pc.M && outCol < pc.N) {
                    writeOutput(outRow, outCol, acc[m][n]);
                }
            }
        }
    }
}
