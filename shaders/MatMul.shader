#version 450
#extension GL_GOOGLE_include_directive : enable

#include "ComputeOpsShared.h"

%DTYPE_DEFINES%

#define TILE_SIZE 16
layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint M;  // rows of A
    uint K;  // cols of A / rows of B
    uint N;  // cols of B
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %SCALAR_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %SCALAR_DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferC {
    %SCALAR_DTYPE% dataC[];
};

shared %SCALAR_DTYPE% tileA[TILE_SIZE][TILE_SIZE];
shared %SCALAR_DTYPE% tileB[TILE_SIZE][TILE_SIZE];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint localRow = gl_LocalInvocationID.y;
    uint localCol = gl_LocalInvocationID.x;

    %SCALAR_DTYPE% sum = %SCALAR_DTYPE%(0);

    // Loop over tiles
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        // Load tile from A
        uint aCol = t * TILE_SIZE + localCol;
        if (row < M && aCol < K) {
            tileA[localRow][localCol] = dataA[row * K + aCol];
        } else {
            tileA[localRow][localCol] = %SCALAR_DTYPE%(0);
        }

        // Load tile from B
        uint bRow = t * TILE_SIZE + localRow;
        if (bRow < K && col < N) {
            tileB[localRow][localCol] = dataB[bRow * N + col];
        } else {
            tileB[localRow][localCol] = %SCALAR_DTYPE%(0);
        }

        barrier();

        // Compute partial sum for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier();
    }

    // Write result
    if (row < M && col < N) {
        dataC[row * N + col] = sum;
    }
}
