// SiLU fusion module for compileCustomShader.
// postProcessImpl replaces the postProcess stub in base matmul shaders.
// dVal is ignored — SiLU only uses the accumulated matmul result.

[noinline] float postProcessImpl(float accum, float dVal) {
    return accum / (1.0 + exp(-accum));
}

groupshared float _keep;

[numthreads(1, 1, 1)]
void main() { _keep = postProcessImpl(0, 0); }
