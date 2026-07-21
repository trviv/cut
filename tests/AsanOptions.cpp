// Default AddressSanitizer / LeakSanitizer options for the test binary.
//
// These weak hooks are consulted by the sanitizer runtime, so they are harmless
// in non-sanitizer builds and are overridable via the ASAN_OPTIONS /
// LSAN_OPTIONS environment variables. Baking them in means the test binary
// behaves correctly even when launched directly (not via scripts/build/run_cpp.sh).

extern "C" const char *__asan_default_options() {
  // protect_shadow_gap=0: the CUDA driver reserves virtual address space that
  //   collides with ASan's shadow gap, which makes cuInit() fail (the CUDA
  //   backend then appears unavailable). Disabling the gap lets it initialize.
  // fast_unwind_on_malloc=0: full allocation stacks so leak suppressions below
  //   reliably match a frame inside the offending third-party driver library.
  return "protect_shadow_gap=0:fast_unwind_on_malloc=0";
}

extern "C" const char *__lsan_default_suppressions() {
  // One-time/thread-local allocations inside the GPU drivers and their system
  // dependencies are never freed and are not actionable from this codebase.
  // Suppress them by library so genuine leaks in CUT code still surface.
  return "leak:libvulkan\n"
         "leak:libdbus-1\n"
         "leak:libdrm\n"
         "leak:libnvidia-ptxjitcompiler\n"  // CUDA/NVRTC PTX JIT compiler
         "leak:libnvidia\n"                  // other NVIDIA driver libraries
         "leak:libcuda\n"
         "leak:libnvrtc\n"
         // One-time lazy init inside the (sometimes unsymbolizable, pip-bundled)
         // NVIDIA JIT libraries goes through pthread_once; these never free.
         "leak:__pthread_once_slow\n";
}
