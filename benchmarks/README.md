# Benchmarks

Everything that measures CUT lives here: the two external comparisons — *how far
off vendor-peak is an operator* and *how far off llama.cpp is a whole model* —
the variant sweeps that build the dispatch table, and the per-operator
micro-benchmarks used while optimising a kernel.

The first table is CUT against the libraries NVIDIA and AMD ship, on the same
shapes, same GPU, same run; the second is CUT against llama.cpp on the same
model, prompt length and generation length. Both are regenerated from recorded
snapshots ([`results/history.json`](results/history.json),
[`results/model_history.json`](results/model_history.json)), so every column is a
run that happened rather than a number someone typed in.

## Results — CUT vs vendor libraries

<!-- BENCH:BEGIN -->
*116 comparisons — 15 at or above the vendor, 31 at parity, 41 within 2x, 28 within 5x, 1 beyond; no correctness failures.*

<p><img src="assets/scorecard.svg" alt="CUT vs vendor, by band" width="620"></p>

<details>
<summary>🔵 <b>Softmax</b> — vs cuDNN · 1.08x avg (16 cases)</summary>

<table>
<tr><th align="left">benchmark</th><th>2026-08-02<br><sub>1c19721</sub></th><th>2026-07-31<br><sub>9bf5e74</sub></th><th>2026-07-23<br><sub>29fb17d</sub></th></tr>
<tr><td align="left"><code>softmax</code> rows=1 cols=152064</td><td align="center">🔵 <b>2.73x</b> ▲<sub>9891%</sub></td><td align="center">🔴 0.027x</td><td align="center">🔴 0.027x</td></tr>
<tr><td align="left"><code>softmax_large</code> gemma2-27b-logits-1k rows=1024 cols=256000</td><td align="center">🔵 <b>1.82x</b> ▲<sub>2977%</sub></td><td align="center">🔴 0.059x</td><td align="center">—</td></tr>
<tr><td align="left"><code>softmax_large</code> qwen2.5-14b-logits-2k rows=2048 cols=152064</td><td align="center">🔵 <b>1.78x</b> ▲<sub>2166%</sub></td><td align="center">🔴 0.078x</td><td align="center">🔴 0.074x</td></tr>
<tr><td align="left"><code>softmax_large</code> llama3-8b-logits-4k rows=4096 cols=128256</td><td align="center">🔵 <b>1.70x</b> ▲<sub>1576%</sub></td><td align="center">🔴 0.10x</td><td align="center">🔴 0.095x</td></tr>
<tr><td align="left"><code>softmax_large</code> llama3-8b-attn-4k rows=131072 cols=4096</td><td align="center">🔵 <b>1.00x</b> ▲<sub>1283%</sub></td><td align="center">🔴 0.072x</td><td align="center">🔴 0.065x</td></tr>
<tr><td align="left"><code>softmax_large</code> llama3-8b-attn-16k-1h rows=16384 cols=16384</td><td align="center">🟢 <b>1.00x</b> ▲<sub>824%</sub></td><td align="center">🔴 0.11x</td><td align="center">—</td></tr>
<tr><td align="left"><code>softmax_large</code> llama3-8b-attn-8k-8h rows=65536 cols=8192</td><td align="center">🟢 <b>1.00x</b> ▲<sub>846%</sub></td><td align="center">🔴 0.11x</td><td align="center">🔴 0.098x</td></tr>
<tr><td align="left"><code>softmax</code> rows=8192 cols=1024</td><td align="center">🟢 <b>0.99x</b> ▲<sub>2189%</sub></td><td align="center">🔴 0.043x</td><td align="center">🔴 0.040x</td></tr>
<tr><td align="left"><code>softmax</code> rows=32768 cols=256</td><td align="center">🟢 <b>0.98x</b> ▲<sub>2190%</sub></td><td align="center">🔴 0.043x</td><td align="center">🔴 0.040x</td></tr>
<tr><td align="left"><code>softmax</code> rows=4096 cols=512</td><td align="center">🟢 <b>0.98x</b> ▲<sub>2263%</sub></td><td align="center">🔴 0.041x</td><td align="center">🔴 0.039x</td></tr>
<tr><td align="left"><code>softmax_large</code> flux-dit-attn rows=110592 cols=4608</td><td align="center">🟡 <b>0.95x</b> ▲<sub>937%</sub></td><td align="center">🔴 0.091x</td><td align="center">🔴 0.086x</td></tr>
<tr><td align="left"><code>softmax_large</code> sd35-large-attn rows=155648 cols=4096</td><td align="center">🟡 <b>0.95x</b> ▲<sub>1171%</sub></td><td align="center">🔴 0.074x</td><td align="center">🔴 0.070x</td></tr>
<tr><td align="left"><code>softmax</code> rows=32 cols=4096</td><td align="center">🟡 <b>0.91x</b> ▲<sub>4685%</sub></td><td align="center">🔴 0.019x</td><td align="center">🔴 0.019x</td></tr>
<tr><td align="left"><code>softmax</code> rows=1 cols=32000</td><td align="center">🟡 <b>0.68x</b> ▲<sub>5286%</sub></td><td align="center">🔴 0.013x</td><td align="center">🔴 0.013x</td></tr>
<tr><td align="left"><code>softmax</code> rows=8 cols=32000</td><td align="center">🟡 <b>0.68x</b> ▲<sub>2573%</sub></td><td align="center">🔴 0.025x</td><td align="center">🔴 0.025x</td></tr>
<tr><td align="left"><code>softmax</code> rows=4096 cols=128</td><td align="center">🟡 <b>0.67x</b> ▲<sub>2058%</sub></td><td align="center">🔴 0.031x</td><td align="center">🔴 0.030x</td></tr>
</table>
</details>

<details>
<summary>🟢 <b>Transpose</b> — vs cuBLAS · 0.98x avg (15 cases)</summary>

<table>
<tr><th align="left">benchmark</th><th>2026-08-02<br><sub>1c19721</sub></th><th>2026-07-31<br><sub>9bf5e74</sub></th><th>2026-07-23<br><sub>29fb17d</sub></th></tr>
<tr><td align="left"><code>transpose_large</code> qwen2.5-14b-lm-head-w M=152064 N=5120</td><td align="center">🔵 <b>1.01x</b></td><td align="center">🔵 1.01x</td><td align="center">—</td></tr>
<tr><td align="left"><code>transpose_large</code> vit-g14-tokens M=65792 N=1408</td><td align="center">🔵 <b>1.01x</b></td><td align="center">🔵 1.01x</td><td align="center">—</td></tr>
<tr><td align="left"><code>transpose</code> M=8192 N=1024</td><td align="center">🟢 <b>0.99x</b></td><td align="center">🟢 0.98x</td><td align="center">🟡 0.78x</td></tr>
<tr><td align="left"><code>transpose_large</code> act-8k M=8192 N=8192</td><td align="center">🟢 <b>0.99x</b></td><td align="center">🟢 0.98x</td><td align="center">🟡 0.85x</td></tr>
<tr><td align="left"><code>transpose_large</code> act-32k-x-16k M=32768 N=16384</td><td align="center">🟢 <b>0.98x</b></td><td align="center">🟢 0.98x</td><td align="center">🟡 0.85x</td></tr>
<tr><td align="left"><code>transpose</code> M=8192 N=8192</td><td align="center">🟢 <b>0.98x</b></td><td align="center">🟢 0.99x</td><td align="center">🟡 0.79x</td></tr>
<tr><td align="left"><code>transpose</code> M=4096 N=4096</td><td align="center">🟢 <b>0.98x</b></td><td align="center">🟢 1.00x</td><td align="center">🟡 0.80x</td></tr>
<tr><td align="left"><code>transpose_large</code> llama3-8b-lm-head-w M=128256 N=4096</td><td align="center">🟢 <b>0.98x</b> ▼<sub>4%</sub></td><td align="center">🔵 1.03x</td><td align="center">—</td></tr>
<tr><td align="left"><code>transpose_large</code> llama3-8b-qkv-act M=32768 N=4096</td><td align="center">🟢 <b>0.98x</b></td><td align="center">🟢 1.00x</td><td align="center">—</td></tr>
<tr><td align="left"><code>transpose</code> M=1024 N=8192</td><td align="center">🟢 <b>0.98x</b></td><td align="center">🟢 0.98x</td><td align="center">🟡 0.78x</td></tr>
<tr><td align="left"><code>transpose_large</code> llama3-70b-qkv-act M=32768 N=8192</td><td align="center">🟢 <b>0.97x</b></td><td align="center">🟢 0.98x</td><td align="center">—</td></tr>
<tr><td align="left"><code>transpose</code> M=1024 N=1024</td><td align="center">🟢 <b>0.97x</b></td><td align="center">🟢 0.99x</td><td align="center">🟡 0.64x</td></tr>
<tr><td align="left"><code>transpose_large</code> flux-dit-act M=18432 N=3072</td><td align="center">🟢 <b>0.96x</b></td><td align="center">🟢 0.96x</td><td align="center">—</td></tr>
<tr><td align="left"><code>transpose_large</code> act-16k M=16384 N=16384</td><td align="center">🟢 <b>0.96x</b></td><td align="center">🟢 0.98x</td><td align="center">🟡 0.85x</td></tr>
<tr><td align="left"><code>transpose_large</code> act-32k M=32768 N=32768</td><td align="center">🟡 <b>0.94x</b> ▼<sub>6%</sub></td><td align="center">🟢 1.00x</td><td align="center">🟡 0.83x</td></tr>
</table>
</details>

<details>
<summary>🟡 <b>Scan</b> — vs CUB · 0.94x avg (28 cases)</summary>

<table>
<tr><th align="left">benchmark</th><th>2026-08-02<br><sub>1c19721</sub></th><th>2026-07-31<br><sub>9bf5e74</sub></th><th>2026-07-23<br><sub>29fb17d</sub></th></tr>
<tr><td align="left"><code>scan_exclusive</code> mixtral-8x7b-moe-32k N=262144</td><td align="center">🔵 <b>1.02x</b> ▼<sub>5%</sub></td><td align="center">🔵 1.08x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> mixtral-8x7b-moe-32k N=262144</td><td align="center">🔵 <b>1.02x</b></td><td align="center">🔵 1.02x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_exclusive</code> gemma2-9b-vocab N=256000</td><td align="center">🔵 <b>1.00x</b> ▼<sub>9%</sub></td><td align="center">🔵 1.10x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> N=16777216</td><td align="center">🟢 <b>0.99x</b></td><td align="center">🟢 0.99x</td><td align="center">🔴 0.043x</td></tr>
<tr><td align="left"><code>scan_inclusive</code> gemma2-9b-vocab N=256000</td><td align="center">🟢 <b>0.99x</b> ▼<sub>5%</sub></td><td align="center">🔵 1.03x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_exclusive</code> N=16777216</td><td align="center">🟢 <b>0.99x</b></td><td align="center">🟢 0.99x</td><td align="center">🔴 0.043x</td></tr>
<tr><td align="left"><code>scan_inclusive</code> N=65536</td><td align="center">🟢 <b>0.98x</b> ▼<sub>2%</sub></td><td align="center">🔵 1.00x</td><td align="center">🟠 0.42x</td></tr>
<tr><td align="left"><code>scan_exclusive</code> N=65536</td><td align="center">🟢 <b>0.98x</b> ▼<sub>4%</sub></td><td align="center">🔵 1.02x</td><td align="center">🟠 0.42x</td></tr>
<tr><td align="left"><code>scan_exclusive</code> qwen2.5-14b-vocab N=152064</td><td align="center">🟢 <b>0.97x</b> ▼<sub>4%</sub></td><td align="center">🔵 1.01x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> llama3-8b-batch64-logits N=8208384</td><td align="center">🟢 <b>0.97x</b></td><td align="center">🟢 0.97x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_exclusive</code> llama3-8b-batch64-logits N=8208384</td><td align="center">🟢 <b>0.97x</b></td><td align="center">🟢 0.97x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_exclusive</code> N=1048576</td><td align="center">🟢 <b>0.96x</b></td><td align="center">🟢 0.97x</td><td align="center">🔴 0.085x</td></tr>
<tr><td align="left"><code>scan_exclusive</code> llama3-8b-vocab N=128256</td><td align="center">🟢 <b>0.96x</b> ▼<sub>4%</sub></td><td align="center">🔵 1.00x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> N=1048576</td><td align="center">🟢 <b>0.95x</b> ▼<sub>3%</sub></td><td align="center">🟢 0.98x</td><td align="center">🔴 0.083x</td></tr>
<tr><td align="left"><code>scan_inclusive</code> llama3-8b-vocab N=128256</td><td align="center">🟢 <b>0.95x</b> ▼<sub>5%</sub></td><td align="center">🔵 1.00x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> qwen2.5-14b-vocab N=152064</td><td align="center">🟢 <b>0.95x</b> ▼<sub>5%</sub></td><td align="center">🟢 1.00x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_exclusive</code> qwen2.5-14b-batch32-logits N=4866048</td><td align="center">🟡 <b>0.95x</b></td><td align="center">🟢 0.95x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> qwen2.5-14b-batch32-logits N=4866048</td><td align="center">🟡 <b>0.95x</b></td><td align="center">🟢 0.95x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_exclusive</code> N=4194304</td><td align="center">🟡 <b>0.94x</b></td><td align="center">🟡 0.94x</td><td align="center">🔴 0.049x</td></tr>
<tr><td align="left"><code>scan_inclusive</code> N=4194304</td><td align="center">🟡 <b>0.94x</b></td><td align="center">🟡 0.94x</td><td align="center">🔴 0.048x</td></tr>
<tr><td align="left"><code>scan_exclusive</code> mixtral-8x7b-moe-4k N=32768</td><td align="center">🟡 <b>0.90x</b> ▼<sub>5%</sub></td><td align="center">🟡 0.94x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> yolov8-nms-640 N=8400</td><td align="center">🟡 <b>0.89x</b> ▼<sub>4%</sub></td><td align="center">🟡 0.93x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> llama2-7b-vocab N=32000</td><td align="center">🟡 <b>0.89x</b> ▼<sub>4%</sub></td><td align="center">🟡 0.93x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> mixtral-8x7b-moe-4k N=32768</td><td align="center">🟡 <b>0.89x</b> ▼<sub>5%</sub></td><td align="center">🟡 0.94x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_exclusive</code> llama2-7b-vocab N=32000</td><td align="center">🟡 <b>0.89x</b> ▼<sub>5%</sub></td><td align="center">🟡 0.93x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_exclusive</code> yolov8-nms-640 N=8400</td><td align="center">🟡 <b>0.88x</b> ▼<sub>5%</sub></td><td align="center">🟡 0.92x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_exclusive</code> llama3-8b-batch16-logits N=2052096</td><td align="center">🟡 <b>0.85x</b></td><td align="center">🟡 0.86x</td><td align="center">—</td></tr>
<tr><td align="left"><code>scan_inclusive</code> llama3-8b-batch16-logits N=2052096</td><td align="center">🟡 <b>0.85x</b></td><td align="center">🟡 0.86x</td><td align="center">—</td></tr>
</table>
</details>

<details>
<summary>🟡 <b>Sort</b> — vs CUB · 0.94x avg (12 cases)</summary>

<table>
<tr><th align="left">benchmark</th><th>2026-08-02<br><sub>1c19721</sub></th><th>2026-07-31<br><sub>9bf5e74</sub></th><th>2026-07-23<br><sub>29fb17d</sub></th></tr>
<tr><td align="left"><code>sort_radix</code> yolov8-nms-640 N=8400</td><td align="center">🔵 <b>1.19x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> llama2-7b-vocab N=32000</td><td align="center">🔵 <b>1.16x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> mixtral-8x7b-moe-4k N=32768</td><td align="center">🔵 <b>1.16x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> N=65536</td><td align="center">🔵 <b>1.05x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> llama3-8b-vocab N=128256</td><td align="center">🟢 <b>0.97x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> qwen2.5-14b-vocab N=152064</td><td align="center">🟡 <b>0.94x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> llama3-8b-batch64-logits N=8208384</td><td align="center">🟡 <b>0.87x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> llama3-8b-batch16-logits N=2052096</td><td align="center">🟡 <b>0.84x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> gemma2-9b-vocab N=256000</td><td align="center">🟡 <b>0.82x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> N=1048576</td><td align="center">🟡 <b>0.82x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> qwen2.5-14b-batch32-logits N=4866048</td><td align="center">🟡 <b>0.81x</b></td><td align="center">—</td><td align="center">—</td></tr>
<tr><td align="left"><code>sort_radix</code> mixtral-8x7b-moe-32k N=262144</td><td align="center">🟡 <b>0.80x</b></td><td align="center">—</td><td align="center">—</td></tr>
</table>
</details>

<details>
<summary>🟡 <b>Convolution</b> — vs cuDNN · 0.61x avg (11 cases)</summary>

<table>
<tr><th align="left">benchmark</th><th>2026-08-02<br><sub>1c19721</sub></th><th>2026-07-31<br><sub>9bf5e74</sub></th><th>2026-07-23<br><sub>29fb17d</sub></th></tr>
<tr><td align="left"><code>conv2d</code> sam-vit-h-patch-embed N=8 C=3 HW=1024 Cout=1280 k=16 s=16</td><td align="center">🟡 <b>0.84x</b> ▲<sub>2808%</sub></td><td align="center">🔴 0.029x</td><td align="center">—</td></tr>
<tr><td align="left"><code>conv2d</code> vit-g14-patch-embed N=24 C=3 HW=336 Cout=1664 k=14 s=14</td><td align="center">🟡 <b>0.77x</b> ▲<sub>1991%</sub></td><td align="center">🔴 0.037x</td><td align="center">🔴 0.037x</td></tr>
<tr><td align="left"><code>conv2d</code> vit-l14-patch-embed N=32 C=3 HW=224 Cout=1024 k=14 s=14</td><td align="center">🟡 <b>0.76x</b> ▲<sub>1745%</sub></td><td align="center">🔴 0.041x</td><td align="center">🔴 0.042x</td></tr>
<tr><td align="left"><code>conv2d</code> resnet50-stage1-1x1 N=256 C=256 HW=56 Cout=64 k=1 s=1</td><td align="center">🟡 <b>0.75x</b> ▲<sub>2238%</sub></td><td align="center">🔴 0.032x</td><td align="center">🔴 0.032x</td></tr>
<tr><td align="left"><code>conv2d</code> resnet50-stage1-3x3 N=256 C=64 HW=56 Cout=64 k=3 s=1</td><td align="center">🟡 <b>0.62x</b> ▲<sub>1348%</sub></td><td align="center">🔴 0.042x</td><td align="center">🔴 0.043x</td></tr>
<tr><td align="left"><code>conv2d</code> sd-vae-decoder-128ch-2048px N=2 C=128 HW=2048 Cout=128 k=3 s=1</td><td align="center">🟡 <b>0.58x</b> ▲<sub>1764%</sub></td><td align="center">🔴 0.031x</td><td align="center">🔴 0.031x</td></tr>
<tr><td align="left"><code>conv2d</code> sd-vae-decoder-128ch-1024px N=6 C=128 HW=1024 Cout=128 k=3 s=1</td><td align="center">🟡 <b>0.58x</b> ▲<sub>1678%</sub></td><td align="center">🔴 0.033x</td><td align="center">🔴 0.033x</td></tr>
<tr><td align="left"><code>conv2d</code> sd-vae-decoder-512ch N=6 C=512 HW=256 Cout=512 k=3 s=1</td><td align="center">🟡 <b>0.55x</b> ▲<sub>1793%</sub></td><td align="center">🔴 0.029x</td><td align="center">🔴 0.029x</td></tr>
<tr><td align="left"><code>conv2d</code> sdxl-unet-1280ch N=24 C=1280 HW=32 Cout=1280 k=3 s=1</td><td align="center">🟡 <b>0.54x</b> ▲<sub>1320%</sub></td><td align="center">🔴 0.038x</td><td align="center">🔴 0.038x</td></tr>
<tr><td align="left"><code>conv2d</code> sdxl-unet-320ch N=16 C=320 HW=128 Cout=320 k=3 s=1</td><td align="center">🟠 <b>0.47x</b> ▲<sub>1324%</sub></td><td align="center">🔴 0.033x</td><td align="center">🔴 0.033x</td></tr>
<tr><td align="left"><code>conv2d</code> resnet50-stage2-3x3 N=256 C=128 HW=28 Cout=128 k=3 s=1</td><td align="center">🟠 <b>0.41x</b> ▲<sub>1355%</sub></td><td align="center">🔴 0.028x</td><td align="center">🔴 0.028x</td></tr>
</table>
</details>

<details>
<summary>🟠 <b>GEMM / GEMV</b> — vs cuBLAS · 0.38x avg (34 cases)</summary>

<table>
<tr><th align="left">benchmark</th><th>2026-08-02<br><sub>1c19721</sub></th><th>2026-07-31<br><sub>9bf5e74</sub></th><th>2026-07-23<br><sub>29fb17d</sub></th></tr>
<tr><td align="left"><code>sgemm</code> M=128 K=128 N=128</td><td align="center">🔵 <b>1.02x</b> ▼<sub>11%</sub></td><td align="center">🔵 1.15x</td><td align="center">🔵 1.14x</td></tr>
<tr><td align="left"><code>sgemv</code> M=1 K=4096 N=4096</td><td align="center">🟡 <b>0.69x</b></td><td align="center">🟡 0.69x</td><td align="center">🟡 0.70x</td></tr>
<tr><td align="left"><code>sgemv</code> M=1 K=2048 N=2048</td><td align="center">🟡 <b>0.66x</b> ▼<sub>3%</sub></td><td align="center">🟡 0.67x</td><td align="center">🟡 0.67x</td></tr>
<tr><td align="left"><code>sgemm</code> M=4096 K=4096 N=4096</td><td align="center">🟡 <b>0.60x</b></td><td align="center">🟡 0.60x</td><td align="center">🟡 0.60x</td></tr>
<tr><td align="left"><code>sgemm</code> M=2048 K=2048 N=2048</td><td align="center">🟡 <b>0.54x</b> ▼<sub>2%</sub></td><td align="center">🟡 0.55x</td><td align="center">🟡 0.55x</td></tr>
<tr><td align="left"><code>sgemm</code> M=512 K=4096 N=4096</td><td align="center">🟡 <b>0.53x</b> ▼<sub>3%</sub></td><td align="center">🟡 0.54x</td><td align="center">🟡 0.55x</td></tr>
<tr><td align="left"><code>sgemm</code> M=1024 K=1024 N=1024</td><td align="center">🟡 <b>0.52x</b> ▼<sub>12%</sub></td><td align="center">🟡 0.59x</td><td align="center">🟡 0.60x</td></tr>
<tr><td align="left"><code>hgemm</code> M=16 K=4096 N=4096</td><td align="center">🟠 <b>0.41x</b></td><td align="center">🟠 0.41x</td><td align="center">🟠 0.41x</td></tr>
<tr><td align="left"><code>sgemm</code> M=512 K=512 N=512</td><td align="center">🟠 <b>0.40x</b> ▼<sub>10%</sub></td><td align="center">🟠 0.44x</td><td align="center">🟠 0.44x</td></tr>
<tr><td align="left"><code>hgemm</code> M=4096 K=4096 N=4096</td><td align="center">🟠 <b>0.38x</b></td><td align="center">🟠 0.38x</td><td align="center">🟠 0.38x</td></tr>
<tr><td align="left"><code>hgemm</code> M=2048 K=2048 N=2048</td><td align="center">🟠 <b>0.37x</b> ▲<sub>3%</sub></td><td align="center">🟠 0.36x</td><td align="center">🟠 0.36x</td></tr>
<tr><td align="left"><code>sgemv</code> M=1 K=8192 N=8192</td><td align="center">🟠 <b>0.36x</b></td><td align="center">🟠 0.36x</td><td align="center">🟠 0.36x</td></tr>
<tr><td align="left"><code>hgemm_large</code> vit-g14-mlp M=65792 K=1408 N=6144</td><td align="center">🟠 <b>0.36x</b> ▲<sub>3%</sub></td><td align="center">🟠 0.34x</td><td align="center">🟠 0.34x</td></tr>
<tr><td align="left"><code>hgemm_large</code> sd35-large-mlp M=16384 K=2432 N=9728</td><td align="center">🟠 <b>0.35x</b></td><td align="center">🟠 0.35x</td><td align="center">🟠 0.35x</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama3-8b-qkv M=32768 K=4096 N=6144</td><td align="center">🟠 <b>0.35x</b> ▲<sub>2%</sub></td><td align="center">🟠 0.34x</td><td align="center">🟠 0.35x</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama3-8b-ffn-down M=32768 K=14336 N=4096</td><td align="center">🟠 <b>0.35x</b></td><td align="center">🟠 0.35x</td><td align="center">🟠 0.35x</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama2-13b-attn-out M=16384 K=5120 N=5120</td><td align="center">🟠 <b>0.35x</b></td><td align="center">🟠 0.35x</td><td align="center">🟠 0.35x</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama3-70b-qkv M=8192 K=8192 N=10240</td><td align="center">🟠 <b>0.35x</b></td><td align="center">🟠 0.34x</td><td align="center">—</td></tr>
<tr><td align="left"><code>sgemm</code> M=16 K=4096 N=4096</td><td align="center">🟠 <b>0.35x</b></td><td align="center">🟠 0.35x</td><td align="center">🟠 0.35x</td></tr>
<tr><td align="left"><code>hgemm_large</code> flux-dit-mlp M=18432 K=3072 N=12288</td><td align="center">🟠 <b>0.34x</b></td><td align="center">🟠 0.34x</td><td align="center">🟠 0.34x</td></tr>
<tr><td align="left"><code>hgemm_large</code> internvit-6b-mlp M=16384 K=3200 N=12800</td><td align="center">🟠 <b>0.34x</b></td><td align="center">🟠 0.34x</td><td align="center">—</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama3-70b-ffn-down M=8192 K=28672 N=8192</td><td align="center">🟠 <b>0.34x</b></td><td align="center">🟠 0.34x</td><td align="center">—</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama3-70b-ffn-up M=8192 K=8192 N=57344</td><td align="center">🟠 <b>0.33x</b> ▲<sub>2%</sub></td><td align="center">🟠 0.33x</td><td align="center">—</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama3-8b-ffn-up M=32768 K=4096 N=28672</td><td align="center">🟠 <b>0.33x</b> ▲<sub>2%</sub></td><td align="center">🟠 0.33x</td><td align="center">🟠 0.33x</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama2-13b-ffn-up M=16384 K=5120 N=27648</td><td align="center">🟠 <b>0.33x</b> ▲<sub>2%</sub></td><td align="center">🟠 0.32x</td><td align="center">🟠 0.33x</td></tr>
<tr><td align="left"><code>hgemm_large</code> gemma2-27b-ffn-up M=8192 K=4608 N=73728</td><td align="center">🟠 <b>0.33x</b> ▲<sub>3%</sub></td><td align="center">🟠 0.32x</td><td align="center">—</td></tr>
<tr><td align="left"><code>hgemm</code> M=1024 K=1024 N=1024</td><td align="center">🟠 <b>0.33x</b></td><td align="center">🟠 0.33x</td><td align="center">🟠 0.32x</td></tr>
<tr><td align="left"><code>hgemm_large</code> qwen2.5-14b-lm-head M=2048 K=5120 N=152064</td><td align="center">🟠 <b>0.33x</b> ▲<sub>3%</sub></td><td align="center">🟠 0.32x</td><td align="center">🟠 0.32x</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama3-70b-lm-head M=2048 K=8192 N=128256</td><td align="center">🟠 <b>0.33x</b> ▲<sub>2%</sub></td><td align="center">🟠 0.32x</td><td align="center">—</td></tr>
<tr><td align="left"><code>hgemm_large</code> llama3-8b-lm-head M=4096 K=4096 N=128256</td><td align="center">🟠 <b>0.32x</b> ▲<sub>2%</sub></td><td align="center">🟠 0.32x</td><td align="center">🟠 0.32x</td></tr>
<tr><td align="left"><code>hgemm</code> M=512 K=4096 N=4096</td><td align="center">🟠 <b>0.32x</b></td><td align="center">🟠 0.31x</td><td align="center">🟠 0.31x</td></tr>
<tr><td align="left"><code>hgemm</code> M=1 K=4096 N=4096</td><td align="center">🟠 <b>0.30x</b> ▲<sub>3%</sub></td><td align="center">🟠 0.30x</td><td align="center">⚠️<br><sub>mismatch</sub></td></tr>
<tr><td align="left"><code>hgemm_large</code> llama3-8b-ffn-up-xl M=65536 K=4096 N=28672</td><td align="center">🟠 <b>0.26x</b> ▼<sub>20%</sub></td><td align="center">🟠 0.33x</td><td align="center">🟠 0.33x</td></tr>
<tr><td align="left"><code>hgemm</code> M=1 K=8192 N=8192</td><td align="center">🔴 <b>0.13x</b><br><sub>8x slower</sub></td><td align="center">🔴 0.13x</td><td align="center">⚠️<br><sub>mismatch</sub></td></tr>
</table>
</details>

<sub>Measured on <b>NVIDIA GeForce RTX 3090</b> · newest column 2026-08-02 (<code>1c19721</code>) · absolute timings for every case are in <a href="results/history.json">results/history.json</a></sub>

<!-- BENCH:END -->
**Cells are speedup — vendor time ÷ CUT time — so bigger is better.** `2.00x` is
twice the vendor's speed, `0.50x` is half of it, `1.00x` is parity. Same
convention `vendor_compare.py` prints on the command line.

🔵 **≥1x — matches or beats the vendor** · 🟢 ≥0.95x parity · 🟡 ≥0.5x (within 2x)
· 🟠 ≥0.2x (within 5x) · 🔴 <0.2x · ⚠️ output disagreed with the reference
(timing not quotable)

Each family's figure is the geometric mean of its cases. Families and rows are
ordered best first; open a family for its per-shape table.

Newest snapshot is the **leftmost** column. ▲ on it means CUT got faster since
the previous column; movement under ±2% is hidden.

## Results — CUT vs llama.cpp

The same question one level up: not how far off vendor-peak an operator is, but
how far off the reference runtime a whole forward pass is. Same convention —
cells are llama.cpp's time over CUT's, so bigger is better — and the same bands.

<!-- MODELBENCH:BEGIN -->
### AMD Radeon RX 9070

*28 comparisons — 0 at or above llama.cpp, 1 at parity, 9 within 2x, 18 within 5x, 0 beyond; no coherence failures.*

<p><img src="assets/model_scorecard-amd-radeon-rx-9070.svg" alt="CUT vs llama.cpp on AMD Radeon RX 9070, by band" width="620"></p>

<details>
<summary>🟡 <b>Decode (tg)</b> — vs llama.cpp · 0.54x avg (14 cases)</summary>

<table>
<tr><th align="left">model · backend · shape</th><th>2026-08-04<br><sub>6bc62da</sub></th></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 vulkan tg32@128</td><td align="center">🟡 <b>0.82x</b><br><sub>162 tok/s</sub></td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 vulkan tg32@128</td><td align="center">🟡 <b>0.74x</b><br><sub>280 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K vulkan tg32@128</td><td align="center">🟡 <b>0.67x</b><br><sub>147 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 vulkan tg128@512</td><td align="center">🟡 <b>0.66x</b><br><sub>127 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S vulkan tg32@128</td><td align="center">🟡 <b>0.63x</b><br><sub>225 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 vulkan tg32@128</td><td align="center">🟡 <b>0.60x</b><br><sub>84 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 vulkan tg32@128</td><td align="center">🟡 <b>0.51x</b><br><sub>39 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 vulkan tg128@512</td><td align="center">🟠 <b>0.48x</b><br><sub>65 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 vulkan tg128@512</td><td align="center">🟠 <b>0.47x</b><br><sub>35 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K vulkan tg128@512</td><td align="center">🟠 <b>0.47x</b><br><sub>97 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M vulkan tg32@128</td><td align="center">🟠 <b>0.46x</b><br><sub>81 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S vulkan tg128@512</td><td align="center">🟠 <b>0.45x</b><br><sub>150 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M vulkan tg128@512</td><td align="center">🟠 <b>0.38x</b><br><sub>64 tok/s</sub></td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 vulkan tg128@512</td><td align="center">🟠 <b>0.38x</b><br><sub>146 tok/s</sub></td></tr>
</table>
</details>

<details>
<summary>🟠 <b>Prefill (pp)</b> — vs llama.cpp · 0.42x avg (14 cases)</summary>

<table>
<tr><th align="left">model · backend · shape</th><th>2026-08-04<br><sub>6bc62da</sub></th></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 vulkan pp128</td><td align="center">🟢 <b>0.97x</b><br><sub>8,057 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K vulkan pp128</td><td align="center">🟡 <b>0.71x</b><br><sub>2,668 tok/s</sub></td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 vulkan pp512</td><td align="center">🟡 <b>0.58x</b><br><sub>13,907 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 vulkan pp128</td><td align="center">🟠 <b>0.48x</b><br><sub>1,663 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M vulkan pp128</td><td align="center">🟠 <b>0.48x</b><br><sub>1,660 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S vulkan pp128</td><td align="center">🟠 <b>0.44x</b><br><sub>3,516 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K vulkan pp512</td><td align="center">🟠 <b>0.44x</b><br><sub>3,304 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 vulkan pp128</td><td align="center">🟠 <b>0.44x</b><br><sub>793 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S vulkan pp512</td><td align="center">🟠 <b>0.36x</b><br><sub>3,734 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 vulkan pp128</td><td align="center">🟠 <b>0.32x</b><br><sub>2,833 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 vulkan pp512</td><td align="center">🟠 <b>0.30x</b><br><sub>3,485 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M vulkan pp512</td><td align="center">🟠 <b>0.30x</b><br><sub>1,451 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 vulkan pp512</td><td align="center">🟠 <b>0.28x</b><br><sub>1,459 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 vulkan pp512</td><td align="center">🟠 <b>0.26x</b><br><sub>710 tok/s</sub></td></tr>
</table>
</details>

<sub>Measured on <b>AMD Radeon RX 9070</b> · newest column 2026-08-04 (<code>6bc62da</code>) · 5 timed runs per measurement, median · absolute tok/s for every case are in <a href="results/model_history.json">results/model_history.json</a></sub>

### NVIDIA GeForce RTX 3090

*56 comparisons — 2 at or above llama.cpp, 1 at parity, 8 within 2x, 39 within 5x, 6 beyond; no coherence failures.*

<p><img src="assets/model_scorecard-nvidia-geforce-rtx-3090.svg" alt="CUT vs llama.cpp on NVIDIA GeForce RTX 3090, by band" width="620"></p>

<details>
<summary>🟠 <b>Decode (tg)</b> — vs llama.cpp · 0.45x avg (28 cases)</summary>

<table>
<tr><th align="left">model · backend · shape</th><th>2026-08-04<br><sub>6bc62da</sub></th></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 vulkan tg32@128</td><td align="center">🔵 <b>1.17x</b><br><sub>492 tok/s</sub></td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 cuda tg32@128</td><td align="center">🟢 <b>0.96x</b><br><sub>764 tok/s</sub></td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 vulkan tg128@512</td><td align="center">🟡 <b>0.75x</b><br><sub>286 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S vulkan tg32@128</td><td align="center">🟡 <b>0.66x</b><br><sub>236 tok/s</sub></td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 cuda tg128@512</td><td align="center">🟡 <b>0.61x</b><br><sub>476 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S vulkan tg128@512</td><td align="center">🟡 <b>0.56x</b><br><sub>188 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S cuda tg32@128</td><td align="center">🟡 <b>0.53x</b><br><sub>317 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 vulkan tg32@128</td><td align="center">🟡 <b>0.52x</b><br><sub>102 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 vulkan tg128@512</td><td align="center">🟠 <b>0.50x</b><br><sub>95 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 cuda tg32@128</td><td align="center">🟠 <b>0.48x</b><br><sub>91 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S cuda tg128@512</td><td align="center">🟠 <b>0.45x</b><br><sub>269 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K vulkan tg32@128</td><td align="center">🟠 <b>0.44x</b><br><sub>95 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 cuda tg128@512</td><td align="center">🟠 <b>0.44x</b><br><sub>83 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K cuda tg32@128</td><td align="center">🟠 <b>0.43x</b><br><sub>156 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 vulkan tg32@128</td><td align="center">🟠 <b>0.40x</b><br><sub>57 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K vulkan tg128@512</td><td align="center">🟠 <b>0.40x</b><br><sub>81 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M vulkan tg32@128</td><td align="center">🟠 <b>0.38x</b><br><sub>66 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 vulkan tg128@512</td><td align="center">🟠 <b>0.38x</b><br><sub>51 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M cuda tg32@128</td><td align="center">🟠 <b>0.38x</b><br><sub>102 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K cuda tg128@512</td><td align="center">🟠 <b>0.38x</b><br><sub>132 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 cuda tg32@128</td><td align="center">🟠 <b>0.36x</b><br><sub>36 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M vulkan tg128@512</td><td align="center">🟠 <b>0.36x</b><br><sub>59 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M cuda tg128@512</td><td align="center">🟠 <b>0.35x</b><br><sub>91 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 cuda tg128@512</td><td align="center">🟠 <b>0.35x</b><br><sub>34 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 vulkan tg32@128</td><td align="center">🟠 <b>0.29x</b><br><sub>22 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 vulkan tg128@512</td><td align="center">🟠 <b>0.28x</b><br><sub>21 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 cuda tg32@128</td><td align="center">🟠 <b>0.27x</b><br><sub>80 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 cuda tg128@512</td><td align="center">🟠 <b>0.27x</b><br><sub>77 tok/s</sub></td></tr>
</table>
</details>

<details>
<summary>🟠 <b>Prefill (pp)</b> — vs llama.cpp · 0.29x avg (28 cases)</summary>

<table>
<tr><th align="left">model · backend · shape</th><th>2026-08-04<br><sub>6bc62da</sub></th></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 vulkan pp128</td><td align="center">🔵 <b>1.18x</b><br><sub>10,153 tok/s</sub></td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 cuda pp128</td><td align="center">🟡 <b>0.73x</b><br><sub>14,999 tok/s</sub></td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 vulkan pp512</td><td align="center">🟡 <b>0.59x</b><br><sub>12,993 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K vulkan pp128</td><td align="center">🟠 <b>0.47x</b><br><sub>1,789 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S vulkan pp128</td><td align="center">🟠 <b>0.45x</b><br><sub>3,560 tok/s</sub></td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 cuda pp512</td><td align="center">🟠 <b>0.41x</b><br><sub>18,663 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 vulkan pp128</td><td align="center">🟠 <b>0.34x</b><br><sub>609 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M vulkan pp128</td><td align="center">🟠 <b>0.34x</b><br><sub>1,173 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 vulkan pp128</td><td align="center">🟠 <b>0.33x</b><br><sub>1,135 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S cuda pp128</td><td align="center">🟠 <b>0.32x</b><br><sub>4,330 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K cuda pp128</td><td align="center">🟠 <b>0.31x</b><br><sub>2,996 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K vulkan pp512</td><td align="center">🟠 <b>0.30x</b><br><sub>2,257 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S vulkan pp512</td><td align="center">🟠 <b>0.29x</b><br><sub>2,990 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 vulkan pp512</td><td align="center">🟠 <b>0.28x</b><br><sub>3,283 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M vulkan pp512</td><td align="center">🟠 <b>0.28x</b><br><sub>1,341 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 vulkan pp512</td><td align="center">🟠 <b>0.27x</b><br><sub>1,362 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 vulkan pp512</td><td align="center">🟠 <b>0.26x</b><br><sub>665 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 cuda pp128</td><td align="center">🟠 <b>0.25x</b><br><sub>1,971 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 vulkan pp128</td><td align="center">🟠 <b>0.25x</b><br><sub>2,182 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M cuda pp128</td><td align="center">🟠 <b>0.24x</b><br><sub>1,730 tok/s</sub></td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K cuda pp512</td><td align="center">🟠 <b>0.24x</b><br><sub>3,748 tok/s</sub></td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S cuda pp512</td><td align="center">🟠 <b>0.22x</b><br><sub>4,348 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 cuda pp128</td><td align="center">🔴 <b>0.20x</b><br><sub>906 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 cuda pp512</td><td align="center">🔴 <b>0.18x</b><br><sub>1,876 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M cuda pp512</td><td align="center">🔴 <b>0.18x</b><br><sub>1,782 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 cuda pp512</td><td align="center">🔴 <b>0.17x</b><br><sub>4,034 tok/s</sub></td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 cuda pp512</td><td align="center">🔴 <b>0.16x</b><br><sub>860 tok/s</sub></td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 cuda pp128</td><td align="center">🔴 <b>0.11x</b><br><sub>2,092 tok/s</sub></td></tr>
</table>
</details>

<details>
<summary>📦 <b>Peak VRAM</b> — CUT holds less on 0 of 14 model/backend pairs</summary>

<table>
<tr><th align="left">model · backend</th><th>CUT</th><th>llama.cpp</th><th>ratio</th></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 cuda</td><td align="center">4032 MiB</td><td align="center">2960 MiB</td><td align="center">0.73x</td></tr>
<tr><td align="left">Llama-3.2-1B-Instruct-f16 vulkan</td><td align="center">3810 MiB</td><td align="center">1399 MiB</td><td align="center">0.37x</td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M cuda</td><td align="center">5914 MiB</td><td align="center">2610 MiB</td><td align="center">0.44x</td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q4_K_M vulkan</td><td align="center">5527 MiB</td><td align="center">1204 MiB</td><td align="center">0.22x</td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 cuda</td><td align="center">6678 MiB</td><td align="center">3944 MiB</td><td align="center">0.59x</td></tr>
<tr><td align="left">Llama-3.2-3B-Instruct-Q8_0 vulkan</td><td align="center">6372 MiB</td><td align="center">2018 MiB</td><td align="center">0.32x</td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 cuda</td><td align="center">13052 MiB</td><td align="center">7748 MiB</td><td align="center">0.59x</td></tr>
<tr><td align="left">Mistral-7B-Instruct-v0.3-Q8_0 vulkan</td><td align="center">12741 MiB</td><td align="center">4657 MiB</td><td align="center">0.37x</td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K cuda</td><td align="center">4712 MiB</td><td align="center">1850 MiB</td><td align="center">0.39x</td></tr>
<tr><td align="left">Qwen2.5-1.5B-Instruct-Q6_K vulkan</td><td align="center">4083 MiB</td><td align="center">778 MiB</td><td align="center">0.19x</td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 cuda</td><td align="center">1348 MiB</td><td align="center">566 MiB</td><td align="center">0.42x</td></tr>
<tr><td align="left">SmolLM2-135M-Instruct-Q8_0 vulkan</td><td align="center">898 MiB</td><td align="center">123 MiB</td><td align="center">0.14x</td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S cuda</td><td align="center">2790 MiB</td><td align="center">990 MiB</td><td align="center">0.35x</td></tr>
<tr><td align="left">tinyllama-1.1b-chat-v1.0.Q4_K_S vulkan</td><td align="center">2436 MiB</td><td align="center">449 MiB</td><td align="center">0.18x</td></tr>
</table>
</details>

<sub>Measured on <b>NVIDIA GeForce RTX 3090</b> · newest column 2026-08-04 (<code>6bc62da</code>) · 5 timed runs per measurement, median · absolute tok/s for every case are in <a href="results/model_history.json">results/model_history.json</a></sub>

<!-- MODELBENCH:END -->

**Prefill** processes N prompt tokens into an empty cache; **decode** generates M
tokens with N already in it. They are separate families because they are
separate regimes — prefill is a GEMM that saturates the card, decode is a GEMV
per token bound by weight bandwidth — and a runtime can be strong at one and
weak at the other.

Both sides prefill the same token count, generate the same token count at the
same KV depth, run fully offloaded, warm up, and report a median of five runs.
What could *not* be held equal, and the one place this comparison is weaker than
the vendor one — its correctness gate detects a runner that has collapsed, not
one that has drifted — is in [`model/README.md`](model/README.md).

## Reproducing

```bash
./scripts/bench/vendor_bench.sh --quick        # build + run every vendor pair
python3 scripts/bench/render_bench.py --snapshot   # record a column, redraw this page

./scripts/bench/model_bench.sh --quick             # build + run every model pair
python3 scripts/bench/render_model_bench.py --snapshot
```

`--quick` skips the two largest `sort_radix` sizes, which otherwise dominate the
wall time; `--no-large` additionally drops the model-scale cases that hold up to
20 GB of VRAM. Cases that do not fit the card are skipped automatically. A run
that a vendor SDK is missing for is skipped rather than failed, and shows up as
blank cells rather than a missing page.

`vendor_bench.sh` rebuilds its targets by default, and that matters: every
benchmark binary compiles in the operator variant table and reads
`tuning_data.json` by index at runtime, so a stale binary silently measures a
different kernel than the one you think you are testing.

## What else is here

| path | what it measures | reference |
|---|---|---|
| [`vendor/`](vendor/README.md) | operator vs cuBLAS / cuDNN / CUB / rocBLAS / rocPRIM | external, absolute |
| [`model/`](model/README.md) | whole-model prefill and decode vs llama.cpp | external, absolute |
| [`autotune/`](autotune/) | CUT's own shader variants against each other, to derive `tuning_data.json` | internal, relative |
| [`op_bench.cpp`](op_bench.cpp) | per-operator, per-variant GPU timestamps → JSON | internal |
| [`benchmark.cpp`](benchmark.cpp) | end-to-end operator timings from the public API | internal |

Both external suites are driven from [`scripts/bench/`](../scripts/bench/), which
also holds the exploratory model tooling the two tables grew out of:
`benchmark_gpus.py` (CUT vs llama.cpp across every GPU in the box, all models at
once), `benchmark_compare.py` (single device), `benchmark_memory.py` (RSS +
VRAM), and `interface/python/benchmarks/run_benchmarks.py` (CUT vs PyTorch,
CuPy, JAX, TensorFlow, Warp).

Methodology — how each pair is timed, how correctness is gated, and how to add a
case — is in [`vendor/README.md`](vendor/README.md) for operators and
[`model/README.md`](model/README.md) for models.
