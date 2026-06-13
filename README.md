# Orthrus llama.cpp

This is an experimental llama.cpp fork with native Orthrus-Qwen3 support.

The main goal is not just loading Orthrus weights as Qwen3. The important part is Orthrus shared-KV self-drafting: the autoregressive path owns the persistent target KV cache, the diffusion view drafts against that same cache, transient diffusion cells are removed before verification, and normal AR verification commits or rolls back the candidate path.

This repository is intended as a working reference implementation. If upstream llama.cpp maintainers want any part of it, they can lift, rewrite, split, or improve it however they prefer.

## Model Files

Prebuilt 8B Q4_K_M GGUF and the official chat template are hosted here:

```text
https://huggingface.co/originalGeek/Orthrus-Qwen3-8B-GGUF
```

The hosted GGUF still requires this fork, or equivalent upstream Orthrus support, for `--spec-type draft-orthrus`.

## What Is Included

- GGUF conversion for `OrthrusLM`.
- Dedicated GGUF architecture name: `orthrus`.
- Orthrus metadata:
  - `orthrus.diffusion.block_size`
  - `tokenizer.ggml.mask_token_id`
- Tensor mapping/loading for all Orthrus diffusion tensors:
  - `q_proj_diff`, `k_proj_diff`, `v_proj_diff`, `o_proj_diff`
  - `q_norm_diff`, `k_norm_diff`
- Qwen3-compatible AR fallback path.
- Orthrus diffusion decoder graph using the `_diff` attention projections/norms.
- `draft-orthrus` speculative decoding with target-context shared-KV drafting.
- `llama-server` wiring so Orthrus drafting uses the target context directly, with no separate draft model or draft historical KV cache.
- Quantization category handling for Orthrus diffusion projection tensors.
- Docs/help listing updates and a small local quantization test.

## Shared-KV Runtime Design

Orthrus is a dual-view Qwen3-derived architecture. At generation time this fork does:

1. Prefill/decode normally through the AR Qwen3 path.
2. Build a diffusion draft block from the last committed token plus mask tokens.
3. Run the Orthrus diffusion graph on the target context with non-causal attention for that pass.
4. Use the same target KV cache as the temporary coordination object.
5. Remove transient diffusion block cells before AR verification.
6. Verify candidates with the normal target AR path.
7. Commit accepted tokens and crop rejected suffixes using existing sequence-removal machinery.

That is the key distinction from ordinary external-draft speculative decoding. There is no second persistent draft KV cache.

## Current Limitations

- `draft-orthrus` is currently wired through `llama-server`.
- Drafting is serialized per slot on the target context.
- At `--temp 0`, verification is greedy token matching and has been validated against AR-only output.
- At `--temp > 0`, this follows existing llama.cpp speculative token-match semantics, not Orthrus's exact residual rejection-sampling variant.
- Acceptance rate is content-dependent. In local tests, Orthrus drafting is not always faster than plain AR for every prompt.

## Build

CPU build:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DLLAMA_CURL=OFF
cmake --build build --config Release --target llama-server llama-quantize
```

CUDA build:

```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build-cuda --config Release --target llama-server
```

The local validation machine used CUDA 13.3 and an RTX 5090. CMake detected `CMAKE_CUDA_ARCHITECTURES=120a-real`.

## Convert Orthrus Models

Install converter dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-convert_hf_to_gguf.txt
```

Download an official Orthrus checkpoint:

```powershell
.\.venv\Scripts\hf.exe download chiennv/Orthrus-Qwen3-8B --local-dir models\Orthrus-Qwen3-8B
```

Convert to F16 GGUF:

```powershell
$env:PYTHONPATH = "$PWD\gguf-py"
.\.venv\Scripts\python.exe convert_hf_to_gguf.py `
  --outtype f16 `
  --outfile models\Orthrus-Qwen3-8B\Orthrus-Qwen3-8B-F16.gguf `
  models\Orthrus-Qwen3-8B
```

Optional Q4_K_M quantization:

```powershell
.\build\bin\Release\llama-quantize.exe `
  models\Orthrus-Qwen3-8B\Orthrus-Qwen3-8B-F16.gguf `
  models\Orthrus-Qwen3-8B\Orthrus-Qwen3-8B-Q4_K_M.gguf `
  Q4_K_M
```

No model weights are committed to this repository.

## Run The Server

Example tuned local run for 8B Q4_K_M on a single RTX 5090:

```powershell
.\build-cuda\bin\Release\llama-server.exe `
  -m models\Orthrus-Qwen3-8B\Orthrus-Qwen3-8B-Q4_K_M.gguf `
  --host 127.0.0.1 `
  --port 8081 `
  -c 40960 `
  -np 1 `
  -b 1024 `
  -ub 512 `
  -ngl 999 `
  -fa on `
  --ui `
  --jinja `
  --chat-template-file models\Orthrus-Qwen3-8B\chat_template.jinja `
  --spec-type draft-orthrus `
  --spec-draft-n-max 15
```

Notes:

- The official Orthrus/Qwen chat template is embedded into the converted GGUF.
- Passing `--chat-template-file` is still useful when the Hugging Face model folder is available, because it makes the intended template explicit.
- For pure AR baseline, use `--spec-type none`.
- `--spec-draft-n-max 15` was the best average tested locally for 8B Q4_K_M. Auto resolves to `block_size - 1` (`31`) and may be better or worse depending on prompts.

## Local Validation

Official checkpoint conversion:

| Model | GGUF | Tensor count | Diff tensors | Result |
| --- | --- | ---: | ---: | --- |
| Orthrus-Qwen3-1.7B | F16 | 478 | 168 | Passed |
| Orthrus-Qwen3-4B | F16 | 614 | 216 | Passed |
| Orthrus-Qwen3-8B | F16 | 615 | 216 | Passed |
| Orthrus-Qwen3-8B | Q4_K_M | 615 | 216 | Passed |

Metadata checked:

- `general.architecture = orthrus`
- `orthrus.diffusion.block_size = 32`
- `tokenizer.ggml.mask_token_id = 151669`
- all expected per-layer `_diff` tensors present
- 8B separate `output.weight` retained

Runtime checks:

- `llama-server` loaded 1.7B F16, 1.7B Q4_K_M, 8B F16, and 8B Q4_K_M.
- `draft-orthrus` startup logged:
  - `shared_kv=yes`
  - `block_size=32`
  - `mask_id=151669`
  - auto `n_max=31` or tuned `n_max=15`
- Temp-0 decoded output matched AR-only output in controlled server tests.
- CUDA build was verified on RTX 5090 with GPU offload enabled.
- The 8B Q4_K_M GGUF chat template matched the provided `chat_template.jinja`.

Build/test checks:

- `git diff --check`
- `python -m py_compile conversion/qwen.py`
- release `llama-server` build
- release `llama-quantize` build
- generated docs build
- `test-orthrus-quant`
- `ctest -R test-orthrus-quant`

## Local Benchmark Snapshot

Official Orthrus runtime cross-check:

The Orthrus team's own Transformers implementation was benchmarked locally with the BF16 source checkpoint, PyTorch 2.11.0+cu128, Transformers 5.12.0, `attn_implementation="sdpa"`, and temp-0 generation on the same RTX 5090. These numbers are not a quantization-to-quantization comparison against the GGUF, but they confirm that the intended shared-KV diffusion path is worthwhile.

| Case | Official HF AR | Official HF Orthrus | Accepted / Drafted | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Raw short, 256 tokens | 32.2 t/s | 352.6 t/s | 245 / 320 | 76.6% |
| Raw technical, 512 tokens | 33.3 t/s | 163.1 t/s | 462 / 1541 | 30.0% |
| Chat technical, 512 tokens | 27.4 t/s | 48.3 t/s | 370 / 4242 | 8.7% |
| Chat long context, 6.5k prompt + 256 output | 3.8 t/s wall / 9.0 t/s decode | 5.9 t/s wall / 16.0 t/s decode | 175 / 2349 | 7.5% |

llama.cpp fork benchmark:

Environment:

- RTX 5090
- CUDA 13.3
- `-np 1`
- `-c 40960`
- `-ngl 999`
- `-fa on`
- raw `/completion` temp-0 benchmark prompts

Wall TPS:

| Config | Short | Technical | Long Context | Average |
| --- | ---: | ---: | ---: | ---: |
| 8B F16 AR | 92 | 84 | 72 | 82 |
| 8B F16 `draft-orthrus` auto | 138 | 74 | 67 | 93 |
| 8B Q4_K_M AR | 220 | 215 | 134 | 190 |
| 8B Q4_K_M `draft-orthrus` auto 31 | 174 | 319 | 89 | 194 |
| 8B Q4_K_M `draft-orthrus` nmax 7 | 173 | 222 | 98 | 164 |
| 8B Q4_K_M `draft-orthrus` nmax 15 | 200 | 340 | 110 | 217 |

Acceptance for the tuned 8B Q4_K_M `nmax 15` run:

| Prompt | Accepted / Drafted | Acceptance |
| --- | ---: | ---: |
| Short | 202 / 778 | 26.0% |
| Technical | 450 / 896 | 50.2% |
| Long context | 182 / 1066 | 17.1% |

Interpretation:

- The tuned 8B Q4_K_M Orthrus config was the best average local benchmark.
- Plain Q4_K_M AR was faster on the long-context raw-completion case.
- Orthrus acceptance varies substantially with prompt/content, so benchmark your intended workload.

## Files Of Interest

- `conversion/qwen.py`
- `conversion/__init__.py`
- `gguf-py/gguf/constants.py`
- `gguf-py/gguf/tensor_mapping.py`
- `src/models/qwen3.cpp`
- `src/llama-context.cpp`
- `common/speculative.cpp`
- `tools/server/server-context.cpp`
- `tests/test-orthrus-quant.cpp`

## Upstream

This fork is based on llama.cpp commit:

```text
70b54e140c90a92285ba699d77e1e32e0868a0e2
```

Original project:

```text
https://github.com/ggml-org/llama.cpp
```

Orthrus models/code:

```text
https://github.com/chiennv2000/orthrus
https://huggingface.co/chiennv
```
