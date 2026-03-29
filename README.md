# ⚡ KVInfer

**A production-grade LLM inference system built from scratch — no llama.cpp, no ONNX Runtime, no inference framework.**  
Custom C++ daemon · Persistent KV-cache · AVX2 SIMD · FastAPI SSE · Live chat UI.

<br/>

[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-KVInfer%20on%20HF%20Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/NOT-OMEGA/KVInfer)
[![Language](https://img.shields.io/badge/C%2B%2B17%20%7C%20Python%203.12-0059CF?style=for-the-badge&logo=cplusplus&logoColor=white)](.)
[![Optimized](https://img.shields.io/badge/AVX2%20%2B%20FMA%20%2B%20OpenMP-FF4444?style=for-the-badge)](.)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)](LICENSE)

<br/>

# System Design

<div align="center">

<img src="https://github.com/user-attachments/assets/aad71f85-c6d8-442a-861c-c38fca648b33" alt="KVInfer Studio" width="100%"/>

<br/>


| Metric | Value |
|:---|:---:|
| 🚀 Throughput | **35.4 tok/s** (CPU-only, 2 vCPU) |
| ⚡ Avg TTFT | **993 ms** |
| 🧠 Model Size | **152.83M parameters** |
| 👥 Concurrent Users | **4 parallel engines** |
| 💾 RAM Budget | **16 GB** (11 GB used) |
| 📦 Dependencies | **Zero inference framework** |

</div>

---

## Table of Contents

- [Why KVInfer](#why-kvinfer)
- [Architecture](#architecture)
- [Performance](#performance)
- [The C++ Inference Engine](#the-c-inference-engine)
- [Session & KV-Cache Design](#session--kv-cache-design)
- [Multi-Engine Pool](#multi-engine-pool)
- [Model & Training](#model--training)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Benchmark Suite](#benchmark-suite)
- [Project Structure](#project-structure)

---

## Why KVInfer

Most inference projects use llama.cpp, ONNX Runtime, or Hugging Face's `generate()`. KVInfer uses none of these.

Every layer of the stack — the matrix kernels, the attention implementation, the KV-cache allocator, the session manager, the streaming protocol — was written from scratch. The goal was not to build the fastest possible chatbot. The goal was to understand, at the systems level, how production inference engines actually work.

The result is a fully functional, deployed chat system that:

- Runs a 152M parameter GPT-2 model at **35.4 tok/s** on 2 CPU cores with no GPU
- Maintains **persistent conversation state** across turns using KV-cache reuse
- Serves **4 concurrent users** without dropping requests
- Streams tokens in **real time** from C++ → FastAPI → browser via SSE
- Costs **$0 to run** on HuggingFace Spaces free/upgraded tier

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KVInfer Studio                           │
│              index.html  ·  dark chat UI                    │
│         live TPS sparkline  ·  benchmark modal              │
└────────────────────────┬────────────────────────────────────┘
                         │  SSE token stream  (text/event-stream)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI  main.py                          │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              EnginePool                             │  │
│   │                                                     │  │
│   │  session-affinity routing  ·  least-load assignment │  │
│   │  asyncio.Lock per engine   ·  TTL-based session GC  │  │
│   │                                                     │  │
│   │   Engine-0   Engine-1   Engine-2   Engine-3         │  │
│   └──────┬───────────┬───────────┬───────────┬──────────┘  │
│          │ stdin/stdout pipe protocol                       │
└──────────┼───────────┼───────────┼───────────┼─────────────┘
           ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────┐
│              inference  (C++ daemon, persistent)            │
│                                                             │
│   ┌──────────────┐   ┌──────────────────────────────────┐  │
│   │  Forward     │   │       Session Manager            │  │
│   │  Pass        │   │                                  │  │
│   │              │   │  SessionState {                  │  │
│   │  LayerNorm   │   │    float* k_cache  // per-layer  │  │
│   │  AVX2 Matmul │   │    float* v_cache  // per-layer  │  │
│   │  MHA         │   │    int    pos                    │  │
│   │  GELU        │   │    double last_used              │  │
│   │  TopK Sample │   │  }                               │  │
│   └──────────────┘   │  LRU eviction @ MAX_SESSIONS     │  │
│                      └──────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │  fread()
                               ▼
                    ┌──────────────────┐
                    │   model.bin      │
                    │  152M × float32  │
                    │  580 MB on disk  │
                    └──────────────────┘
```

### stdin/stdout Protocol

The C++ daemon communicates with Python over a simple line protocol — no gRPC, no sockets, no shared memory:

```
Python → C++   REQUEST|<sess_id>|<token_ids_csv>|<max_new>|<temp>|<top_k>|<stop_csv>
Python → C++   RESET|<sess_id>
Python → C++   QUIT

C++ → Python   READY
C++ → Python   TOKEN <id> <elapsed_ms>
C++ → Python   DONE <count> <total_ms>
C++ → Python   RESET_OK
C++ → Python   ERROR <message>
```

Each `TOKEN` line is forwarded immediately as an SSE event to the browser — achieving true token-by-token streaming from C++ kernel to browser cursor.

---

## Performance

Benchmarks run on HuggingFace Spaces (2 vCPU, 16 GB RAM, Linux, no GPU):

### After Upgrade (16 GB · 4 Engines · 2 vCPU)

| Prompt | Tokens | TTFT | TPS |
|:---|:---:|:---:|:---:|
| Prompt 1 | 8 | 949 ms | 39.4 tok/s |
| Prompt 2 | 50 | 946 ms | 31.0 tok/s |
| Prompt 3 | 50 | 906 ms | 34.8 tok/s |
| Prompt 4 | 50 | 1056 ms | 35.1 tok/s |
| Prompt 5 | 24 | 1109 ms | 36.5 tok/s |
| **Average** | — | **993 ms** | **35.4 tok/s** |

### Before Upgrade (8 GB · 3 Engines · 1 vCPU)

| Average TTFT | Average TPS |
|:---:|:---:|
| 1532 ms | 18.8 tok/s |

**Upgrade result: 1.9× throughput improvement, 35% TTFT reduction — from a single config change (`N_ENGINES=4`).**

### Cross-Model Scalability Benchmark (same hardware, same engine)

| Model | Params | TPS | vs 152M |
|:---|:---:|:---:|:---:|
| KVInfer (this repo) | 152M | 21.25 tok/s | 1× baseline |
| LLaMA 3B | 3B | 0.33 tok/s | **64× slower** |

> This benchmark quantifies empirically why quantization, GPU offloading, and speculative decoding exist. CPU-only FP32 inference hits a hard wall at 3B+ parameter scale.

---

## The C++ Inference Engine

The engine implements the full GPT-2 decoder forward pass — one token at a time, autoregressive, using the KV-cache for all past context.

### AVX2 + FMA Matrix Kernel

The bottleneck in every transformer layer is matrix-vector multiplication (`y = W·x`). The custom kernel processes **8 float32 values per CPU instruction** using 256-bit SIMD registers:

```cpp
static void matmul_vec(float* out, const float* mat, const float* x, int M, int K) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const float* row = mat + (long long)i * K;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j <= K - 8; j += 8)
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(row + j),   // load 8 weights
                _mm256_loadu_ps(x + j),     // load 8 activations
                acc                          // accumulate
            );
        // horizontal reduce + scalar tail
        float tmp[8]; _mm256_storeu_ps(tmp, acc);
        float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; j < K; j++) s += row[j] * x[j];
        out[i] = s;
    }
}
```

OpenMP parallelizes across output rows — each thread handles an independent slice of the output vector with no shared writes or false sharing.

The same AVX2 + OpenMP pattern is applied to: attention projections, output projections, MLP `fc` and `proj` layers, GELU activation, and residual additions.

### Weight Format

Weights are serialized from PyTorch in a flat binary format — no pickle, no safetensors, no framework dependency at inference time:

```
[n_layer | n_head | n_embd | block_size | vocab_size]  ← 5 × int32 header
[wte: vocab_size × n_embd]                             ← token embeddings
[wpe: block_size × n_embd]                             ← position embeddings
[per layer: ln1_w, ln1_b, c_attn_w, c_attn_b,         ← 12 tensors × 16 layers
            c_proj_w, c_proj_b, ln2_w, ln2_b,
            fc_w, fc_b, mlp_proj_w, mlp_proj_b]
[ln_f_w, ln_f_b, lm_head_w]                           ← final norm + head
```

The C++ engine reads this file with a single `fread()` call and maps pointers directly into the buffer — **zero-copy weight loading**.

---

## Session & KV-Cache Design

Each session maintains a private KV-cache that persists across conversation turns:

```cpp
struct SessionState {
    float*  k_cache   = nullptr;   // [n_layer × block_size × n_embd]
    float*  v_cache   = nullptr;   // [n_layer × block_size × n_embd]
    int     pos       = 0;         // current context length
    double  last_used = 0.0;       // for LRU eviction
};
```

**KV-cache size per session:**
```
16 layers × 1024 tokens × 768 dim × 4 bytes × 2 (K+V) = 96 MB
```

**What this means for multi-turn chat:**

| Without KV-cache | With KV-cache |
|:---|:---|
| Turn 3 re-processes turns 1+2+3 | Turn 3 only processes new tokens |
| Prefill cost grows as O(n²) | Prefill cost is O(1) per new turn |
| Latency increases every turn | TTFT stays constant across turns |

**LRU eviction:** when `MAX_SESSIONS` is reached, the session with the oldest `last_used` timestamp is evicted — its KV-cache is `free()`d and its slot is reused for the new session.

**TTL-based GC (Python side):** a background asyncio task runs every 5 minutes and evicts sessions idle for more than 30 minutes, returning memory to the OS.

---

## Multi-Engine Pool

A single C++ process handles one request at a time (no internal concurrency — by design, for predictable latency). True parallelism comes from the engine pool:

```python
class EnginePool:
    engines:       list[InferenceEngine]   # N independent C++ processes
    _locks:        list[asyncio.Lock]      # one lock per engine
    _session_map:  dict[str, int]          # session_id → engine_index
    _engine_load:  list[int]              # active session count per engine
```

**Session affinity:** new sessions are assigned to the least-loaded engine and stay on that engine for their lifetime. This ensures KV-cache on the correct process is always warm.

**Graceful queuing:** when all engines are busy, requests queue behind the asyncio lock and are served in order — no drops, no 503 errors.

**RAM budget (16 GB space):**

```
4 engines × 580 MB model weights    =  2.32 GB
4 engines × 20 sessions × 96 MB KV =  7.68 GB
Python + FastAPI + OS               ~  1.00 GB
─────────────────────────────────────────────
Total                               ≈ 11.00 GB   (5 GB headroom)
```

**Speed mode** (single user, maximum throughput):
```bash
N_ENGINES=1 OMP_NUM_THREADS=2 uvicorn main:app ...
```
Both CPU cores serve one engine → ~1.5–2× faster per-request TPS.

---

## Model & Training

### Architecture

| Parameter | Value |
|:---|:---:|
| Architecture | GPT-2 Decoder-Only |
| Parameters | **152.83M** |
| Layers | 16 |
| Attention heads | 12 |
| Embedding dim | 768 |
| Context length | 1024 tokens |
| Vocab size | 50,304 |

### Training Data

Three instruction-tuning datasets, unified into a consistent plain-text chat format:

| Dataset | Examples | Type |
|:---|:---:|:---|
| Alpaca Cleaned | 51,760 | Single-turn instruction |
| Databricks Dolly 15k | 15,011 | Single-turn instruction |
| OpenHermes 2.5 (subset) | 500,000 | Multi-turn chat |
| **Total tokens** | **219,090,840** | — |

**Chat format (matches inference template exactly):**
```
System: You are a helpful assistant.
User: What is machine learning?
Assistant: Machine learning is a subfield of...
```

Plain-text role markers (`System:`, `User:`, `Assistant:`) were chosen deliberately — GPT-2's tokenizer encodes them as clean, predictable token sequences. Special tokens like `<|user|>` fragment into 5+ pieces and break conversation structure at inference time.

### Training Setup

- **Hardware:** Google Colab T4 GPU (16 GB VRAM)
- **Precision:** float16 mixed precision with gradient scaling
- **Steps:** 20,000 (planned for 50,000; checkpointed every 5,000)
- **Checkpointing:** full state (model + optimizer + scaler) saved to Google Drive every 5,000 steps — Colab session disconnections lose zero progress
- **Data pipeline:** numpy `memmap` — 219M tokens streamed from disk, never fully loaded into RAM

---

## Quick Start

### Prerequisites

- GCC with AVX2 support (Linux) or MSVC (Windows)
- Python 3.10+
- `model.bin` and `tokenizer.bin` from [HuggingFace](https://huggingface.co/NOT-OMEGA/KVInfer-152M)

### 1. Compile the C++ Engine

**Linux (GCC):**
```bash
g++ -O3 -march=native -fopenmp -ffast-math -std=c++17 -o inference inference.cpp
```

**Windows (MSVC):**
```bash
cl /O2 /openmp /arch:AVX2 /fp:fast /std:c++17 /EHsc /Fe:inference.exe inference.cpp
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model Weights

Weights are auto-downloaded from HuggingFace on first startup. Or manually:
```bash
huggingface-cli download NOT-OMEGA/KVInfer-152M model.bin tokenizer.bin --local-dir .
```

### 4. Start the Server

```bash
# Default: 4 engines, 1 OMP thread each (balanced multi-user)
uvicorn main:app --host 0.0.0.0 --port 7860

# Speed mode: 1 engine, 2 OMP threads (single user, max TPS)
N_ENGINES=1 OMP_NUM_THREADS=2 uvicorn main:app --host 0.0.0.0 --port 7860
```

### 5. Open KVInfer Studio

Navigate to `http://localhost:7860` — the dark chat UI loads automatically.

---

## API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Serves KVInfer Studio (index.html) |
| `GET` | `/health` | Engine pool status, RAM usage, uptime |
| `GET` | `/pool/status` | Per-engine load, busy state, session count |
| `POST` | `/chat` | **Streaming SSE chat** (main endpoint) |
| `POST` | `/chat/reset` | Clear session KV-cache |
| `GET` | `/chat/history` | Full turn history + `tokens_in_engine` |
| `GET` | `/metrics` | Server-wide counters (TPS, errors, RAM) |

### POST /chat

```json
{
  "message":        "Explain transformers simply.",
  "session_id":     "uuid-v4",
  "system_prompt":  "You are a helpful assistant.",
  "max_new_tokens": 200,
  "temperature":    0.7,
  "top_k":          40
}
```

**Response:** `text/event-stream`

```
data: {"type": "token", "id": 13466, "text": "Trans", "elapsed_ms": 312.4}
data: {"type": "token", "id": 23914, "text": "formers", "elapsed_ms": 344.1}
...
data: {"type": "done", "total_tokens": 147, "total_ms": 4152.3, "tps": 35.4, "session_id": "..."}
data: [DONE]
```

---

## Benchmark Suite

`benchmark.py` runs 7 measurement phases:

| Phase | What It Measures |
|:---|:---|
| 1 | Warm-up runs (discarded from stats) |
| 2 | **Cold vs Warm TTFT** — quantifies KV-cache speedup across turns |
| 3 | Short / Medium / Long prompt throughput |
| 4 | **Long context stress** (~400 token input, tests O(n²) attention scaling) |
| 5 | **Concurrency** — 2 simultaneous requests, measures interference |
| 6 | **Latency percentiles** — p50, p95, p99 per-token latency |
| 7 | ASCII throughput sparkline over time |

```bash
python benchmark.py
```

The Quick Benchmark in KVInfer Studio (the `⊞ Benchmark` button) runs 5 standardized prompts inline and shows per-prompt TTFT and TPS with color-coded results.

---

## Project Structure

```
kvinfer/
├── inference.cpp      C++ daemon — AVX2+FMA, OpenMP, session KV-cache, LRU eviction
├── inference.exe      Compiled binary (Windows) — compile from source for your platform
├── main.py            FastAPI backend — engine pool, session state, SSE streaming, GC
├── benchmark.py       7-phase benchmark suite — cold/warm/long-ctx/concurrency/percentiles
├── index.html         KVInfer Studio — dark chat UI, live TPS sparkline, benchmark modal
├── model.bin          152M weights in flat float32 binary format (580 MB)
├── tokenizer.bin      GPT-2 vocabulary in binary format (for C++ tokenizer)
├── requirements.txt
├── SETUP_GUIDE.md
└── README.md
```

---

## Design Decisions

**Why write a C++ engine instead of using llama.cpp?**  
llama.cpp is excellent software. Using it would have taught me its API. Writing a custom engine taught me what a transformer forward pass actually computes at the arithmetic level, why KV-cache exists, how session memory is managed, and how to design a streaming protocol. The goal was understanding, not benchmarks.

**Why plain-text role markers instead of special tokens?**  
GPT-2's tokenizer has no special tokens for `<|user|>` — it fragments them into 5+ pieces. Training on fragmented role boundaries and then detecting clean string markers at inference creates a tokenization mismatch that causes role-bleed (the model generates `User:` mid-response). Plain-text markers (`User:`, `Assistant:`) tokenize predictably and the same way in both training and inference.

**Why a stdin/stdout protocol instead of sockets or shared memory?**  
Simplicity and debuggability. The protocol is human-readable, trivial to test with `echo`, and requires no IPC library. For this latency target (sub-second TTFT), pipe overhead is negligible.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

<div align="center">

Built from scratch · No inference framework · Every line is intentional.

**[Live Demo →](https://huggingface.co/spaces/NOT-OMEGA/KVInfer)**

</div>
