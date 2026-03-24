# ⚡ KVInfer

[![Language](https://img.shields.io/badge/language-C%2B%2B17%20%7C%20Python-0059C?style=for-the-badge)](.)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-FCC624?style=for-the-badge&logo=windows)](.)
[![Optimized](https://img.shields.io/badge/optimized-AVX2%20%2B%20OpenMP-red?style=for-the-badge)](.)
[![Frontend](https://img.shields.io/badge/frontend-KVInfer%20Studio-5b8dee?style=for-the-badge)](.)

**KVInfer** is a production-grade **152M parameter chat LM inference system** built from scratch.  
Custom C++ daemon engine · FastAPI SSE backend · KVInfer Studio dark chat frontend.

> **Architecture:** GPT-2 Decoder-Only · 16 layers · 768 embd · 12 heads · 1024 context

---
<img width="1296" height="824" alt="image" src="https://github.com/user-attachments/assets/ee370e1e-7ff5-4fa6-993d-fe5d0bd60205" />

## System Overview

```
KVInfer Studio (index.html)
        │  SSE token stream
        ▼
  FastAPI  main.py          ← session management, string stop detection
        │  asyncio.Lock
        ▼
  inference.exe             ← persistent daemon, model loaded ONCE
        │  stdin/stdout pipe
        │  AVX2 + FMA + OpenMP
        │  per-session KV cache (up to 4, LRU evict)
        ▼
  model.bin                 ← 152M float32 weights
```

---

## What Makes KVInfer Different

| Feature | Detail |
|---|---|
| **Persistent daemon** | C++ process never restarts — model stays in RAM |
| **Session KV cache** | Each session keeps its KV state; new turns only process new tokens |
| **Incremental prefill** | O(1) per turn instead of O(n²) full-history re-encoding |
| **String stop detection** | Catches `"User:"` / `"System:"` role bleed before it reaches the client |
| **AVX2 + FMA matmul** | 8 floats per CPU instruction via `_mm256_fmadd_ps` |
| **OpenMP parallelism** | Multi-threaded matmul, multi-head attention, GELU, residuals |
| **Exact token accounting** | Uses C++ `total_tokens` count — no re-tokenization drift |

---

## Quick Start

```bash
# 1. Compile the daemon
cl /O2 /openmp /arch:AVX2 /fp:fast /std:c++17 /EHsc /Fe:inference.exe inference.cpp

# 2. Install Python deps
pip install -r requirements.txt

# 3. Start server
uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Open KVInfer Studio
#    Open index.html in Chrome / Edge

# 5. Run full benchmark suite
python benchmark.py
```

---

## Benchmark Suite Phases

| Phase | What it tests |
|---|---|
| 1 | Warm-up (discarded) |
| 2 | Cold vs Warm TTFT — KV cache speedup measurement |
| 3 | Short / Medium / Long prompt throughput |
| 4 | Long context stress (~400 token input) |
| 5 | Concurrency (2 simultaneous requests) |
| 6 | p50 / p95 / p99 percentile stats |
| 7 | ASCII throughput chart over time |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Engine + memory status |
| `POST` | `/chat` | SSE streaming chat |
| `POST` | `/chat/reset` | Clear session |
| `GET` | `/chat/history` | Full session history + `tokens_in_engine` |
| `POST` | `/generate` | Non-streaming generation |
| `GET` | `/metrics` | Server-wide counters |
| `GET` | `/benchmark/run` | Quick 5-prompt benchmark (used by Studio) |

---

## Chat Format (matches SFT training)

```
System: You are a helpful assistant.
User: What is machine learning?
Assistant: Machine learning is...
```

---

## Project Structure

```
kvinfer/
├── inference.cpp    C++ daemon — AVX2+FMA+OpenMP, session KV cache, LRU evict
├── inference.exe    Compiled binary (compile from inference.cpp)
├── main.py          FastAPI — persistent engine, session state, string stop detection
├── benchmark.py     Full benchmark suite — cold/warm/long/concurrency/percentiles
├── index.html       KVInfer Studio — dark chat UI, sparkline, benchmark modal
├── model.bin        152M weights (from your SFT training)
├── requirements.txt
├── SETUP_GUIDE.md
└── README.md
```

---

## License

Apache License 2.0
