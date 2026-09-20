# Looped vs. Non-Looped SLM — Benchmark

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Runs on **CPU only** — no GPU required. `torch` auto-detects CUDA/MPS if
present and falls back to CPU otherwise.

## Run

```bash
python benchmark.py
```

First run downloads both models from Hugging Face (~few GB).

## Sample execution

```
========================================================================
LOOPED SLM EXPERIMENT
========================================================================
Python  : 3.11.8
PyTorch : 2.4.0
Device  : cpu
Hardware: CPU

========================================================================
Nanbeige4.2-3B (looped)
========================================================================

Test 1: Ordering
  expected : David
  received : 'David'
  result   : PASS
  latency  : 61.14s

...

Accuracy : 4/4 (100%)
Avg time : 58.30s

========================================================================
SUMMARY
========================================================================
Model                             Accuracy     Avg latency
------------------------------------------------------------------------
Nanbeige4.2-3B (looped)       4/4            32.92s
Qwen3.5-4B                    4/4            26.27s
```

Expect several minutes total on CPU — `MAX_NEW_TOKENS=512` with no GPU
kernel acceleration is slow by design (see script comments).