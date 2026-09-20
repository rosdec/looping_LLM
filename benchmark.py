"""
Tiny experiment: Looped vs. non-looped SLMs

Question:
    Can a smaller looped model achieve similar results to a somewhat
    larger conventional/non-looped model?

This is deliberately NOT a scientific benchmark.
It is a small, reproducible demonstration for running locally.

Install:

    pip install torch transformers accelerate

Run:

    python benchmark.py

The first run downloads the models from Hugging Face.

The models are loaded sequentially, so only one model is resident
in memory at a time.
"""

import re
import sys
import time
import platform

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Community-patched checkpoint: fixes several transformers-compatibility bugs
# in the stock Nanbeige/Nanbeige4.2-3B custom modeling code (a stale
# Cache.get_max_length() call, an inv_freq buffer that silently loads as all
# zeros, etc). Same architecture/weights otherwise. See its model card before
# using it unattended.
NANBEIGE = "johnhalloran/Nanbeige4.2-3B-mps-fix"
QWEN = "Qwen/Qwen3.5-4B"

# Thinking models spend tokens on chain-of-thought before answering. 32 new
# tokens is nowhere near enough for that, so the model runs out of budget
# mid-reasoning and never produces the one-line answer the prompt asks for.
# Bump the budget and turn thinking off so short-answer grading is meaningful.
MAX_NEW_TOKENS = 512


# ---------------------------------------------------------------------------
# Tiny reasoning experiment
# ---------------------------------------------------------------------------

TESTS = [
    {
        "name": "Ordering",
        "prompt": """
Alice is taller than Bob.
Bob is taller than Charlie.
Charlie is taller than David.

Who is the shortest?

Answer with only the person's name.
""",
        "answer": "David",
    },
    {
        "name": "Chain",
        "prompt": """
A implies B.
B implies C.
C implies D.
D implies E.

If A is true, which letter must also be true?

Answer with only the letter.
""",
        "answer": "E",
    },
    {
        "name": "Spatial chain",
        "prompt": """
The red object is north of the blue object.
The blue object is north of the green object.
The green object is north of the yellow object.

Where is the red object relative to the yellow object?

Answer with only: north, south, east, or west.
""",
        "answer": "north",
    },
    {
        "name": "Irrelevant information",
        "prompt": """
Marco owns a red car.
The red car is faster than the blue car.
The blue car is faster than the green car.
Marco's sister owns a yellow bicycle.
The bicycle is newer than the car.

Which car is the slowest?

Answer with only the color of the car.
""",
        "answer": "green",
    },
]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


def device_name():
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(DEVICE)

    if DEVICE.type == "mps":
        return "Apple Silicon / MPS"

    return "CPU"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize(text):
    """
    Normalize model output so that simple answers can be compared
    robustly.

    We deliberately keep this conservative: this is not an LLM judge.
    """
    text = text.strip().lower()

    # Strip a <think>...</think> block if the model produced one but still
    # kept generating past it (thinking models often do both).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove common markdown decoration.
    text = re.sub(r"[*`_]", "", text)

    # Prefer the LAST non-empty line: short-answer prompts often get a
    # rambling restatement first and the actual answer at the end, once
    # there's enough budget for the model to reach it.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[-1] if lines else ""

    return text


def matches(output, expected):
    """
    Check whether the expected answer appears as a whole word in the
    (normalized) last line of output.

    Word-boundary matching matters here: plain substring containment
    (`expected in output`) gives false positives for single-letter or
    short answers -- e.g. expected="E" will "match" almost any English
    sentence, since it's the most common letter in the language. That
    silently turns FAILs into PASSes.
    """
    output = normalize(output)
    expected = normalize(expected)

    if not expected:
        return False

    pattern = r"\b" + re.escape(expected) + r"\b"
    return re.search(pattern, output) is not None


# ---------------------------------------------------------------------------
# Nanbeige
# ---------------------------------------------------------------------------

class NanbeigeDynamicCache(DynamicCache):
    def get_max_length(self, layer_idx=None):
        if layer_idx is None and not self.layers:
            return None
        return super().get_max_length(layer_idx)


def load_nanbeige():
    print(f"Loading {NANBEIGE} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        NANBEIGE,
        trust_remote_code=True,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        NANBEIGE,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer


def run_nanbeige(model, tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            past_key_values=NanbeigeDynamicCache(),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


# ---------------------------------------------------------------------------
# Qwen 3.5
# ---------------------------------------------------------------------------

def load_qwen():
    print(f"Loading {QWEN} ...")

    # Qwen3.5 is a multimodal model, but we use only its text pathway.
    #
    # Import here so that the script fails with a useful message if the
    # installed Transformers version is too old.
    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError:
        print(
            "\nYour version of Transformers does not contain "
            "Qwen3_5ForConditionalGeneration."
        )
        print("Upgrade Transformers with:")
        print("    pip install -U transformers")
        sys.exit(1)

    processor = AutoProcessor.from_pretrained(
        QWEN,
        trust_remote_code=True,
    )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN,
        dtype="auto",
    )

    model.to(DEVICE)
    model.eval()

    return model, processor


def run_qwen(model, processor, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    inputs = processor(
        text=[text],
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
        if hasattr(v, "to")
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]

    return processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_model(name, loader, runner):
    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    model, tokenizer_or_processor = loader()

    results = []
    correct = 0
    total_time = 0.0

    for i, test in enumerate(TESTS, 1):
        print()
        print(f"Test {i}: {test['name']}")

        start = time.perf_counter()

        answer = runner(
            model,
            tokenizer_or_processor,
            test["prompt"],
        )

        elapsed = time.perf_counter() - start

        ok = matches(answer, test["answer"])

        if ok:
            correct += 1

        total_time += elapsed

        print(f"  expected : {test['answer']}")
        print(f"  received : {answer!r}")
        print(f"  result   : {'PASS' if ok else 'FAIL'}")
        print(f"  latency  : {elapsed:.2f}s")

        results.append(ok)

    accuracy = correct / len(TESTS)

    print()
    print(f"Accuracy : {correct}/{len(TESTS)} ({accuracy:.0%})")
    print(f"Avg time : {total_time / len(TESTS):.2f}s")

    # Free memory before loading the next model.
    del model
    del tokenizer_or_processor

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return {
        "name": name,
        "correct": correct,
        "total": len(TESTS),
        "accuracy": accuracy,
        "avg_time": total_time / len(TESTS),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print("LOOPED SLM EXPERIMENT")
    print("=" * 72)

    print(f"Python  : {platform.python_version()}")
    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {DEVICE}")
    print(f"Hardware: {device_name()}")

    print()
    print("The experiment compares:")
    print(f"  Looped model      : {NANBEIGE}")
    print(f"  Non-looped model  : {QWEN}")
    print()
    print("Four small multi-step reasoning problems.")
    print("Models are loaded one at a time.")
    print()

    results = []

    # Run the looped model first.
    results.append(
        run_model(
            "Nanbeige4.2-3B (looped)",
            load_nanbeige,
            run_nanbeige,
        )
    )

    # Then the comparison model.
    results.append(
        run_model(
            "Qwen3.5-4B",
            load_qwen,
            run_qwen,
        )
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"{'Model':<30}"
        f"{'Accuracy':>12}"
        f"{'Avg latency':>16}"
    )

    print("-" * 72)

    for result in results:
        print(
            f"{result['name']:<30}"
            f"{result['correct']}/{result['total']}"
            f"{'':>7}"
            f"{result['avg_time']:>10.2f}s"
        )


if __name__ == "__main__":
    main()