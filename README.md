# MathReasoning-Aligner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework-TRL](https://img.shields.io/badge/Alignment-TRL%20%2F%20DPO-orange.svg)](https://github.com/huggingface/trl)

An end-to-end alignment framework engineered for frontier LLMs to achieve deterministic precision in mathematical and symbolic reasoning tasks.

---

## Key Capabilities

- **Synthetic CoT Generation:** Programmatic creation of high-throughput Chain-of-Thought (CoT) dataset traces using symbolic execution engines (SymPy).
- **Symbolic Ground-Truth Validation:** Zero-hallucination verification pipeline comparing model outputs with canonical symbolic expressions.
- **Alignment Framework:** Fine-tuning pipeline incorporating Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

---

## Repository Structure

```text
MathReasoning-Aligner/
├── data/
│   ├── generator.py    # Synthetic CoT data generation engine (SymPy)
│   └── validator.py    # Deterministic symbolic equivalence verification
├── alignment/
│   ├── train_sft.py    # Supervised Fine-Tuning entry point
│   └── train_dpo.py    # Preference Alignment entry point
├── evaluation/
│   └── eval_math.py    # Mathematical accuracy benchmarks
└── requirements.txt    # Project dependencies
